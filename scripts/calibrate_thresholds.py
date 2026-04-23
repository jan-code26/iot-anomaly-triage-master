"""
Per-dataset alert threshold calibration.

The global threshold of 0.3 was tuned on FD001 (single operating condition).
On FD002, regime-aware causal scores are smaller by design (conditioning
removes regime variance), so the same 0.3 threshold under-alerts.

This script calibrates the optimal alert threshold for FD002 by sweeping
t ∈ {0.05, 0.10, …, 0.50} on the TRAINING set and maximising F1 (W=100).

A single threshold is calibrated per dataset (not per-cluster), because in
FD002 each engine visits multiple clusters across its lifecycle — an engine's
first alert can come from any cluster, so a per-engine threshold makes more
sense than splitting by cluster assignment.

Results written to data/processed/regime_coefficients.json under:
    "alert_threshold_fd002": t*

The per-cluster key "alert_thresholds" is also populated with the same value
for each cluster so that get_alert_threshold() in regime_classifier.py works
without modification.

Usage:
    python scripts/calibrate_thresholds.py

IMPORTANT — data partition:
    Calibration:  train_FD002.txt  ← only file loaded here
    Evaluation:   test_FD002.txt   ← NOT TOUCHED
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR  = ROOT / "data" / "processed"
COEFF_FILE = OUT_DIR / "regime_coefficients.json"

COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

N_CLUSTERS = 6
W = 100           # actionable-window (cycles), same as ablation scripts
THRESHOLD_GRID = [round(t / 20, 2) for t in range(1, 11)]   # 0.05 … 0.50

CAUSAL_SENSORS: dict[str, str] = {
    "sensor_4":  "op_setting_1",
    "sensor_11": "op_setting_2",
    "sensor_15": "op_setting_2",
    "sensor_3":  "op_setting_3",
    "sensor_9":  "op_setting_3",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        print(f"ERROR: {path} not found. Run: python scripts/download_cmapss.py")
        sys.exit(1)
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_kmeans(train: pd.DataFrame) -> KMeans:
    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    km.fit(train[op_cols].values)
    return km


def fit_cluster_coefs(train: pd.DataFrame, km: KMeans) -> dict[int, dict]:
    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
    cluster_labels = km.predict(train[op_cols].values)
    train = train.copy()
    train["_cluster"] = cluster_labels
    cluster_coefs: dict[int, dict] = {}
    for c in range(N_CLUSTERS):
        subset = train[train["_cluster"] == c]
        cluster_coefs[c] = {}
        for sensor, cause in CAUSAL_SENSORS.items():
            X = subset[[cause]].values
            y = subset[sensor].values
            if len(X) < 2:
                cluster_coefs[c][sensor] = {"coef": 0.0, "intercept": float(y.mean()), "residual_std": 1.0}
                continue
            model = LinearRegression().fit(X, y)
            residuals = y - model.predict(X)
            cluster_coefs[c][sensor] = {
                "coef": float(model.coef_[0]),
                "intercept": float(model.intercept_),
                "residual_std": max(float(np.std(residuals)), 1e-6),
            }
    return cluster_coefs


def regime_causal_score(row: dict, km: KMeans, cluster_coefs: dict) -> float:
    """Full regime-aware causal score for one reading."""
    ops = np.array([[row.get("op_setting_1", 0) or 0,
                     row.get("op_setting_2", 0) or 0,
                     row.get("op_setting_3", 0) or 0]])
    cluster = int(km.predict(ops)[0])
    coefs = cluster_coefs[cluster]
    z_scores = []
    for sensor, cause in CAUSAL_SENSORS.items():
        val = row.get(sensor)
        if val is None:
            continue
        c = coefs[sensor]
        cause_val = row.get(cause)
        predicted = c["coef"] * cause_val + c["intercept"] if cause_val is not None else c["intercept"]
        std = c["residual_std"]
        if std == 0:
            continue
        z_scores.append(abs(val - predicted) / std)
    if not z_scores:
        return 0.0
    k = min(3, len(z_scores))
    top_k_mean = sum(sorted(z_scores, reverse=True)[:k]) / k
    return min(top_k_mean / 5.0, 1.0)


# ---------------------------------------------------------------------------
# F1 helpers
# ---------------------------------------------------------------------------

def compute_f1(lead_times: pd.Series) -> float:
    tp = int(((lead_times.notna()) & (lead_times <= W)).sum())
    fp = int(((lead_times.notna()) & (lead_times >  W)).sum())
    fn = int(lead_times.isna().sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not COEFF_FILE.exists():
        print(f"ERROR: {COEFF_FILE} not found. Run scripts/compute_regime_coefficients.py first.")
        sys.exit(1)

    # NOTE: test_FD002.txt is NOT loaded — calibration uses training data only.
    print("Loading train_FD002.txt (calibration set — test_FD002.txt not touched)...")
    train = load("train_FD002.txt")

    print(f"Training KMeans (n_clusters={N_CLUSTERS})...")
    km = train_kmeans(train)

    print("Fitting per-cluster causal coefficients...")
    cluster_coefs = fit_cluster_coefs(train, km)

    # Failure cycle = last recorded training cycle (training data runs to failure)
    failure_cycles = (
        train.groupby("engine_id")["cycle"]
        .max().rename("true_failure_cycle").reset_index()
    )

    # ------------------------------------------------------------------
    # Threshold derivation: 90th percentile of early-life (first 30 cycles)
    # causal scores.
    #
    # Rationale: Early cycles represent healthy operating state — no
    # degradation has accumulated yet. The 90th percentile of scores from
    # this window captures the upper bound of NORMAL operating variability
    # (including sensor noise and regime variation). Setting the alert
    # threshold just above this point fires only when a reading deviates
    # more than 90% of healthy readings ever do.
    #
    # This approach:
    #   1. Uses only training data (no test leakage)
    #   2. Is grounded in the physics of healthy operation, not near-failure
    #   3. Generalises to test data (test engines also start healthy)
    #   4. Avoids the train-overfit problem of "p10 of peak scores" which
    #      selects thresholds only reachable near failure
    # ------------------------------------------------------------------
    print(f"\nComputing early-life (first 30 cycles) causal scores for threshold calibration...")
    early_scores = []
    for engine_id, group in train.groupby("engine_id"):
        early_cycles = group[group["cycle"] <= 30]
        for _, row in early_cycles.iterrows():
            s = regime_causal_score(row.to_dict(), km, cluster_coefs)
            early_scores.append(s)

    early_series = pd.Series(early_scores)
    p75 = float(np.percentile(early_series, 75))
    p90 = float(np.percentile(early_series, 90))
    p95 = float(np.percentile(early_series, 95))
    print(f"  Early-life score percentiles: p75={p75:.4f}, p90={p90:.4f}, p95={p95:.4f}")

    # Round to nearest grid step
    raw_t = p90
    best_t = min(THRESHOLD_GRID, key=lambda t: abs(t - raw_t))
    print(f"\n  → Calibrated threshold = {best_t:.2f} (90th-pct of early-life scores = {raw_t:.4f})")
    print(f"     Fires only when a reading exceeds 90% of healthy operating variation.")
    print(f"     Generalises to test data since it is anchored to healthy-state distribution.")

    # Informational: F1 sweep on full training set
    print(f"\nF1 sweep on training data (informational — higher F1 at high t is train-overfit):")
    print(f"\n  {'Threshold':>10}  {'Coverage':>10}  {'F1':>10}")
    print("  " + "-" * 34)
    results = []
    for t in THRESHOLD_GRID:
        records = []
        for engine_id, group in train.groupby("engine_id"):
            first = None
            for _, row in group.sort_values("cycle").iterrows():
                if regime_causal_score(row.to_dict(), km, cluster_coefs) >= t:
                    first = int(row["cycle"])
                    break
            records.append({"engine_id": engine_id, "first_alert_cycle": first})
        alerts = pd.DataFrame(records)
        merged = failure_cycles.merge(alerts, on="engine_id", how="left")
        lt = merged["true_failure_cycle"] - merged["first_alert_cycle"]
        lt.index = merged["engine_id"]
        f1 = compute_f1(lt)
        cov = lt.notna().mean()
        results.append((t, cov, f1))
        marker = " ← chosen" if t == best_t else ""
        print(f"  {t:>10.2f}  {cov:>9.1%}  {f1:>10.4f}{marker}")

    # Write to regime_coefficients.json
    with open(COEFF_FILE) as f:
        data = json.load(f)

    data["alert_threshold_fd002"] = best_t
    # Also populate per-cluster keys (all set to the same value for now)
    # Regime_classifier.get_alert_threshold() reads from "alert_thresholds"
    data["alert_thresholds"] = {str(c): best_t for c in range(N_CLUSTERS)}

    with open(COEFF_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved alert_threshold_fd002={best_t} to {COEFF_FILE}")


if __name__ == "__main__":
    main()
