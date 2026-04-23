"""
Learned blend weight α optimisation.

The live pipeline blends z-score and causal score as:
    final_score = α × causal_score + (1-α) × z_score

Previously α was hardcoded at 0.5.  This script sweeps α ∈ [0, 1] on the
**training** sets for FD001 and FD002 separately, computing F1 at each α.
The F1 definition matches ablation_study.py and fd002_regime_eval.py:
    TP:  alert fired AND lead_time ≤ W  (alert is actionable)
    FP:  alert fired AND lead_time > W  (premature alert)
    FN:  no alert raised

W = 100 cycles (matches the evaluation scripts).
Alert threshold = 0.3 (matches the live pipeline).

The optimal α for each dataset is written to:
    data/processed/regime_coefficients.json
under keys "blend_alpha_fd001" and "blend_alpha_fd002".

Usage:
    python scripts/optimize_blend_weight.py

IMPORTANT — data partition:
    Calibration: train_FD001.txt, train_FD002.txt
    Evaluation:  test_FD001.txt, test_FD002.txt  ← NOT TOUCHED HERE
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

ALERT_THRESHOLD = 0.3
W = 100          # actionable-window (cycles)
N_CLUSTERS = 6
ALPHA_GRID = [round(a / 20, 2) for a in range(21)]   # 0.00, 0.05, ..., 1.00


# ---------------------------------------------------------------------------
# Global z-score scorer (FD001 means)
# ---------------------------------------------------------------------------

SENSOR_STATS: dict[str, tuple[float, float]] = {
    "sensor_2":  (641.682,  0.501),
    "sensor_3":  (1590.524, 6.132),
    "sensor_4":  (1408.934, 14.806),
    "sensor_7":  (554.027,  0.603),
    "sensor_8":  (2388.099, 0.058),
    "sensor_9":  (9065.252, 20.834),
    "sensor_11": (47.541,   0.396),
    "sensor_12": (521.413,  2.578),
    "sensor_13": (2388.096, 0.051),
    "sensor_14": (8143.750, 14.800),
    "sensor_15": (8.442,    0.035),
    "sensor_17": (392.088,  2.483),
    "sensor_20": (39.234,   0.271),
    "sensor_21": (23.394,   0.178),
}
SENSOR_NOISE_FLOOR: dict[str, float] = {
    "sensor_2":  0.75,
    "sensor_8":  0.15,
    "sensor_13": 0.15,
    "sensor_15": 0.07,
}

CAUSAL_SENSORS: dict[str, str] = {
    "sensor_4":  "op_setting_1",
    "sensor_11": "op_setting_2",
    "sensor_15": "op_setting_2",
    "sensor_3":  "op_setting_3",
    "sensor_9":  "op_setting_3",
}


def score_zscore(row: dict) -> float:
    z_scores = []
    for sensor, (mean, std) in SENSOR_STATS.items():
        val = row.get(sensor)
        if val is None or std == 0:
            continue
        eff_std = max(std, SENSOR_NOISE_FLOOR.get(sensor, 0.0))
        z_scores.append(abs(val - mean) / eff_std)
    if not z_scores:
        return 0.0
    top3_mean = sum(sorted(z_scores, reverse=True)[:3]) / 3
    return min(top3_mean / 5.0, 1.0)


def make_causal_scorer_fd001(coefs: dict) -> object:
    """Return a scorer using single-cluster (FD001) causal coefficients."""
    def score(row: dict) -> float:
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
    return score


def make_causal_scorer_fd002(km: KMeans, cluster_coefs: dict) -> object:
    """Return a regime-aware scorer using per-cluster causal coefficients."""
    def score(row: dict) -> float:
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
    return score


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def load(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        print(f"ERROR: {path} not found. Run: python scripts/download_cmapss.py")
        sys.exit(1)
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def fit_fd001_coefs(train: pd.DataFrame) -> dict:
    """Fit one LinearRegression per causal edge on FD001 training data."""
    coefs = {}
    for sensor, cause in CAUSAL_SENSORS.items():
        X = train[[cause]].values
        y = train[sensor].values
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        coefs[sensor] = {
            "coef": float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "residual_std": max(float(np.std(residuals)), 1e-6),
        }
    return coefs


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


# ---------------------------------------------------------------------------
# F1 sweep helpers
# ---------------------------------------------------------------------------

def first_alerts_blend(
    df: pd.DataFrame,
    score_z_fn, score_c_fn,
    alpha: float,
) -> pd.DataFrame:
    """Return engine_id → first_alert_cycle for a given α."""
    records = []
    for engine_id, group in df.groupby("engine_id"):
        first = None
        for _, row in group.sort_values("cycle").iterrows():
            s = alpha * score_c_fn(row.to_dict()) + (1 - alpha) * score_z_fn(row.to_dict())
            if s >= ALERT_THRESHOLD:
                first = int(row["cycle"])
                break
        records.append({"engine_id": engine_id, "first_alert_cycle": first})
    return pd.DataFrame(records)


def compute_f1(lead_times: pd.Series) -> float:
    tp = int(((lead_times.notna()) & (lead_times <= W)).sum())
    fp = int(((lead_times.notna()) & (lead_times >  W)).sum())
    fn = int(lead_times.isna().sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def sweep_alpha(
    train: pd.DataFrame,
    score_z_fn, score_c_fn,
    label: str,
) -> tuple[float, list[float]]:
    """
    Sweep α on training data, return (optimal_alpha, f1_curve).
    Uses last-cycle of each training engine as the failure cycle
    (training data runs to failure in CMAPSS).
    """
    failure_cycles = (
        train.groupby("engine_id")["cycle"].max()
        .rename("true_failure_cycle").reset_index()
    )

    f1_curve = []
    print(f"\n  {label} — α sweep:")
    print(f"  {'α':>6}  {'Coverage':>10}  {'F1':>8}")
    print("  " + "-" * 28)

    for alpha in ALPHA_GRID:
        alerts = first_alerts_blend(train, score_z_fn, score_c_fn, alpha)
        merged = failure_cycles.merge(alerts, on="engine_id", how="left")
        lt = merged["true_failure_cycle"] - merged["first_alert_cycle"]
        lt.index = merged["engine_id"]
        f1 = compute_f1(lt)
        cov = lt.notna().mean()
        f1_curve.append(f1)
        print(f"  {alpha:>6.2f}  {cov:>9.1%}  {f1:>8.4f}")

    best_idx = int(np.argmax(f1_curve))
    best_alpha = ALPHA_GRID[best_idx]
    print(f"\n  → Optimal α = {best_alpha:.2f}  (F1 = {f1_curve[best_idx]:.4f})")
    return best_alpha, f1_curve


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # FD001
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("FD001 — single operating condition")
    print("=" * 60)

    print("Loading train_FD001.txt...")
    train_fd001 = load("train_FD001.txt")

    print("Fitting FD001 causal coefficients...")
    fd001_coefs = fit_fd001_coefs(train_fd001)
    score_c_fd001 = make_causal_scorer_fd001(fd001_coefs)

    alpha_fd001, f1_fd001 = sweep_alpha(
        train_fd001, score_zscore, score_c_fd001, "FD001"
    )

    # -----------------------------------------------------------------------
    # FD002
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FD002 — six operating conditions")
    print("=" * 60)

    print("Loading train_FD002.txt...")
    train_fd002 = load("train_FD002.txt")

    print(f"Training KMeans (n_clusters={N_CLUSTERS}) on FD002 op_settings...")
    km = train_kmeans(train_fd002)

    print("Fitting per-cluster causal coefficients...")
    cluster_coefs = fit_cluster_coefs(train_fd002, km)
    score_c_fd002 = make_causal_scorer_fd002(km, cluster_coefs)

    alpha_fd002, f1_fd002 = sweep_alpha(
        train_fd002, score_zscore, score_c_fd002, "FD002"
    )

    # -----------------------------------------------------------------------
    # Convergence check
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Convergence check")
    print("=" * 60)
    print(f"  FD001 optimal α = {alpha_fd001:.2f}")
    print(f"  FD002 optimal α = {alpha_fd002:.2f}")
    if abs(alpha_fd001 - alpha_fd002) <= 0.10:
        print("  → Converge within 0.10 — a single α may be sufficient.")
    else:
        print("  → Diverge by > 0.10 — dataset-specific α is warranted.")

    # -----------------------------------------------------------------------
    # Write to regime_coefficients.json
    # -----------------------------------------------------------------------
    if not COEFF_FILE.exists():
        print(f"\nWARNING: {COEFF_FILE} not found. Run scripts/compute_regime_coefficients.py first.")
        return

    with open(COEFF_FILE) as f:
        data = json.load(f)

    data["blend_alpha_fd001"] = alpha_fd001
    data["blend_alpha_fd002"] = alpha_fd002

    with open(COEFF_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved blend_alpha_fd001={alpha_fd001}, blend_alpha_fd002={alpha_fd002}")
    print(f"  → {COEFF_FILE}")


if __name__ == "__main__":
    main()
