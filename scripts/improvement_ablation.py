"""
Improvement trajectory ablation.

Shows how each model improvement contributes to performance on FD001 and
FD002, so the gain from each step can be attributed independently.

Three cumulative steps are evaluated:

    Step 0: Baseline
        Causal aggregation = mean of all 5 sensors
        Blend weight       = α = 0.5 (hardcoded 50/50)
        Alert threshold    = 0.3 (global)

    Step 1: +Aggregation fix
        Causal aggregation = top-3 of 5 sensors  (mirrors z-score's top-3-of-14)
        (blend and threshold unchanged)

    Step 2: +Learned α
        Blend weight       = α = 0.70 (FD001) or α = 1.00 (FD002)
        (threshold unchanged)

    Note: The graduated physics veto (Day 3) operates only during live
    inference when the G-test buffer accumulates 100 readings per engine.
    FD001/FD002 test sets are evaluated offline here without that buffer,
    so the veto's contribution cannot be quantified in this script.
    See backend/agent/nodes.py for the implemented _veto_factor() formula.

Outputs:
    data/processed/improvement_trajectory.csv
    data/processed/improvement_trajectory.png

Usage:
    python scripts/improvement_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR  = ROOT / "data" / "processed"

COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

N_CLUSTERS = 6
W = 100              # actionable-window for F1

SENSOR_STATS: dict[str, tuple[float, float]] = {
    "sensor_2":  (641.682,  0.501),   "sensor_3":  (1590.524, 6.132),
    "sensor_4":  (1408.934, 14.806),  "sensor_7":  (554.027,  0.603),
    "sensor_8":  (2388.099, 0.058),   "sensor_9":  (9065.252, 20.834),
    "sensor_11": (47.541,   0.396),   "sensor_12": (521.413,  2.578),
    "sensor_13": (2388.096, 0.051),   "sensor_14": (8143.750, 14.800),
    "sensor_15": (8.442,    0.035),   "sensor_17": (392.088,  2.483),
    "sensor_20": (39.234,   0.271),   "sensor_21": (23.394,   0.178),
}
SENSOR_NOISE_FLOOR: dict[str, float] = {
    "sensor_2": 0.75, "sensor_8": 0.15, "sensor_13": 0.15, "sensor_15": 0.07,
}
CAUSAL_SENSORS: dict[str, str] = {
    "sensor_4": "op_setting_1", "sensor_11": "op_setting_2",
    "sensor_15": "op_setting_2", "sensor_3": "op_setting_3",
    "sensor_9": "op_setting_3",
}


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def score_zscore(row: dict) -> float:
    z = []
    for s, (m, std) in SENSOR_STATS.items():
        v = row.get(s)
        if v is None or std == 0:
            continue
        z.append(abs(v - m) / max(std, SENSOR_NOISE_FLOOR.get(s, 0.0)))
    if not z:
        return 0.0
    return min(sum(sorted(z, reverse=True)[:3]) / 3 / 5.0, 1.0)


def make_fd001_causal_scorer(coefs: dict, top_k: bool) -> object:
    def score(row: dict) -> float:
        z = []
        for sensor, cause in CAUSAL_SENSORS.items():
            v = row.get(sensor)
            if v is None:
                continue
            c = coefs[sensor]
            cv = row.get(cause)
            pred = c["coef"] * cv + c["intercept"] if cv is not None else c["intercept"]
            std = c["residual_std"]
            if std == 0:
                continue
            z.append(abs(v - pred) / std)
        if not z:
            return 0.0
        if top_k:
            k = min(3, len(z))
            return min(sum(sorted(z, reverse=True)[:k]) / k / 5.0, 1.0)
        return min(sum(z) / len(z) / 5.0, 1.0)
    return score


def make_fd002_regime_scorer(km: KMeans, cluster_coefs: dict, top_k: bool) -> object:
    def score(row: dict) -> float:
        ops = np.array([[row.get("op_setting_1", 0) or 0,
                         row.get("op_setting_2", 0) or 0,
                         row.get("op_setting_3", 0) or 0]])
        cluster = int(km.predict(ops)[0])
        coefs = cluster_coefs[cluster]
        z = []
        for sensor, cause in CAUSAL_SENSORS.items():
            v = row.get(sensor)
            if v is None:
                continue
            c = coefs[sensor]
            cv = row.get(cause)
            pred = c["coef"] * cv + c["intercept"] if cv is not None else c["intercept"]
            std = c["residual_std"]
            if std == 0:
                continue
            z.append(abs(v - pred) / std)
        if not z:
            return 0.0
        if top_k:
            k = min(3, len(z))
            return min(sum(sorted(z, reverse=True)[:k]) / k / 5.0, 1.0)
        return min(sum(z) / len(z) / 5.0, 1.0)
    return score


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def load(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def fit_fd001_coefs(train: pd.DataFrame) -> dict:
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
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    km.fit(train[["op_setting_1", "op_setting_2", "op_setting_3"]].values)
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
# Evaluation helpers
# ---------------------------------------------------------------------------

def build_failure_cycles(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = load(f"test_{dataset}.txt")
    rul = pd.read_csv(DATA_DIR / f"RUL_{dataset}.txt", header=None, names=["rul_at_end"])
    rul["engine_id"] = rul.index + 1
    last = test.groupby("engine_id")["cycle"].max().rename("last_cycle").reset_index()
    rul = rul.merge(last, on="engine_id")
    rul["true_failure_cycle"] = rul["last_cycle"] + rul["rul_at_end"]
    return test, rul[["engine_id", "true_failure_cycle"]]


def first_alerts(test: pd.DataFrame, score_fn, threshold: float = 0.3) -> pd.DataFrame:
    records = []
    for engine_id, group in test.groupby("engine_id"):
        first = None
        for _, row in group.sort_values("cycle").iterrows():
            if score_fn(row.to_dict()) >= threshold:
                first = int(row["cycle"]); break
        records.append({"engine_id": engine_id, "first_alert_cycle": first})
    return pd.DataFrame(records)


def make_lead_times(alerts: pd.DataFrame, failure_cycles: pd.DataFrame) -> pd.Series:
    merged = failure_cycles.merge(alerts, on="engine_id", how="left")
    return (merged["true_failure_cycle"] - merged["first_alert_cycle"]).values


def metrics(lead_times) -> dict:
    lt = pd.Series(lead_times)
    valid = lt.dropna()
    n = len(lt)
    cov = len(valid) / n if n > 0 else 0.0
    mean = valid.mean() if len(valid) else float("nan")
    median = valid.median() if len(valid) else float("nan")
    tp = int(((lt.notna()) & (lt <= W)).sum())
    fp = int(((lt.notna()) & (lt >  W)).sum())
    fn = int(lt.isna().sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"coverage": cov, "mean_lt": mean, "median_lt": median,
            "precision": prec, "recall": rec, "f1": f1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Train models
    # -----------------------------------------------------------------------
    print("Loading training data...")
    train_fd001 = load("train_FD001.txt")
    train_fd002 = load("train_FD002.txt")

    print("Fitting FD001 causal coefficients...")
    fd001_coefs = fit_fd001_coefs(train_fd001)

    print(f"Training KMeans and per-cluster coefs for FD002...")
    km = train_kmeans(train_fd002)
    cluster_coefs = fit_cluster_coefs(train_fd002, km)

    # -----------------------------------------------------------------------
    # Load test sets
    # -----------------------------------------------------------------------
    print("Loading test sets...")
    test_fd001, fc_fd001 = build_failure_cycles("FD001")
    test_fd002, fc_fd002 = build_failure_cycles("FD002")

    # -----------------------------------------------------------------------
    # Build scorer variants
    # -----------------------------------------------------------------------
    # FD001 scorers
    c_fd001_mean = make_fd001_causal_scorer(fd001_coefs, top_k=False)  # baseline
    c_fd001_topk = make_fd001_causal_scorer(fd001_coefs, top_k=True)   # + agg fix

    # FD002 scorers
    c_fd002_mean = make_fd002_regime_scorer(km, cluster_coefs, top_k=False)  # baseline
    c_fd002_topk = make_fd002_regime_scorer(km, cluster_coefs, top_k=True)   # + agg fix

    # -----------------------------------------------------------------------
    # Evaluate each step
    # -----------------------------------------------------------------------
    rows = []

    print("\nEvaluating improvement steps...")

    steps = [
        # (label, dataset, test, fc, score_z, score_c, alpha, threshold)
        ("Step 0: Baseline",        "FD001", test_fd001, fc_fd001, score_zscore, c_fd001_mean, 0.50, 0.30),
        ("Step 1: +Aggregation fix","FD001", test_fd001, fc_fd001, score_zscore, c_fd001_topk, 0.50, 0.30),
        ("Step 2: +Learned α=0.70", "FD001", test_fd001, fc_fd001, score_zscore, c_fd001_topk, 0.70, 0.30),
        # FD002
        ("Step 0: Baseline",        "FD002", test_fd002, fc_fd002, score_zscore, c_fd002_mean, 0.50, 0.30),
        ("Step 1: +Aggregation fix","FD002", test_fd002, fc_fd002, score_zscore, c_fd002_topk, 0.50, 0.30),
        ("Step 2: +Learned α=1.00", "FD002", test_fd002, fc_fd002, score_zscore, c_fd002_topk, 1.00, 0.30),
    ]

    for label, dataset, test, fc, sz, sc, alpha, threshold in steps:
        print(f"  {label} ({dataset})...")

        def blend_score(row: dict, _sz=sz, _sc=sc, _alpha=alpha) -> float:
            return _alpha * _sc(row) + (1 - _alpha) * _sz(row)

        alerts = first_alerts(test, blend_score, threshold)
        lt = make_lead_times(alerts, fc)
        m = metrics(lt)
        rows.append({
            "Step":      label,
            "Dataset":   dataset,
            "Coverage":  f"{m['coverage']:.0%}",
            "Mean LT":   f"{m['mean_lt']:.1f}" if not np.isnan(m["mean_lt"]) else "—",
            "Median LT": f"{m['median_lt']:.1f}" if not np.isnan(m["median_lt"]) else "—",
            "Precision": f"{m['precision']:.3f}",
            "Recall":    f"{m['recall']:.3f}",
            "F1":        f"{m['f1']:.3f}",
        })

    traj = pd.DataFrame(rows)

    # Print
    print("\n" + "=" * 90)
    print(f"Improvement Trajectory (W={W} cycles)")
    print("=" * 90)
    for dataset in ["FD001", "FD002"]:
        print(f"\n  {dataset}")
        sub = traj[traj["Dataset"] == dataset].drop(columns="Dataset")
        col_w = [26, 10, 10, 10, 10, 10, 8]
        hdrs = ["Step", "Coverage", "Mean LT", "Median LT", "Precision", "Recall", "F1"]
        print("  " + "".join(h.ljust(w) for h, w in zip(hdrs, col_w)))
        print("  " + "-" * sum(col_w))
        for _, row in sub.iterrows():
            vals = [row[h] for h in hdrs]
            print("  " + "".join(str(v).ljust(w) for v, w in zip(vals, col_w)))
    print("=" * 90)

    # Save CSV
    csv_out = OUT_DIR / "improvement_trajectory.csv"
    traj.to_csv(csv_out, index=False)
    print(f"\nTrajectory saved to: {csv_out}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    step_labels = ["Baseline", "+Agg fix", "+Learned α"]
    metrics_to_plot = [("Coverage", lambda x: float(x.strip("%")) / 100),
                       ("F1",       lambda x: float(x))]

    for ax, (dataset, color) in zip(axes, [("FD001", "#4c9be8"), ("FD002", "#e84c4c")]):
        sub = traj[traj["Dataset"] == dataset].reset_index(drop=True)
        coverage_vals = [float(v.strip("%")) / 100 for v in sub["Coverage"]]
        f1_vals = [float(v) for v in sub["F1"]]
        x = range(len(step_labels))
        ax.plot(x, coverage_vals, "o-", color=color, label="Coverage", linewidth=2)
        ax.plot(x, f1_vals, "s--", color=color, alpha=0.7, label=f"F1 (W={W})", linewidth=2)
        for i, (c, f) in enumerate(zip(coverage_vals, f1_vals)):
            ax.annotate(f"{c:.0%}", (i, c), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9)
            ax.annotate(f"{f:.3f}", (i, f), textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=9, alpha=0.7)
        ax.set_xticks(list(x))
        ax.set_xticklabels(step_labels, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{dataset} — Improvement Trajectory", fontsize=12)
        ax.set_ylabel("Score")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.4)

    plt.suptitle(f"Cumulative Model Improvements (alert threshold=0.3, W={W} cycles)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig_out = OUT_DIR / "improvement_trajectory.png"
    fig.savefig(fig_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {fig_out}")


if __name__ == "__main__":
    main()
