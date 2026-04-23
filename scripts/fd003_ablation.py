"""
FD003 ablation study — single operating condition, dual fault mode.

FD003 mirrors FD001's single-condition setup but adds a second fault mode:
fan degradation (in addition to HPC degradation).  The causal DAG was designed
for HPC sensors (sensor_4, sensor_11, sensor_15, sensor_3, sensor_9).  This
experiment tests whether the HPC-focused causal scorer still detects degradation
when the underlying failure mode may be fan-related.

Five variants evaluated on test_FD003.txt (100 engines):
    1. Isolation Forest     (trained on train_FD003.txt, inline)
    2. Z-score only         (global sensor means from FD001 — same as ablation_study.py)
    3. Causal only          (op-setting-conditioned residuals; FD001 fallback coefs)
    4. Full pipeline        (α = 0.60 blend, same learned weight as FD001)
    5. Full pipeline + veto (graduated G-test physics veto on sensor_11/sensor_15)

Statistical tests vs. Isolation Forest:
    - Wilcoxon rank-sum  (scipy.stats.mannwhitneyu, two-sided)
    - Fisher's exact     (coverage 2×2 table)

Outputs:
    data/processed/fd003_ablation_table.csv
    data/processed/fd003_ablation_engine_detail.csv
    data/processed/fd003_lead_time_distribution.png

Usage:
    python scripts/fd003_ablation.py
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
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

ALERT_THRESHOLD = 0.3
ALPHA_FD003     = 0.60   # same learned blend weight as FD001 (single-condition)
G_CRITICAL      = 26.30
VETO_BUFFER     = 100

# 14 informative sensors (same as FD001 ablation)
INFORMATIVE_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

# FD001 global sensor stats (used as z-score baseline — same as ablation_study.py)
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

# FD001 causal coefficients (fallback — single condition, same op_settings in FD003)
CAUSAL_COEFS: dict[str, dict] = {
    "sensor_4": {
        "cause": "op_setting_1",
        "coef": 39.27258621831777,
        "intercept": 1408.934130041359,
        "residual_std": 8.999976725237511,
    },
    "sensor_11": {
        "cause": "op_setting_2",
        "coef": 10.654235614477768,
        "intercept": 47.5411430987142,
        "residual_std": 0.2670626746724123,
    },
    "sensor_15": {
        "cause": "op_setting_2",
        "coef": 1.8116380628795663,
        "intercept": 8.442141323035916,
        "residual_std": 0.03750037101730749,
    },
    "sensor_3": {
        "cause": "op_setting_3",
        "coef": 0.0,
        "intercept": 1590.5231186079204,
        "residual_std": 6.131000927188836,
    },
    "sensor_9": {
        "cause": "op_setting_3",
        "coef": 0.0,
        "intercept": 9065.242940720276,
        "residual_std": 22.082344331737627,
    },
}


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

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


def score_causal(row: dict) -> float:
    z_scores = []
    for sensor, coef_dict in CAUSAL_COEFS.items():
        val = row.get(sensor)
        std = coef_dict["residual_std"]
        if val is None or std == 0:
            continue
        cause_val = row.get(coef_dict["cause"])
        predicted = (
            coef_dict["coef"] * cause_val + coef_dict["intercept"]
            if cause_val is not None
            else coef_dict["intercept"]
        )
        z_scores.append(abs(val - predicted) / std)
    if not z_scores:
        return 0.0
    k = min(3, len(z_scores))
    top_k_mean = sum(sorted(z_scores, reverse=True)[:k]) / k
    return min(top_k_mean / 5.0, 1.0)


# ---------------------------------------------------------------------------
# Physics veto helpers
# ---------------------------------------------------------------------------

def _compute_gtest(buf_11: list[float], buf_15: list[float]) -> float:
    if len(buf_11) < 20 or len(buf_15) < 20:
        return 0.0
    arr_11 = np.array(buf_11)
    arr_15 = np.array(buf_15)
    if arr_11.max() - arr_11.min() < 1e-9 or arr_15.max() - arr_15.min() < 1e-9:
        return 0.0
    bins_11 = np.linspace(arr_11.min(), arr_11.max() + 1e-9, 6)
    bins_15 = np.linspace(arr_15.min(), arr_15.max() + 1e-9, 6)
    idx_11 = np.clip(np.digitize(arr_11, bins_11[:-1]) - 1, 0, 4)
    idx_15 = np.clip(np.digitize(arr_15, bins_15[:-1]) - 1, 0, 4)
    table = np.zeros((5, 5))
    for i, j in zip(idx_11, idx_15):
        table[i][j] += 1
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    total = table.sum()
    if total == 0:
        return 0.0
    expected = row_sums * col_sums / total
    mask = (table > 0) & (expected > 0)
    return float(2.0 * np.sum(table[mask] * np.log(table[mask] / expected[mask])))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def build_failure_cycles() -> tuple[pd.DataFrame, pd.DataFrame]:
    test = load("test_FD003.txt")
    rul_labels = pd.read_csv(DATA_DIR / "RUL_FD003.txt", header=None, names=["rul_at_end"])
    rul_labels["engine_id"] = rul_labels.index + 1
    last_cycles = test.groupby("engine_id")["cycle"].max().rename("last_cycle").reset_index()
    rul_labels = rul_labels.merge(last_cycles, on="engine_id")
    rul_labels["true_failure_cycle"] = rul_labels["last_cycle"] + rul_labels["rul_at_end"]
    return test, rul_labels[["engine_id", "true_failure_cycle"]]


# ---------------------------------------------------------------------------
# Per-variant first-alert computation
# ---------------------------------------------------------------------------

def first_alerts_from_scoring(test: pd.DataFrame, score_fn) -> pd.DataFrame:
    records = []
    for engine_id, group in test.groupby("engine_id"):
        first = None
        for _, row in group.sort_values("cycle").iterrows():
            if score_fn(row.to_dict()) >= ALERT_THRESHOLD:
                first = int(row["cycle"])
                break
        records.append({"engine_id": engine_id, "first_alert_cycle": first})
    return pd.DataFrame(records)


def first_alerts_if(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Train IF on FD003 training data, score test set.
    Alert = first cycle where clf.predict() == -1 (anomaly).
    Matches lead_time_baseline.py approach used for FD001.
    """
    X_train = train[INFORMATIVE_SENSORS].values
    clf = IsolationForest(contamination=0.05, n_estimators=100, random_state=42)
    clf.fit(X_train)

    # Batch predict for efficiency
    test = test.copy()
    test["_if_pred"] = clf.predict(test[INFORMATIVE_SENSORS].values)

    records = []
    for engine_id, group in test.groupby("engine_id"):
        anomaly_rows = group[group["_if_pred"] == -1].sort_values("cycle")
        first = int(anomaly_rows["cycle"].iloc[0]) if not anomaly_rows.empty else None
        records.append({"engine_id": engine_id, "first_alert_cycle": first})
    return pd.DataFrame(records)


def first_alerts_vetoed(test: pd.DataFrame) -> pd.DataFrame:
    records = []
    for engine_id, group in test.groupby("engine_id"):
        buf_11: list[float] = []
        buf_15: list[float] = []
        first = None
        for _, row in group.sort_values("cycle").iterrows():
            row_dict = row.to_dict()
            s11 = row_dict.get("sensor_11")
            s15 = row_dict.get("sensor_15")
            if s11 is not None:
                buf_11.append(float(s11))
            if s15 is not None:
                buf_15.append(float(s15))
            if len(buf_11) > VETO_BUFFER:
                buf_11.pop(0)
            if len(buf_15) > VETO_BUFFER:
                buf_15.pop(0)
            z = score_zscore(row_dict)
            c = score_causal(row_dict)
            g_stat = _compute_gtest(buf_11, buf_15)
            veto_factor = 1.0 - 0.5 * min(g_stat / G_CRITICAL, 1.0)
            c_vetoed = c * veto_factor
            score = ALPHA_FD003 * c_vetoed + (1.0 - ALPHA_FD003) * z
            if score >= ALERT_THRESHOLD:
                first = int(row["cycle"])
                break
        records.append({"engine_id": engine_id, "first_alert_cycle": first})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Stats helpers (shared with ablation_study.py)
# ---------------------------------------------------------------------------

def compute_stats(lead_times: pd.Series) -> dict:
    valid = lead_times.dropna()
    return {
        "n_alert": len(valid),
        "coverage": len(valid) / 100,
        "mean": valid.mean() if len(valid) else float("nan"),
        "sd": valid.std(ddof=1) if len(valid) > 1 else float("nan"),
        "median": valid.median() if len(valid) else float("nan"),
    }


def compute_f1(lead_times: pd.Series, W: int = 100) -> dict:
    tp = int(((lead_times.notna()) & (lead_times <= W)).sum())
    fp = int(((lead_times.notna()) & (lead_times >  W)).sum())
    fn = int(lead_times.isna().sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def run_tests(variant_lead: pd.Series, baseline_lead: pd.Series) -> tuple[float, float]:
    v = variant_lead.dropna().values
    b = baseline_lead.dropna().values
    wp = float("nan")
    if len(v) >= 2 and len(b) >= 2:
        _, wp = mannwhitneyu(v, b, alternative="two-sided")
    v_alert = int(variant_lead.notna().sum())
    b_alert = int(baseline_lead.notna().sum())
    table = [[v_alert, 100 - v_alert], [b_alert, 100 - b_alert]]
    _, fp = fisher_exact(table)
    return wp, fp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FD003 data...")
    train = load("train_FD003.txt")
    test, failure_cycles = build_failure_cycles()
    print(f"  {test['engine_id'].nunique()} test engines, {train['engine_id'].nunique()} training engines.")

    # --- Isolation Forest ---
    print("Training Isolation Forest on FD003 training set...")
    if_alerts = first_alerts_if(train, test)

    # --- Z-score only ---
    print("Scoring z-score only...")
    zs_alerts = first_alerts_from_scoring(test, score_zscore)

    # --- Causal only ---
    print("Scoring causal only...")
    ca_alerts = first_alerts_from_scoring(test, score_causal)

    # --- Full pipeline ---
    print(f"Scoring full pipeline (α={ALPHA_FD003})...")
    def score_full(row: dict) -> float:
        return ALPHA_FD003 * score_causal(row) + (1.0 - ALPHA_FD003) * score_zscore(row)
    fp_alerts = first_alerts_from_scoring(test, score_full)

    # --- Full pipeline + veto ---
    print("Scoring full pipeline + veto...")
    fv_alerts = first_alerts_vetoed(test)

    # --- Build lead times ---
    def make_lead_times(alerts: pd.DataFrame, col: str) -> pd.Series:
        merged = failure_cycles.merge(alerts, on="engine_id", how="left")
        lt = merged["true_failure_cycle"] - merged["first_alert_cycle"]
        lt.index = merged["engine_id"]
        return lt.rename(col)

    lt_if = make_lead_times(if_alerts,  "Isolation Forest")
    lt_zs = make_lead_times(zs_alerts,  "Z-score only")
    lt_ca = make_lead_times(ca_alerts,  "Causal only")
    lt_fp = make_lead_times(fp_alerts,  "Full pipeline")
    lt_fv = make_lead_times(fv_alerts,  "Full pipeline + veto")

    engine_detail = pd.concat([lt_if, lt_zs, lt_ca, lt_fp, lt_fv], axis=1)
    engine_detail.index.name = "engine_id"

    # --- Stats table ---
    variants = [
        ("Isolation Forest",     lt_if),
        ("Z-score only",         lt_zs),
        ("Causal only",          lt_ca),
        ("Full pipeline",        lt_fp),
        ("Full pipeline + veto", lt_fv),
    ]

    rows = []
    for name, lt in variants:
        s = compute_stats(lt)
        f = compute_f1(lt)
        if name == "Isolation Forest":
            wp_str, fp_str = "—", "—"
        else:
            wp, fp = run_tests(lt, lt_if)
            wp_str = f"{wp:.3f}" if not (isinstance(wp, float) and np.isnan(wp)) else "nan"
            fp_str = f"{fp:.3f}"
        rows.append({
            "Variant":    name,
            "Coverage":   f"{s['coverage']:.0%}",
            "N_alert":    s["n_alert"],
            "Mean":       f"{s['mean']:.1f}"   if not np.isnan(s["mean"])   else "—",
            "SD":         f"{s['sd']:.1f}"     if not np.isnan(s["sd"])     else "—",
            "Median":     f"{s['median']:.1f}" if not np.isnan(s["median"]) else "—",
            "Precision":  f"{f['precision']:.3f}",
            "Recall":     f"{f['recall']:.3f}",
            "F1":         f"{f['f1']:.3f}",
            "Wilcoxon-p": wp_str,
            "Fisher-p":   fp_str,
        })

    table = pd.DataFrame(rows)

    print("\n" + "=" * 95)
    print("FD003 Ablation — single condition, dual fault mode (100 engines, threshold=0.3, W=100)")
    print("=" * 95)
    col_w = [24, 10, 9, 8, 8, 8, 10, 8, 8, 12, 10]
    headers = ["Variant", "Coverage", "N_alert", "Mean", "SD", "Median",
               "Precision", "Recall", "F1", "Wilcoxon-p", "Fisher-p"]
    print("  " + "".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print("  " + "-" * sum(col_w))
    for _, row in table.iterrows():
        vals = [row[h] for h in headers]
        print("  " + "".join(str(v).ljust(w) for v, w in zip(vals, col_w)))
    print("=" * 95 + "\n")

    table_out = OUT_DIR / "fd003_ablation_table.csv"
    table.to_csv(table_out, index=False)
    print(f"Table saved to: {table_out}")

    detail_out = OUT_DIR / "fd003_ablation_engine_detail.csv"
    engine_detail.reset_index().to_csv(detail_out, index=False)
    print(f"Per-engine detail saved to: {detail_out}")

    # --- Figure ---
    plot_data = (
        engine_detail.reset_index()
        .melt(id_vars="engine_id", var_name="Variant", value_name="Lead Time (cycles)")
        .dropna(subset=["Lead Time (cycles)"])
    )
    order = ["Isolation Forest", "Z-score only", "Causal only", "Full pipeline", "Full pipeline + veto"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=plot_data, x="Variant", y="Lead Time (cycles)",
        order=order, palette="Set2", width=0.5, fliersize=0, ax=ax,
    )
    sns.stripplot(
        data=plot_data, x="Variant", y="Lead Time (cycles)",
        order=order, color="black", alpha=0.35, size=4, jitter=True, ax=ax,
    )
    ax.set_title("Lead Time Distribution — FD003 (single condition, dual fault, threshold=0.3)")
    ax.set_xlabel("")
    ax.set_ylabel("Lead Time (cycles before failure)")
    plt.tight_layout()

    fig_out = OUT_DIR / "fd003_lead_time_distribution.png"
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)
    print(f"Figure saved to: {fig_out}")


if __name__ == "__main__":
    main()
