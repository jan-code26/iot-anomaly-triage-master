"""
FD002 regime-aware evaluation — the paper's central experiment.

FD001 has 1 operating condition, making the regime-aware vs. global comparison
degenerate. FD002 has 6 distinct operating conditions, which is where causal
conditioning on operating regime should demonstrate a clear improvement.

Three variants evaluated on test_FD002.txt:
  1. Global z-score      — uses FD001 global sensor means (ignores operating regime)
  2. Retrained z-score   — uses FD002 global sensor means (same-dataset but no regime)
  3. Regime-aware causal — assigns each reading to 1 of 6 clusters via KMeans,
                           uses cluster-specific causal residuals

Variants 1 vs 2 isolates domain-shift correction (retraining).
Variants 2 vs 3 isolates regime conditioning (the paper's claim).

Both variants use the same alert threshold (score >= 0.3) and same lead-time
formula as ablation_study.py for comparability.

Statistical tests:
  - Wilcoxon rank-sum  on lead-time distributions
  - Fisher's exact     on coverage (2×2 table, all engines in FD002 test set)

Outputs:
  data/processed/fd002_regime_table.csv
  data/processed/fd002_lead_time_distribution.png

Usage:
    python scripts/fd002_regime_eval.py
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
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

ALERT_THRESHOLD = 0.3
N_CLUSTERS = 6

# Same causal DAG as backend/services/causal_scorer.py
CAUSAL_SENSORS: dict[str, str] = {
    "sensor_4":  "op_setting_1",
    "sensor_11": "op_setting_2",
    "sensor_15": "op_setting_2",
    "sensor_3":  "op_setting_3",
    "sensor_9":  "op_setting_3",
}

# Global FD001 sensor stats for the baseline variant
# (from backend/anomaly.py — FD001 training data)
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        print(f"ERROR: {path} not found. Run: python scripts/download_cmapss.py")
        sys.exit(1)
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def build_failure_cycles(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (test_df, failure_cycles_df) for a given dataset (e.g. 'FD002')."""
    test = load(f"test_{dataset}.txt")
    rul_labels = pd.read_csv(
        DATA_DIR / f"RUL_{dataset}.txt", header=None, names=["rul_at_end"]
    )
    rul_labels["engine_id"] = rul_labels.index + 1
    last_cycles = test.groupby("engine_id")["cycle"].max().rename("last_cycle").reset_index()
    rul_labels = rul_labels.merge(last_cycles, on="engine_id")
    rul_labels["true_failure_cycle"] = rul_labels["last_cycle"] + rul_labels["rul_at_end"]
    return test, rul_labels[["engine_id", "true_failure_cycle"]]


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def score_global_zscore(row: dict) -> float:
    """Global z-score — uses FD001 means, ignores operating regime."""
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


def compute_fd002_marginal_stats(
    train: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """
    Compute per-sensor (mean, std) from FD002 training data (all regimes pooled).

    NOTE: We do NOT apply a noise floor here.  For FD002, the cross-engine std at
    cycle 1 spans six operating conditions — using it as a noise floor would inflate
    eff_std by 100–300× (e.g., sensor_4 cycle-1 std ≈ 122, noise_floor = 245),
    suppressing all z-scores to near zero.  The FD002 marginal std already reflects
    the full regime spread; using it directly as eff_std is the correct treatment for
    a "retrained" detector that ignores regime.

    Returns {sensor: (mean, marginal_std)}.
    """
    sensors = list(SENSOR_STATS.keys())
    stats: dict[str, tuple[float, float]] = {}
    for sensor in sensors:
        if sensor not in train.columns:
            continue
        vals = train[sensor].dropna()
        stats[sensor] = (float(vals.mean()), float(vals.std(ddof=1)))
    return stats


def make_fd002_zscore_scorer(stats: dict[str, tuple[float, float]]):
    """
    Return a scoring function using FD002 marginal means/stds.

    Mechanistic role: isolates domain-shift correction (retraining on FD002 data)
    from regime conditioning.  If retrained z-score ≈ regime-aware causal,
    retraining suffices.  If regime-aware causal is superior, regime conditioning
    is the key mechanism — not just retraining.
    """
    def score(row: dict) -> float:
        z_scores = []
        for sensor, (mean, std) in stats.items():
            val = row.get(sensor)
            if val is None or std == 0:
                continue
            z_scores.append(abs(val - mean) / std)
        if not z_scores:
            return 0.0
        top3_mean = sum(sorted(z_scores, reverse=True)[:3]) / 3
        return min(top3_mean / 5.0, 1.0)
    return score


def make_regime_scorer(km: KMeans, cluster_coefs: dict[int, dict]) -> object:
    """Return a scoring function that uses cluster-specific causal coefficients."""
    def score(row: dict) -> float:
        ops = np.array([[
            row.get("op_setting_1", 0.0) or 0.0,
            row.get("op_setting_2", 0.0) or 0.0,
            row.get("op_setting_3", 0.0) or 0.0,
        ]])
        cluster = int(km.predict(ops)[0])
        coefs = cluster_coefs[cluster]

        z_scores = []
        for sensor, cause in CAUSAL_SENSORS.items():
            val = row.get(sensor)
            if val is None:
                continue
            c = coefs[sensor]
            cause_val = row.get(cause)
            if cause_val is None:
                predicted = c["intercept"]
            else:
                predicted = c["coef"] * cause_val + c["intercept"]
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
# Training
# ---------------------------------------------------------------------------

def train_kmeans(train: pd.DataFrame) -> KMeans:
    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    km.fit(train[op_cols].values)
    return km


def fit_cluster_coefs(train: pd.DataFrame, km: KMeans) -> dict[int, dict]:
    """Fit one LinearRegression per (cluster, causal_sensor) pair."""
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
            residual_std = float(np.std(residuals))
            cluster_coefs[c][sensor] = {
                "coef": float(model.coef_[0]),
                "intercept": float(model.intercept_),
                "residual_std": max(residual_std, 1e-6),
            }
    return cluster_coefs


# ---------------------------------------------------------------------------
# Lead time computation
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


def make_lead_times(
    alerts: pd.DataFrame, failure_cycles: pd.DataFrame, col: str
) -> pd.Series:
    merged = failure_cycles.merge(alerts, on="engine_id", how="left")
    lt = merged["true_failure_cycle"] - merged["first_alert_cycle"]
    lt.index = merged["engine_id"]
    return lt.rename(col)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_stats(lead_times: pd.Series) -> dict:
    valid = lead_times.dropna()
    n_engines = len(lead_times)
    return {
        "n_alert": len(valid),
        "n_engines": n_engines,
        "coverage": len(valid) / n_engines,
        "mean": valid.mean() if len(valid) else float("nan"),
        "sd": valid.std(ddof=1) if len(valid) > 1 else float("nan"),
        "median": valid.median() if len(valid) else float("nan"),
    }


def compute_f1(lead_times: pd.Series, W: int = 100) -> dict:
    """
    F1 score for alert quality.
    TP: alerted with lead_time ≤ W (caught engine in degradation zone).
    FP: alerted but lead_time > W (premature alert — alarm fatigue risk).
    FN: not alerted (missed failure).
    W = 100 cycles = maximum RUL at which an alert is actionable.
    """
    tp = int(((lead_times.notna()) & (lead_times <= W)).sum())
    fp = int(((lead_times.notna()) & (lead_times >  W)).sum())
    fn = int(lead_times.isna().sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 3),
        "recall":    round(rec,  3),
        "f1":        round(f1,   3),
    }


def run_tests(
    variant_lead: pd.Series, baseline_lead: pd.Series
) -> tuple[float, float]:
    v = variant_lead.dropna().values
    b = baseline_lead.dropna().values
    n_total = len(variant_lead)

    if len(v) >= 2 and len(b) >= 2:
        _, wp = mannwhitneyu(v, b, alternative="two-sided")
    else:
        wp = float("nan")

    v_alert = int(variant_lead.notna().sum())
    b_alert = int(baseline_lead.notna().sum())
    table = [[v_alert, n_total - v_alert], [b_alert, n_total - b_alert]]
    _, fp = fisher_exact(table)

    return wp, fp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Train on FD002 ---
    print("Loading train_FD002.txt...")
    train_fd002 = load("train_FD002.txt")

    print(f"Training KMeans (n_clusters={N_CLUSTERS}) on op_settings...")
    km = train_kmeans(train_fd002)

    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
    cluster_labels = km.predict(train_fd002[op_cols].values)
    train_fd002 = train_fd002.copy()
    train_fd002["_cluster"] = cluster_labels

    print("\nCluster sanity check (FD002 training set):")
    print(f"{'Cluster':<10} {'N rows':<10} {'op_setting_1':>14} {'op_setting_2':>14} {'op_setting_3':>14}")
    print("-" * 65)
    for c in range(N_CLUSTERS):
        sub = train_fd002[train_fd002["_cluster"] == c]
        means = sub[op_cols].mean()
        print(
            f"  {c:<8} {len(sub):<10} "
            f"{means['op_setting_1']:>14.4f} "
            f"{means['op_setting_2']:>14.4f} "
            f"{means['op_setting_3']:>14.4f}"
        )

    print("\nFitting per-cluster causal coefficients...")
    cluster_coefs = fit_cluster_coefs(train_fd002, km)

    # --- Evaluate on FD002 test set ---
    print("\nLoading test_FD002.txt...")
    test_fd002, failure_cycles = build_failure_cycles("FD002")
    n_engines = test_fd002["engine_id"].nunique()
    print(f"  {n_engines} engines in FD002 test set.")

    print("Scoring: global z-score variant (FD001 means)...")
    global_alerts = first_alerts_from_scoring(test_fd002, score_global_zscore)

    print("Computing FD002 marginal stats for retrained z-score...")
    fd002_stats = compute_fd002_marginal_stats(train_fd002)
    retrained_scorer = make_fd002_zscore_scorer(fd002_stats)
    print("Scoring: retrained z-score variant (FD002 means)...")
    retrained_alerts = first_alerts_from_scoring(test_fd002, retrained_scorer)

    print("Scoring: regime-aware causal variant...")
    regime_scorer = make_regime_scorer(km, cluster_coefs)
    regime_alerts = first_alerts_from_scoring(test_fd002, regime_scorer)

    # --- Lead times ---
    lt_global    = make_lead_times(global_alerts,    failure_cycles, "Global z-score (FD001 means)")
    lt_retrained = make_lead_times(retrained_alerts, failure_cycles, "Retrained z-score (FD002 means)")
    lt_regime    = make_lead_times(regime_alerts,    failure_cycles, "Regime-aware causal")

    # --- Stats ---
    s_global    = compute_stats(lt_global)
    s_retrained = compute_stats(lt_retrained)
    s_regime    = compute_stats(lt_regime)
    f_global    = compute_f1(lt_global)
    f_retrained = compute_f1(lt_retrained)
    f_regime    = compute_f1(lt_regime)

    wp_ret, fp_ret = run_tests(lt_retrained, lt_global)
    wp_reg, fp_reg = run_tests(lt_regime, lt_global)

    # --- Table ---
    rows = []
    for name, s, f, wilcoxon_p, fisher_p in [
        ("Global z-score (FD001 means)",    s_global,    f_global,    "—", "—"),
        ("Retrained z-score (FD002 means)", s_retrained, f_retrained,
         f"{wp_ret:.3f}" if not np.isnan(wp_ret) else "nan", f"{fp_ret:.3f}"),
        ("Regime-aware causal",             s_regime,    f_regime,
         f"{wp_reg:.3f}" if not np.isnan(wp_reg) else "nan", f"{fp_reg:.3f}"),
    ]:
        rows.append({
            "Variant":    name,
            "Coverage":   f"{s['coverage']:.0%}",
            "N_alert":    s["n_alert"],
            "Mean":       f"{s['mean']:.1f}" if not np.isnan(s["mean"]) else "—",
            "SD":         f"{s['sd']:.1f}"   if not np.isnan(s["sd"])   else "—",
            "Median":     f"{s['median']:.1f}" if not np.isnan(s["median"]) else "—",
            "Precision":  f"{f['precision']:.3f}",
            "Recall":     f"{f['recall']:.3f}",
            "F1":         f"{f['f1']:.3f}",
            "Wilcoxon-p": wilcoxon_p,
            "Fisher-p":   fisher_p,
        })
    table = pd.DataFrame(rows)

    print("\n" + "=" * 95)
    print(f"FD002 Regime-Aware Evaluation ({n_engines} engines, alert threshold = 0.3, W=100)")
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

    table_out = OUT_DIR / "fd002_regime_table.csv"
    table.to_csv(table_out, index=False)
    print(f"Table saved to: {table_out}")

    # --- RUL analysis: are non-alerted engines farther from failure? ---
    rul_labels_full = pd.read_csv(
        DATA_DIR / "RUL_FD002.txt", header=None, names=["rul_at_end"]
    )
    rul_labels_full["engine_id"] = rul_labels_full.index + 1
    merged_rul = rul_labels_full.merge(regime_alerts, on="engine_id", how="left")
    alerted     = merged_rul[merged_rul["first_alert_cycle"].notna()]["rul_at_end"].values
    not_alerted = merged_rul[merged_rul["first_alert_cycle"].isna()]["rul_at_end"].values

    print("\n--- RUL analysis: non-alerted vs alerted FD002 engines ---")
    print(f"  Alerted     ({len(alerted)} engines): median RUL={float(np.median(alerted)):.1f}, mean={float(np.mean(alerted)):.1f}")
    print(f"  Not alerted ({len(not_alerted)} engines): median RUL={float(np.median(not_alerted)):.1f}, mean={float(np.mean(not_alerted)):.1f}")
    if len(alerted) >= 2 and len(not_alerted) >= 2:
        from scipy.stats import mannwhitneyu as mwu
        stat, rul_p = mwu(not_alerted, alerted, alternative="greater")
        r_effect = 1 - (2 * stat) / (len(not_alerted) * len(alerted))
        print(f"  H1: non-alerted RUL > alerted RUL — Wilcoxon p={rul_p:.4f}, r={r_effect:.3f}")
        if rul_p < 0.05:
            print("  → Non-alerted engines have significantly higher RUL (p<0.05).")
            print("     System is declining to alarm on engines objectively farther from failure.")
        else:
            print("  → No significant RUL difference.")
    # Save summary
    rul_summary = {
        "n_alerted": len(alerted),
        "n_not_alerted": len(not_alerted),
        "median_rul_alerted": float(np.median(alerted)),
        "median_rul_not_alerted": float(np.median(not_alerted)),
        "mean_rul_alerted": float(np.mean(alerted)),
        "mean_rul_not_alerted": float(np.mean(not_alerted)),
        "wilcoxon_p": float(rul_p) if len(alerted) >= 2 and len(not_alerted) >= 2 else float("nan"),
    }
    pd.DataFrame([rul_summary]).to_csv(OUT_DIR / "fd002_rul_analysis.csv", index=False)
    print(f"RUL analysis saved to: {OUT_DIR / 'fd002_rul_analysis.csv'}")

    # --- Figure ---
    detail = pd.concat([lt_global, lt_retrained, lt_regime], axis=1).reset_index()
    detail.columns = [
        "engine_id",
        "Global z-score (FD001 means)",
        "Retrained z-score (FD002 means)",
        "Regime-aware causal",
    ]
    plot_data = detail.melt(
        id_vars="engine_id", var_name="Variant", value_name="Lead Time (cycles)"
    ).dropna(subset=["Lead Time (cycles)"])

    order = ["Global z-score (FD001 means)", "Retrained z-score (FD002 means)", "Regime-aware causal"]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=plot_data, x="Variant", y="Lead Time (cycles)",
        order=order, palette="Set2", width=0.4, fliersize=0, ax=ax,
    )
    sns.stripplot(
        data=plot_data, x="Variant", y="Lead Time (cycles)",
        order=order, color="black", alpha=0.35, size=4, jitter=True, ax=ax,
    )
    ax.set_title(f"Lead Time Distribution — FD002 ({n_engines} engines, threshold = 0.3)")
    ax.set_xlabel("")
    ax.set_ylabel("Lead Time (cycles before failure)")
    plt.tight_layout()

    fig_out = OUT_DIR / "fd002_lead_time_distribution.png"
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)
    print(f"Figure saved to: {fig_out}")


if __name__ == "__main__":
    main()
