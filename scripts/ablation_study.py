"""
Ablation study — FD001 test set.

Compares five variants on test_FD001.txt to isolate which component drives
the lead-time improvement over the Isolation Forest baseline:

    1. Isolation Forest      (load pre-computed baseline CSV)
    2. Z-score only          (global sensor means from anomaly.py)
    3. Causal only           (op-setting-conditioned residuals; no DB needed)
    4. Full pipeline         (α=0.70 blend of z-score + causal; learned weight)
    5. Full pipeline + veto  (as above + graduated G-test physics veto)

For each variant:
  - First cycle where score >= 0.3 per engine = first_alert_cycle
  - lead_time = true_failure_cycle - first_alert_cycle
  - true_failure_cycle = last_test_cycle + RUL_at_end  (from RUL_FD001.txt)

Statistical tests vs. Isolation Forest:
  - Wilcoxon rank-sum  (scipy.stats.mannwhitneyu) on lead-time distributions
  - Fisher's exact     (scipy.stats.fisher_exact)  on coverage (2×2 table)

Outputs:
  data/processed/ablation_table.csv
  data/processed/ablation_engine_detail.csv
  data/processed/lead_time_distribution.png

Usage:
    python scripts/ablation_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

ALERT_THRESHOLD = 0.3  # matches make_decision UNCERTAIN boundary

# ---------------------------------------------------------------------------
# Inlined scorer components (avoid DB import from causal_scorer.py)
# ---------------------------------------------------------------------------

# From backend/anomaly.py
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

# From backend/services/causal_scorer.py FALLBACK_COEFFICIENTS
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


ALPHA_FD001 = 0.60   # learned blend weight: causal 60%, z-score 40% — maximises F1 on FD001
G_CRITICAL  = 26.30  # χ²(df=16, p=0.05) critical value for 5×5 G-test
VETO_BUFFER = 100    # rolling window size for G-test


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
        if cause_val is None:
            predicted = coef_dict["intercept"]
        else:
            predicted = coef_dict["coef"] * cause_val + coef_dict["intercept"]
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
    """
    G-test for independence between sensor_11 and sensor_15 (HPC coupling).

    Uses a 5×5 contingency table over a rolling window.  Returns 0.0 when
    there are fewer than 20 observations or the range of either sensor is
    effectively zero (no variation → perfect coupling, no veto needed).
    """
    if len(buf_11) < 20 or len(buf_15) < 20:
        return 0.0
    arr_11 = np.array(buf_11)
    arr_15 = np.array(buf_15)
    r11 = arr_11.max() - arr_11.min()
    r15 = arr_15.max() - arr_15.min()
    if r11 < 1e-9 or r15 < 1e-9:
        return 0.0  # no variation → sensors locked together → no coupling break
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


def first_alerts_vetoed(test: pd.DataFrame) -> pd.DataFrame:
    """
    First-alert computation for Full pipeline + physics veto variant.

    Maintains a per-engine rolling buffer of sensor_11 / sensor_15 values
    to compute the G-test.  Applies graduated veto:
        veto_factor = 1.0 - 0.5 * min(G / G_CRITICAL, 1.0)
        score = ALPHA * (causal * veto_factor) + (1 - ALPHA) * zscore
    """
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
            score = ALPHA_FD001 * c_vetoed + (1.0 - ALPHA_FD001) * z

            if score >= ALERT_THRESHOLD:
                first = int(row["cycle"])
                break
        records.append({"engine_id": engine_id, "first_alert_cycle": first})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        print(f"ERROR: {path} not found. Run: python scripts/download_cmapss.py")
        sys.exit(1)
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def build_failure_cycles() -> pd.DataFrame:
    test = load("test_FD001.txt")
    rul_labels = pd.read_csv(DATA_DIR / "RUL_FD001.txt", header=None, names=["rul_at_end"])
    rul_labels["engine_id"] = rul_labels.index + 1
    last_cycles = test.groupby("engine_id")["cycle"].max().rename("last_cycle").reset_index()
    rul_labels = rul_labels.merge(last_cycles, on="engine_id")
    rul_labels["true_failure_cycle"] = rul_labels["last_cycle"] + rul_labels["rul_at_end"]
    return test, rul_labels[["engine_id", "true_failure_cycle"]]


# ---------------------------------------------------------------------------
# Per-variant first-alert computation
# ---------------------------------------------------------------------------

def first_alerts_from_scoring(test: pd.DataFrame, score_fn) -> pd.DataFrame:
    """Return DataFrame with engine_id and first_alert_cycle for a score function."""
    records = []
    for engine_id, group in test.groupby("engine_id"):
        first = None
        for _, row in group.sort_values("cycle").iterrows():
            s = score_fn(row.to_dict())
            if s >= ALERT_THRESHOLD:
                first = int(row["cycle"])
                break
        records.append({"engine_id": engine_id, "first_alert_cycle": first})
    return pd.DataFrame(records)


def first_alerts_from_csv(path: Path) -> pd.DataFrame:
    """Load pre-computed baseline (isolation_forest_baseline.csv)."""
    df = pd.read_csv(path)[["engine_id", "first_alert_cycle"]]
    return df


# ---------------------------------------------------------------------------
# Stats helpers
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
    """
    F1 score for alert quality.

    TP: alert fired AND lead_time ≤ W cycles before failure.
        The engine was in the degradation zone when we caught it.
    FP: alert fired BUT lead_time > W cycles before failure.
        The alert was premature — engine still had significant RUL remaining,
        contributing to alarm fatigue.
    FN: no alert raised — failure missed entirely.

    W = 100 cycles (the maximum RUL at which an alert is "actionable" for
    maintenance scheduling; earlier alerts risk being dismissed as false alarms).
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
    variant_lead: pd.Series,
    baseline_lead: pd.Series,
) -> tuple[float, float, float]:
    """Wilcoxon rank-sum + Fisher's exact vs. IF baseline.
    Returns (wilcoxon_p, fisher_p, rank_biserial_r)."""
    v = variant_lead.dropna().values
    b = baseline_lead.dropna().values

    if len(v) >= 2 and len(b) >= 2:
        u_stat, wilcoxon_p = mannwhitneyu(v, b, alternative="two-sided")
        r_biserial = (2.0 * u_stat) / (len(v) * len(b)) - 1.0
    else:
        wilcoxon_p = float("nan")
        r_biserial = float("nan")

    # Fisher's exact on coverage counts (out of 100 engines)
    v_alert = int(variant_lead.notna().sum())
    b_alert = int(baseline_lead.notna().sum())
    table = [[v_alert, 100 - v_alert], [b_alert, 100 - b_alert]]
    _, fisher_p = fisher_exact(table)

    return wilcoxon_p, fisher_p, r_biserial


def bootstrap_ci_coverage(lead_times: pd.Series, n_bootstrap: int = 10_000) -> tuple[float, float]:
    """Bootstrap 95% CI for coverage proportion (resamples full engine set)."""
    n = len(lead_times)
    alerted = (~lead_times.isna()).astype(float).values
    rng = np.random.default_rng(42)
    boot = np.array([
        rng.choice(alerted, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def bootstrap_ci_f1(lead_times: pd.Series, W: int = 100, n_bootstrap: int = 10_000) -> tuple[float, float]:
    """Bootstrap 95% CI for F1 (resamples full engine set including NaN non-alerts)."""
    n = len(lead_times)
    lt_arr = lead_times.values  # NaN for non-alerted
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = pd.Series(lt_arr[idx])
        boot.append(compute_f1(sample, W=W)["f1"])
    boot = np.array(boot)
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test_FD001.txt...")
    test, failure_cycles = build_failure_cycles()

    # --- Isolation Forest (pre-computed) ---
    baseline_path = OUT_DIR / "isolation_forest_baseline.csv"
    if not baseline_path.exists():
        print(f"ERROR: {baseline_path} not found. Run: python scripts/lead_time_baseline.py")
        sys.exit(1)
    print("Loading IF baseline...")
    if_alerts = first_alerts_from_csv(baseline_path)

    # --- Z-score only ---
    print("Scoring z-score only variant...")
    zs_alerts = first_alerts_from_scoring(test, score_zscore)

    # --- Causal only ---
    print("Scoring causal only variant...")
    ca_alerts = first_alerts_from_scoring(test, score_causal)

    # --- Full pipeline (α=0.60 learned blend weight) ---
    print("Scoring full pipeline variant (α=0.60)...")
    def score_full(row: dict) -> float:
        return ALPHA_FD001 * score_causal(row) + (1.0 - ALPHA_FD001) * score_zscore(row)
    fp_alerts = first_alerts_from_scoring(test, score_full)

    # --- Full pipeline + physics veto ---
    print("Scoring full pipeline + veto variant...")
    fv_alerts = first_alerts_vetoed(test)

    # --- Build lead times ---
    def make_lead_times(alerts: pd.DataFrame, col: str) -> pd.Series:
        merged = failure_cycles.merge(alerts, on="engine_id", how="left")
        lt = merged["true_failure_cycle"] - merged["first_alert_cycle"]
        lt.index = merged["engine_id"]
        return lt.rename(col)

    lt_if = make_lead_times(if_alerts, "Isolation Forest")
    lt_zs = make_lead_times(zs_alerts, "Z-score only")
    lt_ca = make_lead_times(ca_alerts, "Causal only")
    lt_fp = make_lead_times(fp_alerts, "Full pipeline")
    lt_fv = make_lead_times(fv_alerts, "Full pipeline + veto")

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

    # --- Bootstrap CIs (10 000 resamples each) ---
    print("Computing bootstrapped 95% CIs (10 000 resamples per variant)...")
    ci_data: dict[str, dict] = {}
    for name, lt in variants:
        cov_lo, cov_hi = bootstrap_ci_coverage(lt)
        f1_lo,  f1_hi  = bootstrap_ci_f1(lt)
        ci_data[name] = {
            "cov_ci": f"[{cov_lo:.0%}–{cov_hi:.0%}]",
            "f1_ci":  f"[{f1_lo:.3f}–{f1_hi:.3f}]",
        }

    rows = []
    for name, lt in variants:
        s = compute_stats(lt)
        f = compute_f1(lt)
        if name == "Isolation Forest":
            wp, fp_p, r_b = "—", "—", "—"
        else:
            wp, fp_p, r_b = run_tests(lt, lt_if)
            wp  = f"{wp:.3f}"  if not (isinstance(wp,  float) and np.isnan(wp))  else "nan"
            fp_p = f"{fp_p:.3f}"
            r_b  = f"{r_b:.3f}" if not (isinstance(r_b, float) and np.isnan(r_b)) else "nan"
        rows.append({
            "Variant":      name,
            "Coverage":     f"{s['coverage']:.0%}",
            "Coverage_CI":  ci_data[name]["cov_ci"],
            "N_alert":      s["n_alert"],
            "Mean":         f"{s['mean']:.1f}" if not np.isnan(s["mean"]) else "—",
            "SD":           f"{s['sd']:.1f}"   if not np.isnan(s["sd"])   else "—",
            "Median":       f"{s['median']:.1f}" if not np.isnan(s["median"]) else "—",
            "Precision":    f"{f['precision']:.3f}",
            "Recall":       f"{f['recall']:.3f}",
            "F1":           f"{f['f1']:.3f}",
            "F1_CI":        ci_data[name]["f1_ci"],
            "r":            r_b,
            "Wilcoxon-p":   wp,
            "Fisher-p":     fp_p,
        })

    table = pd.DataFrame(rows)

    # Print
    print("\n" + "=" * 110)
    print("Ablation Study — FD001 Test Set (100 engines, alert threshold = 0.3, W=100)")
    print("=" * 110)
    col_w = [22, 10, 18, 9, 8, 8, 8, 10, 8, 8, 18, 8, 12, 10]
    headers = ["Variant", "Coverage", "Coverage_CI", "N_alert", "Mean", "SD", "Median",
               "Precision", "Recall", "F1", "F1_CI", "r", "Wilcoxon-p", "Fisher-p"]
    print("  " + "".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print("  " + "-" * sum(col_w))
    for _, row in table.iterrows():
        vals = [row[h] for h in headers]
        print("  " + "".join(str(v).ljust(w) for v, w in zip(vals, col_w)))
    print("=" * 110 + "\n")

    # --- W sensitivity (full pipeline, W = 50 / 100 / 150) ---
    print("W sensitivity — Full pipeline F1 at W = 50, 100, 150:")
    w_results: dict[int, dict] = {}
    for W in [50, 100, 150]:
        w_results[W] = compute_f1(lt_fp, W=W)
        print(f"  W={W:3d}: P={w_results[W]['precision']:.3f}  R={w_results[W]['recall']:.3f}  F1={w_results[W]['f1']:.3f}")
    print()

    # Save CSV
    table_out = OUT_DIR / "ablation_table.csv"
    table.to_csv(table_out, index=False)
    print(f"Table saved to: {table_out}")

    detail_out = OUT_DIR / "ablation_engine_detail.csv"
    engine_detail.reset_index().to_csv(detail_out, index=False)
    print(f"Per-engine detail saved to: {detail_out}")

    # --- Figure ---
    plot_data = (
        engine_detail
        .reset_index()
        .melt(id_vars="engine_id", var_name="Variant", value_name="Lead Time (cycles)")
        .dropna(subset=["Lead Time (cycles)"])
    )
    order = ["Isolation Forest", "Z-score only", "Causal only", "Full pipeline", "Full pipeline + veto"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=plot_data, x="Variant", y="Lead Time (cycles)",
        order=order, palette="Set2", width=0.5, fliersize=0, ax=ax,
    )
    sns.stripplot(
        data=plot_data, x="Variant", y="Lead Time (cycles)",
        order=order, color="black", alpha=0.35, size=4, jitter=True, ax=ax,
    )
    ax.set_title("Lead Time Distribution — FD001 Ablation (alert threshold = 0.3)")
    ax.set_xlabel("")
    ax.set_ylabel("Lead Time (cycles before failure)")
    plt.tight_layout()

    fig_out = OUT_DIR / "lead_time_distribution.png"
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)
    print(f"Figure saved to: {fig_out}")


if __name__ == "__main__":
    main()
