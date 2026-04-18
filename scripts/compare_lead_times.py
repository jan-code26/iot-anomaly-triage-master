"""
Lead time comparison: causal pipeline vs Isolation Forest baseline.

Queries alert_events to find first ALERT per engine (from streaming test_FD001.txt),
joins with RUL_FD001.txt for true failure cycle, compares against the Isolation
Forest baseline in data/processed/isolation_forest_baseline.csv.

Usage:
    # 1. Start the server and stream test data:
    #    python scripts/simulate_stream.py --file test_FD001.txt --rows 0 --delay 0
    # 2. Then run this script:
    python scripts/compare_lead_times.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set in .env")
    sys.exit(1)

DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]


def main() -> None:
    # --- Query DB: first ALERT cycle per engine ---
    db = create_engine(DATABASE_URL)
    with db.connect() as conn:
        rows = conn.execute(text("""
            SELECT tw.engine_id, MIN(tw.cycle) AS first_alert_cycle
            FROM alert_events ae
            JOIN telemetry_windows tw ON ae.telemetry_window_id = tw.id
            WHERE ae.decision IN ('ALERT', 'UNCERTAIN')
            GROUP BY tw.engine_id
            ORDER BY tw.engine_id
        """)).mappings().all()

    causal = pd.DataFrame([dict(r) for r in rows])
    if causal.empty:
        print("No ALERT or UNCERTAIN decisions found in alert_events.")
        print("Stream test data first:")
        print("  python scripts/simulate_stream.py --file test_FD001.txt --rows 0 --delay 0")
        sys.exit(0)

    # --- Load test set + RUL labels to get true_failure_cycle ---
    test = pd.read_csv(DATA_DIR / "test_FD001.txt", sep=r"\s+", header=None, names=COLUMNS)
    rul_labels = pd.read_csv(DATA_DIR / "RUL_FD001.txt", header=None, names=["rul_at_end"])
    rul_labels["engine_id"] = rul_labels.index + 1
    last_cycles = test.groupby("engine_id")["cycle"].max().rename("last_cycle").reset_index()
    rul_labels = rul_labels.merge(last_cycles, on="engine_id")
    rul_labels["true_failure_cycle"] = rul_labels["last_cycle"] + rul_labels["rul_at_end"]

    # --- Merge causal alerts with true failure cycles ---
    # RIGHT join: keep all 100 engines; engines with no alert get NaN first_alert_cycle
    causal = causal.merge(
        rul_labels[["engine_id", "true_failure_cycle"]], on="engine_id", how="right"
    )
    causal["lead_time_cycles"] = causal["true_failure_cycle"] - causal["first_alert_cycle"]

    # --- Load IF baseline ---
    baseline_path = OUT_DIR / "isolation_forest_baseline.csv"
    if not baseline_path.exists():
        print(f"ERROR: Baseline not found at {baseline_path}")
        print("Run: python scripts/lead_time_baseline.py")
        sys.exit(1)
    baseline = pd.read_csv(baseline_path)

    # --- Compare ---
    c_valid = causal["lead_time_cycles"].dropna()
    b_valid = baseline["lead_time_cycles"].dropna()

    print("\n=== Lead Time Comparison: Causal Pipeline vs Isolation Forest ===\n")
    print(f"{'Metric':<35} {'Causal':>12} {'Iso Forest':>12}")
    print("-" * 61)
    print(f"{'Engines with any alert':<35} {len(c_valid):>12} {len(b_valid):>12}")
    print(f"{'Coverage (out of 100)':<35} {len(c_valid)/100:>11.0%} {len(b_valid)/100:>11.0%}")

    if not c_valid.empty:
        print(f"{'Mean lead time (cycles)':<35} {c_valid.mean():>12.1f} {b_valid.mean():>12.1f}")
        print(f"{'Median lead time (cycles)':<35} {c_valid.median():>12.1f} {b_valid.median():>12.1f}")
        print(f"{'Min lead time':<35} {c_valid.min():>12.0f} {b_valid.min():>12.0f}")
        print(f"{'Max lead time':<35} {c_valid.max():>12.0f} {b_valid.max():>12.0f}")
    else:
        print("  (no valid causal lead times to compute stats)")

    print()

    # --- Save causal results ---
    out_path = OUT_DIR / "causal_lead_times.csv"
    causal[["engine_id", "first_alert_cycle", "true_failure_cycle", "lead_time_cycles"]].to_csv(
        out_path, index=False
    )
    print(f"Causal lead times saved to: {out_path}")


if __name__ == "__main__":
    main()
