"""
Isolation Forest baseline — lead time computation for CMAPSS FD001.

Trains an Isolation Forest on the 14 informative sensors from the FD001
training set, then evaluates it on the test set to measure how many cycles
before failure it raises its first alert.

Output: data/processed/isolation_forest_baseline.csv
Columns: engine_id, first_alert_cycle, true_failure_cycle, lead_time_cycles

Usage:
    python scripts/lead_time_baseline.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"

COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21",
]

# 14 informative sensors (near-constant ones excluded)
INFORMATIVE = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]


def load(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        print(f"ERROR: {path} not found. Run: python scripts/download_cmapss.py")
        sys.exit(1)
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def main() -> None:
    print("Loading training data (FD001)...")
    train = load("train_FD001.txt")

    # Compute RUL for training set
    max_cycles = train.groupby("engine_id")["cycle"].max().rename("max_cycle")
    train = train.join(max_cycles, on="engine_id")
    train["rul"] = train["max_cycle"] - train["cycle"]

    X_train = train[INFORMATIVE].values
    print(f"  Training set: {len(X_train)} rows, {len(INFORMATIVE)} features")

    print("Training Isolation Forest (contamination=0.05)...")
    clf = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    clf.fit(X_train)
    print("  Done.")

    print("Loading test data and RUL labels...")
    test = load("test_FD001.txt")
    rul_labels = pd.read_csv(
        DATA_DIR / "RUL_FD001.txt", header=None, names=["rul_at_end"]
    )
    rul_labels["engine_id"] = rul_labels.index + 1

    # true failure cycle = last cycle in test + rul_at_end
    last_cycles = test.groupby("engine_id")["cycle"].max().rename("last_cycle").reset_index()
    rul_labels = rul_labels.merge(last_cycles, on="engine_id")
    rul_labels["true_failure_cycle"] = rul_labels["last_cycle"] + rul_labels["rul_at_end"]

    print("Scoring test data and finding first alert per engine...")
    X_test = test[INFORMATIVE].values
    preds = clf.predict(X_test)  # -1 = anomaly, 1 = normal
    test = test.copy()
    test["is_anomaly"] = preds == -1

    records = []
    for engine_id, group in test.groupby("engine_id"):
        anomaly_rows = group[group["is_anomaly"]]
        if anomaly_rows.empty:
            first_alert_cycle = None
        else:
            first_alert_cycle = int(anomaly_rows["cycle"].min())

        true_row = rul_labels[rul_labels["engine_id"] == engine_id]
        true_failure_cycle = int(true_row["true_failure_cycle"].values[0]) if len(true_row) else None

        if first_alert_cycle is not None and true_failure_cycle is not None:
            lead_time = true_failure_cycle - first_alert_cycle
        else:
            lead_time = None

        records.append({
            "engine_id": engine_id,
            "first_alert_cycle": first_alert_cycle,
            "true_failure_cycle": true_failure_cycle,
            "lead_time_cycles": lead_time,
        })

    df_out = pd.DataFrame(records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "isolation_forest_baseline.csv"
    df_out.to_csv(out_path, index=False)

    valid = df_out["lead_time_cycles"].dropna()
    print(f"\nBaseline results ({len(valid)} engines with alerts):")
    print(f"  Mean lead time  : {valid.mean():.1f} cycles")
    print(f"  Median lead time: {valid.median():.1f} cycles")
    print(f"  Min / Max       : {valid.min():.0f} / {valid.max():.0f} cycles")
    print(f"\nSaved to: {out_path}")
    print("\nThis is your Phase 3 target — your causal pipeline must beat these numbers.")


if __name__ == "__main__":
    main()
