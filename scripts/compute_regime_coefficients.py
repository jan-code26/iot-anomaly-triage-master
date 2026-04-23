"""
Compute KMeans cluster centroids and per-cluster causal coefficients from
train_FD002.txt and save to data/processed/regime_coefficients.json.

This JSON file is loaded at startup by backend/services/regime_classifier.py
so the live backend can classify operating regimes and score readings with
per-cluster causal coefficients — without needing the raw training data present.

Usage:
    python scripts/compute_regime_coefficients.py

Output:
    data/processed/regime_coefficients.json
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
OUT_DIR = ROOT / "data" / "processed"

COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

N_CLUSTERS = 6

# Same causal DAG as causal_scorer.py
CAUSAL_SENSORS: dict[str, str] = {
    "sensor_4":  "op_setting_1",
    "sensor_11": "op_setting_2",
    "sensor_15": "op_setting_2",
    "sensor_3":  "op_setting_3",
    "sensor_9":  "op_setting_3",
}


def main() -> None:
    train_path = DATA_DIR / "train_FD002.txt"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found.")
        sys.exit(1)

    print("Loading train_FD002.txt...")
    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMNS)
    print(f"  {len(train):,} rows, {train['engine_id'].nunique()} engines")

    # --- KMeans on op_settings ---
    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
    print(f"\nTraining KMeans (n_clusters={N_CLUSTERS})...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    km.fit(train[op_cols].values)
    centroids = km.cluster_centers_.tolist()  # shape (6, 3)

    cluster_labels = km.predict(train[op_cols].values)
    train = train.copy()
    train["_cluster"] = cluster_labels

    print(f"\n{'Cluster':<10} {'N rows':<10} {'op1':>8} {'op2':>8} {'op3':>8}")
    print("-" * 50)
    for c in range(N_CLUSTERS):
        sub = train[train["_cluster"] == c]
        m = sub[op_cols].mean()
        print(f"  {c:<8} {len(sub):<10} {m['op_setting_1']:>8.2f} {m['op_setting_2']:>8.3f} {m['op_setting_3']:>8.2f}")

    # --- Per-cluster causal coefficients ---
    print("\nFitting per-cluster causal coefficients...")
    cluster_coefficients: dict[str, dict] = {}

    for c in range(N_CLUSTERS):
        subset = train[train["_cluster"] == c]
        cluster_coefficients[str(c)] = {}
        for sensor, cause in CAUSAL_SENSORS.items():
            X = subset[[cause]].values
            y = subset[sensor].values
            if len(X) < 2:
                cluster_coefficients[str(c)][sensor] = {
                    "cause": cause,
                    "coef": 0.0,
                    "intercept": float(y.mean()),
                    "residual_std": 1.0,
                }
                continue
            model = LinearRegression().fit(X, y)
            residuals = y - model.predict(X)
            residual_std = float(np.std(residuals))
            cluster_coefficients[str(c)][sensor] = {
                "cause": cause,
                "coef": float(model.coef_[0]),
                "intercept": float(model.intercept_),
                "residual_std": max(residual_std, 1e-6),
            }
        print(f"  Cluster {c}: {len(subset):,} rows — coefficients fitted")

    # --- Save ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "regime_coefficients.json"
    payload = {
        "n_clusters": N_CLUSTERS,
        "op_cols": op_cols,
        "centroids": centroids,
        "cluster_coefficients": cluster_coefficients,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
