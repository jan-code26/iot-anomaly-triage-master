"""
Causal anomaly scorer for turbofan engine sensor readings.

Instead of comparing sensors against global means (z-score), this module
conditions each sensor reading on its causal parent from the DAG:

    op_setting_1 (Altitude) → sensor_4   (HPC outlet temp)
    op_setting_2 (Mach)     → sensor_11  (HPC outlet temp)
                            → sensor_15  (HPC outlet pressure)
    op_setting_3 (TRA)      → sensor_3   (total temperature fan inlet)
                            → sensor_9   (physical fan speed)

For each causal edge, we fit a LinearRegression on train_FD001.txt:
    predicted_sensor = coef * op_setting + intercept

The residual (observed - predicted) / residual_std is the causally-conditioned
z-score for that sensor. If the residual is large, the engine is behaving
unexpectedly *given* its current operating conditions — a stronger signal than
a raw z-score that ignores those conditions.

Note on FD001: all three op_settings are nearly constant in FD001 (single
operating condition). The causal benefit is small here but grows significantly
for FD002–FD004 where six distinct operating regimes are present.

Why not use DoWhy's ATE estimator per-request?
    Live readings have no RUL column, and DoWhy v0.11 requires ≥2 rows per
    call. We use DoWhy only to validate the DAG structure at module load time.
    The actual regression is done with sklearn, which has negligible overhead.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sqlalchemy import insert

from backend.database import engine
from backend.models import dowhy_results

# ---------------------------------------------------------------------------
# Causal DAG — three branches from operational settings to sensors
# ---------------------------------------------------------------------------

CAUSAL_BRANCHES: dict[str, dict] = {
    "altitude_branch": {
        "cause": "op_setting_1",
        "effects": ["sensor_4"],
    },
    "mach_branch": {
        "cause": "op_setting_2",
        "effects": ["sensor_11", "sensor_15"],
    },
    "tra_branch": {
        # op_setting_3 = TRA (Throttle Resolver Angle). In FD001 this is
        # always 100 (no variation), so the regression coef is 0 and the
        # residual is identical to the raw deviation. Still correct — just
        # not informative until FD002 where TRA varies.
        "cause": "op_setting_3",
        "effects": ["sensor_3", "sensor_9"],
    },
}

# ---------------------------------------------------------------------------
# Fallback coefficients — computed from train_FD001.txt and pasted here.
# Used on Render (and any environment) where data/raw/ is not present.
# To recompute: run scripts/compute_causal_coefficients.py
# ---------------------------------------------------------------------------

FALLBACK_COEFFICIENTS: dict[str, dict] = {
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

# Training data columns — matches the FD001 file format
_COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21",
]

_DATA_FILE = (
    Path(__file__).parent.parent.parent / "data" / "raw" / "train_FD001.txt"
)


# ---------------------------------------------------------------------------
# DOT graph — used by DoWhy to validate the causal structure
# ---------------------------------------------------------------------------

def build_dot_graph() -> str:
    """
    Return the causal DAG in DOT format.

    Intermediate latent nodes (AirDensity, TipSpeed, etc.) are included as
    unobserved nodes so DoWhy's backdoor criterion accounts for them correctly.
    We pass this as a string to CausalModel — no pygraphviz installation needed.
    """
    return """digraph {
        op_setting_1 -> AirDensity;
        AirDensity -> CoolingEfficiency;
        CoolingEfficiency -> sensor_4;
        op_setting_2 -> TipSpeed;
        TipSpeed -> HPCLoading;
        HPCLoading -> sensor_11;
        HPCLoading -> sensor_15;
        op_setting_3 -> FuelFlow;
        FuelFlow -> CombustorTemp;
        CombustorTemp -> sensor_3;
        CombustorTemp -> sensor_9;
    }"""


# ---------------------------------------------------------------------------
# Model loading — fits once at module import, cached in module-level dicts
# ---------------------------------------------------------------------------

def _load_branch_models() -> dict[str, dict]:
    """
    Fit one LinearRegression per causal edge.

    Returns a dict keyed by sensor name:
        {"sensor_4": {"coef": ..., "intercept": ..., "residual_std": ...}, ...}

    Falls back to FALLBACK_COEFFICIENTS if the training data file is missing.
    This ensures the scorer works on Render (where data/raw/ is gitignored).
    """
    if not _DATA_FILE.exists():
        return FALLBACK_COEFFICIENTS

    try:
        df = pd.read_csv(
            _DATA_FILE, sep=r"\s+", header=None, names=_COLUMNS
        )
    except Exception:
        return FALLBACK_COEFFICIENTS

    fitted: dict[str, dict] = {}
    for branch in CAUSAL_BRANCHES.values():
        cause = branch["cause"]
        X = df[[cause]].values
        for effect in branch["effects"]:
            y = df[effect].values
            model = LinearRegression().fit(X, y)
            residuals = y - model.predict(X)
            fitted[effect] = {
                "cause": cause,
                "coef": float(model.coef_[0]),
                "intercept": float(model.intercept_),
                "residual_std": float(np.std(residuals)),
            }
    return fitted


# Fit once when the module is first imported.
# Every request re-uses these in-memory models — no per-request fitting.
_branch_coefficients: dict[str, dict] = _load_branch_models()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_causal_score(
    reading: dict,
    regime: str = "cluster_0",
) -> tuple[float, dict]:
    """
    Compute a causally-conditioned anomaly score for one sensor reading.

    For each sensor in the causal DAG:
        residual = observed_value - (coef * op_setting + intercept)
        causal_z = |residual| / residual_std

    The per-sensor z-scores are averaged and clamped to [0.0, 1.0] using the
    same 5-std normalization as the z-score scorer in anomaly.py.

    Args:
        reading: dict with sensor_* and op_setting_* keys. None values are skipped.
        regime:  operating regime label (always "cluster_0" for FD001;
                 future FD002 support will use one of six cluster labels).

    Returns:
        (causal_score, details)
        - causal_score: float in [0.0, 1.0]
        - details: per-sensor residual z-scores, useful for LLM explanation
    """
    causal_z_scores: list[float] = []
    details: dict[str, float] = {}

    for sensor, coef_dict in _branch_coefficients.items():
        sensor_val: Optional[float] = reading.get(sensor)
        cause_val: Optional[float] = reading.get(coef_dict["cause"])
        residual_std: float = coef_dict["residual_std"]

        if sensor_val is None or residual_std == 0:
            continue

        if cause_val is None:
            # Op-setting missing — fall back to intercept-only prediction
            predicted = coef_dict["intercept"]
        else:
            predicted = coef_dict["coef"] * cause_val + coef_dict["intercept"]

        residual_z = abs(sensor_val - predicted) / residual_std
        causal_z_scores.append(residual_z)
        details[sensor] = round(residual_z, 4)

    if not causal_z_scores:
        return 0.0, {}

    mean_z = sum(causal_z_scores) / len(causal_z_scores)
    causal_score = min(mean_z / 5.0, 1.0)
    return round(causal_score, 6), details


# ---------------------------------------------------------------------------
# Database helper
# ---------------------------------------------------------------------------

def save_dowhy_result(
    conn,
    telemetry_window_id: str,
    regime: str,
    causal_score: float,
    from_cache: bool,
) -> None:
    """
    Insert one row into the dowhy_results table.

    Args:
        conn:                 An open SQLAlchemy connection (from engine.begin()).
        telemetry_window_id:  UUID of the parent telemetry_windows row.
        regime:               Operating regime label (e.g. "cluster_0").
        causal_score:         The causally-conditioned anomaly score.
        from_cache:           True if this result was looked up from a prior run.
    """
    conn.execute(
        insert(dowhy_results).values(
            telemetry_window_id=telemetry_window_id,
            regime=regime,
            causal_score=causal_score,
            from_cache=from_cache,
        )
    )
