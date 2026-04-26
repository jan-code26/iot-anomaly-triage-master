"""
Centralised scoring constants shared across the backend.

All magic numbers used in anomaly scoring, physics veto, and decision
thresholds live here.  Import from this module instead of repeating literals.
"""
from __future__ import annotations

# ── Blend weights α (causal × α + z-score × (1-α)) ──────────────────────────
# Calibrated on the CMAPSS evaluation set (see paper §4.2).
# FD002/FD004 use pure-causal (α=1.0) because z-score degrades precision
# when operating conditions shift across the 6-regime space.
BLEND_ALPHA: dict[str, float] = {
    "FD001": 0.60,
    "FD002": 1.00,
    "FD003": 0.60,
    "FD004": 1.00,
}
BLEND_ALPHA_DEFAULT: float = 0.50   # live-streaming default (dataset-agnostic)

# ── Decision thresholds ───────────────────────────────────────────────────────
ALERT_THRESHOLD: float = 0.30       # combined score ≥ this → ALERT
UNCERTAIN_THRESHOLD: float = 0.20   # combined score ≥ this → UNCERTAIN (else NORMAL)

# ── Physics veto (G-test on sensor_11 / sensor_15 isentropic coupling) ───────
G_TEST_CHI2_CRITICAL: float = 26.30  # χ²(df=16, p=0.05)  — (5-1)×(5-1) contingency table
G_TEST_BUFFER_SIZE: int = 100        # minimum readings before the test runs
G_TEST_NUM_BINS: int = 5             # bins per sensor axis
VETO_MAX_REDUCTION: float = 0.50    # maximum fraction the score can be reduced

# ── Causal DAG sensors ────────────────────────────────────────────────────────
CAUSAL_SENSORS: tuple[str, ...] = (
    "sensor_3",   # HPC outlet temperature
    "sensor_4",   # LPT outlet temperature
    "sensor_9",   # Core speed (N2)
    "sensor_11",  # HPC outlet static pressure (Ps30)
    "sensor_15",  # Bypass ratio (BPR)
)

# ── Anomaly scoring ───────────────────────────────────────────────────────────
RESIDUAL_NOISE_FLOOR: float = 3.0   # σ — residuals below this are treated as noise
STALE_SENSOR_THRESHOLD: int = 3     # max causal sensors allowed to be stale
COLD_START_CYCLES: int = 100        # physics veto inactive during engine warm-up
