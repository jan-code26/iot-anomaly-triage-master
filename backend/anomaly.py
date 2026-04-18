"""
Rule-based anomaly scorer for turbofan engine sensor readings.

Uses z-scores on the 14 informative sensors identified in the EDA
(sensors 1, 5, 6, 10, 16, 18, 19 are near-constant and are ignored).

Sensor statistics (mean, std) are computed from FD001 training data.
"""

# Means and standard deviations from train_FD001.txt (.describe() output)
# Only the 14 sensors that show meaningful variation with RUL are included.
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

# Per-sensor noise floors: derived from 2 × cross-engine std at cycle 1 in train_FD001.
# Sensors with very tight training std generate inflated z-scores for physically
# insignificant deviations. The floor prevents measurement precision artifacts from
# triggering false anomaly classifications.
#   sensor_2:  2 × 0.358 ≈ 0.72  → 0.75
#   sensor_8:  2 × 0.055 ≈ 0.11  → 0.15
#   sensor_13: 2 × 0.054 ≈ 0.11  → 0.15
#   sensor_15: 2 × 0.027 ≈ 0.05  → 0.07
SENSOR_NOISE_FLOOR: dict[str, float] = {
    "sensor_2":  0.75,
    "sensor_8":  0.15,
    "sensor_13": 0.15,
    "sensor_15": 0.07,
}


def compute_anomaly_score(reading: dict) -> float:
    """
    Return a score in [0.0, 1.0] for one sensor reading.

    0.0 = indistinguishable from normal training data
    1.0 = 5+ standard deviations away from normal on average

    reading: dict with keys like 'sensor_2', 'sensor_3', etc.
             None values (missing sensors) are skipped.
    """
    z_scores = []
    for sensor, (mean, std) in SENSOR_STATS.items():
        value = reading.get(sensor)
        if value is None or std == 0:
            continue
        effective_std = max(std, SENSOR_NOISE_FLOOR.get(sensor, 0.0))
        z_scores.append(abs(value - mean) / effective_std)

    if not z_scores:
        return 0.0

    # Mean of the top-3 worst sensors: sensitive to clustered degradation
    # (real wear affects multiple related sensors) but robust to single-sensor noise.
    top3_mean = sum(sorted(z_scores, reverse=True)[:3]) / 3
    return min(top3_mean / 5.0, 1.0)


def make_decision(score: float) -> tuple[str, float]:
    """
    Convert a numeric anomaly score to a human-readable decision + confidence.

    Returns (decision, confidence) where:
        decision   — "NORMAL" | "UNCERTAIN" | "ALERT"
        confidence — how sure we are (0.0–1.0)
    """
    if score < 0.3:
        return ("NORMAL", round(1.0 - score, 4))
    if score < 0.6:
        return ("UNCERTAIN", 0.5)
    return ("ALERT", round(score, 4))
