"""
Unit tests for the anomaly scorer — no database required.
"""
from backend.anomaly import compute_anomaly_score, make_decision

NORMAL_READING = {
    "sensor_2": 641.82,
    "sensor_3": 1589.70,
    "sensor_4": 1400.60,
    "sensor_7": 554.36,
    "sensor_8": 2388.06,
    "sensor_9": 9046.19,
    "sensor_11": 47.47,
    "sensor_12": 521.66,
    "sensor_13": 2388.02,
    "sensor_14": 8138.62,
    "sensor_15": 8.4195,
    "sensor_17": 391.0,
    "sensor_20": 39.06,
    "sensor_21": 23.419,
}

DEGRADED_READING = {
    "sensor_2": 642.80,   # +2 std
    "sensor_3": 1620.0,   # +5 std
    "sensor_4": 1480.0,   # +5 std
    "sensor_7": 557.0,    # +5 std
    "sensor_8": 2388.40,  # +5 std
    "sensor_9": 9200.0,   # +7 std
    "sensor_11": 49.5,    # +5 std
    "sensor_12": 535.0,   # +5 std
    "sensor_13": 2388.40, # +5 std
    "sensor_14": 8220.0,  # +5 std
    "sensor_15": 8.60,    # +5 std
    "sensor_17": 404.0,   # +5 std
    "sensor_20": 40.6,    # +5 std
    "sensor_21": 24.3,    # +5 std
}


def test_normal_score_is_low():
    score = compute_anomaly_score(NORMAL_READING)
    assert score < 0.3, f"Expected score < 0.3 for normal reading, got {score}"


def test_degraded_score_is_high():
    score = compute_anomaly_score(DEGRADED_READING)
    assert score > 0.3, f"Expected score > 0.3 for degraded reading, got {score}"


def test_empty_reading_returns_zero():
    assert compute_anomaly_score({}) == 0.0


def test_all_none_returns_zero():
    reading = {f"sensor_{i}": None for i in range(1, 22)}
    assert compute_anomaly_score(reading) == 0.0


def test_score_clipped_to_one():
    extreme = {s: v * 100 for s, v in NORMAL_READING.items()}
    score = compute_anomaly_score(extreme)
    assert score <= 1.0


def test_make_decision_normal():
    decision, confidence = make_decision(0.1)
    assert decision == "NORMAL"
    assert confidence > 0.5


def test_make_decision_uncertain():
    decision, confidence = make_decision(0.45)
    assert decision == "UNCERTAIN"
    assert confidence == 0.5


def test_make_decision_alert():
    decision, confidence = make_decision(0.75)
    assert decision == "ALERT"
    assert confidence > 0.5
