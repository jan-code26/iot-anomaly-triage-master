"""
Unit tests for the causal anomaly scorer.

No database, no network. All tests use the module-level FALLBACK_COEFFICIENTS
so they pass whether or not data/raw/train_FD001.txt is present.
"""
from backend.services.causal_scorer import (
    CAUSAL_BRANCHES,
    FALLBACK_COEFFICIENTS,
    build_dot_graph,
    compute_causal_score,
)


# ---------------------------------------------------------------------------
# Helper: build a "normal" reading using the expected values (mean residual ≈ 0)
# ---------------------------------------------------------------------------

def _normal_reading() -> dict:
    """Return a reading at the predicted value for each causal sensor."""
    reading = {}
    for sensor, coefs in FALLBACK_COEFFICIENTS.items():
        # Use the intercept as the op_setting value (op = 0 → predicted = intercept)
        reading[coefs["cause"]] = 0.0
        reading[sensor] = coefs["intercept"]  # residual = 0 → z = 0
    return reading


def _degraded_reading(multiplier: float = 5.0) -> dict:
    """Return a reading where each causal sensor is pushed multiplier × std away."""
    reading = _normal_reading()
    for sensor, coefs in FALLBACK_COEFFICIENTS.items():
        reading[sensor] = coefs["intercept"] + multiplier * coefs["residual_std"]
    return reading


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_causal_score_normal_reading_is_low():
    """A reading at the predicted value should score near 0."""
    score, _ = compute_causal_score(_normal_reading())
    assert score < 0.1, f"Expected score < 0.1, got {score}"


def test_causal_score_degraded_reading_is_higher():
    """A reading 5 std away from prediction should score near 1.0."""
    score, _ = compute_causal_score(_degraded_reading(multiplier=5.0))
    assert score > 0.5, f"Expected score > 0.5, got {score}"


def test_causal_score_returns_tuple():
    """compute_causal_score must return (float, dict)."""
    result = compute_causal_score(_normal_reading())
    assert isinstance(result, tuple) and len(result) == 2
    score, details = result
    assert isinstance(score, float)
    assert isinstance(details, dict)


def test_causal_score_details_contain_sensor_names():
    """The details dict should have entries for each causal sensor that was present."""
    _, details = compute_causal_score(_normal_reading())
    for sensor in FALLBACK_COEFFICIENTS:
        assert sensor in details, f"Expected {sensor} in details"


def test_causal_score_handles_none_sensor_value():
    """None sensor values should be skipped — score still valid from remaining sensors."""
    reading = _normal_reading()
    reading["sensor_4"] = None  # remove one sensor
    score, details = compute_causal_score(reading)
    assert isinstance(score, float)
    assert "sensor_4" not in details


def test_causal_score_handles_none_op_setting():
    """None op_setting falls back to intercept-only prediction — should not crash."""
    reading = _normal_reading()
    reading["op_setting_1"] = None  # altitude missing
    # sensor_4's predicted value = intercept (coef * 0 is same as intercept-only for op=0)
    score, _ = compute_causal_score(reading)
    assert isinstance(score, float)


def test_causal_score_empty_reading_returns_zero():
    """No usable sensor values → score of 0.0."""
    score, details = compute_causal_score({})
    assert score == 0.0
    assert details == {}


def test_causal_score_clamped_to_one():
    """Extremely high residuals should be clamped to 1.0."""
    score, _ = compute_causal_score(_degraded_reading(multiplier=100.0))
    assert score == 1.0, f"Expected 1.0, got {score}"


def test_build_dot_graph_contains_expected_edges():
    """The DOT graph must mention all causal sensors and op_settings."""
    dot = build_dot_graph()
    assert "op_setting_1" in dot
    assert "op_setting_2" in dot
    assert "op_setting_3" in dot
    assert "sensor_4" in dot
    assert "sensor_11" in dot
    assert "sensor_15" in dot
    assert "sensor_3" in dot
    assert "sensor_9" in dot


def test_causal_branches_cover_expected_sensors():
    """CAUSAL_BRANCHES should cover exactly the 5 sensors in FALLBACK_COEFFICIENTS."""
    all_effects = [
        s for branch in CAUSAL_BRANCHES.values() for s in branch["effects"]
    ]
    assert set(all_effects) == set(FALLBACK_COEFFICIENTS.keys())
