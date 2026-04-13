"""
Unit tests for backend.agent.nodes — no database required.

_write_trace() inside each node opens its own engine.begin() connection.
When DATABASE_URL points at an unreachable server (our test dummy) that call
fails silently (try/except: pass) — which is the correct production behaviour.

We set DATABASE_URL before any backend import so database.py doesn't raise
a RuntimeError at module load time.
"""
import os

# Must be set before any backend.* import — database.py reads it at load time.
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test_db")

import pytest  # noqa: E402

from backend.agent.nodes import (  # noqa: E402
    causal_reasoner,
    decision_writer,
    ingest_validator,
    physics_veto,
    regime_classifier,
)

# ---------------------------------------------------------------------------
# Shared base state — all tests copy this and override what they need
# ---------------------------------------------------------------------------

BASE_STATE = {
    "engine_id": 1,
    "cycle": 50,
    "telemetry_window_id": "00000000-0000-0000-0000-000000000001",
    "alert_event_id": "00000000-0000-0000-0000-000000000002",
    "z_score": 0.4,
    "causal_score": 0.5,
    "combined_score": 0.45,
    "causal_details": {
        "sensor_3": 0.9,
        "sensor_4": 1.2,
        "sensor_9": 0.3,
        "sensor_11": 0.5,
        "sensor_15": 0.1,
    },
    "reading": {
        "sensor_3": 1590.0,
        "sensor_4": 1410.0,
        "sensor_9": 9070.0,
        "sensor_11": 47.6,
        "sensor_15": 8.44,
    },
    "stale_sensors": [],
    "data_quality_ok": True,
    "stale_causal_count": 0,
    "regime": "cluster_0",
    "causal_score_refined": 0.5,
    "physics_veto_applied": False,
    "from_cache": False,
    "cache_penalty": 1.0,
    "llm_explanation": None,
    "final_score": 0.0,
    "final_decision": "NORMAL",
    "final_confidence": 0.0,
    "agent_warnings": [],
}


# ---------------------------------------------------------------------------
# Node 1: ingest_validator
# ---------------------------------------------------------------------------

def test_ingest_validator_clean():
    """All 5 causal sensors present and not stale → data_quality_ok=True, count=0."""
    state = {**BASE_STATE, "stale_sensors": [], "reading": BASE_STATE["reading"]}
    result = ingest_validator(state)
    assert result["data_quality_ok"] is True
    assert result["stale_causal_count"] == 0


def test_ingest_validator_many_stale():
    """4 of 5 causal sensors stale → data_quality_ok=False, stale_causal_count=4."""
    state = {
        **BASE_STATE,
        "stale_sensors": ["sensor_3", "sensor_4", "sensor_9", "sensor_11"],
    }
    result = ingest_validator(state)
    assert result["data_quality_ok"] is False
    assert result["stale_causal_count"] == 4


def test_ingest_validator_none_reading_counts_as_stale():
    """A causal sensor with None in reading (not in stale_sensors) still counts."""
    reading = dict(BASE_STATE["reading"])
    reading["sensor_3"] = None
    reading["sensor_4"] = None
    state = {**BASE_STATE, "stale_sensors": [], "reading": reading}
    result = ingest_validator(state)
    assert result["stale_causal_count"] == 2
    assert result["data_quality_ok"] is True  # 2 <= 3 threshold


# ---------------------------------------------------------------------------
# Node 2: regime_classifier
# ---------------------------------------------------------------------------

def test_regime_classifier_returns_cluster_0():
    result = regime_classifier(BASE_STATE)
    assert result["regime"] == "cluster_0"


# ---------------------------------------------------------------------------
# Node 3: causal_reasoner
# ---------------------------------------------------------------------------

def test_causal_reasoner_passes_through_score():
    state = {**BASE_STATE, "causal_score": 0.42}
    result = causal_reasoner(state)
    assert result["causal_score_refined"] == 0.42


def test_causal_reasoner_missing_key_defaults_to_zero():
    """If causal_score not in state, defaults to 0.0."""
    state = {k: v for k, v in BASE_STATE.items() if k != "causal_score"}
    result = causal_reasoner(state)
    assert result["causal_score_refined"] == 0.0


# ---------------------------------------------------------------------------
# Node 4: physics_veto
# ---------------------------------------------------------------------------

def test_physics_veto_no_buffer():
    """
    engine_id=999 has no G-test readings → should_run(999) is False → no veto.
    causal_score_refined passes through unchanged.
    """
    state = {**BASE_STATE, "engine_id": 999, "causal_score_refined": 0.8}
    result = physics_veto(state)
    # Empty buffer → should_run returns False → no veto
    assert result["physics_veto_applied"] is False
    assert result["causal_score_refined"] == 0.8


# ---------------------------------------------------------------------------
# Node 7: decision_writer
# ---------------------------------------------------------------------------

def test_decision_writer_score_math():
    """final_score = round(0.5 * z_score + 0.5 * causal_score_refined, 6)."""
    state = {**BASE_STATE, "z_score": 0.4, "causal_score_refined": 0.6, "cache_penalty": 1.0}
    result = decision_writer(state)
    expected = round(0.5 * 0.4 + 0.5 * 0.6, 6)
    assert result["final_score"] == expected  # 0.5


def test_decision_writer_cache_penalty():
    """cache_penalty=0.7 reduces final_confidence by 30%."""
    state = {
        **BASE_STATE,
        "z_score": 0.7,
        "causal_score_refined": 0.7,
        "cache_penalty": 0.7,
        "from_cache": True,
    }
    result = decision_writer(state)
    # score = round(0.5*0.7 + 0.5*0.7, 6) = 0.7
    # make_decision(0.7) → ("ALERT", 0.7)
    # confidence after penalty = round(0.7 * 0.7, 4) = 0.49
    assert result["final_decision"] == "ALERT"
    assert result["final_confidence"] == pytest.approx(round(0.7 * 0.7, 4))


def test_decision_writer_no_penalty_when_one():
    """cache_penalty=1.0 → final_confidence is unchanged."""
    from backend.anomaly import make_decision

    state = {
        **BASE_STATE,
        "z_score": 0.7,
        "causal_score_refined": 0.7,
        "cache_penalty": 1.0,
        "from_cache": False,
    }
    result = decision_writer(state)
    _, expected_conf = make_decision(0.7)
    assert result["final_confidence"] == expected_conf
