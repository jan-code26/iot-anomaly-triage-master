"""
LangGraph node implementations for the IoT Anomaly Triage agent.

7 nodes run in a linear sequence for every reading with combined_score >= 0.3:
    ingest_validator → regime_classifier → causal_reasoner → physics_veto
    → cache_lookup → llm_explainer → decision_writer

Each node:
  - Takes the accumulated AgentState dict
  - Returns a PARTIAL dict of only the keys it sets (LangGraph merges it in)
  - Calls _write_trace() to log its execution to reasoning_traces

_write_trace() opens its own connection so a trace write failure never
affects the main /ingest response.
"""
from __future__ import annotations

import os
import time

from sqlalchemy import func as sa_func
from sqlalchemy import insert, select, update

from backend.anomaly import make_decision
from backend.database import engine
from backend.models import (
    alert_events,
    dowhy_results,
    human_feedback,
    reasoning_traces,
    telemetry_windows,
)
from backend.services.gtest_monitor import gtest_monitor

# The 5 sensors in the causal DAG — used by ingest_validator
CAUSAL_SENSORS = {"sensor_3", "sensor_4", "sensor_9", "sensor_11", "sensor_15"}


# ---------------------------------------------------------------------------
# Trace helper — called by every node
# ---------------------------------------------------------------------------

def _write_trace(
    alert_event_id: str,
    node_name: str,
    input_snapshot: dict,
    output_snapshot: dict,
    latency_ms: int,
) -> None:
    """
    Persist one reasoning_traces row.

    Uses its own engine.begin() connection so a failure here never rolls back
    the telemetry insert or the alert_events row from the main /ingest flow.
    Failures are silently swallowed — trace writes are best-effort.

    Pass plain Python dicts for input_snapshot / output_snapshot.
    psycopg2 serialises them to JSONB automatically — do NOT json.dumps() first.
    """
    try:
        with engine.begin() as conn:
            conn.execute(
                insert(reasoning_traces).values(
                    alert_event_id=alert_event_id,
                    node_name=node_name,
                    input_state=input_snapshot,
                    output_state=output_snapshot,
                    latency_ms=latency_ms,
                )
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Node 1: ingest_validator
# ---------------------------------------------------------------------------

def ingest_validator(state: dict) -> dict:
    """
    Check how many of the 5 causal DAG sensors are stale or missing.

    A sensor counts as bad if it appears in stale_sensors OR if its value
    in the reading dict is None.  data_quality_ok = (count <= 3).
    """
    t0 = time.monotonic()
    alert_event_id = state["alert_event_id"]
    stale_sensors = state.get("stale_sensors", [])
    reading = state.get("reading", {})

    stale_set = set(stale_sensors)
    stale_causal_count = sum(
        1 for s in CAUSAL_SENSORS
        if s in stale_set or reading.get(s) is None
    )
    data_quality_ok = stale_causal_count <= 3

    out = {
        "data_quality_ok": data_quality_ok,
        "stale_causal_count": stale_causal_count,
    }
    _write_trace(
        alert_event_id=alert_event_id,
        node_name="ingest_validator",
        input_snapshot={
            "stale_sensors": stale_sensors,
            "causal_sensors_checked": sorted(CAUSAL_SENSORS),
        },
        output_snapshot=out,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    return out


# ---------------------------------------------------------------------------
# Node 2: regime_classifier
# ---------------------------------------------------------------------------

def regime_classifier(state: dict) -> dict:
    """
    Classify the operating regime.

    FD001 has a single operating condition so this always returns cluster_0.
    Phase 4 will replace this with a KMeans classifier trained on FD002's
    six op_setting clusters.
    """
    t0 = time.monotonic()
    regime = "cluster_0"
    out = {"regime": regime}
    _write_trace(
        alert_event_id=state["alert_event_id"],
        node_name="regime_classifier",
        input_snapshot={"engine_id": state.get("engine_id")},
        output_snapshot=out,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    return out


# ---------------------------------------------------------------------------
# Node 3: causal_reasoner
# ---------------------------------------------------------------------------

def causal_reasoner(state: dict) -> dict:
    """
    Pass through the pre-computed causal_score as causal_score_refined.

    The score was already computed by compute_causal_score() in /ingest
    before the agent was invoked.  In Phase 4 this node will re-run the
    scorer with regime-specific coefficients (e.g. different linear
    regression fits per FD002 cluster).
    """
    t0 = time.monotonic()
    causal_score_refined = state.get("causal_score", 0.0)
    out = {"causal_score_refined": causal_score_refined}
    _write_trace(
        alert_event_id=state["alert_event_id"],
        node_name="causal_reasoner",
        input_snapshot={
            "causal_score": state.get("causal_score"),
            "regime": state.get("regime"),
        },
        output_snapshot=out,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    return out


# ---------------------------------------------------------------------------
# Node 4: physics_veto
# ---------------------------------------------------------------------------

def physics_veto(state: dict) -> dict:
    """
    Apply a physics-based veto if the sensor_11/sensor_15 coupling is broken.

    The G-test (computed by gtest_monitor) checks whether sensor_11 (HPC
    outlet temp) and sensor_15 (HPC outlet pressure) are still correlated.
    If they appear independent AND the causal score is high (>= 0.5), the
    anomaly is more likely a sensor fault than real engine degradation — so
    we halve the causal score.

    should_run() returns True only when the per-engine deque has hit its
    maxlen (100 readings).  Most test or dev calls will skip the veto.
    """
    t0 = time.monotonic()
    engine_id = state["engine_id"]
    causal_score_refined = state.get("causal_score_refined", 0.0)
    physics_veto_applied = False

    buffer_full = gtest_monitor.should_run(engine_id)
    if buffer_full:
        _g_stat, is_decorrelated = gtest_monitor.run_gtest(engine_id)
        if is_decorrelated and causal_score_refined >= 0.5:
            causal_score_refined = round(causal_score_refined * 0.5, 6)
            physics_veto_applied = True

    out = {
        "causal_score_refined": causal_score_refined,
        "physics_veto_applied": physics_veto_applied,
    }
    _write_trace(
        alert_event_id=state["alert_event_id"],
        node_name="physics_veto",
        input_snapshot={
            "engine_id": engine_id,
            "causal_score_refined_in": state.get("causal_score_refined"),
            "buffer_full": buffer_full,
        },
        output_snapshot=out,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    return out


# ---------------------------------------------------------------------------
# Node 5: cache_lookup
# ---------------------------------------------------------------------------

def cache_lookup(state: dict) -> dict:
    """
    Check for a recent similar result and for operator false-positive feedback.

    Query 1 — cache hit:
        Look for dowhy_results rows for the same engine_id whose causal_score
        is within ±0.05 of the current reading.  from_cache = True if more
        than 1 row is found (the current row was just inserted, so any
        additional rows are prior matches).

    Query 2 — false-positive penalty:
        If the engine has ≥ 2 human_feedback rows labelled FALSE_POSITIVE,
        set cache_penalty = 0.7 so decision_writer reduces confidence by 30%.

    Both queries are wrapped in a single try/except — failure is non-fatal.
    """
    t0 = time.monotonic()
    from_cache = False
    cache_penalty = 1.0
    agent_warnings = list(state.get("agent_warnings", []))

    engine_id = state["engine_id"]
    current_causal = state.get("causal_score_refined", state.get("causal_score", 0.0))

    try:
        with engine.begin() as conn:
            # Query 1 — cache hit
            j1 = dowhy_results.join(
                telemetry_windows,
                dowhy_results.c.telemetry_window_id == telemetry_windows.c.id,
            )
            stmt1 = (
                select(dowhy_results.c.id)
                .select_from(j1)
                .where(telemetry_windows.c.engine_id == engine_id)
                .where(
                    sa_func.abs(dowhy_results.c.causal_score - current_causal) <= 0.05
                )
                .order_by(dowhy_results.c.computed_at.desc())
                .limit(10)
            )
            rows = conn.execute(stmt1).fetchall()
            from_cache = len(rows) > 1

            # Query 2 — false-positive penalty
            # Join chain: human_feedback → alert_events → telemetry_windows
            j2 = human_feedback.join(
                alert_events,
                human_feedback.c.alert_event_id == alert_events.c.id,
            ).join(
                telemetry_windows,
                alert_events.c.telemetry_window_id == telemetry_windows.c.id,
            )
            stmt2 = (
                select(sa_func.count())
                .select_from(j2)
                .where(telemetry_windows.c.engine_id == engine_id)
                .where(human_feedback.c.label == "FALSE_POSITIVE")
            )
            fp_count = conn.execute(stmt2).scalar_one()
            if fp_count >= 2:
                cache_penalty = 0.7

    except Exception as exc:
        agent_warnings.append(f"cache_lookup failed (non-fatal): {str(exc)[:80]}")

    out = {
        "from_cache": from_cache,
        "cache_penalty": cache_penalty,
        "agent_warnings": agent_warnings,
    }
    _write_trace(
        alert_event_id=state["alert_event_id"],
        node_name="cache_lookup",
        input_snapshot={
            "engine_id": engine_id,
            "current_causal": current_causal,
        },
        output_snapshot={
            "from_cache": from_cache,
            "cache_penalty": cache_penalty,
        },
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    return out


# ---------------------------------------------------------------------------
# Node 6: llm_explainer
# ---------------------------------------------------------------------------

def _build_prompt(state: dict) -> str:
    details = state.get("causal_details", {})
    top_sensors = sorted(details.items(), key=lambda kv: kv[1], reverse=True)[:3]
    veto_note = (
        " Physics veto was applied (G-test decorrelation detected)."
        if state.get("physics_veto_applied")
        else ""
    )
    sensors_str = ", ".join(f"{s}={v:.2f}" for s, v in top_sensors)
    return (
        f"You are an IoT anomaly analyst for a turbofan engine. "
        f"Engine {state.get('engine_id')} at cycle {state.get('cycle')} "
        f"has anomaly score {state.get('combined_score', 0.0):.3f} "
        f"(refined causal score: {state.get('causal_score_refined', 0.0):.3f}).{veto_note} "
        f"Top causal residuals: {sensors_str}. "
        f"In 1-2 sentences, explain the likely cause of this anomaly to a maintenance engineer."
    )


def _call_llm(state: dict) -> str:
    """Call Groq or Gemini depending on LLM_PROVIDER env var."""
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        # Lazy import so the module loads without a GROQ_API_KEY
        from groq import Groq  # noqa: PLC0415
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[{"role": "user", "content": _build_prompt(state)}],
            max_tokens=150,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    elif provider == "gemini":
        import google.generativeai as genai  # noqa: PLC0415
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel(
            os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        )
        return model.generate_content(_build_prompt(state)).text.strip()

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def _rule_based_explanation(state: dict) -> str:
    """Fallback explanation when the LLM is unavailable."""
    details = state.get("causal_details", {})
    top_2 = sorted(details.items(), key=lambda kv: kv[1], reverse=True)[:2]
    parts = []
    if top_2:
        sensors_str = " and ".join(
            f"{s} (residual z={v:.2f})" for s, v in top_2
        )
        parts.append(f"Elevated causal residuals detected in {sensors_str}.")
    if state.get("physics_veto_applied"):
        parts.append(
            "Physics veto applied: sensor_11/sensor_15 coupling is decorrelated, "
            "suggesting a possible sensor fault rather than true engine degradation."
        )
    if not parts:
        return "Anomaly detected; insufficient sensor detail for explanation."
    return " ".join(parts)


def llm_explainer(state: dict) -> dict:
    """
    Generate a plain-English explanation of the anomaly.

    Tries the configured LLM provider (Groq by default, Gemini if
    LLM_PROVIDER=gemini).  Falls back to a rule-based template string
    on any exception (rate limit, missing API key, network error).
    """
    t0 = time.monotonic()
    agent_warnings = list(state.get("agent_warnings", []))
    llm_explanation = None

    try:
        llm_explanation = _call_llm(state)
    except Exception as exc:
        agent_warnings.append(f"LLM explainer error: {str(exc)[:80]}")
        llm_explanation = _rule_based_explanation(state)

    out = {
        "llm_explanation": llm_explanation,
        "agent_warnings": agent_warnings,
    }
    _write_trace(
        alert_event_id=state["alert_event_id"],
        node_name="llm_explainer",
        input_snapshot={
            "causal_score_refined": state.get("causal_score_refined"),
            "physics_veto_applied": state.get("physics_veto_applied"),
        },
        output_snapshot={"llm_explanation": llm_explanation},
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    return out


# ---------------------------------------------------------------------------
# Node 7: decision_writer
# ---------------------------------------------------------------------------

def decision_writer(state: dict) -> dict:
    """
    Compute the final blended score, apply the cache penalty to confidence,
    and UPDATE the alert_events row with the refined values.

    final_score = round(0.5 * z_score + 0.5 * causal_score_refined, 6)

    If cache_penalty < 1.0 (operator marked ≥ 2 prior readings FALSE_POSITIVE),
    the confidence is reduced by that factor.

    The UPDATE is non-fatal: wrapped in try/except so a transient DB error
    does not fail the /ingest response.
    """
    t0 = time.monotonic()
    z_score = state.get("z_score", 0.0)
    causal_score_refined = state.get("causal_score_refined", 0.0)
    cache_penalty = state.get("cache_penalty", 1.0)
    from_cache = state.get("from_cache", False)
    alert_event_id = state["alert_event_id"]

    final_score = round(0.5 * z_score + 0.5 * causal_score_refined, 6)
    final_decision, final_confidence = make_decision(final_score)

    if cache_penalty < 1.0:
        final_confidence = round(final_confidence * cache_penalty, 4)

    try:
        with engine.begin() as conn:
            conn.execute(
                update(alert_events)
                .where(alert_events.c.id == alert_event_id)
                .values(
                    anomaly_score=final_score,
                    decision=final_decision,
                    confidence=final_confidence,
                    cache_hit=from_cache,
                )
            )
    except Exception:
        pass

    out = {
        "final_score": final_score,
        "final_decision": final_decision,
        "final_confidence": final_confidence,
    }
    _write_trace(
        alert_event_id=alert_event_id,
        node_name="decision_writer",
        input_snapshot={
            "z_score": z_score,
            "causal_score_refined": causal_score_refined,
            "cache_penalty": cache_penalty,
            "from_cache": from_cache,
        },
        output_snapshot=out,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    return out
