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

import json
import os
import time
from pathlib import Path

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
from backend.services import regime_classifier as _regime_svc

# The 5 sensors in the causal DAG — used by ingest_validator
CAUSAL_SENSORS = {"sensor_3", "sensor_4", "sensor_9", "sensor_11", "sensor_15"}

_COEFF_FILE = Path(__file__).parent.parent.parent / "data" / "processed" / "regime_coefficients.json"


def _load_blend_alpha(is_multi_cluster: bool) -> float:
    """
    Load the learned blend weight α from regime_coefficients.json.

    α controls:  final_score = α × causal_score + (1-α) × z_score
    - FD001 (single condition): α = 0.70 — causal weighted higher, z-score retained
    - FD002 (multi condition):  α = 1.00 — pure causal; z-score is harmful here
    Falls back to 0.5 if the JSON is missing or malformed.
    """
    try:
        data = json.loads(_COEFF_FILE.read_text())
        key = "blend_alpha_fd002" if is_multi_cluster else "blend_alpha_fd001"
        return float(data.get(key, data.get("blend_alpha", 0.5)))
    except Exception:
        return 0.5


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
    Classify the operating regime using nearest-centroid assignment.

    Uses KMeans centroids from data/processed/regime_coefficients.json
    (trained on FD002's 6 operating conditions).  Falls back to "cluster_0"
    (single-condition FD001 mode) if the coefficients file is absent.
    """
    t0 = time.monotonic()
    reading = state.get("reading", {})
    op1 = reading.get("op_setting_1") or 0.0
    op2 = reading.get("op_setting_2") or 0.0
    op3 = reading.get("op_setting_3") or 0.0

    regime = _regime_svc.classify(op1, op2, op3)

    out = {
        "regime": regime,
        "regime_multi_cluster": _regime_svc.is_multi_cluster,
    }
    _write_trace(
        alert_event_id=state["alert_event_id"],
        node_name="regime_classifier",
        input_snapshot={
            "engine_id": state.get("engine_id"),
            "op_setting_1": op1,
            "op_setting_2": op2,
            "op_setting_3": op3,
            "n_clusters": _regime_svc.n_clusters,
        },
        output_snapshot=out,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    return out


# ---------------------------------------------------------------------------
# Node 3: causal_reasoner
# ---------------------------------------------------------------------------

def causal_reasoner(state: dict) -> dict:
    """
    Recompute the causal score using per-cluster regime coefficients.

    Uses the regime label assigned by regime_classifier to select the
    appropriate LinearRegression coefficients for this operating condition.
    On single-condition data (FD001 / cluster_0) this is equivalent to the
    original scorer; on multi-condition data (FD002) each cluster's
    coefficients reflect the expected sensor values for that regime.
    """
    t0 = time.monotonic()
    reading = state.get("reading", {})
    regime = state.get("regime", "cluster_0")

    causal_score_refined, causal_details = _regime_svc.compute_causal_score(
        reading, regime
    )

    out = {
        "causal_score_refined": causal_score_refined,
        "causal_details": causal_details,
    }
    _write_trace(
        alert_event_id=state["alert_event_id"],
        node_name="causal_reasoner",
        input_snapshot={
            "causal_score_pre": state.get("causal_score"),
            "regime": regime,
        },
        output_snapshot={
            "causal_score_refined": causal_score_refined,
            "top_sensors": sorted(causal_details.items(), key=lambda kv: kv[1], reverse=True)[:3],
        },
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    return out


# ---------------------------------------------------------------------------
# Node 4: physics_veto
# ---------------------------------------------------------------------------

_CHI2_CRITICAL = 26.30   # df=16, α=0.05 — same critical value used by gtest_monitor


def _veto_factor(chi2_stat: float) -> float:
    """
    Graduated veto multiplier based on the actual G-test χ² value.

    Formula:  veto_factor = 1.0 - 0.5 × min(chi2_stat / CHI2_CRITICAL, 1.0)

    Behaviour:
        chi2 = 0               → veto_factor = 1.00  (no penalty)
        chi2 = CHI2_CRITICAL/2 → veto_factor = 0.75  (25% reduction)
        chi2 = CHI2_CRITICAL   → veto_factor = 0.50  (50% reduction, same as old binary)
        chi2 > CHI2_CRITICAL   → veto_factor = 0.50  (clamped — no stronger than 50%)

    The veto is applied to ALL causal scores, removing the old >= 0.5 gate
    that prevented vetoing moderately anomalous readings.
    """
    return 1.0 - 0.5 * min(chi2_stat / _CHI2_CRITICAL, 1.0)


def physics_veto(state: dict) -> dict:
    """
    Apply a graduated physics-based veto if the sensor_11/sensor_15 coupling
    is degraded or broken.

    The G-test (computed by gtest_monitor) checks whether sensor_11 (HPC
    outlet temp) and sensor_15 (HPC outlet pressure) are still correlated.
    In normal operation these sensors obey the isentropic compression relation
        T2/T1 = (P2/P1)^((gamma-1)/gamma), gamma ~ 1.4 for air,
    so a loss of correlation signals a sensor fault, not engine degradation.
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
    g_stat = None
    is_decorrelated = False
    if buffer_full:
        g_stat, is_decorrelated = gtest_monitor.run_gtest(engine_id)
        if g_stat is not None:
            factor = _veto_factor(g_stat)
            if factor < 1.0:
                causal_score_refined = round(causal_score_refined * factor, 6)
                physics_veto_applied = (factor < 0.95)   # log if non-trivial

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
            "g_stat": g_stat,
            "is_decorrelated": is_decorrelated,
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

    final_score = α × causal_score_refined + (1-α) × z_score
    where α is loaded from regime_coefficients.json:
        FD001 (single condition): α = 0.70
        FD002 (multi  condition): α = 1.00 (pure causal; z-score is harmful)

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

    alpha = _load_blend_alpha(is_multi_cluster=_regime_svc.is_multi_cluster)
    final_score = round(alpha * causal_score_refined + (1 - alpha) * z_score, 6)
    # Use per-regime calibrated threshold when available (falls back to 0.3)
    cluster_label = state.get("cluster_label", "cluster_0")
    threshold = _regime_svc.get_alert_threshold(cluster_label)
    final_decision, final_confidence = make_decision(final_score, threshold=threshold)

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
