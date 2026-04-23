import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import func, insert, select, update

from backend.anomaly import compute_anomaly_score, make_decision
from backend.database import engine
from backend.logging_config import get_logger, setup_logging
from backend.models import alert_events, human_feedback, maintenance_events, reasoning_traces, telemetry_windows
from backend.schemas import (
    AlertEventOut, DemoEngineOut, EngineSummaryOut, FeedbackOut,
    FeedbackRequest, IngestOut, PipelineNodeOut,
    TelemetryReading, TelemetryWindowOut,
)
from backend.services.causal_scorer import compute_causal_score, save_dowhy_result
from backend.services.gtest_monitor import G_THRESHOLD, gtest_monitor
from backend.services.psi_monitor import psi_monitor
from backend.services.sensor_service import sensor_service

_DASHBOARD = Path(__file__).parent.parent / "dashboard"
_DEMO_DATA_PATH = Path(__file__).parent / "demo_data.json"
_demo_engines: list[dict] = []

def _load_demo_data() -> None:
    global _demo_engines
    if _DEMO_DATA_PATH.exists():
        _demo_engines = json.loads(_DEMO_DATA_PATH.read_text())

setup_logging()
log = get_logger("backend.main")
_load_demo_data()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Don't Trust the Sensors — IoT Anomaly Triage",
    version="0.1.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dashboard pages
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return FileResponse(_DASHBOARD / "showcase.html")

@app.get("/dashboard")
def dashboard():
    return FileResponse(_DASHBOARD / "index.html")

@app.get("/showcase")
def showcase():
    return FileResponse(_DASHBOARD / "showcase.html")

@app.get("/tutorial")
def tutorial():
    return FileResponse(_DASHBOARD / "tutorial.html")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Sensor triage system is running"}


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestOut, status_code=201)
@limiter.limit("10/second")
def ingest(request: Request, reading: TelemetryReading):
    """
    Accept one sensor reading, forward-fill missing values, score it,
    write to telemetry_windows + alert_events, return the saved row.

    Rate limited to 10 requests/second per IP address.
    """
    raw = reading.model_dump()

    # --- forward-fill ---
    sensor_values = {f"sensor_{i}": raw.get(f"sensor_{i}") for i in range(1, 22)}
    filled, stale_sensors, warnings = sensor_service.process(
        reading.engine_id, reading.cycle, sensor_values
    )

    # Recompute imputation density after fill
    imputed_count = sum(
        1 for s in sensor_values
        if sensor_values[s] is None and filled.get(s) is not None
    )
    imputation_density = imputed_count / 21

    # Build the row to insert (use filled sensor values)
    row_data = {
        "engine_id": reading.engine_id,
        "cycle": reading.cycle,
        "op_setting_1": raw.get("op_setting_1"),
        "op_setting_2": raw.get("op_setting_2"),
        "op_setting_3": raw.get("op_setting_3"),
        "imputation_density": imputation_density,
        "stale_sensors": stale_sensors,
        **filled,
    }

    # --- G-test: add reading and check coupling ---
    gtest_monitor.add(reading.engine_id, filled.get("sensor_11"), filled.get("sensor_15"))
    if gtest_monitor.should_run(reading.engine_id):
        g_stat, is_decorrelated = gtest_monitor.run_gtest(reading.engine_id)
        if is_decorrelated:
            warnings.append(
                f"G-test: sensor_11/sensor_15 coupling broken "
                f"(G={g_stat}, threshold={G_THRESHOLD}) — possible sensor fault"
            )

    # --- PSI: feed current readings ---
    for s in [f"sensor_{i}" for i in range(1, 22)]:
        psi_monitor.add_reading(s, filled.get(s))

    try:
        with engine.begin() as conn:
            # 1. Save telemetry row
            result = conn.execute(
                insert(telemetry_windows).values(**row_data).returning(
                    telemetry_windows.c.id,
                    telemetry_windows.c.engine_id,
                    telemetry_windows.c.cycle,
                    telemetry_windows.c.imputation_density,
                    telemetry_windows.c.stale_sensors,
                    telemetry_windows.c.created_at,
                )
            )
            row = result.mappings().one()

            # 2. Score: z-score (global) + causal (conditioned on op_settings)
            reading_dict = {**filled, **{
                "op_setting_1": raw.get("op_setting_1"),
                "op_setting_2": raw.get("op_setting_2"),
                "op_setting_3": raw.get("op_setting_3"),
            }}
            z_score = compute_anomaly_score(reading_dict)
            causal_score, causal_details = compute_causal_score(reading_dict)

            # Blend 50/50. In Phase 4 adjust weights based on lead-time results.
            combined_score = round(0.5 * z_score + 0.5 * causal_score, 6)
            decision, confidence = make_decision(combined_score)

            # 3. Save causal score to dowhy_results
            save_dowhy_result(
                conn,
                telemetry_window_id=str(row["id"]),
                regime="cluster_0",
                causal_score=causal_score,
                from_cache=False,
            )

            # 4. Write alert — RETURNING id so the LangGraph agent can reference it
            alert_result = conn.execute(
                insert(alert_events).values(
                    telemetry_window_id=row["id"],
                    anomaly_score=combined_score,
                    decision=decision,
                    confidence=confidence,
                    cache_hit=False,
                ).returning(alert_events.c.id)
            )
            alert_event_id = str(alert_result.scalar_one())
    except Exception as exc:
        log.error("ingest_db_error", extra={
            "engine_id": reading.engine_id, "cycle": reading.cycle, "error": str(exc),
        })
        raise HTTPException(status_code=500, detail=str(exc))

    log.info("ingest_scored", extra={
        "engine_id": reading.engine_id,
        "cycle": reading.cycle,
        "z_score": round(z_score, 4),
        "causal_score": round(causal_score, 4),
        "combined_score": combined_score,
        "decision": decision,
        "confidence": round(confidence, 4),
        "imputation_density": round(imputation_density, 4),
        "stale_count": len(stale_sensors),
    })

    # --- LangGraph agent (runs outside the main DB transaction) ---
    # LLM latency (0.5-2s) must not hold the connection open.
    # Agent is non-fatal: if anything raises, we fall back to pre-agent decision.
    llm_explanation = None
    if combined_score >= 0.3:
        try:
            from backend.agent.graph import run_triage_agent
            agent_result = run_triage_agent({
                "engine_id": reading.engine_id,
                "cycle": reading.cycle,
                "telemetry_window_id": str(row["id"]),
                "alert_event_id": alert_event_id,
                "z_score": z_score,
                "causal_score": causal_score,
                "combined_score": combined_score,
                "causal_details": causal_details,
                "reading": reading_dict,
                "stale_sensors": stale_sensors,
                "agent_warnings": [],
            })
            llm_explanation = agent_result.get("llm_explanation")
            agent_warnings = agent_result.get("agent_warnings", [])
            for w in agent_warnings:
                warnings.append(w)
            log.info("agent_complete", extra={
                "engine_id": reading.engine_id,
                "cycle": reading.cycle,
                "regime": agent_result.get("regime"),
                "has_explanation": bool(llm_explanation),
                "agent_warnings": agent_warnings,
            })
        except Exception as exc:
            warnings.append(
                f"Agent unavailable (pre-agent decision kept): {str(exc)[:80]}"
            )
            log.warning("agent_failed", extra={
                "engine_id": reading.engine_id,
                "cycle": reading.cycle,
                "error": str(exc)[:120],
            })

    return IngestOut(
        id=row["id"],
        engine_id=row["engine_id"],
        cycle=row["cycle"],
        imputation_density=row["imputation_density"],
        z_score=round(z_score, 6),
        causal_score=round(causal_score, 6),
        combined_score=combined_score,
        decision=decision,
        confidence=round(confidence, 6),
        alert_event_id=alert_event_id,
        warnings=warnings,
        llm_explanation=llm_explanation,
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# Retrieve a single telemetry window by UUID
# ---------------------------------------------------------------------------

@app.get("/telemetry/{telemetry_id}", response_model=TelemetryWindowOut)
def get_telemetry(telemetry_id: str):
    """Return one telemetry_windows row by its UUID."""
    try:
        with engine.connect() as conn:
            result = conn.execute(
                select(
                    telemetry_windows.c.id,
                    telemetry_windows.c.engine_id,
                    telemetry_windows.c.cycle,
                    telemetry_windows.c.imputation_density,
                    telemetry_windows.c.stale_sensors,
                    telemetry_windows.c.created_at,
                ).where(telemetry_windows.c.id == telemetry_id)
            )
            row = result.mappings().one_or_none()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if row is None:
        raise HTTPException(status_code=404, detail="Telemetry window not found")

    return TelemetryWindowOut(
        id=row["id"],
        engine_id=row["engine_id"],
        cycle=row["cycle"],
        imputation_density=row["imputation_density"],
        stale_sensors=row["stale_sensors"] or [],
        warnings=[],
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# PSI endpoints
# ---------------------------------------------------------------------------

@app.get("/psi/status")
def psi_status():
    """Return current PSI score and status for all sensors."""
    return {"sensors": psi_monitor.all_status()}


class BaselineResetRequest(BaseModel):
    engine_id: int


@app.post("/baselines/reset", status_code=200)
def reset_baseline(body: BaselineResetRequest):
    """
    Log a maintenance event and clear PSI baselines for all sensors.
    Call this after physical maintenance on an engine.
    """
    try:
        with engine.begin() as conn:
            conn.execute(
                insert(maintenance_events).values(
                    engine_id=body.engine_id,
                    event_type="baseline_reset",
                    notes="PSI baseline cleared via API",
                )
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    for s in [f"sensor_{i}" for i in range(1, 22)]:
        psi_monitor.clear_baseline(s)

    return {"status": "ok", "message": f"Baseline reset for engine {body.engine_id}"}


# ---------------------------------------------------------------------------
# Human feedback loop
# ---------------------------------------------------------------------------

@app.post("/feedback", response_model=FeedbackOut, status_code=201)
def submit_feedback(body: FeedbackRequest):
    """
    Submit an operator label for an alert event.

    label must be one of: TRUE_POSITIVE, FALSE_POSITIVE, UNCERTAIN.
    Set override=True to also update the alert_events decision immediately
    (sets confidence=1.0 — operator is treated as ground truth).

    The LangGraph cache_lookup node reads these labels: ≥2 FALSE_POSITIVE
    labels for the same engine triggers a 0.7 confidence penalty on future alerts.
    """
    try:
        with engine.begin() as conn:
            alert_row = conn.execute(
                select(alert_events.c.id)
                .where(alert_events.c.id == body.alert_event_id)
            ).mappings().one_or_none()
            if alert_row is None:
                raise HTTPException(status_code=404, detail="Alert event not found")

            result = conn.execute(
                insert(human_feedback).values(
                    alert_event_id=body.alert_event_id,
                    label=body.label,
                    override=body.override,
                ).returning(
                    human_feedback.c.id,
                    human_feedback.c.submitted_at,
                )
            )
            row = result.mappings().one()

            if body.override:
                conn.execute(
                    update(alert_events)
                    .where(alert_events.c.id == body.alert_event_id)
                    .values(decision=body.label, confidence=1.0)
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return FeedbackOut(
        id=row["id"],
        alert_event_id=body.alert_event_id,
        label=body.label,
        override=body.override,
        submitted_at=row["submitted_at"],
    )


@app.get("/alerts/recent", response_model=list[AlertEventOut])
def get_recent_alerts(limit: int = 50):
    """Return the most recent alert events with engine_id and cycle, newest first."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                select(
                    alert_events.c.id,
                    alert_events.c.telemetry_window_id,
                    alert_events.c.triggered_at,
                    alert_events.c.anomaly_score,
                    alert_events.c.decision,
                    alert_events.c.confidence,
                    alert_events.c.cache_hit,
                    telemetry_windows.c.engine_id,
                    telemetry_windows.c.cycle,
                )
                .join(telemetry_windows, alert_events.c.telemetry_window_id == telemetry_windows.c.id)
                .order_by(alert_events.c.triggered_at.desc())
                .limit(limit)
            ).mappings().all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return [AlertEventOut(**dict(row)) for row in rows]


@app.get("/alerts/{alert_id}/trace", response_model=list[PipelineNodeOut])
def get_alert_trace(alert_id: str):
    """Return the full LangGraph pipeline trace for a given alert."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                select(
                    reasoning_traces.c.node_name,
                    reasoning_traces.c.output_state,
                    reasoning_traces.c.latency_ms,
                    reasoning_traces.c.created_at,
                )
                .where(reasoning_traces.c.alert_event_id == alert_id)
                .order_by(reasoning_traces.c.id.asc())
            ).mappings().all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return [PipelineNodeOut(**dict(row)) for row in rows]


@app.get("/engines/summary", response_model=list[EngineSummaryOut])
def get_engines_summary():
    """Return current status for every engine seen, sorted by risk (highest score first)."""
    try:
        with engine.connect() as conn:
            # Latest alert per engine
            subq = (
                select(
                    telemetry_windows.c.engine_id,
                    func.max(alert_events.c.triggered_at).label("last_seen"),
                )
                .join(alert_events, alert_events.c.telemetry_window_id == telemetry_windows.c.id)
                .group_by(telemetry_windows.c.engine_id)
                .subquery()
            )

            rows = conn.execute(
                select(
                    telemetry_windows.c.engine_id,
                    alert_events.c.anomaly_score,
                    alert_events.c.decision,
                    alert_events.c.confidence,
                    telemetry_windows.c.cycle,
                    alert_events.c.triggered_at,
                    alert_events.c.id.label("alert_id"),
                )
                .join(alert_events, alert_events.c.telemetry_window_id == telemetry_windows.c.id)
                .join(subq, (subq.c.engine_id == telemetry_windows.c.engine_id) &
                             (alert_events.c.triggered_at == subq.c.last_seen))
                .order_by(alert_events.c.anomaly_score.desc())
            ).mappings().all()

            # Count total alerts per engine
            counts = conn.execute(
                select(
                    telemetry_windows.c.engine_id,
                    func.count(alert_events.c.id).label("alert_count"),
                )
                .join(alert_events, alert_events.c.telemetry_window_id == telemetry_windows.c.id)
                .group_by(telemetry_windows.c.engine_id)
            ).mappings().all()
            count_map = {r["engine_id"]: r["alert_count"] for r in counts}

            # Fetch regime for each latest alert from reasoning_traces
            regime_map: dict[int, str] = {}
            for row in rows:
                regime_row = conn.execute(
                    select(reasoning_traces.c.output_state)
                    .where(reasoning_traces.c.alert_event_id == str(row["alert_id"]))
                    .where(reasoning_traces.c.node_name == "regime_classifier")
                    .limit(1)
                ).mappings().one_or_none()
                if regime_row:
                    regime_map[row["engine_id"]] = (regime_row["output_state"] or {}).get("regime")

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    seen = set()
    result = []
    for row in rows:
        eid = row["engine_id"]
        if eid in seen:
            continue
        seen.add(eid)
        result.append(EngineSummaryOut(
            engine_id=eid,
            latest_score=row["anomaly_score"],
            latest_decision=row["decision"],
            latest_confidence=row["confidence"],
            latest_cycle=row["cycle"],
            alert_count=count_map.get(eid, 0),
            last_seen=row["triggered_at"],
            regime=regime_map.get(eid),
        ))
    return result


@app.get("/demo/readings", response_model=list[DemoEngineOut])
def get_demo_readings():
    """Return the 3 curated FD001 demo engines for the dashboard demo mode."""
    if not _demo_engines:
        raise HTTPException(status_code=404, detail="Demo data not available")
    return [DemoEngineOut(**e) for e in _demo_engines]


@app.get("/alerts/{alert_id}/explanation")
def get_alert_explanation(alert_id: str):
    """
    Return the LLM explanation and regime for a given alert event,
    sourced from the reasoning_traces rows written by the LangGraph agent.
    Returns {} if the agent did not run (score < 0.3) or traces are absent.
    """
    try:
        with engine.connect() as conn:
            # LLM explanation lives in the llm_explainer node's output_state
            exp_row = conn.execute(
                select(reasoning_traces.c.output_state, reasoning_traces.c.latency_ms)
                .where(reasoning_traces.c.alert_event_id == alert_id)
                .where(reasoning_traces.c.node_name == "llm_explainer")
                .order_by(reasoning_traces.c.id.desc())
                .limit(1)
            ).mappings().one_or_none()

            regime_row = conn.execute(
                select(reasoning_traces.c.output_state)
                .where(reasoning_traces.c.alert_event_id == alert_id)
                .where(reasoning_traces.c.node_name == "regime_classifier")
                .order_by(reasoning_traces.c.id.desc())
                .limit(1)
            ).mappings().one_or_none()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    explanation = None
    latency_ms = None
    regime = None
    if exp_row:
        explanation = (exp_row["output_state"] or {}).get("llm_explanation")
        latency_ms = exp_row["latency_ms"]
    if regime_row:
        regime = (regime_row["output_state"] or {}).get("regime")

    return {
        "alert_id": alert_id,
        "llm_explanation": explanation,
        "regime": regime,
        "llm_latency_ms": latency_ms,
    }


@app.get("/alerts/{alert_id}/feedback", response_model=list[FeedbackOut])
def get_alert_feedback(alert_id: str):
    """Return all operator feedback submitted for a given alert event."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                select(
                    human_feedback.c.id,
                    human_feedback.c.alert_event_id,
                    human_feedback.c.label,
                    human_feedback.c.override,
                    human_feedback.c.submitted_at,
                )
                .where(human_feedback.c.alert_event_id == alert_id)
                .order_by(human_feedback.c.submitted_at.desc())
            ).mappings().all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return [FeedbackOut(**dict(row)) for row in rows]
