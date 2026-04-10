from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import insert, select

from backend.anomaly import compute_anomaly_score, make_decision
from backend.database import engine
from backend.models import alert_events, maintenance_events, telemetry_windows
from backend.schemas import TelemetryReading, TelemetryWindowOut
from backend.services.causal_scorer import compute_causal_score, save_dowhy_result
from backend.services.gtest_monitor import gtest_monitor
from backend.services.psi_monitor import psi_monitor
from backend.services.sensor_service import sensor_service

app = FastAPI(
    title="Don't Trust the Sensors — IoT Anomaly Triage",
    version="0.1.0"
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Sensor triage system is running"}


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=TelemetryWindowOut, status_code=201)
def ingest(reading: TelemetryReading):
    """
    Accept one sensor reading, forward-fill missing values, score it,
    write to telemetry_windows + alert_events, return the saved row.
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
                f"(G={g_stat}, threshold=9.49) — possible sensor fault"
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
        raise HTTPException(status_code=500, detail=str(exc))

    return TelemetryWindowOut(
        id=row["id"],
        engine_id=row["engine_id"],
        cycle=row["cycle"],
        imputation_density=row["imputation_density"],
        stale_sensors=row["stale_sensors"] or [],
        warnings=warnings,
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
