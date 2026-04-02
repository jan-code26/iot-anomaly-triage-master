from fastapi import FastAPI, HTTPException
from sqlalchemy import insert, select

from backend.database import engine
from backend.models import telemetry_windows
from backend.schemas import TelemetryReading, TelemetryWindowOut

app = FastAPI(
    title="Don't Trust the Sensors — IoT Anomaly Triage",
    version="0.1.0"
)


@app.get("/health")
def health_check():
    """
    A simple health check endpoint.
    This is the first thing Render will ping to confirm your app is running.
    If this returns 200 OK, the deployment succeeded.
    """
    return {"status": "ok", "message": "Sensor triage system is running"}


@app.post("/ingest", response_model=TelemetryWindowOut, status_code=201)
def ingest(reading: TelemetryReading):
    """
    Accepts a single engine sensor reading, validates it, and saves it to
    the telemetry_windows table in Neon Postgres.

    Returns the saved row including the database-generated id and created_at.
    """
    data = reading.model_dump()

    try:
        with engine.begin() as conn:
            result = conn.execute(
                insert(telemetry_windows).values(**data).returning(
                    telemetry_windows.c.id,
                    telemetry_windows.c.engine_id,
                    telemetry_windows.c.cycle,
                    telemetry_windows.c.imputation_density,
                    telemetry_windows.c.created_at,
                )
            )
            row = result.mappings().one()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return TelemetryWindowOut(**row)
