"""
Pydantic v2 schemas for the IoT Anomaly Triage ingestion pipeline.

TelemetryReading    — validates incoming sensor data (POST /ingest body)
SensorStatus        — per-sensor health status
TelemetryWindowOut  — what the API returns after a successful insert
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TelemetryReading(BaseModel):
    """
    A single engine sensor reading.

    All 21 sensor fields are Optional — real IoT sensors fail.
    A None value means the reading is missing and will be imputed later.
    imputation_density is computed automatically: it's the fraction of the
    21 sensors that are None (0.0 = all present, 1.0 = all missing).
    """

    engine_id: int = Field(..., ge=1, description="Engine identifier")
    cycle: int = Field(..., ge=1, description="Current operating cycle number")

    # Operational settings
    op_setting_1: Optional[float] = None
    op_setting_2: Optional[float] = None
    op_setting_3: Optional[float] = None

    # 21 sensor readings
    sensor_1: Optional[float] = None
    sensor_2: Optional[float] = None
    sensor_3: Optional[float] = None
    sensor_4: Optional[float] = None
    sensor_5: Optional[float] = None
    sensor_6: Optional[float] = None
    sensor_7: Optional[float] = None
    sensor_8: Optional[float] = None
    sensor_9: Optional[float] = None
    sensor_10: Optional[float] = None
    sensor_11: Optional[float] = None
    sensor_12: Optional[float] = None
    sensor_13: Optional[float] = None
    sensor_14: Optional[float] = None
    sensor_15: Optional[float] = None
    sensor_16: Optional[float] = None
    sensor_17: Optional[float] = None
    sensor_18: Optional[float] = None
    sensor_19: Optional[float] = None
    sensor_20: Optional[float] = None
    sensor_21: Optional[float] = None

    # Auto-computed — do not pass this in; the validator fills it in
    imputation_density: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def compute_imputation_density(self) -> "TelemetryReading":
        sensor_fields = [f"sensor_{i}" for i in range(1, 22)]
        missing = sum(1 for f in sensor_fields if getattr(self, f) is None)
        self.imputation_density = missing / len(sensor_fields)
        return self


class SensorStatus(BaseModel):
    """Per-sensor health status, populated by the forward-fill service."""

    sensor_id: str
    status: Literal["ok", "stale", "offline"]
    last_valid_value: Optional[float] = None
    last_valid_cycle: Optional[int] = None


class TelemetryWindowOut(BaseModel):
    """
    Response model returned after a telemetry reading is saved to the database.
    Confirms the row ID, what was stored, imputation info, and any warnings.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    engine_id: int
    cycle: int
    imputation_density: float
    stale_sensors: list[str] = []
    warnings: list[str] = []
    created_at: datetime
