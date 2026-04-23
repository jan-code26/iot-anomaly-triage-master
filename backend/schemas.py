"""
Pydantic v2 schemas for the IoT Anomaly Triage ingestion pipeline.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TelemetryReading(BaseModel):
    engine_id: int = Field(..., ge=1)
    cycle: int = Field(..., ge=1)
    op_setting_1: Optional[float] = None
    op_setting_2: Optional[float] = None
    op_setting_3: Optional[float] = None
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
    imputation_density: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def compute_imputation_density(self) -> "TelemetryReading":
        sensor_fields = [f"sensor_{i}" for i in range(1, 22)]
        missing = sum(1 for f in sensor_fields if getattr(self, f) is None)
        self.imputation_density = missing / len(sensor_fields)
        return self


class SensorStatus(BaseModel):
    sensor_id: str
    status: Literal["ok", "stale", "offline"]
    last_valid_value: Optional[float] = None
    last_valid_cycle: Optional[int] = None


class TelemetryWindowOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    engine_id: int
    cycle: int
    imputation_density: float
    stale_sensors: list[str] = []
    warnings: list[str] = []
    llm_explanation: Optional[str] = None
    created_at: datetime


class IngestOut(BaseModel):
    """Full response from POST /ingest — includes score breakdown for dashboard."""
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    engine_id: int
    cycle: int
    z_score: float
    causal_score: float
    combined_score: float
    decision: str
    confidence: float
    alert_event_id: Optional[str] = None
    llm_explanation: Optional[str] = None
    warnings: list[str] = []
    imputation_density: float
    created_at: datetime


class FeedbackRequest(BaseModel):
    alert_event_id: UUID
    label: Literal["TRUE_POSITIVE", "FALSE_POSITIVE", "UNCERTAIN"]
    override: bool = False


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    alert_event_id: UUID
    label: str
    override: bool
    submitted_at: datetime


class AlertEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    telemetry_window_id: UUID
    triggered_at: datetime
    anomaly_score: float
    decision: str
    confidence: float
    cache_hit: bool
    engine_id: Optional[int] = None
    cycle: Optional[int] = None


class PipelineNodeOut(BaseModel):
    node_name: str
    output_state: dict[str, Any] = {}
    latency_ms: Optional[int] = None
    created_at: Optional[datetime] = None


class EngineSummaryOut(BaseModel):
    engine_id: int
    latest_score: float
    latest_decision: str
    latest_confidence: float
    latest_cycle: Optional[int] = None
    alert_count: int
    last_seen: datetime
    regime: Optional[str] = None


class DemoEngineOut(BaseModel):
    label: str
    engine_id: int
    rul_at_end: int
    total_test_cycles: int
    readings: list[dict[str, Any]]
