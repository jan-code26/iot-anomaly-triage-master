from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    # ── inputs (set by main.py before graph.invoke) ────────────────────────
    engine_id: int
    cycle: int
    telemetry_window_id: str   # UUID as string
    alert_event_id: str        # UUID as string
    z_score: float
    causal_score: float
    combined_score: float
    causal_details: dict       # {"sensor_4": 1.2, "sensor_11": 0.5, ...}
    reading: dict              # filled sensor values + op_settings
    stale_sensors: list        # e.g. ["sensor_3"]

    # ── node outputs (accumulated as graph runs) ───────────────────────────
    data_quality_ok: bool
    stale_causal_count: int
    regime: str
    causal_score_refined: float
    physics_veto_applied: bool
    from_cache: bool
    cache_penalty: float
    llm_explanation: str
    final_score: float
    final_decision: str
    final_confidence: float
    agent_warnings: list
