"""
SQLAlchemy Core table definitions for the IoT Anomaly Triage system.

All 8 tables are defined here in dependency order (parents before children).
Run scripts/create_schema.py to apply these to your Neon Postgres database.
"""
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    Table,
    Text,
    TIMESTAMP,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.sql import text

metadata = MetaData()

# ---------------------------------------------------------------------------
# 1. telemetry_windows — one row per engine reading window (root table)
# ---------------------------------------------------------------------------
telemetry_windows = Table(
    "telemetry_windows",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("engine_id", Integer, nullable=False),
    Column("cycle", Integer, nullable=False),
    # 21 sensor readings
    Column("sensor_1", Float),
    Column("sensor_2", Float),
    Column("sensor_3", Float),
    Column("sensor_4", Float),
    Column("sensor_5", Float),
    Column("sensor_6", Float),
    Column("sensor_7", Float),
    Column("sensor_8", Float),
    Column("sensor_9", Float),
    Column("sensor_10", Float),
    Column("sensor_11", Float),
    Column("sensor_12", Float),
    Column("sensor_13", Float),
    Column("sensor_14", Float),
    Column("sensor_15", Float),
    Column("sensor_16", Float),
    Column("sensor_17", Float),
    Column("sensor_18", Float),
    Column("sensor_19", Float),
    Column("sensor_20", Float),
    Column("sensor_21", Float),
    # 3 operational settings
    Column("op_setting_1", Float),
    Column("op_setting_2", Float),
    Column("op_setting_3", Float),
    # fraction of sensor values that were imputed (0.0 = no imputation)
    Column("imputation_density", Float, nullable=False, server_default=text("0.0")),
    # list of sensor names that were stale (> 5 cycles since last valid reading)
    Column("stale_sensors", ARRAY(Text), nullable=False, server_default=text("'{}'")),
    Column(
        "created_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
)

# ---------------------------------------------------------------------------
# 2. psi_baselines — baseline distributions for Population Stability Index
# ---------------------------------------------------------------------------
psi_baselines = Table(
    "psi_baselines",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("sensor_name", Text, nullable=False),
    # JSON blob storing bin edges and expected frequencies
    Column("baseline_dist", JSONB, nullable=False),
    Column(
        "created_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
    # set when this baseline is superseded by a newer one
    Column("invalidated_at", TIMESTAMP(timezone=True)),
)

# ---------------------------------------------------------------------------
# 3. maintenance_events — human-logged maintenance records per engine
# ---------------------------------------------------------------------------
maintenance_events = Table(
    "maintenance_events",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("engine_id", Integer, nullable=False),
    Column("event_type", Text, nullable=False),
    Column(
        "logged_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
    Column("notes", Text),
)

# ---------------------------------------------------------------------------
# 4. alert_events — one row per anomaly alert raised by the agent
#    FK → telemetry_windows
# ---------------------------------------------------------------------------
alert_events = Table(
    "alert_events",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column(
        "telemetry_window_id",
        UUID,
        nullable=False,
    ),
    Column(
        "triggered_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
    Column("anomaly_score", Float, nullable=False),
    # "ALERT" | "NORMAL" | "UNCERTAIN"
    Column("decision", Text, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("cache_hit", Boolean, nullable=False, server_default=text("false")),
)

# ---------------------------------------------------------------------------
# 5. reasoning_traces — one row per LangGraph node execution for an alert
#    FK → alert_events
# ---------------------------------------------------------------------------
reasoning_traces = Table(
    "reasoning_traces",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("alert_event_id", UUID, nullable=False),
    # LangGraph node name, e.g. "causal_reasoner", "physics_veto"
    Column("node_name", Text, nullable=False),
    Column("input_state", JSONB),
    Column("output_state", JSONB),
    # wall-clock time this node took to run
    Column("latency_ms", Integer),
    Column(
        "created_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
)

# ---------------------------------------------------------------------------
# 6. human_feedback — operator label corrections on alerts
#    FK → alert_events
# ---------------------------------------------------------------------------
human_feedback = Table(
    "human_feedback",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("alert_event_id", UUID, nullable=False),
    # operator's label: "TRUE_POSITIVE" | "FALSE_POSITIVE" | "UNCERTAIN"
    Column("label", Text, nullable=False),
    Column(
        "submitted_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
    # True if the operator overrode the agent's decision
    Column("override", Boolean, nullable=False, server_default=text("false")),
)

# ---------------------------------------------------------------------------
# 7. dowhy_results — causal inference scores per telemetry window
#    FK → telemetry_windows
# ---------------------------------------------------------------------------
dowhy_results = Table(
    "dowhy_results",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("telemetry_window_id", UUID, nullable=False),
    # operating regime label, e.g. "high_altitude", "ground_idle"
    Column("regime", Text, nullable=False),
    Column("causal_score", Float, nullable=False),
    Column(
        "computed_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("now()"),
    ),
    Column("from_cache", Boolean, nullable=False, server_default=text("false")),
)

# ---------------------------------------------------------------------------
# 8. lead_time_measurements — how many cycles before failure the alert fired
#    FK → alert_events
# ---------------------------------------------------------------------------
lead_time_measurements = Table(
    "lead_time_measurements",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("gen_random_uuid()")),
    Column("engine_id", Integer, nullable=False),
    Column("alert_event_id", UUID, nullable=False),
    # cycle at which degradation is considered to have started
    Column("failure_onset_cycle", Integer, nullable=False),
    # cycle at which the alert was triggered
    Column("alert_cycle", Integer, nullable=False),
    # alert_cycle - failure_onset_cycle (positive = alert was early)
    Column("lead_time_cycles", Integer, nullable=False),
    # method used to determine failure onset, e.g. "rul_threshold_30"
    Column("method", Text, nullable=False),
)
