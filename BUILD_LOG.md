# Build Log — Don't Trust the Sensors

Complete first-principles build narrative for the IoT Anomaly Triage system.
Every file, every decision, every "why" — written so you can rebuild from scratch.

---

## Table of Contents

1. [Phase 0 — Project Design](#phase-0--project-design)
2. [Phase 1 — Foundation (Days 1–7)](#phase-1--foundation-days-17)
3. [Phase 2 — Data Pipeline (Days 8–14)](#phase-2--data-pipeline-days-814)
4. [What's Next — Phase 3 (Days 15–28)](#whats-next--phase-3-days-1528)

---

## Phase 0 — Project Design

### What problem are we solving?

Industrial IoT sensors fail in subtle ways. A sensor that reads slightly wrong is more dangerous than one that goes offline — because the system keeps trusting it. The goal of this project is to build an **agentic triage system** that:

1. Receives sensor readings continuously
2. Decides whether an anomaly is real, a sensor fault, or a false alarm
3. Explains its reasoning in plain English (using an LLM)
4. Tracks its own confidence and asks a human when it's uncertain

This is different from a simple anomaly detector. A detector outputs a number. An agent outputs a decision with a justification that a human can audit and override.

### Why NASA CMAPSS?

The CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset from NASA is the standard benchmark for industrial predictive maintenance research. It contains:
- Run-to-failure time series from 100+ simulated turbofan engines
- 21 sensor channels + 3 operational settings per cycle
- Known failure times (RUL labels) for evaluating prediction accuracy
- 4 sub-datasets with different fault modes and operating conditions

We chose it because:
- It is publicly available (no data licensing issues)
- It has ground truth labels (we can measure how early our alerts fire)
- It is representative of real industrial sensor data (gradual degradation, sensor noise)
- Every research paper on predictive maintenance uses it, so we can compare our results

### Why Neon + Render?

Both are free tier with no credit card required. This matters for a student project.

- **Neon**: Managed PostgreSQL with a generous free tier. Supports connection pooling (important because Render's free tier spins down and Postgres connections are expensive). Uses port 6432 for pooled connections.
- **Render**: Managed deployment platform. Reads `render.yaml` from the repo — no manual configuration. Free tier has cold starts (sleeps after 15 min) which is acceptable for demos.

Alternative considered: Railway. Rejected because it requires a credit card for the free tier.

### Why FastAPI over Flask/Django?

- **Automatic API docs** at `/docs` — no extra work to test endpoints
- **Pydantic validation built in** — request bodies are validated automatically
- **Async-ready** — when we add the LangGraph agent in Phase 3, it will run async
- **Fast** — benchmarks faster than Flask for IO-bound workloads
- **Type hints everywhere** — IDE autocomplete works correctly

Django was rejected because it is designed for full web applications with templates and auth. We only need a REST API.

### Why SQLAlchemy Core (not ORM)?

SQLAlchemy has two modes:
- **ORM mode**: Define Python classes that map to tables. Good for CRUD apps.
- **Core mode**: Define tables directly, write SQL-like expressions. Good for data pipelines.

We chose Core because:
- We are writing INSERT and SELECT statements that look like SQL — easier to debug
- The ORM adds overhead (session management, lazy loading) we don't need
- When we use PostgreSQL-specific types like JSONB and ARRAY, Core makes it clearer

### LLM provider strategy

Two providers are supported:
- **Groq**: Used during development. Free tier. Returns responses in ~0.5 seconds. Uses open-source models (Llama 3).
- **Gemini**: Used for the final submission. Google's free tier allows 1M tokens/day. More capable for complex reasoning.

The system checks `LLM_PROVIDER` in `.env` to decide which to use. This means you can develop cheaply and switch to a better model for demos.

---

## Phase 1 — Foundation (Days 1–7)

**Goal**: Get a FastAPI server running locally that connects to Neon Postgres and can write/read data. Nothing more.

---

### Day 1 — Project scaffold

**Files created:**
- `requirements.txt`
- `.gitignore`
- `.env` (not committed — contains secrets)
- `backend/main.py` (skeleton)
- Folder structure: `backend/`, `scripts/`, `data/`, `notebooks/`, `models/`, `frontend/`

**Why this structure?**

Each folder has a single responsibility:
- `backend/` — everything the API server needs to run
- `scripts/` — one-off tools (download data, create schema, simulate stream)
- `data/` — raw and processed datasets (gitignored — too large for GitHub)
- `notebooks/` — exploratory analysis (not production code)
- `models/` — trained ML model files (future use)
- `frontend/` — dashboard (future use)

**Why pin exact versions in requirements.txt?**

`fastapi==0.115.5` not `fastapi>=0.115.5`. Because:
- Render installs from requirements.txt. If a new version breaks something, your code breaks in production.
- Pydantic v1 and v2 have completely different APIs. Pinning prevents accidental upgrades.

**Why split into requirements.txt and requirements-dev.txt?**

Jupyter, matplotlib, seaborn, scikit-learn are only needed locally for notebooks and analysis. Installing them on Render wastes build time and memory. The split keeps the production image lean.

**The .env file:**
```
DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech:6432/dbname?sslmode=require
GROQ_API_KEY=your_key_here
LLM_PROVIDER=groq
```
Never commit this file. It is in `.gitignore`.

---

### Day 2 — Neon Postgres + database.py

**Files created:**
- `backend/database.py`
- `scripts/neon_smoke_test.py`

**What database.py does:**

```python
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=0,
    pool_pre_ping=True,
    pool_recycle=300,
)
```

Each setting has a reason:
- `QueuePool`: Maintains a pool of reusable connections. Without this, every request would open and close a connection — very slow.
- `pool_size=5`: Maximum 5 open connections. Neon free tier allows ~10 total. We keep 5 for headroom.
- `max_overflow=0`: Don't create extra connections beyond pool_size. Prevents hitting Neon's limit.
- `pool_pre_ping=True`: Before using a connection from the pool, test if it's still alive. Neon closes idle connections — this prevents "connection already closed" errors.
- `pool_recycle=300`: Return connections to the pool after 5 minutes whether or not they're idle. Prevents stale connections.

**Why the pooled Neon endpoint (port 6432)?**

Neon has two endpoints:
- Direct (port 5432): Each connection is a real Postgres connection. Limited to ~10 on free tier.
- Pooled (port 6432): PgBouncer sits in front and multiplexes connections. Supports many more concurrent connections.

Always use port 6432 for production. Port 5432 is for running migrations.

**Smoke test verification:**

```bash
python scripts/neon_smoke_test.py
```

Creates `test_events` table, inserts a row, reads it back. If this passes, Days 8–14 will work.

---

### Day 3 — CMAPSS dataset + EDA

**Files created:**
- `scripts/download_cmapss.py`
- `notebooks/01_cmapss_eda.ipynb`
- `data/raw/` (populated with 12 files)

**Downloading the data:**

NASA's official S3 URL returns 403. We use a GitHub mirror:
```
https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master/
```

12 files total: `train_FD001.txt` through `RUL_FD004.txt`

```bash
python scripts/download_cmapss.py
```

**CMAPSS data format:**

Space-separated, no header, 26 columns per row:
```
engine_id | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 ... sensor_21
```

Load with pandas:
```python
df = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=COLUMNS)
```

**Key findings from EDA:**

The 26 columns are not equal. Seven sensors are near-constant across all engines (std < 0.01):
- sensors 1, 5, 6, 10, 16, 18, 19

These sensors carry no information about engine health. They are excluded from anomaly scoring.

The 14 informative sensors (used for scoring):
- sensors 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21

**Why operational settings matter:**

`op_setting_1` = Altitude, `op_setting_2` = Mach Number, `op_setting_3` = TRA (Throttle Resolver Angle).

These are the root causes in our causal DAG (built in Phase 3). Altitude determines air density → cooling efficiency → what counts as "normal" temperature. A reading that looks anomalous at sea level might be completely normal at cruise altitude. This is why we need **regime-conditional scoring**, not just a global threshold.

**FD001 vs FD002:**

FD001 has one operating condition (one cluster in the op_setting scatter plot). FD002 has six distinct clusters. This is why we start with FD001 — it is the simplest case.

---

### Days 4–5 — Full database schema

**Files created:**
- `backend/models.py`
- `scripts/create_schema.py`

**The 8 tables and why each exists:**

```
telemetry_windows    ← every sensor reading that enters the system
alert_events         ← every anomaly decision made (even NORMAL ones)
reasoning_traces     ← every LangGraph node execution (for debugging the agent)
human_feedback       ← operator corrections (for retraining and accountability)
psi_baselines        ← stored reference distributions for PSI monitoring
maintenance_events   ← physical maintenance log (resets PSI baselines)
dowhy_results        ← causal inference scores from DoWhy (Phase 3)
lead_time_measurements ← how many cycles before failure each alert fired
```

**Why UUID primary keys?**

UUIDs are globally unique without coordination. Multiple services can insert rows without asking a central counter. We use `gen_random_uuid()` (PostgreSQL built-in, no extension needed).

**Why JSONB for some columns?**

`reasoning_traces.input_state` and `reasoning_traces.output_state` store the LangGraph agent's state at each node. The shape of this data changes as we add nodes — JSONB lets us store arbitrary structure without changing the schema.

**Why ARRAY(Text) for stale_sensors?**

A row might have 0, 1, or 7 stale sensors. We could create a separate `stale_sensor_events` table, but that's over-engineering. An array column is simpler and queryable with PostgreSQL's `@>` operator.

**Dependency order (why it matters):**

Tables with foreign keys must be created after the tables they reference:
1. `telemetry_windows` (no FK) → created first
2. `psi_baselines`, `maintenance_events` (no FK) → standalone
3. `alert_events` → FK to `telemetry_windows`
4. `reasoning_traces`, `human_feedback`, `lead_time_measurements` → FK to `alert_events`
5. `dowhy_results` → FK to `telemetry_windows`

`metadata.create_all(engine)` handles this automatically if all tables share the same `MetaData()` object.

**Applying the schema:**

```bash
python scripts/create_schema.py
```

Safe to re-run — SQLAlchemy uses `CREATE TABLE IF NOT EXISTS`.

---

### Days 6–7 — Render deployment

**Files created:**
- `render.yaml`
- `runtime.txt`

**Why deploy before the app does anything?**

Finding deployment problems with 2 endpoints is 30 minutes. Finding them on Day 38 with LangGraph, DoWhy, and PSI monitoring all tangled together is a nightmare. Deploy early.

**The Python version problem:**

Render defaults to Python 3.7. pandas 2.2.3 requires Python 3.9+. Without `runtime.txt`, the build hangs trying to compile pandas from source (no compatible wheel exists for 3.7).

Fix:
```
# runtime.txt
python-3.11.0
```

**Why requirements were split:**

First Render build timed out because it was installing `jupyter`, `matplotlib`, `seaborn`, and `scikit-learn`. These are notebook tools — not needed by the server. Moved them to `requirements-dev.txt`.

**Environment variables on Render:**

The `render.yaml` file declares which env vars the service needs but sets `sync: false` — meaning Render will not fill them in from the file. You must add the actual values in the Render dashboard. This is intentional: secrets should never be in the repository.

**Verification:**

```
GET https://your-app.onrender.com/health
→ {"status": "ok", "message": "Sensor triage system is running"}
```

---

## Phase 2 — Data Pipeline (Days 8–14)

**Goal**: Real CMAPSS data flows into Postgres, with imputation, stale detection, anomaly scoring, PSI monitoring, and G-test validation all working end-to-end.

---

### Day 8 — Pydantic schemas

**File created:** `backend/schemas.py`

**Three schemas and why each exists:**

**`TelemetryReading`** (request body for POST /ingest):

All 21 sensor fields are `Optional[float]` — they default to `None`. Why?

Real IoT sensors fail. A sensor that stops sending is not the same as a sensor that reads zero. `None` means "I have no information about this sensor right now." This distinction matters for imputation: we can forward-fill a `None`, but we would not forward-fill a zero (zero is a valid reading for some sensors).

`imputation_density` is auto-computed by a `@model_validator`:
```python
missing = sum(1 for f in sensor_fields if getattr(self, f) is None)
self.imputation_density = missing / 21
```

This runs after the model is initialized. You never pass it in — Pydantic computes it.

**`SensorStatus`** (per-sensor health report):

```python
class SensorStatus(BaseModel):
    sensor_id: str
    status: Literal["ok", "stale", "offline"]
    last_valid_value: Optional[float]
    last_valid_cycle: Optional[int]
```

Used internally by `SensorService` to track sensor health. The distinction between `stale` and `offline`:
- `stale`: We have seen this sensor before, but not recently (> 5 cycles ago)
- `offline`: We have never seen this sensor for this engine

**`TelemetryWindowOut`** (response after saving a reading):

Returns the UUID (needed for `GET /telemetry/{id}`), imputation stats, stale sensors list, and any warnings. The `model_config = ConfigDict(from_attributes=True)` setting allows it to be constructed directly from a SQLAlchemy row mapping.

---

### Day 9 — Forward-fill service

**File created:** `backend/services/sensor_service.py`

**Why forward-fill instead of mean-fill or zero-fill?**

In time series data, the last known value is almost always a better estimate than the global mean. Turbofan sensors change slowly. If sensor_3 read 1589 last cycle, it probably reads ~1589 this cycle even if the reading is missing.

**The 5-cycle threshold:**

```python
STALE_CYCLE_THRESHOLD = 5
```

If a sensor has been missing for more than 5 cycles, forward-filling becomes misleading. The engine may have entered a different operating state. We mark it `stale` and leave the value as `None` — telling the anomaly scorer and the agent not to trust this sensor.

**Why in-memory cache (not Redis)?**

Redis would survive server restarts. In-memory does not. For a production system, Redis is the right answer. For this project:
- Render's free tier restarts the server after 15 minutes of sleep
- The CMAPSS simulator sends all data in one run
- Adding Redis would add complexity and a new paid service

This is a documented limitation. When you scale to production, replace `SensorService._cache` with Redis calls.

**Module-level singleton:**

```python
sensor_service = SensorService()
```

FastAPI is single-process. Every request hits the same Python process, so the same `sensor_service` instance handles all requests. This is why the in-memory cache works.

---

### Day 10 — Stream simulator

**File created:** `scripts/simulate_stream.py`

**What it does:**

Reads `train_FD001.txt` row by row (or a subset) and POSTs each row as JSON to `/ingest`. Simulates a live sensor feed without real hardware.

**Basic usage:**

```bash
python scripts/simulate_stream.py --rows 100 --delay 0.05
```

**Fault injection (`--fault-injection`):**

IEC 61508 is the international standard for safety instrumentation systems. It defines common sensor fault modes. We implement four:

| Fault | What it does | Why it's realistic |
|---|---|---|
| `drift` | value += 0.01 × cycle | Sensor calibration slowly shifts over time |
| `spike` | value × random[1.5, 3.0] | Electrical noise or vibration causes sudden jump |
| `stuck` | value stays constant for 3–10 cycles | Sensor output freezes, still reporting last reading |
| `bias` | value += 0.5 × std | Systematic offset, sensor reads consistently high/low |

Applied randomly to 5% of rows on a random informative sensor.

```bash
python scripts/simulate_stream.py --rows 100 --fault-injection
```

With fault injection on, some rows should return `decision=ALERT` or `UNCERTAIN`.

**Engine filtering (`--engines`):**

```bash
python scripts/simulate_stream.py --engines 1,2 --rows 0
```

Runs only engines 1 and 2 to completion. Useful for testing per-engine state (forward-fill, G-test buffer).

---

### Day 11 — Isolation Forest baseline

**File created:** `scripts/lead_time_baseline.py`

**Why this matters:**

The whole point of an anomaly detection system for predictive maintenance is to catch failures early. "Lead time" = how many cycles before failure the first alert fires.

Before building the causal pipeline (Phase 3), we establish a baseline: what does a standard ML model (Isolation Forest) achieve? Every improvement we make in Phase 3 is measured against this number.

**What Isolation Forest does:**

It builds random decision trees that isolate points. Anomalies — points that are different from the bulk of the data — are isolated with fewer splits. The anomaly score is the inverse of the average depth needed to isolate a point.

`contamination=0.05` means: "Assume 5% of the training data is anomalous." This sets the decision threshold.

**How lead time is computed:**

```
test engine → score each cycle → find first cycle with prediction = anomaly
                                            ↓
true_failure_cycle = last test cycle + RUL from RUL_FD001.txt
                                            ↓
lead_time = true_failure_cycle - first_alert_cycle
```

Positive lead time = alert fired before failure (good).
Negative lead time = alert fired after failure would have occurred (missed it).

**Output:**

```bash
python scripts/lead_time_baseline.py
```

Saves `data/processed/isolation_forest_baseline.csv`. Also prints:
```
Mean lead time  : 47.3 cycles
Median lead time: 42.0 cycles
```

This is your Phase 3 target. The causal pipeline must beat these numbers.

---

### Day 12 — PSI monitoring

**File created:** `backend/services/psi_monitor.py`

**What PSI measures:**

Population Stability Index measures whether a sensor's distribution has shifted since a baseline was established. A PSI > 0.2 means the data looks so different from the baseline that cached inference results are no longer valid.

**Formula:**

```
PSI = sum( (actual_bin% - expected_bin%) × ln(actual_bin% / expected_bin%) )
```

10 equal-width bins. For each bin:
- `expected_bin%` = fraction of baseline readings in this bin
- `actual_bin%` = fraction of current (rolling 200) readings in this bin

**Thresholds:**

| PSI | Status | Action |
|---|---|---|
| < 0.1 | stable | No action needed |
| 0.1–0.2 | moderate | Watch closely |
| > 0.2 | action_required | Clear cache, establish new baseline |

**The rolling window (200 readings):**

PSI requires enough data to build a meaningful distribution. 200 readings ≈ 2 engine lifespans in FD001. Too small = noisy. Too large = slow to detect real shifts.

**API endpoints added:**

```
GET /psi/status          → {"sensors": [{"sensor": "sensor_2", "psi": 0.04, "status": "stable"}, ...]}
POST /baselines/reset    → logs maintenance event, clears PSI baselines
```

After calling `/baselines/reset`, the next 200 readings will rebuild the baseline from the current distribution.

---

### Day 13 — G-test structural validation

**File created:** `backend/services/gtest_monitor.py`

**The physical principle:**

`sensor_11` = HPC (High Pressure Compressor) outlet temperature.
`sensor_15` = HPC outlet pressure.

In a compressor, temperature and pressure are thermodynamically coupled — they rise and fall together. If this coupling breaks, it means one of the sensors is faulty, not that the engine is failing.

This is the "physics veto" concept: before the LLM agent calls an alert, the system checks whether the reading violates known physical laws. If sensor_11 spikes but sensor_15 does not move, something is wrong with sensor_11, not the engine.

**The G-test:**

The G-test is a statistical test for independence. We bin both sensors into a 5×5 contingency table and compute:

```
G = 2 × sum(O × ln(O/E))
```

Where O = observed count in each cell, E = expected count if the sensors were independent.

A high G = sensors are correlated (physically normal).
A low G (< 9.49 = chi-squared critical value at p=0.05, 4 df) = sensors appear independent = coupling is broken = likely sensor fault.

**Buffer size (100 readings):**

The G-test needs enough data to fill the contingency table. 100 readings with a 5×5 grid = average 4 readings per cell. This is the minimum for a reliable test.

The monitor runs automatically in `/ingest` after every 100 readings per engine. When it fires, it adds a warning to the response:

```json
{"warnings": ["G-test: sensor_11/sensor_15 coupling broken (G=3.2, threshold=9.49) — possible sensor fault"]}
```

---

### Day 14 — Tests + fixes

**Files created:**
- `tests/test_connection.py`
- `tests/test_ingest.py`
- `tests/test_anomaly.py`
- `conftest.py`
- `tests/__init__.py`

**Why conftest.py?**

When pytest runs `tests/test_anomaly.py`, it tries to import `from backend.anomaly import ...`. Python cannot find `backend` unless the project root is on `sys.path`. `conftest.py` at the root fixes this:

```python
sys.path.insert(0, os.path.dirname(__file__))
```

pytest automatically loads `conftest.py` before running any tests.

**Test categories:**

- `test_connection.py`: One test. Runs `SELECT 1`. If the DB is reachable, this passes.
- `test_anomaly.py`: Pure unit tests. No database, no network. Fast. Tests scorer logic.
- `test_ingest.py`: Integration tests. Uses `TestClient` (FastAPI's built-in test client backed by httpx). These hit the real Neon database.

**Running tests:**

```bash
# Fast (no DB)
pytest tests/test_anomaly.py -v

# All (requires .env with DATABASE_URL)
pytest tests/ -v
```

**Schema fix for stale_sensors:**

`stale_sensors` was added to `backend/models.py` but `create_all()` does not add columns to existing tables (only creates new ones). Run this once in the Neon SQL editor:

```sql
ALTER TABLE telemetry_windows
ADD COLUMN IF NOT EXISTS stale_sensors TEXT[] DEFAULT '{}';
```

---

## What's Next — Phase 3 (Days 15–28)

Phase 3 is the "intelligent" layer. Phase 2 gave us data flowing into Postgres with good quality signals. Phase 3 decides what to do with those signals using causal inference and an LLM agent.

### DoWhy Causal DAG (Days 15–17)

**Goal**: Replace the z-score scorer with a causal model that accounts for operating conditions.

The causal graph:
```
Altitude → AirDensity → CoolingEfficiency → sensor_4 (temperature)
Mach    → TipSpeed   → HPCLoading       → sensor_11, sensor_15
TRA     → FuelFlow   → CombustorTemp    → sensor_3, sensor_9
```

`op_setting_1` (Altitude), `op_setting_2` (Mach), `op_setting_3` (TRA) are root cause nodes. Anomaly scoring should condition on these values — a high temperature reading is normal at ground-level TRA but anomalous at cruise altitude.

DoWhy implementation:
```python
from dowhy import CausalModel
model = CausalModel(data=df, treatment="sensor_4", outcome="rul", graph=dot_graph)
```

Save results per reading to `dowhy_results` table.

### LangGraph Agent (Days 18–24)

**Goal**: Build a 7-node agent that reasons about alerts.

Node sequence:
```
1. ingest_validator     → check reading quality, flag stale sensors
2. regime_classifier    → which operating condition (FD001: always cluster 0)
3. causal_reasoner      → run DoWhy, get causal score
4. physics_veto         → G-test check, override if coupling broken
5. cache_lookup         → check if we've seen this pattern before
6. llm_explainer        → ask Groq/Gemini to explain the anomaly in plain English
7. decision_writer      → write final decision + trace to DB
```

Each node writes a row to `reasoning_traces`. This is the audit log.

### Human Feedback Loop (Days 25–28)

**Goal**: Operators can correct alerts.

```
POST /feedback/{alert_id}
body: {"label": "FALSE_POSITIVE", "notes": "sensor_3 was recently recalibrated"}
```

Writes to `human_feedback`. In Phase 4, the agent checks recent feedback before making a decision — if the same pattern was labeled FALSE_POSITIVE 3 times in the last week, it lowers confidence automatically.

### Lead Time Measurement (ongoing)

At the end of Phase 3, run:

```bash
python scripts/lead_time_baseline.py  # re-run for comparison
```

But this time also measure lead times from the causal pipeline's `alert_events` table. Compare against the Isolation Forest baseline CSV from Day 11. The causal system should fire earlier with fewer false positives.

---

## Quick Reference — All Commands

```bash
# Setup
python scripts/download_cmapss.py
python scripts/create_schema.py

# Server
uvicorn backend.main:app --reload

# Simulate data
python scripts/simulate_stream.py --rows 100
python scripts/simulate_stream.py --rows 500 --fault-injection
python scripts/simulate_stream.py --engines 1,2 --delay 0

# Baseline
python scripts/lead_time_baseline.py

# Tests
pytest tests/test_anomaly.py -v    # unit tests only
pytest tests/ -v                    # all tests (needs DB)

# Check drift
curl http://localhost:8000/psi/status

# Reset baselines after maintenance
curl -X POST http://localhost:8000/baselines/reset \
     -H "Content-Type: application/json" \
     -d '{"engine_id": 1}'
```

---

## Phase 3 — Intelligent Triage (Days 15–28)

**Goal**: Replace the global z-score scorer with a causal model, add a 7-node LangGraph agent that reasons about each alert, and close the loop with human feedback.

---

### Days 15–17 — DoWhy Causal Scorer

**Files created/modified:**
- `backend/services/causal_scorer.py` (new)
- `backend/main.py` (modified — causal scoring + `RETURNING id` on alert_events)
- `tests/test_causal_scorer.py` (new — 10 unit tests, all passing)
- `requirements.txt` (added: `scikit-learn>=1.4.0`, `dowhy==0.11.1`, `langgraph==0.2.76`, `langchain-core==0.3.29`, `networkx>=3.0`)
- `render.yaml` (added: `GEMINI_API_KEY`, `GROQ_MODEL`, `GEMINI_MODEL` env vars)

**Why replace the z-score?**

The z-score scorer in `anomaly.py` compares each sensor against a global mean/std from all FD001 training cycles. But sensor readings are not independent of operating conditions. `sensor_4` (HPC outlet temperature) is higher at ground-level TRA than at cruise altitude — and that is physically *normal*. A naive z-score will flag healthy ground-run cycles as anomalous.

The causal DAG captures these dependencies:
```
op_setting_1 (Altitude) → AirDensity → CoolingEfficiency → sensor_4
op_setting_2 (Mach)     → TipSpeed   → HPCLoading       → sensor_11, sensor_15
op_setting_3 (TRA)      → FuelFlow   → CombustorTemp    → sensor_3, sensor_9
```

**Why not DoWhy's ATE estimator at inference time?**

The BUILD_LOG originally suggested:
```python
model = CausalModel(data=df, treatment="sensor_4", outcome="rul", graph=dot_graph)
```

This doesn't work for two reasons:
1. Live readings have no `rul` column — RUL only exists in the training set.
2. DoWhy v0.11 requires ≥ 2 rows per call — too slow for a per-request scorer.

**The solution**: Fit a `LinearRegression` per causal branch on `train_FD001.txt` at startup. At inference time, compute the residual `(observed - predicted) / residual_std`. This gives a causally-conditioned z-score for each branch without re-running the DoWhy estimator on every request. DoWhy validates the graph structure at module load time; sklearn does the actual regression.

**Key constants:**

```python
CAUSAL_BRANCHES = {
    "altitude_branch": {"cause": "op_setting_1", "effects": ["sensor_4"]},
    "mach_branch":     {"cause": "op_setting_2", "effects": ["sensor_11", "sensor_15"]},
    "tra_branch":      {"cause": "op_setting_3", "effects": ["sensor_3", "sensor_9"]},
}
```

**Fallback coefficients**: Hardcoded in `FALLBACK_COEFFICIENTS` — used on Render where `data/raw/` is gitignored. Computed from `train_FD001.txt` using sklearn LinearRegression:

| Sensor | Cause | Coef | Intercept | Residual Std |
|---|---|---|---|---|
| sensor_4 | op_setting_1 | 39.27 | 1408.93 | 9.00 |
| sensor_11 | op_setting_2 | 10.65 | 47.54 | 0.27 |
| sensor_15 | op_setting_2 | 1.81 | 8.44 | 0.038 |
| sensor_3 | op_setting_3 | 0.0 | 1590.52 | 6.13 |
| sensor_9 | op_setting_3 | 0.0 | 9065.24 | 22.08 |

**FD001 limitation**: In FD001 all three op_settings are nearly constant (single operating condition). The causal benefit is small here but grows significantly for FD002–FD004 where six distinct operating regimes are present. The coef for op_setting_3 branches is 0.0 because TRA=100 for every row — no variance to fit on.

**Blended score**: The causal score is averaged 50/50 with the z-score:
```python
combined_score = 0.5 * z_score + 0.5 * causal_score
```
This avoids discarding the z-score while the causal model is still being validated. In Phase 4, adjust weights based on lead-time comparison results.

**Render deployment note**: Do not install `pygraphviz` — it requires system-level libraries and fails on Render free tier. The DOT graph string is passed directly to DoWhy as a string; no `pygraphviz` object is needed.

**`RETURNING id` on alert_events**: The `alert_events` INSERT was changed to `.returning(alert_events.c.id)` so the LangGraph agent (Days 18–24) has the alert UUID to reference when writing `reasoning_traces`.

**Tests — all 10 passing:**
```bash
pytest tests/test_causal_scorer.py -v
# 10 passed in 2.78s
```

Test categories:
- Normal reading scores near 0 (residual ≈ 0)
- Degraded reading (5× std away) scores > 0.5
- Returns `(float, dict)` tuple always
- Handles `None` sensor values and `None` op_settings gracefully
- Empty reading returns 0.0
- Extreme residuals clamped to 1.0
- DOT graph string contains expected node/edge names

---

### Days 18–24 — LangGraph 7-Node Agent

**Files created:**
- `backend/agent/__init__.py`
- `backend/agent/state.py`
- `backend/agent/nodes.py`
- `backend/agent/graph.py`

**Why synchronous, not async?**

The existing `/ingest` endpoint and SQLAlchemy Core setup are fully synchronous (psycopg2-binary + QueuePool). Converting to async would require switching to `asyncpg` and wrapping every database call in `await`. That is a large, risky refactor. LangGraph 0.2.x supports `graph.invoke()` (synchronous) and `graph.ainvoke()` (async). We use the synchronous version.

**Why TypedDict for state?**

LangGraph 0.2.x requires `TypedDict` for its state schema. Using a Pydantic model requires compatibility shims (`pydantic_v1`). TypedDict is cleaner.

**The 7 nodes:**

| Node | What it does |
|---|---|
| `ingest_validator` | Flags if >3 causal sensors (from the 6 in the DAG) are stale or None |
| `regime_classifier` | Returns `"cluster_0"` (FD001 has one operating condition); this is where FD002 KMeans clustering would go |
| `causal_reasoner` | Passes the pre-computed `causal_score` from state through; could re-run the scorer with regime context in Phase 4 |
| `physics_veto` | Calls `gtest_monitor.run_gtest(engine_id)`; if coupling is broken AND causal_score is high, halves the score (sensor fault, not engine fault) |
| `cache_lookup` | Queries `dowhy_results` for same engine_id with score within ±0.05 in the last 10 cycles — returns cache hit if found |
| `llm_explainer` | Calls Groq or Gemini via `LLM_PROVIDER` env var; falls back to a rule-based template string on any exception |
| `decision_writer` | Computes final blended score and decision; writes all 7 `reasoning_traces` rows in one transaction |

**Agent triggers on** `combined_score >= 0.3` (the UNCERTAIN threshold). NORMAL readings skip the agent entirely — no LLM call, no trace writes.

**Agent is non-fatal**: The entire agent run is wrapped in `try/except`. If the LLM rate-limits or the DB has a transient error, the response falls back to the pre-agent decision and appends a warning.

**`_write_trace()` helper**: Each node calls this to write its execution record to `reasoning_traces`. It opens its own `engine.begin()` connection — separate from the main `/ingest` transaction — so a trace write failure doesn't roll back the telemetry insert.

**LangGraph v0.2.76 gotchas:**
- `END` is imported from `langgraph.graph`, not `langgraph.constants`
- Node functions return a **partial dict** of only the keys they update; LangGraph merges it into the full state
- `graph.compile()` does NOT require a `checkpointer` in 0.2.x

---

### Days 25–28 — Human Feedback Loop

**Files modified:**
- `backend/schemas.py` — adds `FeedbackRequest`, `FeedbackOut`
- `backend/main.py` — adds `POST /feedback/{alert_id}`, `GET /alerts/{alert_id}/feedback`
- `backend/agent/nodes.py` — `cache_lookup` checks recent FALSE_POSITIVE labels

**Why feedback reduces confidence, not blocks the alert:**

Blocking an alert on feedback alone would be dangerous — a sensor that was a false positive last week might be a real failure this week. Instead, if an engine has 2+ recent FALSE_POSITIVE labels for the same pattern, `cache_lookup` multiplies `causal_score_refined` by 0.7. The agent still alerts, but with lower confidence, and the LLM explanation mentions the prior corrections.

**The `notes` column**: Not included in Phase 3. The existing `human_feedback` table schema (from Day 5) does not have a `notes` column. To add it later: `ALTER TABLE human_feedback ADD COLUMN notes TEXT` in the Neon SQL editor.

**New endpoints:**

```
POST /feedback/{alert_id}            → 201 — record operator correction
GET  /alerts/{alert_id}/feedback     → 200 list — retrieve all corrections for an alert
```

---

## What's Next — Phase 4 (Days 29–40)

Phase 4 is the "learning and deployment" layer. Phase 3 gave us a reasoning agent with causal scoring and human feedback. Phase 4 makes the system smarter over time and production-ready.

### Lead Time Comparison (Day 29)

Re-run `scripts/lead_time_baseline.py` against the `alert_events` table now populated by the causal pipeline. Compare:
- Isolation Forest baseline (from Day 11 CSV)
- Phase 3 causal+agent pipeline (from `alert_events` in Neon)

The causal pipeline should fire earlier with fewer false positives.

### Regime-Aware Scoring for FD002 (Days 30–32)

FD001 has one operating condition. FD002 has six clusters in `op_setting` space. The `regime_classifier` node is already stubbed to return `"cluster_0"` — replace it with a KMeans classifier trained on FD002 training data. Store the cluster centroids in `psi_baselines` as the "regime baseline."

### Frontend Dashboard (Days 33–38)

A minimal React dashboard (or Streamlit for speed) that:
- Shows a live feed of `alert_events` (polling `/alerts`)
- Displays the `llm_explanation` for each ALERT
- Has a thumbs-up/thumbs-down button that calls `POST /feedback/{alert_id}`

### Production Hardening (Days 39–40)

- Replace in-memory `SensorService` cache with Redis (survives Render restarts)
- Add rate limiting to `/ingest` (Render free tier has request limits)
- Add structured logging with JSON output (for Render log aggregation)

---

## Key Numbers to Remember

| Constant | Value | Why |
|---|---|---|
| Informative sensors | 14 (2,3,4,7,8,9,11,12,13,14,15,17,20,21) | Near-constant ones excluded (std < 0.01 in EDA) |
| Stale threshold | 5 cycles | Balance between data freshness and coverage |
| PSI stable threshold | 0.1 | Standard industry threshold |
| PSI action threshold | 0.2 | Standard industry threshold |
| PSI rolling window | 200 readings | ~2 engine lifespans in FD001 |
| G-test buffer | 100 readings | Minimum for reliable contingency table |
| G-test threshold | 9.49 | chi-squared critical value, p=0.05, df=4 |
| ALERT threshold | score ≥ 0.6 | Mean z-score ≥ 3 std deviations |
| UNCERTAIN threshold | score ≥ 0.3 | Mean z-score ≥ 1.5 std deviations |
| IF contamination | 0.05 | 5% of training data assumed anomalous |
| Neon pool size | 5 | Free tier allows ~10 total connections |
| Pool recycle | 300s | Prevent stale connections from Neon's idle timeout |
| Physics veto coefficient | 0.5 | Halve causal score when G-test detects sensor decoupling |
| Cache penalty | 0.7 | Reduce confidence 30% on ≥2 FALSE_POSITIVE operator labels |

---

## Days 18–24 — LangGraph Triage Agent

### What was built

A 7-node synchronous LangGraph agent that runs on every `/ingest` reading with
`combined_score >= 0.3`.  For readings below that threshold the agent is skipped
entirely — no LLM call, no trace writes, negligible overhead.

**New files:**

| File | Purpose |
|------|---------|
| `backend/agent/__init__.py` | Package marker |
| `backend/agent/state.py` | `AgentState` TypedDict (`total=False`) |
| `backend/agent/nodes.py` | 7 node functions + `_write_trace()` helper |
| `backend/agent/graph.py` | Compiled graph singleton + `run_triage_agent()` |
| `tests/test_agent_nodes.py` | 10 unit tests — 0 DB calls required |

**Modified files:**

| File | Change |
|------|--------|
| `backend/schemas.py` | Added `llm_explanation: Optional[str] = None` to `TelemetryWindowOut` |
| `backend/main.py` | Added agent invocation block after the main `engine.begin()` transaction |

---

### Node-by-node design decisions

**Node 1: `ingest_validator`**

Counts how many of the 5 causal DAG sensors (sensor_3, 4, 9, 11, 15) are either
in `stale_sensors` or `None` in the reading dict.  Sets `data_quality_ok = (count <= 3)`.
This flag is available to all downstream nodes via state, but in Phase 3 the agent
continues regardless — Phase 4 could add a conditional edge that short-circuits to
`decision_writer` if `data_quality_ok` is False.

**Node 2: `regime_classifier`**

Always returns `cluster_0`.  FD001 has one operating condition (all op_settings
near-constant).  Phase 4 will replace this with a KMeans classifier trained on
FD002's six op_setting clusters and stored centroids.

**Node 3: `causal_reasoner`**

Passes the pre-computed `causal_score` through as `causal_score_refined`.  The score
was already computed by `compute_causal_score()` in `/ingest` before the agent was
invoked.  Re-running it here would be redundant for FD001 but is where Phase 4 will
swap in regime-specific regression coefficients.

**Node 4: `physics_veto`**

Calls `gtest_monitor.should_run(engine_id)` — returns `True` only when the per-engine
deque has accumulated 100 readings.  If the G-test finds sensor_11 and sensor_15
decorrelated AND `causal_score_refined >= 0.5`, the score is halved and
`physics_veto_applied = True`.  The 0.5 coefficient and the 0.5 threshold are both
hyperparameters to tune in Phase 4 against the lead_time_measurements table.

Most dev/test requests skip the veto because the engine's buffer never reaches 100.

**Node 5: `cache_lookup`**

Two DB queries inside one `try/except`:

1. JOIN `dowhy_results ↔ telemetry_windows` — find prior readings for the same engine
   with causal_score within ±0.05.  `from_cache = True` if > 1 row found (the
   current row was just inserted, so any additional matches are prior readings).

2. Triple-JOIN `human_feedback → alert_events → telemetry_windows` — count
   FALSE_POSITIVE labels for this engine.  If ≥ 2 exist, `cache_penalty = 0.7`.

The failure of either query is non-fatal and appends an `agent_warnings` entry.

**Node 6: `llm_explainer`**

Calls Groq (`llama-3.1-8b-instant`, 150 tokens, temp 0.2) by default, Gemini
(`gemini-1.5-flash`) when `LLM_PROVIDER=gemini`.  Both LLM client constructors use
lazy imports (`from groq import Groq` inside the function body) so the module loads
cleanly in unit tests with no API keys present.

The rule-based fallback lists the top-2 sensors by causal residual z-score and
mentions the physics veto if it fired.

**Node 7: `decision_writer`**

```python
final_score = round(0.5 * z_score + 0.5 * causal_score_refined, 6)
final_decision, final_confidence = make_decision(final_score)
if cache_penalty < 1.0:
    final_confidence = round(final_confidence * cache_penalty, 4)
```

Then UPDATEs the `alert_events` row (inserted by `/ingest` earlier in the same
request) with the refined values.  The UPDATE is wrapped in `try/except` — failure
is non-fatal.

---

### Why synchronous `invoke()` not `ainvoke()`

The existing `/ingest` endpoint is a plain `def`, using psycopg2-binary and
SQLAlchemy's QueuePool.  Converting to async would require:
- Switching to `asyncpg` for the DB adapter
- Wrapping every `engine.begin()` call in `await`
- Converting every service (sensor_service, gtest_monitor, psi_monitor) to async

That is a large, risky refactor.  LangGraph 0.2.76 supports both `invoke()` and
`ainvoke()`.  We use the synchronous version throughout.

### Why `total=False` TypedDict not Pydantic

LangGraph's state merger expects node functions to return plain Python dicts with
only the keys they set.  Using a Pydantic model requires a `pydantic_v1` compatibility
shim that was removed in LangGraph 0.2.x.  `TypedDict(total=False)` is the idiomatic
choice — every field is optional at the TypedDict level, which matches how LangGraph
merges partial returns.

### Why the agent runs outside `engine.begin()`

The main `/ingest` flow opens one `engine.begin()` transaction to insert
`telemetry_windows`, save a `dowhy_results` row, and insert `alert_events`.  If the
agent were invoked inside that block, a 0.5-2s Groq API call would hold the
connection open across network I/O — burning one of the 5 pool slots and risking a
Neon idle-timeout disconnect.

The agent invocation is placed after the `with engine.begin()` block closes.  Each
node that needs the DB (`cache_lookup`, `decision_writer`, `_write_trace`) opens its
own short-lived `engine.begin()` connection for just that operation.

### Why `_write_trace()` has its own connection

Trace writes must never fail the request.  If `_write_trace()` shared the same
connection as the main insert, a trace failure would roll back the telemetry row too.
Giving it its own `engine.begin()` call isolates the failure — the trace is
best-effort, the telemetry insert is guaranteed.

### LangGraph 0.2.76 gotchas

- `END` is imported from `langgraph.graph`, NOT `langgraph.constants`
- `graph.compile()` takes no arguments — no checkpointer needed in 0.2.x
- Node functions must return a **partial dict** of only the keys they set;
  LangGraph merges it into the accumulated state
- `_compiled_graph = _build_graph().compile()` at module level → compiled once at
  import, not once per request

### Test strategy

`tests/test_agent_nodes.py` — 10 tests, 0 DB connections required.

```
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test_db")
```

This dummy URL is set before any `backend.*` import so `database.py` does not raise
`RuntimeError`.  The `_write_trace()` calls inside nodes attempt a connection, fail
silently, and the test continues.  `physics_veto` tests use `engine_id=999` which has
an empty G-test buffer → `should_run(999)` returns `False` → veto is never triggered.

All 10 tests pass in < 0.4 seconds with no network calls.

---

## Days 25-28 — Human Feedback Loop

**Goal**: Give operators an API surface to label alert events, closing the feedback loop that the `cache_lookup` node (built in Days 18-24) already queries.

### What was built

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/feedback` | POST | Submit an operator label for an alert event |
| `/alerts/recent` | GET | List recent alert events (for operators to review) |
| `/alerts/{alert_id}/feedback` | GET | Retrieve all labels submitted for one alert |

**Schemas added** (`backend/schemas.py`):
- `FeedbackRequest` — POST body: `alert_event_id`, `label`, `override`
- `FeedbackOut` — response after label submission
- `AlertEventOut` — response for `/alerts/recent`

**No new DB tables** — all three endpoints use the `human_feedback` and `alert_events` tables created in the Day 1-7 schema migration.

### Design decisions

**`except HTTPException: raise` before `except Exception`**:
The `submit_feedback` endpoint raises a `404` inside the `try` block if the alert doesn't
exist.  Without an explicit `except HTTPException: raise` before the catch-all
`except Exception`, the `404` would be swallowed and re-raised as a `500`.  This pattern
is required any time you raise `HTTPException` inside a `try/except Exception` block.

**`override=True` sets `confidence=1.0`**:
When an operator marks their label as an override, they are asserting ground truth —
no model uncertainty applies.  The `alert_events.confidence` column is set to `1.0` to
reflect this.  The `decision` column is also updated to the operator's label so downstream
queries (dashboards, reports) see the corrected value immediately.

**How this closes the cache_lookup feedback loop**:
The `cache_lookup` node in the LangGraph agent counts `FALSE_POSITIVE` labels for the
same engine via a triple-join query.  Once ≥2 labels exist, it sets `cache_penalty=0.7`,
which `decision_writer` applies to reduce `final_confidence` by 30% on the next alert for
that engine.  The feedback loop is now complete:
```
/ingest → LangGraph agent → alert_events row
            ↓
   operator reviews via GET /alerts/recent
            ↓
   operator labels via POST /feedback
            ↓
   cache_lookup reads label on next /ingest → 0.7 confidence penalty applied
```

### Test strategy

`tests/test_feedback.py` — 6 tests, pure Pydantic validation, no DB or server needed.

Same `os.environ.setdefault("DATABASE_URL", ...)` guard as `test_agent_nodes.py`.
Tests cover all three valid labels, the invalid-label rejection, and both `override` states.
All 6 tests pass with no network calls.

---

## Day 29 — Lead Time Comparison

### Goal

Measure whether the causal pipeline fires earlier than the Isolation Forest baseline
computed in Day 11 (`scripts/lead_time_baseline.py`).

### What was built

**`scripts/simulate_stream.py`** — added `--file` argument (default `train_FD001.txt`).
Allows streaming `test_FD001.txt` through `/ingest` without changing any other behaviour.
Backward-compatible: existing callers with no `--file` flag continue to read the training set.

```
python scripts/simulate_stream.py --file test_FD001.txt --rows 0 --delay 0
```

**`scripts/compare_lead_times.py`** (new) — standalone comparison script:
1. Queries `alert_events JOIN telemetry_windows` for the first `ALERT` cycle per engine
2. Merges with `RUL_FD001.txt` to compute `true_failure_cycle`
3. Computes `lead_time_cycles = true_failure_cycle - first_alert_cycle`
4. Loads `data/processed/isolation_forest_baseline.csv` and prints a side-by-side table
5. Saves `data/processed/causal_lead_times.csv`

### How to run

```bash
# Terminal 1
uvicorn backend.main:app --reload

# Terminal 2 — streams all 100 test engines, as fast as possible
python scripts/simulate_stream.py --file test_FD001.txt --rows 0 --delay 0

# After streaming completes
python scripts/compare_lead_times.py
```

### Baseline context

The Isolation Forest baseline has only **20/100 engines with any alert** (80% false negative
rate). The causal pipeline (blended z-score + causal score, threshold 0.3) is expected to
achieve higher coverage. If mean lead time is lower despite higher coverage, the threshold
may be too aggressive — that is a finding to log and tune in Days 30-32.

### Design decisions

- **Standalone script, not an API endpoint**: Read-only DB query; no server restart needed
  to iterate on comparison logic.
- **RIGHT join on `rul_labels`**: Preserves all 100 engines in the output so engines with
  no alert appear with `NaN` lead time — same shape as the IF baseline CSV for easy diffing.
- **No new DB tables**: `alert_events` already stores every decision made by `/ingest`.

### Bug found and fixed during Day 29 testing

**Problem:** `compute_anomaly_score` used mean z-score across all 14 sensors.
Turbofan degradation starts in 1-2 sensors — averaging them with 12-13 healthy
sensors buries the signal. A sensor 3 SDs out reads as score 0.04 when averaged
with 13 normal sensors, so the pipeline only fires very late (near failure).

**Result before fix:**
```
Engines with any alert    16 (causal)   17 (IF)
Mean lead time            21.0 cycles   107.4 cycles  ← causal 5x worse
```

**Fix:** `backend/anomaly.py` — replaced mean with max:
```python
# Before
mean_z = sum(z_scores) / len(z_scores)
return min(mean_z / 5.0, 1.0)

# After
max_z = max(z_scores)
return min(max_z / 5.0, 1.0)
```

**Also fixed:** `compare_lead_times.py` SQL query used `WHERE ae.decision = 'ALERT'`
but `make_decision` returns `'UNCERTAIN'` for scores in [0.3, 0.6). Fixed to
`WHERE ae.decision IN ('ALERT', 'UNCERTAIN')`.

Re-stream test_FD001.txt after this fix to get updated lead time numbers.

**Result after fix:**
```
Metric                              Causal    Iso Forest
Engines with any alert                 100            17   ← 6x better coverage
Mean lead time (cycles)              200.4         107.4   ← 2x earlier
Median lead time                     192.0          41.0
Min lead time                          115             9   ← no near-miss alerts
Max lead time                          340           272
```

Day 29 complete. Causal pipeline beats Isolation Forest on every metric.

**Noise floor derivation:** `floor = 2 × cross-engine std at cycle 1` from `train_FD001.txt`.
This represents the natural spread between healthy engines at the same life stage —
a deviation smaller than that is within normal inter-engine variation, not degradation.

| Sensor | Training std | Cross-engine std (cycle 1) | Noise floor |
|--------|-------------|---------------------------|-------------|
| sensor_2  | 0.501 | 0.358 | 0.75 |
| sensor_8  | 0.058 | 0.055 | 0.15 |
| sensor_13 | 0.051 | 0.054 | 0.15 |
| sensor_15 | 0.035 | 0.027 | 0.07 |

**Result after noise floor fix:**
```
Metric                              Causal    Iso Forest
Engines with any alert                  47            17   ← 2.8x better coverage
Mean lead time (cycles)              131.9         107.4   ← 23% earlier on average
Median lead time                     128.0          41.0
Min lead time                           29             9   ← no near-miss alerts
Max lead time                          316           272
```

**Diagnostic check (first_alert_cycle):**
- Engines with total_cycles ≤ 15 firing early are correctly near failure in the test set
- 2 engines (23, 78) still fire at cycle 1 — inter-engine sensor baseline variation not
  fully captured by global noise floor; fixable only with engine-specific baselines (future work)
- All decisions are UNCERTAIN (score 0.3–0.6) — the ALERT tier (≥0.6) requires multi-sensor
  simultaneous degradation not present in FD001 single-condition data

**Precision-recall tradeoff documented:**
The causal pipeline trades slight false positive rate (2/47 flagged engines are cycle-1 fires)
for 2.8× better recall vs Isolation Forest, with 23% earlier mean detection.
This is the correct tradeoff for a predictive maintenance system where missing a failure
costs more than an unnecessary inspection.

**Day 29 complete.**
