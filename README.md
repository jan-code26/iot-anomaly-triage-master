# Don't Trust the Sensors — IoT Anomaly Triage

An agentic IoT anomaly triage system for turbofan engine sensors.
Built for INFO 7390 (Northeastern University) using NASA CMAPSS data.

- Ingests real-time sensor readings, forward-fills missing values, and detects anomalies
- Scores each reading using a **blended causal + z-score model** (conditioned on operating conditions)
- Monitors sensor distribution drift with Population Stability Index (PSI)
- Validates physical sensor coupling with G-tests (thermodynamic consistency)
- Writes every decision to PostgreSQL for auditability and LLM reasoning traces
- Returns NORMAL / UNCERTAIN / ALERT decisions with per-sensor causal residuals

---

## Architecture

```
CMAPSS Data
    │
    ▼
simulate_stream.py ──POST /ingest──► FastAPI (backend/main.py)
                                          │
                              ┌───────────┼────────────────┐
                              ▼           ▼                ▼
                       SensorService  PSIMonitor     GTestMonitor
                       (forward-fill) (drift detect) (coupling check)
                              │
                    ┌─────────┴──────────────────┐
                    ▼                            ▼
             anomaly.py                  causal_scorer.py
             (z-score scorer)            (residual scorer —
                    │                    conditions on op_settings)
                    └─────────┬──────────────────┘
                              │ blended 50/50
                              ▼
                    ┌─────────┴──────────┐
                    ▼                    ▼
             telemetry_windows      alert_events
                    │                    │
                    ▼              dowhy_results
              Neon PostgreSQL (8 tables)
```

**Causal DAG** — the scorer conditions each sensor on its physical cause:
```
op_setting_1 (Altitude) → sensor_4   (HPC outlet temperature)
op_setting_2 (Mach)     → sensor_11  (HPC outlet temperature)
                        → sensor_15  (HPC outlet pressure)
op_setting_3 (TRA)      → sensor_3   (fan inlet temperature)
                        → sensor_9   (physical fan speed)
```

---

## Quick Start

### Prerequisites
- Python 3.11
- A [Neon](https://neon.tech) PostgreSQL project (free tier)
- A [Groq](https://console.groq.com) API key (free tier)

### Setup

```bash
# 1. Clone
git clone https://github.com/jan-code26/iot-anomaly-triage.git
cd iot-anomaly-triage-master

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements-dev.txt   # includes jupyter, pytest, scikit-learn
# or for server only:
pip install -r requirements.txt

# 4. Environment variables
cp .env.example .env
# Edit .env and fill in:
#   DATABASE_URL=postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech:6432/dbname?sslmode=require
#   GROQ_API_KEY=your_key_here
#   LLM_PROVIDER=groq              # or gemini

# 5. Download dataset
python scripts/download_cmapss.py

# 6. Create database schema
python scripts/create_schema.py

# 7. Start the server
uvicorn backend.main:app --reload

# 8. Test it
curl http://localhost:8000/health
# → {"status":"ok","message":"Sensor triage system is running"}
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | Neon pooled connection string (port 6432, sslmode=require) |
| `GROQ_API_KEY` | Yes | API key from console.groq.com |
| `LLM_PROVIDER` | Yes | `groq` or `gemini` |
| `GEMINI_API_KEY` | If `LLM_PROVIDER=gemini` | API key from Google AI Studio |
| `GROQ_MODEL` | No | Groq model name (default: `llama-3.3-70b-versatile`) |
| `GEMINI_MODEL` | No | Gemini model name (default: `gemini-2.0-flash`) |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check — used by Render for liveness |
| POST | `/ingest` | Submit one sensor reading; returns blended anomaly decision |
| GET | `/telemetry/{id}` | Retrieve a saved telemetry window by UUID |
| GET | `/psi/status` | Current PSI drift score per sensor |
| POST | `/baselines/reset` | Log maintenance event, clear PSI baselines |

API docs (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)

### `/ingest` response fields

```json
{
  "id": "uuid",
  "engine_id": 1,
  "cycle": 42,
  "imputation_density": 0.05,
  "stale_sensors": [],
  "warnings": [],
  "created_at": "2026-04-10T12:00:00Z"
}
```

---

## Running Tests

Always run from the project root so `conftest.py` is loaded:

```bash
# Unit tests — no database required
pytest tests/test_anomaly.py tests/test_causal_scorer.py -v

# All tests (requires DATABASE_URL in .env)
pytest tests/ -v
```

In VSCode, use the **Testing sidebar** (beaker icon) — it runs pytest correctly.
Do not use the play button on individual test files (it skips `conftest.py`).

---

## Key Scripts

| Script | Purpose |
|---|---|
| `scripts/download_cmapss.py` | Download 12 CMAPSS data files from GitHub mirror |
| `scripts/create_schema.py` | Apply all 8 tables to Neon (safe to re-run) |
| `scripts/simulate_stream.py` | POST CMAPSS rows to /ingest as a live sensor feed |
| `scripts/lead_time_baseline.py` | Train Isolation Forest, save lead-time CSV |
| `scripts/neon_smoke_test.py` | Quick DB connectivity check |

Simulator examples:
```bash
python scripts/simulate_stream.py --rows 100
python scripts/simulate_stream.py --rows 500 --fault-injection
python scripts/simulate_stream.py --engines 1,2,3 --delay 0
```

---

## Project Structure

```
iot-anomaly-triage-master/
│
├── backend/
│   ├── main.py                  # FastAPI app — all endpoints
│   ├── database.py              # SQLAlchemy engine (QueuePool → Neon)
│   ├── models.py                # 8 SQLAlchemy Core table definitions
│   ├── schemas.py               # Pydantic v2 request/response schemas
│   ├── anomaly.py               # Z-score scorer + decision logic
│   └── services/
│       ├── sensor_service.py    # Forward-fill imputation (5-cycle stale threshold)
│       ├── causal_scorer.py     # Causal residual scorer (op_settings → sensors)
│       ├── psi_monitor.py       # PSI drift detection (rolling 200 readings)
│       └── gtest_monitor.py     # G-test sensor coupling (sensor_11 vs sensor_15)
│
├── scripts/
│   ├── download_cmapss.py       # Download NASA dataset
│   ├── create_schema.py         # Apply DB schema to Neon
│   ├── simulate_stream.py       # Stream simulator with fault injection
│   ├── lead_time_baseline.py    # Isolation Forest baseline CSV
│   └── neon_smoke_test.py       # DB smoke test
│
├── tests/
│   ├── test_connection.py       # DB connectivity
│   ├── test_ingest.py           # /ingest + /telemetry integration tests
│   ├── test_anomaly.py          # Z-score scorer unit tests (8 tests)
│   └── test_causal_scorer.py    # Causal scorer unit tests (10 tests)
│
├── notebooks/
│   └── 01_cmapss_eda.ipynb      # Exploratory data analysis
│
├── data/
│   ├── raw/                     # CMAPSS files (gitignored)
│   └── processed/               # baseline CSVs (gitignored)
│
├── requirements.txt             # Server deps (installed on Render)
├── requirements-dev.txt         # Dev deps (jupyter, pytest, scikit-learn)
├── render.yaml                  # Render deployment config
└── conftest.py                  # pytest path setup (project root → sys.path)
```

---

## Database Schema (8 tables)

```
telemetry_windows      ← one row per sensor reading (root table)
    │
    ├── alert_events   ← blended anomaly score + NORMAL/UNCERTAIN/ALERT decision
    │       │
    │       ├── reasoning_traces      ← LangGraph node execution log (Phase 3)
    │       ├── human_feedback        ← operator label corrections (Phase 3)
    │       └── lead_time_measurements ← cycles-before-failure metric
    │
    └── dowhy_results  ← causal residual scores per reading (Phase 3)

psi_baselines          ← reference distributions for PSI (standalone)
maintenance_events     ← physical maintenance log (standalone)
```

---

## Deployment on Render

1. Push to GitHub
2. Create a **Web Service** on [render.com](https://render.com) → connect repo
3. Render auto-reads `render.yaml` — build and start commands are set
4. Add env vars in the Render dashboard: `DATABASE_URL`, `GROQ_API_KEY`, `LLM_PROVIDER`
5. Verify: `https://your-app.onrender.com/health`

Note: free tier sleeps after 15 min of inactivity — first request takes ~30s to wake up.
