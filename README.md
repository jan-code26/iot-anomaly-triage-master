# Don't Trust the Sensors — Regime-Aware Causal Anomaly Triage for Industrial IoT

**INFO 7390 · Northeastern University · Jahnavi Patel**

An agentic anomaly triage system for turbofan engine sensor streams. The system detects anomalies using a causal scoring model conditioned on operating regime, validates readings against a physics-based sensor coupling test, and generates operator-readable explanations via a 7-node LangGraph agent.

Evaluated on the [NASA CMAPSS benchmark](https://www.nasa.gov/intelligent-systems-division) across all four sub-datasets (FD001–FD004).

---

## Key Results

| Metric | Value | Dataset |
|--------|-------|---------|
| Coverage vs Isolation Forest | **4.1×** (69% vs 17%) | FD001 |
| Best F1 | **0.352** | FD004 |
| False positive rate — global z-score baseline | **100%** (F1 = 0.000) | FD002, FD004 |
| False positive rate — regime-aware causal | **34–75%** (F1 = 0.279–0.352) | FD002, FD004 |
| Mean alert lead time vs Isolation Forest | **165 vs 107 cycles (+53%)** | FD001 |

---

## Live Demo

- **Application:** [https://iot-anomaly-triage.onrender.com](https://iot-anomaly-triage.onrender.com) *(may take ~30 s to wake from free-tier sleep)*
- **API docs:** [https://iot-anomaly-triage.onrender.com/docs](https://iot-anomaly-triage.onrender.com/docs)

---

## Problem Statement

Industrial IoT anomaly detectors commonly compute z-scores against global training-set means. A turbofan engine running at high altitude produces temperature and pressure readings that are *normal for that regime* but anomalous relative to sea-level means — triggering false alarms unrelated to engine health.

This project addresses that with three layers:

1. **Regime-aware causal scoring** — anomaly score is the residual from a causally-predicted value conditioned on the current operating regime (KMeans k=6 on altitude/Mach/throttle settings), not a deviation from the global mean.
2. **Physics-based veto** — a G-test on the sensor_11 / sensor_15 coupling (isentropic compression relation) distinguishes real degradation from sensor faults.
3. **Human-in-the-loop** — operators submit TRUE_POSITIVE / FALSE_POSITIVE / UNCERTAIN labels; the cache_lookup node applies a confidence penalty to engines with repeated false-positive histories.

---

## Architecture

```
Sensor stream (POST /ingest)
        │
        ▼
  FastAPI backend
  ├── SensorService      forward-fill missing values (5-cycle stale threshold)
  ├── PSIMonitor         distribution drift detection (PSI > 0.2 = auto-reset)
  ├── GTestMonitor       sensor coupling validation (χ²(df=16) = 26.30 critical)
  ├── anomaly.py         z-score against per-regime rolling mean
  └── causal_scorer.py   residual from LinearRegression on causal DAG branches
        │
        ▼ blended score (α · causal + (1-α) · z-score)
        │
  LangGraph 7-node agent
  ┌─────────────────────────────────────────────┐
  │ ingest_validator → regime_classifier        │
  │   → causal_reasoner → physics_veto          │
  │   → cache_lookup → llm_explainer            │
  │   → decision_writer                         │
  └─────────────────────────────────────────────┘
        │
        ▼
  PostgreSQL (Neon) — 8 tables
  React + Vite dashboard
```

**Causal DAG** (DoWhy-validated structure, LinearRegression inference):

```
op_setting_1 (Altitude) → sensor_4   (LPC outlet temperature)
op_setting_2 (Mach)     → sensor_11  (HPC outlet temperature)
                        → sensor_15  (HPC outlet pressure)
op_setting_3 (TRA)      → sensor_3   (fan inlet temperature)
                        → sensor_9   (physical fan speed)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI, Python 3.11 |
| Agent pipeline | LangGraph 0.2, LangChain Core |
| Causal reasoning | DoWhy 0.11 (DAG validation), scikit-learn (inference) |
| LLM | Groq / Llama-3.3-70B or Gemini 2.0 Flash |
| Database | PostgreSQL on Neon (serverless) |
| Frontend | React 19, Vite, TypeScript, Recharts |
| Deployment | Render (backend + frontend bundled) |
| Data | NASA CMAPSS turbofan benchmark (FD001–FD004) |

---

## Quick Start (Local)

### Prerequisites
- Python 3.11+
- Node.js 18+
- A [Neon](https://neon.tech) PostgreSQL project (free tier)
- A [Groq](https://console.groq.com) API key (free tier)

### Backend setup

```bash
# 1. Clone
git clone https://github.com/jan-code26/iot-anomaly-triage.git
cd iot-anomaly-triage-master

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements-dev.txt

# 4. Environment variables
cp .env.example .env
# Fill in DATABASE_URL, GROQ_API_KEY, LLM_PROVIDER=groq

# 5. Download CMAPSS dataset
python scripts/download_cmapss.py

# 6. Create database schema
python scripts/create_schema.py

# 7. Compute regime coefficients
python scripts/compute_regime_coefficients.py

# 8. Start backend
uvicorn backend.main:app --reload
# → http://localhost:8000/docs
```

### Frontend setup

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Stream a dataset

```bash
# Stream 100 readings from FD001 to the local backend
python scripts/simulate_stream.py --rows 100

# With fault injection
python scripts/simulate_stream.py --rows 500 --fault-injection

# Specific engines only
python scripts/simulate_stream.py --engines 1,2,3 --delay 0
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | Neon pooled connection string (port 6432, `sslmode=require`) |
| `GROQ_API_KEY` | Yes | From [console.groq.com](https://console.groq.com) |
| `LLM_PROVIDER` | Yes | `groq` or `gemini` |
| `GEMINI_API_KEY` | If `gemini` | From Google AI Studio |
| `GROQ_MODEL` | No | Default: `llama-3.3-70b-versatile` |
| `GEMINI_MODEL` | No | Default: `gemini-2.0-flash` |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/ingest` | Submit one sensor reading → decision + LLM explanation |
| GET | `/cmapss/{dataset}/engines` | Fleet summary for FD001–FD004 |
| GET | `/cmapss/{dataset}/engines/{id}` | Single engine detail + score history |
| GET | `/cmapss/{dataset}/alerts` | Alert events for a dataset |
| GET | `/cmapss/{dataset}/chat` | LLM assistant for fleet-level questions |
| GET | `/alerts/recent` | Most recent alert events |
| GET | `/alerts/{id}/explanation` | LLM explanation + reasoning trace |
| POST | `/feedback` | Submit operator label (TRUE_POSITIVE / FALSE_POSITIVE / UNCERTAIN) |
| GET | `/admin/retraining-status` | Override count + retraining recommendation |
| GET | `/psi/status` | PSI drift score per sensor |

---

## Running Tests

```bash
# Unit tests — no database required
pytest tests/test_anomaly.py tests/test_causal_scorer.py -v

# All tests (requires DATABASE_URL in .env)
pytest tests/ -v
```

---

## Project Structure

```
iot-anomaly-triage-master/
├── backend/
│   ├── main.py                  # FastAPI app + all endpoints
│   ├── anomaly.py               # Z-score scorer
│   ├── cmapss_api.py            # CMAPSS evaluation endpoints + G-stat computation
│   ├── agent/
│   │   ├── graph.py             # LangGraph pipeline
│   │   └── nodes.py             # 7 node implementations
│   └── services/
│       ├── causal_scorer.py     # Causal residual scoring
│       ├── regime_classifier.py # KMeans regime classifier
│       ├── psi_monitor.py       # PSI drift monitor
│       ├── gtest_monitor.py     # G-test sensor coupling
│       └── sensor_service.py   # Forward-fill imputation
├── frontend/
│   └── src/
│       ├── pages/               # Overview, Engines, EngineDetail, Alerts, About, Methodology
│       ├── components/          # ChatWidget, ScoreBar, Sparkline, Badge, Card
│       └── lib/                 # API client, types, utils
├── scripts/
│   ├── download_cmapss.py       # Download NASA dataset
│   ├── create_schema.py         # Apply DB schema
│   ├── simulate_stream.py       # Live stream simulator
│   ├── compute_regime_coefficients.py
│   ├── ablation_study.py        # FD001 4-variant ablation
│   ├── fd002_regime_eval.py     # FD002 multi-condition evaluation
│   └── fd004_regime_eval.py     # FD004 multi-condition evaluation
├── data/processed/              # Ablation CSVs + regime_coefficients.json
├── tests/                       # pytest unit + integration tests
├── render.yaml                  # Render deployment config
└── requirements.txt
```

---

## Evaluation

Evaluation scripts are in `scripts/`. Pre-computed results are in `data/processed/`:

| File | Contents |
|------|----------|
| `ablation_table.csv` | FD001: IF vs z-score vs causal vs full pipeline |
| `fd002_regime_table.csv` | FD002: global z-score vs regime-aware causal |
| `fd004_regime_table.csv` | FD004: global z-score vs regime-aware causal |
| `fd003_ablation_table.csv` | FD003: same 4-variant ablation as FD001 |
| `causal_lead_times.csv` | Per-engine first alert cycle vs true failure cycle |
| `regime_coefficients.json` | KMeans centroids + per-cluster LinearRegression coefficients |

---

## Deployment (Render)

The `render.yaml` builds the React frontend and serves it from the FastAPI backend:

```
buildCommand: pip install -r requirements.txt && cd frontend && npm ci && npm run build
startCommand: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Add env vars (`DATABASE_URL`, `GROQ_API_KEY`, `LLM_PROVIDER`) in the Render dashboard. Free tier sleeps after 15 min of inactivity — first request takes ~30 s to wake.

---

## Author

Jahnavi Patel · College of Engineering, Northeastern University · patel.jahnavi@northeastern.edu
