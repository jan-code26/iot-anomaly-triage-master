# Don't Trust the Sensors — IoT Anomaly Triage

An IoT anomaly triage system that uses LLMs to help identify and prioritize sensor anomalies.

## Features

- FastAPI backend with health check endpoint
- Support for multiple LLM providers: **Groq** (fast dev iteration) and **Gemini**
- PostgreSQL database (Neon) for persistence
- REST API docs at `/docs` when running locally

## Quick Start

### Prerequisites

- Python 3.10+
- API keys for Groq and/or Gemini
- Neon PostgreSQL connection string

### Setup

1. **Clone and enter the project:**
  ```bash
  cd iot-anomaly-triage-master
  ```
2. **Create and activate a virtual environment:**
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```
3. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
4. **Set up environment variables:**
  - Copy .env.example to .env
  - Add your Groq API key, Gemini API key (if using Gemini), and DATABASE_URL
  - Set LLM_PROVIDER to groq or gemini
5. **Run the application:**
  ```bash
  uvicorn backend.main:app --reload
  ```
6. **Access the API docs:**
  - Health: [http://localhost:8000/health](http://localhost:8000/health)
  - API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Environment Variables


| Variable       | Description                               |
| -------------- | ----------------------------------------- |
| LLM_PROVIDER   | groq or gemini                            |
| GROQ_API_KEY   | API key for Groq                          |
| GROQ_MODEL     | Model name (e.g. llama-3.3-70b-versatile) |
| GEMINI_API_KEY | API key for Gemini                        |
| GEMINI_MODEL   | Model name (e.g. gemini-2.0-flash)        |
| DATABASE_URL   | Neon PostgreSQL connection string         |


## Project Structure

```
├── backend/
│   └── main.py          # FastAPI application
│   └── database.py      # SQLAlchemy engine
├── requirements.txt
├── scripts/
│   └── neon_smoke_test.py # Neon smoke test script
│   ├── .env.example
│   └── README.md
├── .env.example
└── README.md
```

## Database (Neon PostgreSQL)

This project uses **Neon Postgres** with the **pooled** connection endpoint to handle free-tier connection limits.

- Set `DATABASE_URL` in `.env` (use pooled host/port **6432**, and `sslmode=require`)
- SQLAlchemy engine uses `QueuePool` with `pool_size=5`, `pool_pre_ping=True`, `pool_recycle=300`

### Smoke test (creates `test_events`, inserts a row, reads it back)

```bash
python scripts/neon_smoke_test.py
```

