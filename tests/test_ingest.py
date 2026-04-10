"""
Integration test for POST /ingest and GET /telemetry/{id}.
Uses FastAPI TestClient (backed by httpx) — hits the real Neon database.
"""
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

VALID_READING = {
    "engine_id": 1,
    "cycle": 999,
    "op_setting_1": 0.0,
    "op_setting_2": 0.0003,
    "op_setting_3": 100.0,
    "sensor_1": 518.67,
    "sensor_2": 641.82,
    "sensor_3": 1589.70,
    "sensor_4": 1400.60,
    "sensor_7": 554.36,
    "sensor_8": 2388.06,
    "sensor_9": 9046.19,
    "sensor_11": 47.47,
    "sensor_12": 521.66,
    "sensor_13": 2388.02,
    "sensor_14": 8138.62,
    "sensor_15": 8.4195,
    "sensor_17": 391.0,
    "sensor_20": 39.06,
    "sensor_21": 23.419,
}


def test_ingest_returns_201():
    resp = client.post("/ingest", json=VALID_READING)
    assert resp.status_code == 201


def test_ingest_response_fields():
    resp = client.post("/ingest", json=VALID_READING)
    body = resp.json()
    assert "id" in body
    assert body["engine_id"] == VALID_READING["engine_id"]
    assert body["cycle"] == VALID_READING["cycle"]
    assert "imputation_density" in body
    assert "stale_sensors" in body
    assert "created_at" in body


def test_ingest_imputation_density_zero_when_all_present():
    resp = client.post("/ingest", json=VALID_READING)
    body = resp.json()
    assert body["imputation_density"] == 0.0


def test_get_telemetry_by_id():
    # Insert a row then retrieve it
    resp = client.post("/ingest", json=VALID_READING)
    assert resp.status_code == 201
    telemetry_id = resp.json()["id"]

    get_resp = client.get(f"/telemetry/{telemetry_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == telemetry_id


def test_get_telemetry_not_found():
    resp = client.get("/telemetry/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_ingest_invalid_missing_engine_id():
    bad = {k: v for k, v in VALID_READING.items() if k != "engine_id"}
    resp = client.post("/ingest", json=bad)
    assert resp.status_code == 422
