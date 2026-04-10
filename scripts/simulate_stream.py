"""
Stream simulator for the IoT Anomaly Triage pipeline.

Reads the CMAPSS training data row by row and POSTs each row to the
/ingest endpoint as if it were a live sensor feed.

Usage:
    python scripts/simulate_stream.py                          # 100 rows, localhost
    python scripts/simulate_stream.py --rows 500               # 500 rows
    python scripts/simulate_stream.py --rows 0                 # all rows
    python scripts/simulate_stream.py --delay 0                # as fast as possible
    python scripts/simulate_stream.py --fault-injection        # inject IEC 61508 faults
    python scripts/simulate_stream.py --engines 1,2,3          # only these engine IDs
    python scripts/simulate_stream.py --url http://app.onrender.com/ingest
"""
import argparse
import random
import time
from pathlib import Path

import pandas as pd
import requests

DATA_FILE = Path(__file__).parent.parent / "data" / "raw" / "train_FD001.txt"

COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21",
]

SENSOR_COLS = [c for c in COLUMNS if c.startswith("sensor_")]

# Approximate std per sensor (from EDA) — used for bias fault magnitude
SENSOR_STD = {
    "sensor_2": 0.501, "sensor_3": 6.132, "sensor_4": 14.806,
    "sensor_7": 0.603, "sensor_8": 0.058, "sensor_9": 20.834,
    "sensor_11": 0.396, "sensor_12": 2.578, "sensor_13": 0.051,
    "sensor_14": 14.800, "sensor_15": 0.035, "sensor_17": 2.483,
    "sensor_20": 0.271, "sensor_21": 0.178,
}

# Fault injection state: engine_id → {sensor: stuck_value, stuck_remaining}
_stuck_state: dict[int, dict] = {}


def inject_fault(payload: dict, cycle: int) -> dict:
    """
    Apply one IEC 61508 fault model to a randomly chosen informative sensor.
    Fault types: drift, spike, stuck, bias. Applied to ~5% of rows.
    """
    if random.random() > 0.05:
        return payload

    sensor = random.choice(list(SENSOR_STD.keys()))
    if payload.get(sensor) is None:
        return payload

    fault_type = random.choice(["drift", "spike", "stuck", "bias"])
    engine_id = payload["engine_id"]

    if fault_type == "drift":
        payload[sensor] = round(payload[sensor] + 0.01 * cycle, 4)

    elif fault_type == "spike":
        payload[sensor] = round(payload[sensor] * random.uniform(1.5, 3.0), 4)

    elif fault_type == "stuck":
        state = _stuck_state.setdefault(engine_id, {})
        if sensor not in state:
            state[sensor] = {"value": payload[sensor], "remaining": random.randint(3, 10)}
        state[sensor]["remaining"] -= 1
        payload[sensor] = state[sensor]["value"]
        if state[sensor]["remaining"] <= 0:
            del state[sensor]

    elif fault_type == "bias":
        std = SENSOR_STD.get(sensor, 1.0)
        payload[sensor] = round(payload[sensor] + 0.5 * std, 4)

    return payload


def load_data(engines: list[int] | None, limit: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=COLUMNS)
    if engines:
        df = df[df["engine_id"].isin(engines)]
    if limit > 0:
        df = df.head(limit)
    return df


def send_row(url: str, row: dict) -> tuple[int, dict]:
    resp = requests.post(url, json=row, timeout=10)
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    return resp.status_code, body


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a live sensor stream")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/ingest",
        help="Target /ingest endpoint URL",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Number of rows to send (0 = all rows)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Seconds to wait between rows (default 0.05)",
    )
    parser.add_argument(
        "--fault-injection",
        action="store_true",
        help="Inject IEC 61508 fault models (drift/spike/stuck/bias) into ~5%% of rows",
    )
    parser.add_argument(
        "--engines",
        type=str,
        default=None,
        help="Comma-separated engine IDs to include (e.g. 1,2,3). Default: all.",
    )
    args = parser.parse_args()

    if not DATA_FILE.exists():
        print(f"ERROR: Data file not found at {DATA_FILE}")
        print("Run: python scripts/download_cmapss.py")
        return

    engines = [int(e) for e in args.engines.split(",")] if args.engines else None
    df = load_data(engines, args.rows)
    total = len(df)
    fault_tag = " [FAULT INJECTION ON]" if args.fault_injection else ""
    print(f"Sending {total} rows to {args.url}{fault_tag}\n")

    sent = 0
    errors = 0

    for _, row in df.iterrows():
        payload = {col: (None if pd.isna(val) else val) for col, val in row.items()}
        payload["engine_id"] = int(payload["engine_id"])
        payload["cycle"] = int(payload["cycle"])

        if args.fault_injection:
            payload = inject_fault(payload, payload["cycle"])

        status, body = send_row(args.url, payload)

        imputation = body.get("imputation_density", "?")
        stale = body.get("stale_sensors", [])
        stale_tag = f" stale={stale}" if stale else ""

        if status == 201:
            print(
                f"  engine={payload['engine_id']:<3} "
                f"cycle={payload['cycle']:<4} "
                f"→ {status}  "
                f"imputation={imputation:.3f}"
                f"{stale_tag}"
            )
            sent += 1
        else:
            print(
                f"  engine={payload['engine_id']:<3} "
                f"cycle={payload['cycle']:<4} "
                f"→ {status}  ERROR: {body}"
            )
            errors += 1

        if args.delay > 0:
            time.sleep(args.delay)

    print(f"\nDone. Sent {sent}/{total} rows successfully. Errors: {errors}")


if __name__ == "__main__":
    main()
