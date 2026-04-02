"""
Stream simulator for the IoT Anomaly Triage pipeline.

Reads the CMAPSS training data row by row and POSTs each row to the
/ingest endpoint as if it were a live sensor feed.

Usage:
    python scripts/simulate_stream.py                      # 100 rows, localhost
    python scripts/simulate_stream.py --rows 500           # 500 rows
    python scripts/simulate_stream.py --rows 0             # all rows
    python scripts/simulate_stream.py --delay 0            # as fast as possible
    python scripts/simulate_stream.py --url http://your-app.onrender.com/ingest
"""
import argparse
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


def load_data(limit: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=COLUMNS)
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
    args = parser.parse_args()

    if not DATA_FILE.exists():
        print(f"ERROR: Data file not found at {DATA_FILE}")
        print("Run: python scripts/download_cmapss.py")
        return

    df = load_data(args.rows)
    total = len(df)
    print(f"Sending {total} rows to {args.url}\n")

    sent = 0
    errors = 0

    for _, row in df.iterrows():
        payload = {col: (None if pd.isna(val) else val) for col, val in row.items()}
        # engine_id and cycle must be int
        payload["engine_id"] = int(payload["engine_id"])
        payload["cycle"] = int(payload["cycle"])

        status, body = send_row(args.url, payload)

        imputation = body.get("imputation_density", "?")
        if status == 201:
            print(
                f"  engine={payload['engine_id']:<3} "
                f"cycle={payload['cycle']:<4} "
                f"→ {status}  "
                f"imputation={imputation:.3f}"
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
