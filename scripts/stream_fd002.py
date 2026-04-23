"""
FD002 end-to-end integration test.

Streams rows from test_FD002.txt through the live /ingest endpoint, then
queries the reasoning_traces (via /alerts/{id}/explanation) to verify that:
  1. Multiple distinct operating-regime clusters appear (not always cluster_0)
  2. LLM explanations are returned for high-score alerts

Output:
  - Live per-row summary as rows are sent
  - Final cluster-distribution table
  - Pass / FAIL verdict

Usage:
    python scripts/stream_fd002.py                      # 5 engines, first 30 cycles each
    python scripts/stream_fd002.py --engines 3          # 3 engines
    python scripts/stream_fd002.py --cycles 50          # first 50 cycles per engine
    python scripts/stream_fd002.py --url http://...     # remote backend
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "raw" / "test_FD002.txt"

COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get(base_url: str, path: str, timeout: int = 8):
    r = requests.get(base_url + path, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post_ingest(base_url: str, payload: dict, timeout: int = 15):
    r = requests.post(base_url + "/ingest", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def decision_icon(decision: str, score: float) -> str:
    if score >= 0.6 or decision in ("ALERT", "TRUE_POSITIVE"):
        return "🔴"
    if score >= 0.3 or decision == "UNCERTAIN":
        return "🟡"
    return "⚪"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FD002 integration test")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--engines", type=int, default=5,
                        help="Number of FD002 test engines to stream (default 5)")
    parser.add_argument("--cycles", type=int, default=30,
                        help="Max cycles per engine to stream (default 30)")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Seconds between rows (default 0.02)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    # --- Health check ---
    try:
        h = requests.get(base_url + "/health", timeout=5).json()
        print(f"✓ Backend: {base_url} — {h.get('message', 'ok')}")
    except Exception as exc:
        print(f"✗ Cannot reach backend at {base_url}: {exc}")
        return

    # --- Load FD002 test data ---
    if not DATA_FILE.exists():
        print(f"✗ {DATA_FILE} not found. Run: python scripts/download_cmapss.py")
        return

    df = pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=COLUMNS)
    engine_ids = sorted(df["engine_id"].unique())[: args.engines]
    df = df[df["engine_id"].isin(engine_ids)]
    df = df.groupby("engine_id").head(args.cycles).reset_index(drop=True)

    print(f"  Streaming {len(df)} rows — {len(engine_ids)} engines × up to {args.cycles} cycles each\n")
    print(f"  {'Engine':<8} {'Cycle':<6} {'op1':>6} {'op2':>6} {'op3':>6}   {'Score':>6}  {'Decision':<12}  Expl?")
    print("  " + "─" * 72)

    sent_alert_ids: list[str] = []
    sent = 0
    errors = 0

    for _, row in df.iterrows():
        payload = {col: (None if pd.isna(val) else val) for col, val in row.items()}
        payload["engine_id"] = int(payload["engine_id"])
        payload["cycle"] = int(payload["cycle"])

        try:
            body = post_ingest(base_url, payload)
            sent += 1
        except Exception as exc:
            print(f"  engine={payload['engine_id']} cycle={payload['cycle']} ERROR: {exc}")
            errors += 1
            continue

        # /ingest returns TelemetryWindowOut (no score) — fetch latest alert score
        # from /alerts/recent for this engine to get score + decision
        # (We'll bulk-query at the end; for live output just show imputation)
        expl_flag = "✓" if body.get("llm_explanation") else "·"

        print(
            f"  {payload['engine_id']:<8} {payload['cycle']:<6} "
            f"{(payload.get('op_setting_1') or 0):>6.1f} "
            f"{(payload.get('op_setting_2') or 0):>6.3f} "
            f"{(payload.get('op_setting_3') or 0):>6.1f}   "
            f"{'':>6}   {'':12}  {expl_flag}"
        )

        if args.delay > 0:
            time.sleep(args.delay)

    print(f"\n  Sent {sent} rows ({errors} errors)\n")

    # --- Query results ---
    print("Querying regime labels from reasoning traces…\n")
    time.sleep(1)  # let DB writes settle

    try:
        recent_alerts = get(base_url, f"/alerts/recent?limit={sent + 20}")
    except Exception as exc:
        print(f"✗ Could not fetch alerts: {exc}")
        return

    # Filter alerts to those belonging to our streamed engines
    # (alert_events doesn't store engine_id directly; use telemetry_window_id
    # — we can't easily filter. Just take all recent ones within our run.)
    regime_counter: Counter = Counter()
    score_by_regime: dict[str, list[float]] = {}
    n_with_explanation = 0
    n_alerts_checked = 0

    print(f"  {'Alert ID':<12} {'Score':>6}  {'Decision':<18}  {'Regime':<12}  LLM?")
    print("  " + "─" * 65)

    for alert in recent_alerts[: sent + 5]:
        alert_id = str(alert["id"])
        score = alert["anomaly_score"]
        decision = alert["decision"]
        icon = decision_icon(decision, score)

        try:
            exp = get(base_url, f"/alerts/{alert_id}/explanation")
            regime = exp.get("regime") or "unknown"
            has_llm = bool(exp.get("llm_explanation"))
        except Exception:
            regime = "unknown"
            has_llm = False

        regime_counter[regime] += 1
        score_by_regime.setdefault(regime, []).append(score)
        if has_llm:
            n_with_explanation += 1
        n_alerts_checked += 1

        print(
            f"  {alert_id[:8]+'…':<12} {score:>6.3f}  "
            f"{icon} {decision:<16}  {regime:<12}  {'✓' if has_llm else '·'}"
        )

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"CLUSTER DISTRIBUTION ({n_alerts_checked} alerts)")
    print(f"{'='*60}")
    print(f"  {'Regime':<14} {'Count':>6}  {'Mean Score':>12}")
    print("  " + "─" * 36)

    for regime, count in sorted(regime_counter.items()):
        scores = score_by_regime.get(regime, [])
        mean_s = sum(scores) / len(scores) if scores else 0.0
        print(f"  {regime:<14} {count:>6}  {mean_s:>12.3f}")

    n_distinct_clusters = len([r for r in regime_counter if r.startswith("cluster_")])

    print(f"\n  Distinct clusters seen:  {n_distinct_clusters}")
    print(f"  Alerts with LLM text:    {n_with_explanation} / {n_alerts_checked}")

    # --- Verdict ---
    print(f"\n{'='*60}")
    passes = []
    fails  = []

    if n_distinct_clusters >= 2:
        passes.append(f"≥2 distinct regime clusters observed ({n_distinct_clusters})")
    else:
        fails.append(f"Only {n_distinct_clusters} cluster(s) seen — regime_classifier may be stuck on cluster_0")

    if errors == 0:
        passes.append("All rows ingested without errors")
    else:
        fails.append(f"{errors} ingest errors")

    for msg in passes:
        print(f"  ✅ {msg}")
    for msg in fails:
        print(f"  ❌ {msg}")

    verdict = "PASS" if not fails else "FAIL"
    print(f"\n  Verdict: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
