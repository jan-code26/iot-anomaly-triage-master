"""
IoT Anomaly Triage — Streamlit Dashboard

Three tabs:
  1. Live Alert Feed    — recent alerts table, auto-refresh
  2. Alert Detail       — LLM explanation + operator feedback buttons
  3. Sensor Health      — PSI drift status per sensor

Usage (local):
    streamlit run dashboard/app.py

Usage (pointing at Render):
    Change the Backend URL in the sidebar to your Render deployment URL.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="IoT Anomaly Triage",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")
    backend_url = st.text_input(
        "Backend URL",
        value="http://localhost:8000",
        help="FastAPI backend (local or Render URL)",
    ).rstrip("/")

    st.divider()
    auto_refresh = st.toggle("Auto-refresh", value=False)
    refresh_interval = st.slider(
        "Refresh interval (s)", min_value=5, max_value=60, value=10, step=5,
        disabled=not auto_refresh,
    )
    alert_limit = st.slider("Alerts to fetch", min_value=10, max_value=100, value=20, step=10)

    st.divider()
    st.caption("Built with FastAPI + LangGraph + Streamlit")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(path: str, timeout: int = 5) -> dict | list | None:
    """GET from backend; return parsed JSON or None on error."""
    try:
        r = requests.get(f"{backend_url}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach backend at {backend_url}. Is the server running?")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def _post(path: str, payload: dict, timeout: int = 5) -> dict | None:
    """POST to backend; return parsed JSON or None on error."""
    try:
        r = requests.post(f"{backend_url}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"POST failed: {exc}")
        return None


def _decision_badge(decision: str) -> str:
    """Return a coloured emoji prefix for a decision label."""
    return {
        "ALERT":          "🔴 ALERT",
        "TRUE_POSITIVE":  "🔴 TRUE_POSITIVE",
        "FALSE_POSITIVE": "🟢 FALSE_POSITIVE",
        "UNCERTAIN":      "🟡 UNCERTAIN",
        "NORMAL":         "⚪ NORMAL",
    }.get(decision, decision)


def _score_color(score: float) -> str:
    if score >= 0.6:
        return "🔴"
    if score >= 0.3:
        return "🟡"
    return "🟢"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "selected_alert_id" not in st.session_state:
    st.session_state.selected_alert_id = None
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = 0.0
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = {}


# ---------------------------------------------------------------------------
# Auto-refresh trigger
# ---------------------------------------------------------------------------

now = time.time()
if auto_refresh and (now - st.session_state.last_refresh) >= refresh_interval:
    st.session_state.last_refresh = now
    st.rerun()


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("🛩️ IoT Anomaly Triage Dashboard")
health = _get("/health")
if health:
    st.caption(f"Backend: {backend_url} — {health.get('message', 'connected')}")
else:
    st.stop()

tab_feed, tab_detail, tab_health = st.tabs(
    ["📋 Live Alert Feed", "🔍 Alert Detail", "📊 Sensor Health"]
)


# ---------------------------------------------------------------------------
# Tab 1 — Live Alert Feed
# ---------------------------------------------------------------------------

with tab_feed:
    col_head, col_refresh = st.columns([5, 1])
    with col_head:
        st.subheader("Recent Alerts")
    with col_refresh:
        if st.button("↺ Refresh", use_container_width=True):
            st.session_state.last_refresh = time.time()
            st.rerun()

    alerts = _get(f"/alerts/recent?limit={alert_limit}")
    if not alerts:
        st.info("No alerts found. Ingest some readings via POST /ingest.")
        st.stop()

    # Build display dataframe
    rows = []
    for a in alerts:
        triggered = a.get("triggered_at", "")
        if triggered:
            try:
                dt = datetime.fromisoformat(triggered.replace("Z", "+00:00"))
                triggered = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
        rows.append({
            "Select": False,
            "Time": triggered,
            "Alert ID": str(a["id"])[:8] + "…",
            "_full_id": str(a["id"]),
            "Score": round(a["anomaly_score"], 3),
            "Decision": _decision_badge(a["decision"]),
            "Confidence": f"{a['confidence']:.2%}",
            "Cache hit": "✓" if a.get("cache_hit") else "",
        })

    df = pd.DataFrame(rows)

    # Color score column
    def _color_row(row):
        score = row["Score"]
        if score >= 0.6:
            return ["background-color: #ffe0e0"] * len(row)
        if score >= 0.3:
            return ["background-color: #fff8e0"] * len(row)
        return [""] * len(row)

    display_cols = ["Time", "Alert ID", "Score", "Decision", "Confidence", "Cache hit"]
    styled = (
        df[display_cols + ["_full_id"]]
        .drop(columns=["_full_id"])
        .style.apply(_color_row, axis=1)
        .format({"Score": "{:.3f}"})
    )
    st.dataframe(styled, width="stretch", hide_index=True)

    # Alert selector
    st.divider()
    alert_options = {f"{r['Time']} — {r['_full_id'][:8]}…  [{r['Decision']}]": r["_full_id"]
                     for r in rows}
    chosen_label = st.selectbox(
        "Select alert to inspect →",
        options=list(alert_options.keys()),
        index=0,
    )
    if st.button("Open in Alert Detail tab →", type="primary"):
        st.session_state.selected_alert_id = alert_options[chosen_label]
        st.rerun()

    # Summary metrics
    st.divider()
    total = len(alerts)
    n_alert = sum(1 for a in alerts if a["decision"] in ("ALERT", "TRUE_POSITIVE"))
    n_uncertain = sum(1 for a in alerts if a["decision"] == "UNCERTAIN")
    n_fp = sum(1 for a in alerts if a["decision"] == "FALSE_POSITIVE")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total shown", total)
    c2.metric("🔴 ALERT / TP", n_alert)
    c3.metric("🟡 UNCERTAIN", n_uncertain)
    c4.metric("🟢 FALSE_POSITIVE", n_fp)

    if auto_refresh:
        st.caption(
            f"Auto-refresh every {refresh_interval}s — "
            f"next in {max(0, refresh_interval - int(time.time() - st.session_state.last_refresh))}s"
        )


# ---------------------------------------------------------------------------
# Tab 2 — Alert Detail + Feedback
# ---------------------------------------------------------------------------

with tab_detail:
    alert_id = st.session_state.selected_alert_id

    if not alert_id:
        st.info("Select an alert from the Live Alert Feed tab and click 'Open in Alert Detail'.")
    else:
        st.subheader(f"Alert `{alert_id[:8]}…`")

        # Find this alert in the already-fetched list
        alert_data = next(
            (a for a in (alerts or []) if str(a["id"]) == alert_id), None
        )

        if alert_data:
            score = alert_data["anomaly_score"]
            col_score, col_decision, col_conf, col_cache = st.columns(4)
            col_score.metric(
                "Anomaly Score",
                f"{score:.3f}",
                delta=f"{_score_color(score)}",
                delta_color="off",
            )
            col_decision.metric("Decision", alert_data["decision"])
            col_conf.metric("Confidence", f"{alert_data['confidence']:.2%}")
            col_cache.metric("Cache hit", "Yes" if alert_data.get("cache_hit") else "No")

            # Score progress bar
            st.progress(min(score, 1.0), text=f"Anomaly score: {score:.3f}")

        # Fetch LLM explanation + regime
        st.divider()
        with st.spinner("Fetching explanation…"):
            exp = _get(f"/alerts/{alert_id}/explanation")

        if exp:
            regime = exp.get("regime") or "—"
            llm_text = exp.get("llm_explanation")
            latency = exp.get("llm_latency_ms")

            col_r, col_l = st.columns([1, 3])
            col_r.metric("Operating Regime", regime)
            if latency:
                col_l.metric("LLM latency", f"{latency} ms")

            st.markdown("**LLM Explanation**")
            if llm_text:
                st.info(llm_text)
            else:
                st.warning(
                    "No LLM explanation available for this alert. "
                    "The agent runs only when combined_score ≥ 0.3."
                )

        # Feedback section
        st.divider()
        st.markdown("**Operator Feedback**")

        if alert_id in st.session_state.feedback_submitted:
            st.success(
                f"Feedback submitted: **{st.session_state.feedback_submitted[alert_id]}**"
            )
        else:
            override = st.checkbox(
                "Override decision (sets confidence = 1.0, treats this label as ground truth)",
                value=False,
            )
            fb_col1, fb_col2, fb_col3 = st.columns(3)
            with fb_col1:
                if st.button("✅ TRUE_POSITIVE", use_container_width=True, type="primary"):
                    result = _post("/feedback", {
                        "alert_event_id": alert_id,
                        "label": "TRUE_POSITIVE",
                        "override": override,
                    })
                    if result:
                        st.session_state.feedback_submitted[alert_id] = "TRUE_POSITIVE"
                        st.rerun()
            with fb_col2:
                if st.button("❌ FALSE_POSITIVE", use_container_width=True):
                    result = _post("/feedback", {
                        "alert_event_id": alert_id,
                        "label": "FALSE_POSITIVE",
                        "override": override,
                    })
                    if result:
                        st.session_state.feedback_submitted[alert_id] = "FALSE_POSITIVE"
                        st.rerun()
            with fb_col3:
                if st.button("❓ UNCERTAIN", use_container_width=True):
                    result = _post("/feedback", {
                        "alert_event_id": alert_id,
                        "label": "UNCERTAIN",
                        "override": override,
                    })
                    if result:
                        st.session_state.feedback_submitted[alert_id] = "UNCERTAIN"
                        st.rerun()

        # Prior feedback for this alert
        prior = _get(f"/alerts/{alert_id}/feedback")
        if prior:
            st.divider()
            st.markdown(f"**Prior feedback ({len(prior)} entries)**")
            prior_rows = [
                {
                    "Label": _decision_badge(p["label"]),
                    "Override": "✓" if p.get("override") else "",
                    "Submitted": p.get("submitted_at", "")[:19],
                }
                for p in prior
            ]
            st.dataframe(pd.DataFrame(prior_rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3 — Sensor Health (PSI)
# ---------------------------------------------------------------------------

with tab_health:
    st.subheader("Sensor Drift Status (PSI Monitor)")
    psi = _get("/psi/status")

    if not psi or not psi.get("sensors"):
        st.info("PSI monitor has no data yet. Ingest readings to populate sensor baselines.")
    else:
        sensor_list = psi["sensors"]  # list of {sensor, psi, status}

        statuses = [v.get("status", "unknown") for v in sensor_list]
        n_stable = statuses.count("stable")
        n_moderate = statuses.count("moderate")
        n_action = statuses.count("action_required")

        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Stable", n_stable)
        c2.metric("⚠️ Moderate drift", n_moderate)
        c3.metric("🔴 Action required", n_action)

        st.divider()

        status_icons = {"stable": "✅", "moderate": "⚠️", "action_required": "🔴"}
        psi_rows = [
            {
                "Sensor": v["sensor"],
                "Status": f"{status_icons.get(v['status'], '❓')} {v['status']}",
                "PSI Score": round(v["psi"], 4) if v.get("psi") is not None else "—",
            }
            for v in sorted(sensor_list, key=lambda x: x["sensor"])
        ]

        st.dataframe(pd.DataFrame(psi_rows), width="stretch", hide_index=True)

        st.caption(
            "PSI < 0.1: stable | 0.1–0.2: moderate drift | > 0.2: action required"
        )
