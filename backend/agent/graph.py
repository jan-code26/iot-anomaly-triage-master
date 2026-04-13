"""
LangGraph graph for the IoT Anomaly Triage agent.

Compiles a 7-node linear state machine at module import time.
The compiled graph is a module-level singleton — compile() is called
once, not once per request.

Usage:
    from backend.agent.graph import run_triage_agent
    result = run_triage_agent(initial_state_dict)

LangGraph 0.2.76 notes:
  - END is imported from langgraph.graph (NOT langgraph.constants)
  - graph.compile() requires no checkpointer argument in 0.2.x
  - Node functions return PARTIAL dicts; LangGraph merges them into state
  - Use graph.invoke() (synchronous) — the existing stack is psycopg2-based
    and non-async; graph.ainvoke() would require asyncpg + async FastAPI
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from backend.agent.nodes import (
    cache_lookup,
    causal_reasoner,
    decision_writer,
    ingest_validator,
    llm_explainer,
    physics_veto,
    regime_classifier,
)
from backend.agent.state import AgentState


def _build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Register all 7 nodes
    g.add_node("ingest_validator", ingest_validator)
    g.add_node("regime_classifier", regime_classifier)
    g.add_node("causal_reasoner", causal_reasoner)
    g.add_node("physics_veto", physics_veto)
    g.add_node("cache_lookup", cache_lookup)
    g.add_node("llm_explainer", llm_explainer)
    g.add_node("decision_writer", decision_writer)

    # Entry point
    g.set_entry_point("ingest_validator")

    # Linear chain — no conditional edges in Phase 3
    _sequence = [
        "ingest_validator",
        "regime_classifier",
        "causal_reasoner",
        "physics_veto",
        "cache_lookup",
        "llm_explainer",
        "decision_writer",
    ]
    for src, dst in zip(_sequence, _sequence[1:]):
        g.add_edge(src, dst)
    g.add_edge("decision_writer", END)

    return g


# Module-level singleton — compiled once at import time
_compiled_graph = _build_graph().compile()


def run_triage_agent(initial_state: dict) -> dict:
    """
    Run the triage graph synchronously and return the final accumulated state.

    Args:
        initial_state: Dict with keys matching AgentState — at minimum:
            engine_id, cycle, telemetry_window_id, alert_event_id,
            z_score, causal_score, combined_score, causal_details,
            reading, stale_sensors, agent_warnings.

    Returns:
        Final AgentState dict after all 7 nodes have run.
        Key of interest: "llm_explanation", "final_score", "final_decision",
        "final_confidence", "agent_warnings".
    """
    return _compiled_graph.invoke(initial_state)
