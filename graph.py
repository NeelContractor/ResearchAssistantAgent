"""graph.py — LangGraph multi-agent research pipeline.

Graph topology:
  START → planner → researcher → analyst → writer → END

Each node is a specialized agent. The graph passes ResearchState
through the pipeline, accumulating results at each stage.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, START, END

from agents.state import ResearchState
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.writer import writer_node
from config.settings import RECURSION_LIMIT


def should_continue_after_researcher(state: ResearchState) -> Literal["analyst", "writer"]:
    """Decide whether to analyze results or skip to writing (on error)."""
    if state.get("error") or not state.get("search_results"):
        return "writer"
    return "analyst"


def should_continue_after_analyst(state: ResearchState) -> Literal["writer", "end"]:
    """Decide whether to write the report or end (on critical error)."""
    # FIX: return the string key "end" that maps to END in add_conditional_edges,
    # NOT the END sentinel itself (which is not a valid return value from a routing fn).
    if state.get("error") and not state.get("analysis"):
        return "end"
    return "writer"


def build_graph() -> StateGraph:
    """Build and compile the multi-agent research graph."""

    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("writer", writer_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")

    builder.add_conditional_edges(
        "researcher",
        should_continue_after_researcher,
        {"analyst": "analyst", "writer": "writer"},
    )

    builder.add_conditional_edges(
        "analyst",
        should_continue_after_analyst,
        # FIX: key "end" (lowercase string) maps to the END sentinel constant.
        {"writer": "writer", "end": END},
    )

    builder.add_edge("writer", END)

    graph = builder.compile()
    return graph


def get_graph() -> StateGraph:
    """Always build a fresh graph so env-var changes (model, URL) take effect.

    The original code cached _GRAPH at module level, which meant any model or
    Ollama URL changes made in the sidebar after first run were silently ignored.
    Building is cheap compared to inference, so we skip the cache entirely.
    """
    return build_graph()


def run_research(query: str, progress_callback=None) -> dict:
    """Run the full research pipeline for a query.

    Args:
        query: The research question/topic.
        progress_callback: Optional callable(agent_name, message) for UI updates.

    Returns:
        Final ResearchState dict with report and all intermediate results.
    """
    graph = get_graph()

    initial_state: ResearchState = {
        "messages": [],
        "query": query,
        "research_plan": None,
        "search_results": None,
        "analysis": None,
        "final_report": None,
        "current_agent": None,
        "iteration": 0,
        "is_complete": False,
        "error": None,
    }

    final_state = initial_state
    agent_order = ["planner", "researcher", "analyst", "writer"]

    try:
        for event in graph.stream(
            initial_state,
            config={"recursion_limit": RECURSION_LIMIT},
        ):
            for node_name, node_state in event.items():
                final_state = {**final_state, **node_state}
                if progress_callback and node_name in agent_order:
                    progress_callback(
                        node_name,
                        node_state.get("messages", []),
                    )
    except Exception as e:
        final_state["error"] = f"Graph execution error: {e}"
        final_state["is_complete"] = True

    return final_state