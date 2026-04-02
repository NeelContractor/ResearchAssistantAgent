"""agents/state.py — Shared LangGraph state for the multi-agent graph."""

from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ResearchState(TypedDict):
    """State object that flows through the entire agent graph.

    Fields
    ------
    messages        : Full conversation + tool call history (auto-appended).
    query           : The original user research query.
    research_plan   : Strucutred plan from the Planner agent.
    search_results  : Raw results collected by the Researcher agent.
    analysis        : Analysis output from the Analyst agent.
    final_report    : Polished report produced by the Writer agent.
    current_agent   : Which agent node is currently active.
    iteration       : Number of research iterations completed.
    is_complete     : Whether the research pipeline has finished.
    error           : Last error message (if any).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    research_plan: Optional[str]
    search_results: Optional[list[dict[str, Any]]]
    analysis: Optional[str]
    final_report: Optional[str]
    current_agent: Optional[str]
    iteration: int
    is_complete: bool
    error: Optional[str]