"""agents/planner.py — Planner Agent."""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from agents.state import ResearchState
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL

# Kept deliberately short — small models lose the plot with long system prompts
SYSTEM_PROMPT = """You are a Research Planner. Given a research query, output a concise plan with EXACTLY these four sections:

GOAL: One sentence describing what we want to learn.

QUESTIONS:
- Question 1
- Question 2
- Question 3

SEARCHES:
- search query 1
- search query 2
- search query 3

TOOLS: List which tools to use: web_search, wikipedia_search, arxiv_search

Keep it short and specific. Do not add extra sections."""


def planner_node(state: ResearchState) -> ResearchState:
    """Generate a research plan for the given query."""

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.1,
        # Limit output length — planners don't need long responses
        num_predict=512,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Create a research plan for: {state['query']}"),
    ]

    try:
        response = llm.invoke(messages)
        plan = response.content.strip()

        # Sanity check — if the model returned something very short or empty,
        # generate a minimal fallback plan so downstream agents aren't starved
        if len(plan) < 50:
            plan = (
                f"GOAL: Research {state['query']}.\n\n"
                f"QUESTIONS:\n- What is {state['query']}?\n"
                f"- What are recent developments?\n"
                f"- What are the key findings?\n\n"
                f"SEARCHES:\n- {state['query']}\n"
                f"- {state['query']} recent research\n"
                f"- {state['query']} latest findings\n\n"
                f"TOOLS: web_search, wikipedia_search"
            )

        return {
            **state,
            "research_plan": plan,
            "current_agent": "planner",
            "messages": [AIMessage(content=f"📋 **Research Plan**\n\n{plan}", name="planner")],
        }
    except Exception as e:
        # Don't stop the pipeline — use a minimal fallback plan
        fallback = (
            f"GOAL: Research {state['query']}.\n\n"
            f"SEARCHES:\n- {state['query']}\n- {state['query']} overview\n\n"
            f"TOOLS: web_search, wikipedia_search"
        )
        return {
            **state,
            "research_plan": fallback,
            "current_agent": "planner",
            "error": f"Planner error (using fallback): {e}",
            "messages": [AIMessage(content=f"📋 **Plan (fallback)**\n\n{fallback}", name="planner")],
        }