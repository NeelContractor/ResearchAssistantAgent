"""agents/analyst.py — Analyst Agent."""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from agents.state import ResearchState
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL

# Short, directive prompt that works well with smaller models
SYSTEM_PROMPT = """You are a Research Analyst. You will be given raw search results.
Your job is to extract and organize the key information.

Write your analysis with these sections:
## Key Findings
(The most important facts from the sources — be specific, cite numbers/dates where available)

## Main Themes
(Recurring ideas across multiple sources)

## Gaps & Uncertainties
(What the sources didn't cover or contradicted each other on)

## Notable Facts
(Specific data points, statistics, names, dates worth highlighting)

Be factual and specific. Do not invent information not present in the sources."""


def analyst_node(state: ResearchState) -> ResearchState:
    """Analyze and synthesize the research results."""

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.1,
        num_predict=1536,
    )

    query = state["query"]
    results = state.get("search_results", [])

    if not results:
        no_data_msg = "No research data available to analyze."
        return {
            **state,
            "analysis": no_data_msg,
            "current_agent": "analyst",
            "messages": [AIMessage(content=f"🔬 **Analysis**\n\n{no_data_msg}", name="analyst")],
        }

    # Format sources — give the analyst the FULL result text (up to a safe cap)
    # so it has real content to work with rather than truncated snippets.
    # We budget ~300 tokens per source; 6 sources × 300 ≈ 1800 tokens input.
    source_blocks = []
    total_chars = 0
    char_budget = 8000  # roughly 2000 tokens, safe for 8 GB machines

    for i, r in enumerate(results, 1):
        header = f"[Source {i}: {r['tool']} | query: {r['args']}]"
        content = r["result"]
        # How much of this source can we still fit?
        remaining = char_budget - total_chars
        if remaining <= 100:
            break
        snippet = content[:min(len(content), remaining)]
        block = f"{header}\n{snippet}"
        source_blocks.append(block)
        total_chars += len(block)

    sources_text = "\n\n".join(source_blocks)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Research query: {query}\n\n"
                f"=== RAW SOURCES ===\n{sources_text}\n=== END SOURCES ===\n\n"
                "Write a structured analysis of the above sources."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        analysis = response.content.strip()

        if len(analysis) < 50:
            analysis = f"Analysis could not be generated. Raw sources collected: {len(results)}."

        return {
            **state,
            "analysis": analysis,
            "current_agent": "analyst",
            "messages": [AIMessage(content=f"🔬 **Analysis**\n\n{analysis}", name="analyst")],
        }
    except Exception as e:
        err_analysis = f"Analyst error: {e}. Raw sources were collected ({len(results)} results)."
        return {
            **state,
            "analysis": err_analysis,
            "current_agent": "analyst",
            "error": err_analysis,
            "messages": [AIMessage(content=f"🔬 **Analysis**\n\n{err_analysis}", name="analyst")],
        }