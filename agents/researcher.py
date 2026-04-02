"""agents/researcher.py — Researcher Agent."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from agents.state import ResearchState
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, MAX_ITERATIONS
from tools.research_tools import web_search, wikipedia_search, arxiv_search, scrape_webpage, calculator

RESEARCHER_TOOLS = [web_search, wikipedia_search, arxiv_search, scrape_webpage, calculator]

SYSTEM_PROMPT = """You are a Research Agent. Use tools to gather information.

Rules:
- Call tools one at a time
- Use web_search for current info, wikipedia_search for background, arxiv_search for papers
- After each result, decide what to search next
- Stop after 4-5 tool calls OR when you have enough information
- Do NOT repeat the same search twice

When done, reply with: RESEARCH COMPLETE"""

# These strings in a tool result mean the tool produced nothing useful.
# Used by _is_bad_result() to filter out empty/error results.
_BAD_RESULT_MARKERS = (
    "no search backend",
    "not installed",
    "search failed",
    "tool error",
    "unknown tool",
    "no results found",
    "could not retrieve",
    "error:",
    "failed to",
    "no matching",
    "no arxiv",
    "no wikipedia",
    "modulenotfounderror",
    "importerror",
)


def _is_bad_result(text: str) -> bool:
    """Return True if the tool result is an error / empty / unavailable."""
    lower = text.lower()
    return any(marker in lower for marker in _BAD_RESULT_MARKERS) or len(text.strip()) < 30


def _run_tool(tool_name: str, tool_args: dict) -> str:
    """Execute a tool by name, returning its string result."""
    tool_fn_map = {t.name: t for t in RESEARCHER_TOOLS}
    if tool_name not in tool_fn_map:
        return f"Unknown tool: {tool_name}"
    try:
        result = tool_fn_map[tool_name].invoke(tool_args)
        return str(result)
    except Exception as e:
        return f"Tool error: {e}"


def _check_backends() -> list[str]:
    """Return a list of human-readable warnings about missing search backends."""
    warnings = []
    try:
        from duckduckgo_search import DDGS  # noqa: F401
    except ImportError:
        warnings.append("duckduckgo-search not installed (run: pip install duckduckgo-search)")

    try:
        import wikipedia  # noqa: F401
    except ImportError:
        warnings.append("wikipedia not installed (run: pip install wikipedia)")

    try:
        import arxiv  # noqa: F401
    except ImportError:
        warnings.append("arxiv not installed (run: pip install arxiv)")

    try:
        import bs4  # noqa: F401
    except ImportError:
        warnings.append("beautifulsoup4 not installed — web scraping will use regex fallback")

    return warnings


def _direct_search(query: str, plan: str) -> list[dict[str, Any]]:
    """
    Fallback: run searches directly without LLM tool-calling.

    Called when the model produces zero tool calls (common with small models like
    llama3.2:1b that don't reliably emit structured tool-call JSON).

    Parses the SEARCHES section of the plan for specific queries, then runs
    web_search and wikipedia_search directly.
    """
    results: list[dict[str, Any]] = []

    # Parse SEARCHES section from plan
    search_terms: list[str] = []
    in_searches = False
    for line in plan.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SEARCHES"):
            in_searches = True
            continue
        if in_searches:
            if stripped.startswith("-"):
                term = stripped.lstrip("- ").strip()
                if term:
                    search_terms.append(term)
            elif stripped and not stripped.startswith("-"):
                in_searches = False

    # Guarantee at least 3 queries
    if not search_terms:
        search_terms = [
            query,
            f"{query} recent research 2024",
            f"{query} causes effects statistics",
        ]

    # Cap at 3 to keep RAM usage down on 8 GB machines
    for term in search_terms[:3]:
        # Web search
        web_result = _run_tool("web_search", {"query": term})
        if not _is_bad_result(web_result):
            results.append({
                "tool": "web_search",
                "args": {"query": term},
                "result": web_result[:2000],
            })

        # Wikipedia for background
        wiki_result = _run_tool("wikipedia_search", {"query": term})
        if not _is_bad_result(wiki_result):
            results.append({
                "tool": "wikipedia_search",
                "args": {"query": term},
                "result": wiki_result[:2000],
            })

    return results


def researcher_node(state: ResearchState) -> ResearchState:
    """Execute tool-calling loop to gather research data."""

    plan = state.get("research_plan", "No plan provided.")
    query = state["query"]

    # ── Pre-flight: warn about missing backends ───────────────────────────────
    backend_warnings = _check_backends()

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.1,
        num_predict=1024,
    ).bind_tools(RESEARCHER_TOOLS)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Research query: {query}\n\n"
                f"Plan:\n{plan}\n\n"
                "Use the tools to gather information now. Start with web_search."
            )
        ),
    ]

    collected_results: list[dict[str, Any]] = []
    max_tool_calls = min(MAX_ITERATIONS, 6)
    used_queries: set[str] = set()

    # ── Tool-calling loop ─────────────────────────────────────────────────────
    for _turn in range(max_tool_calls):
        try:
            response = llm.invoke(messages)
        except Exception:
            break

        messages.append(response)

        content_lower = (response.content or "").lower()
        if "research complete" in content_lower and not response.tool_calls:
            break
        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            cache_key = f"{tool_name}:{tool_args}"
            if cache_key in used_queries:
                tool_result_content = "Duplicate search skipped."
            else:
                used_queries.add(cache_key)
                tool_result_content = _run_tool(tool_name, tool_args)
                if not _is_bad_result(tool_result_content):
                    collected_results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": tool_result_content[:2000],
                    })

            messages.append(
                ToolMessage(
                    content=tool_result_content[:1500],
                    tool_call_id=tool_id,
                    name=tool_name,
                )
            )

    # ── Fallback: direct search if tool-calling produced nothing ──────────────
    if not collected_results:
        collected_results = _direct_search(query, plan)

    # ── If still nothing, surface a clear error in the state ─────────────────
    if not collected_results:
        missing = "; ".join(backend_warnings) if backend_warnings else "unknown reason"
        error_msg = (
            f"No search results collected. Missing dependencies: {missing}. "
            f"Run: pip install duckduckgo-search wikipedia"
        )
        summary_msg = AIMessage(
            content=f"🔍 **Research Failed**\n\n{error_msg}",
            name="researcher",
        )
        return {
            **state,
            "search_results": [],
            "current_agent": "researcher",
            "iteration": state.get("iteration", 0) + 1,
            "error": error_msg,
            "messages": [summary_msg],
        }

    # ── Compact summary ───────────────────────────────────────────────────────
    tools_used = len(set(r["tool"] for r in collected_results))
    summary_lines = "\n".join(
        f"- **{r['tool']}** `{r['args']}`: {r['result'][:120]}..."
        for r in collected_results
    )
    # Include any backend warnings as an informational note
    warning_note = ""
    if backend_warnings:
        warning_note = "\n\n⚠️ Some tools unavailable:\n" + "\n".join(f"- {w}" for w in backend_warnings)

    summary_msg = AIMessage(
        content=(
            f"🔍 **Research Complete** — {len(collected_results)} results "
            f"from {tools_used} tool(s).{warning_note}\n\n{summary_lines}"
        ),
        name="researcher",
    )

    return {
        **state,
        "search_results": collected_results,
        "current_agent": "researcher",
        "iteration": state.get("iteration", 0) + 1,
        "messages": [summary_msg],
    }