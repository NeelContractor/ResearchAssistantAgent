"""agents/writer.py — Writer Agent."""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from agents.state import ResearchState
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL

SYSTEM_PROMPT = """You are a Research Writer. Write a clear, well-structured research report.

IMPORTANT: Base your report ONLY on the analysis and sources provided below. 
Do NOT use your general knowledge to fill gaps — if something wasn't in the sources, note it as unknown.

Report format:
# [Title]

## Executive Summary
2-3 sentences covering the main findings from the research.

## Background
Brief context about the topic (from sources only).

## Key Findings
The specific facts, data, and insights found in the research. Use bullet points. Be concrete.

## Analysis
What the findings mean and how they connect.

## Limitations
What the research didn't cover or couldn't confirm.

## Conclusion
Brief summary and takeaway.

## Sources Used
List the tools and search queries that produced this report.

Write clearly for a general educated audience. Use markdown."""


def writer_node(state: ResearchState) -> ResearchState:
    """Produce the final research report grounded in actual research data."""

    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.3,
        num_predict=2048,
    )

    query = state["query"]
    analysis = state.get("analysis", "")
    results = state.get("search_results", [])

    # Build a sources index so the writer knows exactly what was found
    sources_index = ""
    if results:
        lines = []
        for i, r in enumerate(results, 1):
            # Give the writer a short excerpt from each source
            snippet = r["result"][:300].replace("\n", " ")
            lines.append(f"{i}. [{r['tool']}] {r['args']} → {snippet}...")
        sources_index = "\n".join(lines)
    else:
        sources_index = "No sources were collected."

    # If there's no analysis AND no sources, produce a clear error report
    if not analysis and not results:
        error_report = (
            f"# Research Report: {query}\n\n"
            "## Status\n"
            "The research pipeline was unable to collect data for this query. "
            "This is likely due to network issues, missing tool dependencies, "
            "or the Ollama model not supporting tool-calling.\n\n"
            "## Troubleshooting\n"
            "- Ensure `duckduckgo-search` or Tavily API key is configured\n"
            "- Try a larger model: `ollama pull llama3.2:3b`\n"
            "- Check Ollama is running: `ollama serve`\n"
        )
        return {
            **state,
            "final_report": error_report,
            "is_complete": True,
            "current_agent": "writer",
            "messages": [AIMessage(content=error_report, name="writer")],
        }

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Research query: {query}\n\n"
                f"=== ANALYSIS (from analyst agent) ===\n"
                f"{analysis if analysis else 'No analysis available.'}\n\n"
                f"=== SOURCES COLLECTED ===\n"
                f"{sources_index}\n\n"
                f"Write the research report based strictly on the above."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        report = response.content.strip()

        if len(report) < 100:
            report = (
                f"# Research Report: {query}\n\n"
                f"## Summary\n{analysis or 'See sources below.'}\n\n"
                f"## Sources\n{sources_index}"
            )

        # Append a proper sources section if the model forgot to include one
        if "## Sources" not in report and results:
            source_list = "\n".join(
                f"- {r['tool']}: `{r['args']}`" for r in results
            )
            report += f"\n\n## Sources Used\n{source_list}"

        return {
            **state,
            "final_report": report,
            "is_complete": True,
            "current_agent": "writer",
            "messages": [AIMessage(content=f"✍️ **Report Ready**\n\n{report[:300]}…", name="writer")],
        }
    except Exception as e:
        fallback = (
            f"# Research Report: {query}\n\n"
            f"## Writer Error\n`{e}`\n\n"
            f"## Analysis\n{analysis}\n\n"
            f"## Sources\n{sources_index}"
        )
        return {
            **state,
            "final_report": fallback,
            "is_complete": True,
            "current_agent": "writer",
            "error": f"Writer error: {e}",
            "messages": [AIMessage(content="✍️ Writer encountered an error — using fallback report.", name="writer")],
        }