"""agents/writer.py — Writer Agent."""

from __future__ import annotations

import re
import urllib.request
import urllib.error

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
# from langchain_ollama import ChatOllama

from agents.state import ResearchState
# from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from langchain_groq import ChatGroq
from config.settings import GROQ_API_KEY, GROQ_MODEL

SYSTEM_PROMPT = """You are an Academic Research Writer. Write a formal research paper.

CRITICAL RULES:
- Base the report ONLY on the analysis and sources provided below.
- Do NOT copy source metadata (tool names, query strings, URLs) into the body text.
- Do NOT invent facts, statistics, or citations not present in the sources.
- Number references sequentially: [1], [2], [3] — each source gets a unique number.
- Do NOT include code examples unless the research query explicitly asks for code.

Use this EXACT structure:

# [Full Title of Research Paper]

**Authors:** AI Research Assistant  
**Date:** [Current Date]  
**Keywords:** [3-5 relevant keywords]

---

## Abstract
3-4 sentences covering: objective, methods, key findings, significance.

---

## 1. Introduction
Background and motivation. End with: *This paper investigates...*

---

## 2. Methodology
Describe what sources were searched and how findings were gathered.

---

## 3. Results

### 3.1 [First major finding]
Findings with citations like [1] or [2].

### 3.2 [Second major finding]
Additional findings.

---

## 4. Discussion
Interpret results. What do they mean? What are the implications?

---

## 5. Limitations
Constraints and unanswered questions.

---

## 6. Conclusion
Key takeaways and future research directions.

---

## References
[1] Author/Source. (Year). *Title*. Retrieved from: URL
[2] Author/Source. (Year). *Title*. Source: [tool] search.

(Each source gets a unique sequential number. Do not repeat numbers.)

---

## Appendix A: Research Data Sources

| # | Tool | Query | Key Finding |
|---|------|-------|-------------|
| 1 | ... | ... | ... |
"""


# ── URL validation ─────────────────────────────────────────────────────────────

# Domains that reject bots but are reliably live
_ALWAYS_LIVE_DOMAINS = {
    "en.wikipedia.org",
    "wikipedia.org",
    "arxiv.org",
    "scholar.google.com",
    "pubmed.ncbi.nlm.nih.gov",
    "doi.org",
}


def _domain(url: str) -> str:
    m = re.match(r'https?://([^/]+)', url)
    return m.group(1).lower() if m else ""


def _check_url(url: str, timeout: int = 5) -> bool:
    """Return True if the URL is reachable. Trusted domains are assumed live."""
    url = url.strip().rstrip(".,)")
    if not url.startswith("http"):
        return False
    if _domain(url) in _ALWAYS_LIVE_DOMAINS:
        return True
    try:
        req = urllib.request.Request(
            url, method="HEAD",
            headers={"User-Agent": "Mozilla/5.0 (research-bot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 400
    except urllib.error.HTTPError as e:
        if e.code == 405:
            try:
                req2 = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0 (research-bot/1.0)"},
                )
                with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                    return resp2.status < 400
            except Exception:
                return False
        return False
    except Exception:
        return False


def _extract_url_from_result(result_text: str) -> str:
    """Pull the first http URL out of a raw result string."""
    for line in result_text.splitlines():
        candidate = line.strip().replace("URL:", "").replace("url:", "").strip()
        if candidate.startswith("http"):
            return candidate
    return ""


def _validate_sources(results: list[dict]) -> list[dict]:
    """Attach _ref_num and _url (validated) to each source dict."""
    validated = []
    for i, r in enumerate(results, 1):
        raw_url = _extract_url_from_result(r["result"])
        live_url = raw_url if (raw_url and _check_url(raw_url)) else ""
        validated.append({**r, "_ref_num": i, "_url": live_url})
    return validated


# ── Source summary builder ─────────────────────────────────────────────────────

def _safe_query_str(args) -> str:
    if isinstance(args, dict):
        return args.get("query", str(args))
    return str(args)


def _build_sources_block(validated: list[dict]) -> str:
    """
    Clean, human-readable sources block for the writer prompt.
    Strips raw tool metadata lines so they cannot leak into the paper body.
    """
    if not validated:
        return "No sources were collected."

    lines = []
    for r in validated:
        content_lines = []
        for line in r["result"].splitlines():
            stripped = line.strip()
            if any(stripped.lower().startswith(x) for x in [
                "url:", "[web search", "[wikipedia", "[arxiv",
                "source:", "http", "query:", "authors:", "published:"
            ]):
                continue
            if stripped:
                content_lines.append(stripped)

        content_snippet = " ".join(content_lines)[:400]
        url_str = r["_url"] if r["_url"] else "not available"

        lines.append(
            f"[{r['_ref_num']}] Tool: {r['tool']} | "
            f"Query: {_safe_query_str(r['args'])} | "
            f"URL: {url_str}\n"
            f"    Content: {content_snippet}"
        )
    return "\n\n".join(lines)


# ── Post-processing ────────────────────────────────────────────────────────────

def _clean_report(report: str, validated: list[dict]) -> str:
    """Remove leaked metadata, [object Object], and duplicate sections."""

    # Remove [object Object] garbage
    report = re.sub(r',?\s*\[object Object\]\s*,?', '', report)
    report = re.sub(r'\n{3,}', '\n\n', report)

    # Remove leaked "Source: Source: [tool name]..." blocks
    report = re.sub(
        r'Source:\s*Source:\s*\[tool[^\n]*\n?(?:.*\n)*?(?=\n#|\Z)',
        '',
        report,
        flags=re.IGNORECASE,
    )

    # Strip individual lines that are raw source dumps
    clean_lines = []
    for line in report.splitlines():
        s = line.strip()
        if re.match(r'Source:\s*(Source:|Web|Wikipedia|\[)', s, re.IGNORECASE):
            continue
        if re.match(r'URL:\s*(N/A|http|\(link)', s, re.IGNORECASE):
            continue
        if re.match(r'Content:\s*Source:', s, re.IGNORECASE):
            continue
        clean_lines.append(line)
    report = "\n".join(clean_lines)

    # Keep only the FIRST occurrence of each ## section heading
    seen: set[str] = set()
    final_lines: list[str] = []
    skipping = False
    for line in report.splitlines():
        m = re.match(r'^(#{1,3})\s+(.*)', line)
        if m:
            key = re.sub(r'[^a-z0-9 ]', '', m.group(2).strip().lower())
            if key in seen:
                skipping = True
                continue
            seen.add(key)
            skipping = False
        if skipping:
            continue
        final_lines.append(line)
    report = "\n".join(final_lines)

    # Linkify bare live URLs only
    live_urls = {r["_url"] for r in validated if r.get("_url")}
    def linkify(m: re.Match) -> str:
        url = m.group(1)
        return f"[{url}]({url})" if url in live_urls else url
    report = re.sub(r'(?<!\()(?<!\[)(https?://[^\s\)\]\,]+)(?!\))', linkify, report)

    return report.strip()


def _ensure_references(report: str, validated: list[dict]) -> str:
    """Remove any existing References section and rebuild it cleanly."""
    report = re.sub(r'\n#{1,3} References.*', '', report, flags=re.DOTALL)
    report = re.sub(r'\nREFERENCES.*', '', report, flags=re.DOTALL)

    if not validated:
        return report

    ref_lines = []
    for r in validated:
        query_str = _safe_query_str(r['args'])
        url = r.get("_url", "")
        if url:
            ref_lines.append(
                f"[{r['_ref_num']}] {r['tool'].replace('_', ' ').title()}. "
                f"*{query_str}*. Retrieved from: [{url}]({url})"
            )
        else:
            ref_lines.append(
                f"[{r['_ref_num']}] {r['tool'].replace('_', ' ').title()}. "
                f"*{query_str}*. Source: {r['tool']} search."
            )

    return report.rstrip() + "\n\n## References\n\n" + "\n\n".join(ref_lines)


def _ensure_appendix(report: str, validated: list[dict]) -> str:
    """Remove any existing Appendix and rebuild it cleanly."""
    report = re.sub(r'\n#{1,3} Appendix.*', '', report, flags=re.DOTALL)
    report = re.sub(r'\nAPPENDIX.*', '', report, flags=re.DOTALL)

    if not validated:
        return report

    rows = []
    for r in validated:
        query_str = _safe_query_str(r['args'])[:45].replace("|", "/")
        snippet = " ".join(r["result"].split())[:80].replace("|", "/")
        url = r.get("_url", "")
        url_cell = f"[link]({url})" if url else "—"
        rows.append(f"| {r['_ref_num']} | {r['tool']} | {query_str} | {snippet}… | {url_cell} |")

    table = (
        "\n\n---\n\n## Appendix A: Research Data Sources\n\n"
        "| # | Tool | Query | Key Data | URL |\n"
        "|---|------|-------|----------|-----|\n"
        + "\n".join(rows)
    )
    return report.rstrip() + table


# ── Main node ──────────────────────────────────────────────────────────────────

def writer_node(state: ResearchState) -> ResearchState:
    """Produce the final research paper grounded in actual research data."""

    # llm = ChatOllama(
    #     base_url=OLLAMA_BASE_URL,
    #     model=OLLAMA_MODEL,
    #     temperature=0.3,
    #     num_predict=3000,
    # )
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=3000,
    )

    query = state["query"]
    analysis = state.get("analysis", "")
    results = state.get("search_results", [])

    validated = _validate_sources(results) if results else []
    sources_block = _build_sources_block(validated)

    if not analysis and not results:
        error_report = (
            f"# Research Report: {query}\n\n"
            "## Status\n"
            "The research pipeline was unable to collect data for this query.\n\n"
            "## Troubleshooting\n"
            "- Ensure `duckduckgo-search` is installed: `pip install duckduckgo-search`\n"
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
                f"=== ANALYSIS ===\n"
                f"{analysis or 'No analysis available.'}\n\n"
                f"=== NUMBERED SOURCES ===\n"
                f"{sources_block}\n\n"
                "Instructions:\n"
                "- Write the paper body using [1], [2] etc. for in-text citations.\n"
                "- Do NOT copy tool names, query strings, or raw URLs into the body text.\n"
                "- Use unique sequential numbers in the References section — no duplicates.\n"
                "- Do NOT add a code examples section.\n"
                "- Stop after Appendix A."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        report = response.content.strip()

        if len(report) < 100:
            report = (
                f"# Research Paper: {query}\n\n"
                f"## Abstract\n{analysis or 'See sources below.'}\n\n"
            )

        report = _clean_report(report, validated)
        report = _ensure_references(report, validated)
        report = _ensure_appendix(report, validated)

        return {
            **state,
            "final_report": report,
            "is_complete": True,
            "current_agent": "writer",
            "messages": [AIMessage(
                content=f"✍️ **Report Ready**\n\n{report[:300]}…",
                name="writer",
            )],
        }

    except Exception as e:
        fallback = (
            f"# Research Paper: {query}\n\n"
            f"## Writer Error\n`{e}`\n\n"
            f"## Analysis\n{analysis}\n\n"
            f"## Sources\n{sources_block}"
        )
        return {
            **state,
            "final_report": fallback,
            "is_complete": True,
            "current_agent": "writer",
            "error": f"Writer error: {e}",
            "messages": [AIMessage(
                content="✍️ Writer encountered an error — using fallback report.",
                name="writer",
            )],
        }