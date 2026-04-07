"""agents/analyst.py — Analyst Agent."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
# from langchain_ollama import ChatOllama

from agents.state import ResearchState
# from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from langchain_groq import ChatGroq
from config.settings import GROQ_API_KEY, GROQ_MODEL

# Short, directive prompt that works well with smaller models
SYSTEM_PROMPT = """You are a Research Analyst. You will be given raw search results.
Your job is to extract and organize the key information.

Write your analysis with these FOUR sections and NO MORE:
## Key Findings
(The most important facts from the sources — be specific, cite numbers/dates where available)

## Main Themes
(Recurring ideas across multiple sources)

## Gaps & Uncertainties
(What the sources didn't cover or contradicted each other on)

## Notable Facts
(Specific data points, statistics, names, dates worth highlighting)

IMPORTANT RULES:
- Write each section ONCE. Do NOT repeat any section.
- Stop writing after ## Notable Facts is complete.
- Do not add conclusions, recommendations, or extra sections.
- Be factual and specific. Do not invent information not present in the sources."""


def _deduplicate_analysis(text: str) -> str:
    """
    Remove repeated sections that small models (llama3.2:1b) tend to emit.

    Strategy:
    1. Keep only the FIRST occurrence of each ## heading block.
    2. Remove duplicate paragraphs (exact or near-exact).
    3. Hard-stop after the last expected section.
    """
    if not text:
        return text

    # ── Step 1: keep only the first occurrence of each ## section ────────────
    seen_headings: set[str] = set()
    lines = text.split("\n")
    filtered: list[str] = []
    skip_until_next_heading = False
    current_heading: str | None = None

    for line in lines:
        heading_match = re.match(r'^(#{1,3})\s+(.*)', line)
        if heading_match:
            heading_text = heading_match.group(2).strip().lower()
            # Normalize slight variations (e.g. "Key Findings" vs "Key findings")
            canonical = re.sub(r'[^a-z0-9 ]', '', heading_text)
            if canonical in seen_headings:
                # Duplicate heading — skip this whole block
                skip_until_next_heading = True
                current_heading = canonical
                continue
            else:
                seen_headings.add(canonical)
                skip_until_next_heading = False
                current_heading = canonical
                filtered.append(line)
        else:
            if not skip_until_next_heading:
                filtered.append(line)

    deduped_by_heading = "\n".join(filtered)

    # ── Step 2: remove duplicate paragraphs ──────────────────────────────────
    paragraphs = re.split(r'\n{2,}', deduped_by_heading)
    seen_paragraphs: set[str] = set()
    unique_paragraphs: list[str] = []

    for para in paragraphs:
        # Normalize for comparison: lowercase, collapse whitespace
        norm = re.sub(r'\s+', ' ', para.strip().lower())
        if len(norm) < 20:
            # Keep short lines (headings, bullets, blanks) unconditionally
            unique_paragraphs.append(para)
            continue
        if norm not in seen_paragraphs:
            seen_paragraphs.add(norm)
            unique_paragraphs.append(para)
        # else: silently drop the duplicate paragraph

    result = "\n\n".join(unique_paragraphs)

    # ── Step 3: hard-stop after the last expected section ────────────────────
    # Expected order: Key Findings → Main Themes → Gaps & Uncertainties → Notable Facts
    # If the model kept going past Notable Facts, chop it there.
    stop_after_pattern = re.compile(
        r'(##\s*Notable Facts.*?)(\n##\s+(?!Notable Facts))',
        re.DOTALL | re.IGNORECASE,
    )
    match = stop_after_pattern.search(result)
    if match:
        # Find where the unwanted heading starts and truncate
        cut_pos = match.start(2)
        result = result[:cut_pos].rstrip()

    return result.strip()


def analyst_node(state: ResearchState) -> ResearchState:
    """Analyze and synthesize the research results."""

    # llm = ChatOllama(
    #     base_url=OLLAMA_BASE_URL,
    #     model=OLLAMA_MODEL,
    #     temperature=0.1,
    #     num_predict=1024,   # Reduced from 1536 — less room for the model to ramble
    # )
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=1024,
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
    source_blocks = []
    total_chars = 0
    char_budget = 6000  # Tightened slightly to keep the prompt smaller for 1b model

    for i, r in enumerate(results, 1):
        header = f"[Source {i}: {r['tool']} | query: {r['args']}]"
        content = r["result"]
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
                "Write a structured analysis with EXACTLY the four sections listed above. "
                "Write each section once and then STOP."
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        analysis = response.content.strip()

        if len(analysis) < 50:
            analysis = f"Analysis could not be generated. Raw sources collected: {len(results)}."
        else:
            # Post-process to remove repetition introduced by small models
            analysis = _deduplicate_analysis(analysis)

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