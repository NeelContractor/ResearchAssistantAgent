"""tools/research_tools.py

All tool definitions for the research agents.
Tools are wrapped as LangChain @tool functions so LangGraph agents
can call them automatically through the tool-calling loop.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
import time
from typing import Optional

import requests
from langchain_core.tools import tool


# Optional imports

try:
    from tavily import TavilyClient
    _TAVILY_AVAILABLE = True
except ImportError:
    _TAVILY_AVAILABLE = False


try:
    from duckduckgo_search import DDGS
    _DDG_AVAILABLE = True
except ImportError:
    _DDG_AVAILABLE = False

try:
    import wikipedia as _wiki_lib
    _WIKI_AVAILABLE = True
except ImportError:
    _WIKI_AVAILABLE = False

try:
    import arxiv as _arxiv_lib
    _ARXIV_AVAILABLE = True
except ImportError:
    _ARXIV_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

from config.settings import TAVILY_API_KEY, MAX_SEARCH_RESULTS

# 1. WEB SEARCH (Tavily -> DuckDuckGo fallback)
@tool
def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """Search the web for current information.

    Tries Tavily first (higher quality), falls back to DuckDuckGo.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        Formatted string with titles, URLs, and snippets.
    """
    # Tavily
    if _TAVILY_AVAILABLE and TAVILY_API_KEY:
        try:
            client = TavilyClient(api_key=TAVILY_API_KEY)
            response = client.search(query, max_results=max_results)
            results = response.get("results", [])
            if results:
                lines = [f"[Web Search Results for: {query}]\n"]
                for i, r in enumerate(results, 1):
                    lines.append(
                        f"{i}. **{r.get('title', 'No title')}**\n"
                        f"   URL: {r.get('url', '')}\n"
                        f"   {r.get('content', '')[:300]}\n"
                    )
                return "\n".join(lines)
        except Exception as e:
            pass # fall through to DDG

    # DuckDuckGo
    if _DDG_AVAILABLE:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if results:
                lines = [f"[Web Search Results for: {query}]\n"]
                for i, r in enumerate(results, 1):
                    lines.append(
                        f"{i}. **{r.get('title', 'No title')}**\n"
                        f"   URL: {r.get('href', '')}\n"
                        f"   {r.get('body', '')[:300]}\n"
                    )
                return "\n".join(lines)
        except Exception as e:
            return f"Search failed: {e}"

    return "No search backend available. Install tavily-python or duckduckgo-search."

# 2. WIKIPEDIA LOOKUP
@tool
def wikipedia_search(query: str, sentences: int = 5) -> str:
    """Search Wikipedia for encyclopedic information about a topic.

    Args:
        query: The topic to look up on Wikipedia.
        sentences: Number of summary sentences to return (default 5).

    Returns:
        Wikipedia summary text and URL.
    """
    if not _WIKI_AVAILABLE:
        return "wikipedia package not installed. Run: pip install wikipedia"

    try:
        _wiki_lib.set_lang("en")
        results = _wiki_lib.search(query, results=3)
        if not results:
            return f"No Wikipedia results found for: {query}"

        # Try each result until one succeeds
        for title in results:
            try:
                page = _wiki_lib.page(title, auto_suggest=False)
                summary = _wiki_lib.summary(title, sentences=sentences, auto_suggest=False)
                return (
                    f"[Wikipedia: {page.title}]\n"
                    f"URL: {page.url}\n\n"
                    f"{summary}"
                )
            except _wiki_lib.exceptions.DisambiguationError as e:
                try:
                    page = _wiki_lib.page(e.options[0], auto_suggest=False)
                    summary = _wiki_lib.summary(e.options[0], sentences=sentences, auto_suggest=False)
                    return (
                        f"[Wikipedia: {page.title}]\n"
                        f"URL: {page.url}\n\n"
                        f"{summary}"
                    )
                except Exception:
                    continue
            except Exception:
                continue

        return f"Could not retrieve Wikipedia content for: {query}"
    except Exception as e:
        return f"Wikipedia search error: {e}"

# 3. ARXIV PAPER SEARCH
@tool
def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for academic papers and preprints.

    Best for cutting-edge research in CS, physics, math, biology, etc.

    Args:
        query: Research topic or keywords.
        max_results: Number of papers to retrieve (default 5).

    Returns:
        List of paper titles, authors, abstracts, and arXiv URLs.
    """
    if not _ARXIV_AVAILABLE:
        return "arxiv package not installed. Run: pip install arxiv"
    
    try:
        client = _arxiv_lib.Client()
        search = _arxiv_lib.Search(
            query=query,
            max_results=max_results,
            sort_by=_arxiv_lib.SortCriterion.Relevance,
        )
        papers = list(client.results(search))

        if not papers:
            return f"No arXiv papers found for: {query}"

        lines = [f"[arXiv Papers for: {query}]\n"]
        for i, paper in enumerate(papers, 1):
            authors = ", ".join(str(a) for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            abstract_snippet = paper.summary[:250].replace("\n", " ")
            lines.append(
                f"{i}. **{paper.title}**\n"
                f"   Authors: {authors}\n"
                f"   Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"   URL: {paper.entry_id}\n"
                f"   Abstract: {abstract_snippet}...\n"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"arXiv search error: {e}"

# 4. WEB PAGE SCRAPER
@tool
def scrape_webpage(url: str, max_chars: int = 3000) -> str:
    """Fetch and extract readable text content from a web page URL.

    Useful for reading full articles, blog posts, or documentation.

    Args:
        url: The full URL of the web page to scrape.
        max_chars: Maximum characters of content to return (default 3000).

    Returns:
        Extracted text content from the page.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")

        if "text/html" not in content_type and "text/plain" not in content_type:
            return f"Cannot scrape non-HTML content (type: {content_type})"

        if _BS4_AVAILABLE:
            soup = BeautifulSoup(response.text, "html.parser")
            # Remove noise tags
            for tag in soup(["script", "style", "nav", "footer",
                              "header", "aside", "form", "iframe"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        else:
            # Minimal regex fallback
            text = re.sub(r"<[^>]+>", "", response.text)
            text = re.sub(r"\s+", " ", text)

        # Collapse blank lines
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        clean = "\n".join(lines)
        truncated = clean[:max_chars]
        if len(clean) > max_chars:
            truncated += f"\n\n[... content truncated at {max_chars} chars ...]"

        return f"[Scraped: {url}]\n\n{truncated}"

    except requests.exceptions.Timeout:
        return f"Timeout fetching: {url}"
    except requests.exceptions.HTTPError as e:
        return f"HTTP error {e.response.status_code} for: {url}"
    except Exception as e:
        return f"Scraping error: {e}"

# 5. CALCULATOR
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports: +, -, *, /, **, sqrt, log, sin, cos, tan, pi, e, abs, round.

    Args:
        expression: A mathematical expression string, e.g. "sqrt(144) + 2**10".

    Returns:
        The result as a string, or an error message.
    """
    # Whitelist-safe evaluation
    safe_globals = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "pow": pow,
        "floor": math.floor,
        "ceil": math.ceil,
        "factorial": math.factorial,
    }
    try:
        # Strip dangerous patterns
        cleaned = re.sub(r"[^0-9+\-*/().,%\s_a-zA-Z]", "", expression)
        result = eval(cleaned, safe_globals)  # noqa: S307
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Calculator error: {e}"

# 6. TEXT SUMMARIZER (calls local ollama directly)
@tool
def summarize_text(text: str, style: str = "concise") -> str:
    """Summarize a long piece of text.

    Args:
        text: The text to summarize (up to ~4000 chars).
        style: Summary style — "concise", "bullet", or "detailed".

    Returns:
        A summarized version of the text.
    """
    from config.settings import GROQ_API_KEY, GROQ_MODEL
    from langchain_groq import ChatGroq

    style_instructions = {
        "concise": "Write a concise 2-3 sentence summary.",
        "bullet": "Summarize as 5-7 bullet points highlighting key facts.",
        "detailed": "Write a thorough summary preserving all important details.",
    }
    instruction = style_instructions.get(style, style_instructions["concise"])

    try:
        llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
        response = llm.invoke(f"{instruction}\n\nText to summarize:\n{text[:4000]}")
        return response.content
    except Exception as e:
        return f"Summarization error: {e}"

# Tool registry
ALL_TOOLS = [
    web_search,
    wikipedia_search,
    arxiv_search,
    scrape_webpage,
    calculator, 
    summarize_text,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}