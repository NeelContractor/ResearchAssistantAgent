# 🔬 Multi-Agent Research Assistant

A fully local, privacy-first research assistant powered by **LangGraph** (multi-agent orchestration), **Ollama** (local LLMs), and **Streamlit** (frontend UI).

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────────┐
│   PLANNER   │────▶│  RESEARCHER  │────▶│   ANALYST   │────▶│   WRITER   │
│             │     │              │     │             │     │            │
│ Breaks query│     │ Calls tools  │     │ Synthesizes │     │ Produces   │
│ into a plan │     │ to gather    │     │ findings &  │     │ polished   │
│             │     │ information  │     │ insights    │     │ report     │
└─────────────┘     └──────────────┘     └─────────────┘     └────────────┘
                           │
                    ┌──────┴───────┐
                    │    TOOLS     │
                    ├──────────────┤
                    │ web_search   │  DuckDuckGo / Tavily
                    │ wikipedia    │  Wikipedia API
                    │ arxiv        │  arXiv papers
                    │ scrape_page  │  BeautifulSoup
                    │ calculator   │  Safe math eval
                    │ summarize    │  Ollama summarizer
                    └──────────────┘
```

---

## Quick Start

### 1. Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model 
ollama pull llama3.2:1b
```

### 2. Install Python dependencies

```bash
cd research_assistant_agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure (optional)

```bash
cp .env.example .env
# Edit .env to set your model, Tavily API key, etc.
```

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Tools

| Tool | Description | Requires |
|------|-------------|----------|
| `web_search` | Searches the web (Tavily → DuckDuckGo fallback) | `duckduckgo-search` |
| `wikipedia_search` | Looks up Wikipedia articles | `wikipedia` |
| `arxiv_search` | Searches arXiv for academic papers | `arxiv` |
| `scrape_webpage` | Fetches and extracts text from URLs | `beautifulsoup4` |
| `calculator` | Evaluates math expressions safely | built-in |
| `summarize_text` | Summarizes text via Ollama | Ollama |

For higher-quality web search, get a free API key at [tavily.com](https://tavily.com) and add it in the sidebar.

---

## Project Structure

```
research_assistant/
├── app.py                  # Streamlit frontend
├── graph.py                # LangGraph pipeline definition
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py         # Central config (reads .env)
├── agents/
│   ├── state.py            # Shared ResearchState TypedDict
│   ├── planner.py          # Planner agent
│   ├── researcher.py       # Researcher agent (tool-calling loop)
│   ├── analyst.py          # Analyst agent
│   └── writer.py           # Writer agent
└── tools/
    └── research_tools.py   # All tool definitions
```

---


