"""app.py — Multi-Agent Research Assistant — Streamlit Frontend."""

from __future__ import annotations

import time

import streamlit as st

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;600;800&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg: #0a0c10; --bg-card: #111318; --bg-input: #161a22;
    --border: #1e2330; --border-glow: #2a3555;
    --accent: #4f8ef7; --accent2: #7c5cbf; --accent3: #2ec4b6;
    --text-primary: #e8ecf4; --text-muted: #6b7890;
    --success: #2ec4b6; --warning: #f4a261; --danger: #e76f51;
    font-family: 'Inter', sans-serif;
}
.stApp { background: var(--bg); color: var(--text-primary); }
.block-container { padding-top: 1.5rem !important; max-width: 1400px; }
[data-testid="stSidebar"] { background: var(--bg-card) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--bg-input) !important; border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius: 8px;
}
.app-header {
    font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 60%, var(--accent3) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem; letter-spacing: -0.02em;
}
.app-subtitle { color: var(--text-muted); font-size: 0.9rem; font-weight: 300; margin-bottom: 1.5rem; font-family: 'IBM Plex Mono', monospace; }
.stTextInput > div > div > input {
    background: var(--bg-input) !important; border: 1.5px solid var(--border) !important;
    border-radius: 12px !important; color: var(--text-primary) !important;
    font-size: 1rem !important; padding: 0.85rem 1rem !important; transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(79,142,247,0.12) !important; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif; font-weight: 600; font-size: 1rem;
    padding: 0.7rem 2rem !important; transition: all 0.2s ease; letter-spacing: 0.02em;
}
.stButton > button[kind="primary"]:hover { opacity: 0.88; transform: translateY(-1px); box-shadow: 0 6px 24px rgba(79,142,247,0.3); }
.agent-pipeline { display: flex; align-items: center; gap: 8px; margin: 1.2rem 0; padding: 1rem; background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; overflow-x: auto; }
.agent-pill { display: flex; align-items: center; gap: 6px; padding: 6px 14px; border-radius: 20px; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 600; white-space: nowrap; border: 1.5px solid transparent; transition: all 0.3s ease; }
.agent-pill.waiting  { background: #1a1e28; color: var(--text-muted); border-color: var(--border); }
.agent-pill.active   { background: rgba(79,142,247,0.15); color: var(--accent); border-color: var(--accent); animation: pulse 1.5s infinite; }
.agent-pill.complete { background: rgba(46,196,182,0.12); color: var(--success); border-color: var(--success); }
.agent-pill.error    { background: rgba(231,111,81,0.12); color: var(--danger); border-color: var(--danger); }
.arrow { color: var(--border-glow); font-size: 0.9rem; }
@keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(79,142,247,0.4); } 50% { box-shadow: 0 0 0 6px rgba(79,142,247,0); } }
.metric-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; text-align: center; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 600; color: var(--accent); line-height: 1; }
.metric-label { font-size: 0.72rem; color: var(--text-muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }
.tool-badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace; font-weight: 600; margin: 2px; }
.tool-web_search       { background: rgba(79,142,247,0.15);  color: #6ea8ff; border: 1px solid rgba(79,142,247,0.3); }
.tool-wikipedia_search { background: rgba(124,92,191,0.15);  color: #b088f5; border: 1px solid rgba(124,92,191,0.3); }
.tool-arxiv_search     { background: rgba(244,162,97,0.15);  color: #f4a261; border: 1px solid rgba(244,162,97,0.3); }
.tool-scrape_webpage   { background: rgba(46,196,182,0.15);  color: #2ec4b6; border: 1px solid rgba(46,196,182,0.3); }
.tool-calculator       { background: rgba(231,111,81,0.15);  color: #e76f51; border: 1px solid rgba(231,111,81,0.3); }
.dep-ok  { color: #2ec4b6; font-size: 0.78rem; font-family: 'IBM Plex Mono', monospace; }
.dep-err { color: #e76f51; font-size: 0.78rem; font-family: 'IBM Plex Mono', monospace; }
.dep-warn{ color: #f4a261; font-size: 0.78rem; font-family: 'IBM Plex Mono', monospace; }
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: var(--bg-card); border-radius: 8px 8px 0 0; color: var(--text-muted); font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; border: 1px solid var(--border); border-bottom: none; padding: 0.5rem 1rem; }
.stTabs [aria-selected="true"] { background: var(--bg-input) !important; color: var(--accent) !important; border-color: var(--border-glow) !important; }
.stExpander { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Dependency checker ────────────────────────────────────────────────────────
def check_dependencies() -> dict:
    """Check which optional packages are available and return a status dict."""
    deps = {}

    # Search backends
    try:
        from duckduckgo_search import DDGS  # noqa: F401
        deps["duckduckgo-search"] = ("ok", "Web search ✓")
    except ImportError:
        deps["duckduckgo-search"] = ("error", "pip install duckduckgo-search")

    try:
        import wikipedia  # noqa: F401
        deps["wikipedia"] = ("ok", "Wikipedia ✓")
    except ImportError:
        deps["wikipedia"] = ("error", "pip install wikipedia")

    try:
        import arxiv  # noqa: F401
        deps["arxiv"] = ("ok", "arXiv ✓")
    except ImportError:
        deps["arxiv"] = ("warn", "pip install arxiv  (optional)")

    try:
        from bs4 import BeautifulSoup  # noqa: F401
        deps["beautifulsoup4"] = ("ok", "Web scraping ✓")
    except ImportError:
        deps["beautifulsoup4"] = ("warn", "pip install beautifulsoup4  (optional)")

    return deps


# ── Session state init ────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "research_state": None,
        "is_running": False,
        "agent_statuses": {k: "waiting" for k in ["planner", "researcher", "analyst", "writer"]},
        "agent_messages": {},
        "history": [],
        "run_time": 0.0,
        "error_msg": None,
        "pending_query": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;'
        'color:#4f8ef7;margin-bottom:1rem;">⚙️ Configuration</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Ollama Model**")
    model_choice = st.selectbox(
        "Model",
        [
            # "llama3.2:3b", 
            "llama3.2:1b", 
            # "mistral", "qwen2.5", "gemma2", "phi3"
        ],
        label_visibility="collapsed",
    )

    st.markdown("**Ollama URL**")
    ollama_url = st.text_input(
        "Ollama URL", value="http://localhost:11434", label_visibility="collapsed",
    )

    st.markdown("**Tavily API Key** *(optional, better search)*")
    tavily_key = st.text_input(
        "Tavily Key", type="password", placeholder="tvly-...", label_visibility="collapsed",
    )

    st.divider()

    st.markdown("**Active Tools**")
    use_web    = st.checkbox("🌐 Web Search",    value=True)
    use_wiki   = st.checkbox("📖 Wikipedia",     value=True)
    use_arxiv  = st.checkbox("📄 arXiv Papers",  value=True)
    use_scrape = st.checkbox("🔗 Web Scraper",   value=True)
    use_calc   = st.checkbox("🧮 Calculator",    value=True)

    st.divider()

    # ── Dependency status panel ───────────────────────────────────────────────
    st.markdown("**Dependencies**")
    deps = check_dependencies()
    all_critical_ok = all(
        status == "ok"
        for pkg, (status, _) in deps.items()
        if pkg in ("duckduckgo-search", "wikipedia")
    )

    for pkg, (status, msg) in deps.items():
        css = {"ok": "dep-ok", "warn": "dep-warn", "error": "dep-err"}[status]
        icon = {"ok": "●", "warn": "◐", "error": "○"}[status]
        st.markdown(f'<div class="{css}">{icon} {msg}</div>', unsafe_allow_html=True)

    if not all_critical_ok:
        st.markdown(
            '<div style="margin-top:8px;padding:8px;background:#1a1215;border:1px solid #e76f5140;'
            'border-radius:8px;font-size:0.72rem;color:#e76f51;font-family:IBM Plex Mono,monospace;">'
            '⚠️ Missing required packages.<br>Run in your terminal:<br><br>'
            '<code>pip install duckduckgo-search wikipedia</code>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Ollama status ─────────────────────────────────────────────────────────
    st.markdown("**Ollama Status**")
    try:
        import requests as _req
        r = _req.get(f"{ollama_url}/api/tags", timeout=3)
        available_models = [m["name"] for m in r.json().get("models", [])]
        model_present = any(model_choice in m for m in available_models)
        st.markdown(
            f'<span style="color:#2ec4b6;font-size:0.8rem;">● Connected</span><br>'
            f'<span style="color:#6b7890;font-size:0.72rem;">{len(available_models)} model(s)</span>',
            unsafe_allow_html=True,
        )
        if not model_present:
            st.markdown(
                f'<div style="margin-top:6px;padding:6px 8px;background:#1a150a;border:1px solid #f4a26140;'
                f'border-radius:8px;font-size:0.72rem;color:#f4a261;font-family:IBM Plex Mono,monospace;">'
                f'⚠️ Model not found locally.<br>Run: <code>ollama pull {model_choice}</code></div>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            '<span style="color:#e76f51;font-size:0.8rem;">● Not reachable</span><br>'
            '<span style="color:#6b7890;font-size:0.72rem;">Run: <code>ollama serve</code></span>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("**Research History**")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
            q_short = item["query"][:38] + ("…" if len(item["query"]) > 38 else "")
            st.markdown(
                f'<div style="font-size:0.75rem;color:#6b7890;padding:4px 0;border-bottom:1px solid #1e2330;">'
                f'{i}. {q_short}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div style="font-size:0.75rem;color:#6b7890;">No history yet</div>', unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-header">🔬 Research Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">multi-agent · langgraph + ollama · powered by local LLMs</div>',
    unsafe_allow_html=True,
)

# ── Query input ───────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Research Query",
        placeholder="e.g. What are the latest advances in quantum computing?",
        label_visibility="collapsed",
        key="query_input",
    )
with col_btn:
    start_btn = st.button("Research →", type="primary", use_container_width=True)

# ── Example queries ───────────────────────────────────────────────────────────
st.markdown(
    '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:0.5rem;margin-bottom:1rem;">'
    '<span style="font-size:0.72rem;color:#6b7890;">Try:</span>',
    unsafe_allow_html=True,
)
examples = [
    "Mobile addiction in today's generation",
    "How does CRISPR gene editing work?",
    "Impact of AI on software engineering jobs",
    "Recent breakthroughs in Alzheimer's research",
]
ex_cols = st.columns(len(examples))
for col, ex in zip(ex_cols, examples):
    with col:
        if st.button(ex, key=f"ex_{ex[:12]}", use_container_width=True):
            st.session_state.pending_query = ex
            st.rerun()

if st.session_state.pending_query and not st.session_state.is_running:
    query = st.session_state.pending_query
    start_btn = True
    st.session_state.pending_query = ""

# ── Block run if critical dependencies missing ────────────────────────────────
if start_btn and not all_critical_ok:
    st.error(
        "⛔ Cannot run research — required packages are missing.\n\n"
        "Open a terminal in your project folder and run:\n"
        "```\npip install duckduckgo-search wikipedia\n```\n"
        "Then restart Streamlit."
    )
    start_btn = False  # prevent the run from starting


# ── Pipeline visualizer ───────────────────────────────────────────────────────
def render_pipeline(statuses: dict[str, str]):
    icons  = {"planner": "📋", "researcher": "🔍", "analyst": "🔬", "writer": "✍️"}
    labels = {"planner": "Planner", "researcher": "Researcher", "analyst": "Analyst", "writer": "Writer"}
    pills  = ""
    for i, agent in enumerate(statuses):
        pills += f'<div class="agent-pill {statuses[agent]}">{icons[agent]} {labels[agent]}</div>'
        if i < len(statuses) - 1:
            pills += '<span class="arrow">→</span>'
    st.markdown(f'<div class="agent-pipeline">{pills}</div>', unsafe_allow_html=True)


render_pipeline(st.session_state.agent_statuses)


# ── Research runner ───────────────────────────────────────────────────────────
def run_research_with_updates(research_query: str):
    import os, importlib
    os.environ["OLLAMA_BASE_URL"] = ollama_url
    os.environ["OLLAMA_MODEL"]    = model_choice
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    import config.settings as cfg_mod
    importlib.reload(cfg_mod)

    from tools import research_tools as rt
    tools_active = []
    if use_web:    tools_active.append(rt.web_search)
    if use_wiki:   tools_active.append(rt.wikipedia_search)
    if use_arxiv:  tools_active.append(rt.arxiv_search)
    if use_scrape: tools_active.append(rt.scrape_webpage)
    if use_calc:   tools_active.append(rt.calculator)

    import agents.researcher as res_mod
    res_mod.RESEARCHER_TOOLS = tools_active if tools_active else [rt.web_search]

    from graph import run_research
    t0 = time.time()

    def progress_cb(agent_name, messages):
        order = ["planner", "researcher", "analyst", "writer"]
        idx = order.index(agent_name) if agent_name in order else -1
        for a in order[:idx]:
            st.session_state.agent_statuses[a] = "complete"
        st.session_state.agent_statuses[agent_name] = "active"
        if messages:
            st.session_state.agent_messages[agent_name] = messages

    try:
        final_state = run_research(research_query, progress_callback=progress_cb)
        st.session_state.research_state = final_state
        st.session_state.run_time = time.time() - t0
        for agent in st.session_state.agent_statuses:
            st.session_state.agent_statuses[agent] = (
                "error" if final_state.get("error") else "complete"
            )
        st.session_state.history.append({
            "query": research_query,
            "report": final_state.get("final_report", ""),
            "time": st.session_state.run_time,
        })
    except Exception as e:
        st.session_state.error_msg = str(e)
        for agent in st.session_state.agent_statuses:
            if st.session_state.agent_statuses[agent] == "active":
                st.session_state.agent_statuses[agent] = "error"
    finally:
        st.session_state.is_running = False


# ── Kick off ──────────────────────────────────────────────────────────────────
if start_btn and query and not st.session_state.is_running:
    st.session_state["active_query"] = query
    st.session_state.is_running      = True
    st.session_state.research_state  = None
    st.session_state.error_msg       = None
    st.session_state.agent_messages  = {}
    st.session_state.agent_statuses  = {k: "waiting" for k in ["planner", "researcher", "analyst", "writer"]}
    st.session_state.agent_statuses["planner"] = "active"
    st.rerun()

if st.session_state.is_running:
    active_query = st.session_state.get("active_query", "")
    if active_query:
        with st.spinner(f'Researching: "{active_query}"…'):
            run_research_with_updates(active_query)
        st.rerun()
    else:
        st.session_state.is_running = False
        st.error("No query found. Please enter a research question and try again.")


# ── Error display ─────────────────────────────────────────────────────────────
if st.session_state.error_msg:
    st.error(f"❌ {st.session_state.error_msg}")


# ── Results ───────────────────────────────────────────────────────────────────
if state := st.session_state.research_state:
    results    = state.get("search_results") or []
    tools_used = set(r["tool"] for r in results) if results else set()
    report     = state.get("final_report") or ""

    m1, m2, m3, m4 = st.columns(4)
    for col, val, label in [
        (m1, len(results),           "Sources Found"),
        (m2, len(tools_used),        "Tools Used"),
        (m3, len(report.split()),    "Words in Report"),
        (m4, f"{st.session_state.run_time:.1f}s", "Total Time"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    tab_report, tab_analysis, tab_plan, tab_sources, tab_raw = st.tabs(
        ["📄 Final Report", "🔬 Analysis", "📋 Research Plan", "🗂 Sources", "⚙️ Raw State"]
    )

    with tab_report:
        if report:
            st.markdown(report)
            st.download_button(
                "⬇️ Download Report (.md)", data=report,
                file_name=f"research_report_{int(time.time())}.md", mime="text/markdown",
            )
        else:
            st.info("No report generated yet.")

    with tab_analysis:
        analysis = state.get("analysis")
        # st.markdown(analysis) if analysis else st.info("No analysis available.")
        if analysis:
            st.markdown(analysis)
        else:
            st.info("No analysis available.")

    with tab_plan:
        plan = state.get("research_plan")
        # st.markdown(plan) if plan else st.info("No plan available.")
        if plan:
            st.markdown(plan)
        else:
            st.info("No plan available.")

    with tab_sources:
        if results:
            badges = " ".join(
                f'<span class="tool-badge tool-{t}">{t.replace("_", " ")}</span>'
                for t in tools_used
            )
            st.markdown(badges, unsafe_allow_html=True)
            st.markdown("")
            for i, r in enumerate(results, 1):
                with st.expander(f"**{i}.** `{r['tool']}` — {str(r.get('args',''))[:80]}", expanded=False):
                    st.markdown(
                        f'<span class="tool-badge tool-{r["tool"]}">{r["tool"]}</span>',
                        unsafe_allow_html=True,
                    )
                    st.code(r.get("result", "")[:1500], language="text")
        else:
            st.info("No sources collected.")

    with tab_raw:
        st.markdown("**Full ResearchState**")
        st.json({k: v for k, v in state.items() if k != "messages"})
        if state.get("messages"):
            st.markdown(f"**Messages** ({len(state['messages'])} total)")
            for msg in state["messages"][-10:]:
                role    = getattr(msg, "type", "unknown")
                name    = getattr(msg, "name", "")
                content = getattr(msg, "content", str(msg))
                st.markdown(
                    f'<div style="background:#161a22;border:1px solid #1e2330;border-radius:8px;'
                    f'padding:0.8rem;margin:4px 0;font-size:0.78rem;font-family:IBM Plex Mono,monospace;">'
                    f'<span style="color:#4f8ef7">[{role}{" · " + name if name else ""}]</span>'
                    f'<br>{str(content)[:400]}</div>',
                    unsafe_allow_html=True,
                )

elif not st.session_state.is_running:
    st.markdown(
        '<div style="text-align:center;padding:4rem 2rem;color:#6b7890;">'
        '<div style="font-size:3.5rem;margin-bottom:1rem;">🔬</div>'
        '<div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:600;color:#2a3555;margin-bottom:0.5rem;">'
        'Enter a research query to begin</div>'
        '<div style="font-size:0.85rem;max-width:480px;margin:0 auto;line-height:1.7;">'
        'The assistant will plan your research, gather information from multiple sources, '
        'analyze the findings, and produce a comprehensive report — all automatically.</div></div>',
        unsafe_allow_html=True,
    )