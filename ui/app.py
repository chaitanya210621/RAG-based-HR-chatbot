"""
╔══════════════════════════════════════════════════════════════════╗
║              PeopleDesk — ui/app.py (BEAUTIFUL UI + FIX)         ║
║   Company HR Intelligence Chatbot — Streamlit Frontend           ║
║   Run with:  streamlit run ui/app.py                             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import time
import streamlit as st

# ─── Path Setup ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ─── Backend Import ─────────────────────────────────────────────────────────────
try:
    from src.search import RAGSearch
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    IMPORT_ERROR = str(e)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PeopleDesk · HR Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — Warm, editorial, HR-grade design
# ══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,600;1,9..144,300;1,9..144,400&family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base:       #0f1117;
    --bg-surface:    #161b27;
    --bg-elevated:   #1c2335;
    --bg-input:      #1a2030;
    --border-subtle: #252d40;
    --border-mid:    #2e3a52;
    --border-strong: #3d4f6e;

    /* Brand: warm amber-gold + cool slate */
    --amber:         #f5c842;
    --amber-dim:     rgba(245,200,66,0.12);
    --amber-glow:    rgba(245,200,66,0.06);
    --teal:          #3ecfb2;
    --teal-dim:      rgba(62,207,178,0.12);
    --rose:          #f0736a;
    --rose-dim:      rgba(240,115,106,0.12);
    --slate-hi:      #7b9cce;

    --text-bright:   #eef1f8;
    --text-main:     #c8d0e0;
    --text-soft:     #8896b0;
    --text-ghost:    #4a5672;

    --user-bg:       #1d2b45;
    --user-border:   #2e4270;
    --bot-bg:        #1a1f2e;
    --bot-border:    #252d40;

    --font-display:  'Fraunces', Georgia, serif;
    --font-body:     'Plus Jakarta Sans', system-ui, sans-serif;
    --font-mono:     'JetBrains Mono', monospace;

    --r-xs: 4px;
    --r-sm: 8px;
    --r-md: 14px;
    --r-lg: 22px;
    --r-xl: 32px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    color: var(--text-main) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stAppViewContainer"] > .main {
    background-color: var(--bg-base) !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-mid); border-radius: 99px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.6rem 1.3rem 2rem; }

.sb-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 22px;
    margin-bottom: 6px;
    border-bottom: 1px solid var(--border-subtle);
}
.sb-brand-mark {
    width: 42px; height: 42px;
    background: linear-gradient(140deg, #f5c842 0%, #f0a030 100%);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
    box-shadow: 0 4px 18px rgba(245,200,66,0.25);
}
.sb-brand-text { line-height: 1.15; }
.sb-brand-text .name {
    font-family: var(--font-display);
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-bright);
    letter-spacing: -0.01em;
}
.sb-brand-text .sub {
    font-size: 0.6rem;
    color: var(--text-ghost);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 500;
    margin-top: 2px;
}

.sb-label {
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-ghost);
    margin: 22px 0 9px;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.63rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.pill-amber { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(245,200,66,0.25); }
.pill-teal  { background: var(--teal-dim);  color: var(--teal);  border: 1px solid rgba(62,207,178,0.25); }
.pill-slate { background: rgba(123,156,206,0.1); color: var(--slate-hi); border: 1px solid rgba(123,156,206,0.2); }

.info-panel {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-md);
    padding: 13px 15px;
    margin: 7px 0;
    font-size: 0.79rem;
    color: var(--text-soft);
    line-height: 1.65;
}
.info-panel strong { color: var(--text-main); font-weight: 500; }
.info-panel .row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 9px; }
.info-panel .row:last-child { margin-bottom: 0; }

.status-row {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.78rem;
    color: var(--text-soft);
    margin: 10px 0;
}
.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-green { background: var(--teal); box-shadow: 0 0 6px rgba(62,207,178,0.6); animation: pulse-teal 2.5s infinite; }
.dot-red   { background: var(--rose); }
.dot-amber { background: var(--amber); animation: pulse-amber 2.5s infinite; }
@keyframes pulse-teal  { 0%,100%{box-shadow:0 0 0 0 rgba(62,207,178,0.5)} 50%{box-shadow:0 0 0 5px rgba(62,207,178,0)} }
@keyframes pulse-amber { 0%,100%{box-shadow:0 0 0 0 rgba(245,200,66,0.5)} 50%{box-shadow:0 0 0 5px rgba(245,200,66,0)} }

.hairline {
    height: 1px;
    background: var(--border-subtle);
    margin: 14px 0;
}

/* ── Main layout ── */
.main-col {
    max-width: 820px;
    margin: 0 auto;
    padding: 0 1.2rem;
}

/* ── Hero header ── */
.hero {
    padding: 2.8rem 1rem 2rem;
    text-align: center;
    position: relative;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: 0; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-mid), transparent);
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 99px;
    background: var(--amber-dim);
    border: 1px solid rgba(245,200,66,0.2);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 16px;
}
.hero h1 {
    font-family: var(--font-display);
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    color: var(--text-bright);
    font-weight: 500;
    letter-spacing: -0.03em;
    line-height: 1.05;
    margin-bottom: 12px;
}
.hero h1 .accent {
    font-style: italic;
    color: var(--amber);
}
.hero h1 .accent2 {
    font-style: italic;
    color: var(--teal);
}
.hero-sub {
    font-size: 0.88rem;
    color: var(--text-ghost);
    letter-spacing: 0.02em;
    line-height: 1.6;
}

/* ── Empty state ── */
.empty-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 4rem 2rem;
    text-align: center;
    color: var(--text-ghost);
}
.empty-icon-ring {
    width: 72px; height: 72px;
    border-radius: 50%;
    background: var(--bg-elevated);
    border: 1px solid var(--border-mid);
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 40px rgba(245,200,66,0.06);
}
.empty-wrap h3 {
    font-family: var(--font-display);
    font-size: 1.4rem;
    font-weight: 500;
    color: var(--text-soft);
    margin-bottom: 6px;
    letter-spacing: -0.01em;
}
.empty-wrap p { font-size: 0.84rem; line-height: 1.7; max-width: 380px; }

.suggestion-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    margin-top: 1.8rem;
    max-width: 480px;
    width: 100%;
}
.suggestion-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-md);
    padding: 12px 15px;
    font-size: 0.78rem;
    color: var(--text-soft);
    cursor: pointer;
    text-align: left;
    transition: all .2s;
    line-height: 1.5;
}
.suggestion-card .sq-icon { font-size: 1.1rem; display: block; margin-bottom: 5px; }
.suggestion-card:hover {
    border-color: var(--amber);
    color: var(--amber);
    background: var(--amber-glow);
}

/* ── Messages ── */
.msg-row {
    display: flex;
    gap: 13px;
    margin-bottom: 1.4rem;
    animation: fade-up .28s ease-out both;
}
@keyframes fade-up {
    from { opacity:0; transform: translateY(8px); }
    to   { opacity:1; transform: translateY(0); }
}
.msg-row.user-row { flex-direction: row-reverse; }

.av {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    margin-top: 1px;
}
.av-user {
    background: linear-gradient(135deg, #f5c842, #e08030);
    box-shadow: 0 3px 14px rgba(245,200,66,0.22);
}
.av-bot {
    background: var(--bg-elevated);
    border: 1px solid var(--border-mid);
}

.bubble-col { max-width: 78%; }

.bubble {
    padding: 13px 17px;
    border-radius: var(--r-md);
    font-size: 0.875rem;
    line-height: 1.8;
    word-break: break-word;
    color: var(--text-main);
}
.bubble-user {
    background: var(--user-bg);
    border: 1px solid var(--user-border);
    border-radius: var(--r-md) var(--r-xs) var(--r-md) var(--r-md);
}
.bubble-bot {
    background: var(--bot-bg);
    border: 1px solid var(--bot-border);
    border-radius: var(--r-xs) var(--r-md) var(--r-md) var(--r-md);
}

.msg-meta {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-top: 5px;
    font-size: 0.66rem;
    color: var(--text-ghost);
    font-family: var(--font-mono);
}
.msg-meta.right { justify-content: flex-end; }
.tag-time {
    padding: 1px 7px;
    border-radius: 99px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
}
.tag-speed {
    padding: 1px 7px;
    border-radius: 99px;
    background: var(--teal-dim);
    border: 1px solid rgba(62,207,178,0.2);
    color: var(--teal);
}

/* ── Sources ── */
.sources-label {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-ghost);
    padding: 9px 0 6px;
    border-top: 1px solid var(--border-subtle);
    margin-top: 10px;
}
.count-badge {
    padding: 1px 8px;
    border-radius: 99px;
    background: var(--amber-dim);
    color: var(--amber);
    border: 1px solid rgba(245,200,66,0.25);
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}

.src-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--r-sm);
    padding: 13px 15px 13px 18px;
    margin: 6px 0;
    font-size: 0.79rem;
    color: var(--text-soft);
    line-height: 1.65;
    position: relative;
    overflow: hidden;
}
.src-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, var(--amber), var(--teal));
    border-radius: 2px 0 0 2px;
}
.src-card-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 7px;
}
.src-rank {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-ghost);
}
.src-text { color: var(--text-soft); font-size: 0.79rem; line-height: 1.65; }
.src-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.src-tag {
    padding: 2px 8px;
    border-radius: 99px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    font-size: 0.62rem;
    color: var(--text-ghost);
    font-family: var(--font-mono);
}

/* ── Input bar ── */
.input-wrap {
    position: sticky;
    bottom: 0;
    background: linear-gradient(0deg, var(--bg-base) 72%, transparent);
    padding: 1rem 0 1.8rem;
    margin-top: 0.8rem;
}
[data-testid="stTextInput"] input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--r-xl) !important;
    color: var(--text-bright) !important;
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
    padding: 15px 22px !important;
    transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px var(--amber-glow) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--text-ghost) !important; }

[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--amber), #e08030) !important;
    border: none !important;
    color: #0f1117 !important;
    font-weight: 700 !important;
    font-family: var(--font-body) !important;
    font-size: 0.82rem !important;
    border-radius: var(--r-md) !important;
    box-shadow: 0 4px 18px rgba(245,200,66,0.25) !important;
    letter-spacing: 0.02em !important;
}
[data-testid="baseButton-secondary"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-mid) !important;
    color: var(--text-soft) !important;
    font-family: var(--font-body) !important;
    font-size: 0.78rem !important;
    border-radius: var(--r-sm) !important;
    transition: all .2s !important;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: var(--amber) !important;
    color: var(--amber) !important;
    background: var(--amber-glow) !important;
}

/* ── Thinking dots ── */
.thinking-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: var(--bot-bg);
    border: 1px solid var(--bot-border);
    border-radius: var(--r-xs) var(--r-md) var(--r-md) var(--r-md);
    width: fit-content;
    font-size: 0.8rem;
    color: var(--text-ghost);
    font-family: var(--font-body);
}
.thinking-dots span {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--amber);
    animation: dot-pop .9s infinite;
}
.thinking-dots span:nth-child(2) { animation-delay: .16s; }
.thinking-dots span:nth-child(3) { animation-delay: .32s; }
@keyframes dot-pop {
    0%,80%,100% { transform: scale(.55); opacity:.35; }
    40%         { transform: scale(1.05); opacity:1; }
}

/* ── Alert banners ── */
.banner {
    padding: 11px 16px;
    border-radius: var(--r-sm);
    font-size: 0.8rem;
    margin-bottom: 14px;
    border-left: 3px solid;
    line-height: 1.6;
}
.banner-error   { background: var(--rose-dim);  border-color: var(--rose);  color: #f09898; }
.banner-success { background: var(--teal-dim);  border-color: var(--teal);  color: #7de8d2; }
.banner-info    { background: var(--amber-dim); border-color: var(--amber); color: var(--amber); }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--r-sm) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-soft) !important;
    font-size: 0.79rem !important;
}

/* ── Slider ── */
[data-testid="stSlider"] .stSlider > div > div > div { background: var(--amber) !important; }

.footer-note {
    text-align: center;
    font-size: 0.65rem;
    color: var(--text-ghost);
    margin-top: 8px;
    letter-spacing: 0.04em;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "messages":     [],
        "rag_instance": None,
        "top_k":        5,
        "input_key":    0,
        "last_query":   "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ══════════════════════════════════════════════════════════════════════════════
#  RAG INIT — WITH LOADING INDICATOR (THE FIX!)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=True)  # ✅ CHANGED: show_spinner=True
def load_rag():
    return RAGSearch()

if BACKEND_AVAILABLE and st.session_state.rag_instance is None:
    try:
        with st.spinner("⏳ Loading HR knowledge base..."):  # ✅ ADDED: custom message
            st.session_state.rag_instance = load_rag()
    except Exception as e:
        st.session_state.rag_instance = None
        st.session_state["load_error"] = str(e)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_timestamp() -> str:
    return time.strftime("%H:%M")

def score_color(score: float) -> str:
    if score >= 0.85: return "#3ecfb2"
    elif score >= 0.70: return "#f5c842"
    elif score >= 0.55: return "#f0a030"
    else: return "#f0736a"

def score_label(score: float) -> str:
    if score >= 0.85: return "Excellent"
    elif score >= 0.70: return "Good"
    elif score >= 0.55: return "Fair"
    else: return "Weak"

def run_rag_query(query: str, top_k: int) -> dict:
    rag: RAGSearch = st.session_state.rag_instance
    t0 = time.perf_counter()
    raw = rag.search_and_summarize(query, top_k)
    elapsed = round(time.perf_counter() - t0, 2)
    if isinstance(raw, str):
        return {"answer": raw, "sources": [], "elapsed": elapsed}
    elif isinstance(raw, dict):
        return {
            "answer":  raw.get("answer", raw.get("response", str(raw))),
            "sources": raw.get("sources", raw.get("chunks", raw.get("results", []))),
            "elapsed": elapsed,
        }
    return {"answer": str(raw), "sources": [], "elapsed": elapsed}

def render_source_cards(sources: list):
    if not sources:
        return
    st.markdown(
        f'<div class="sources-label">📎 Source Chunks <span class="count-badge">{len(sources)}</span></div>',
        unsafe_allow_html=True,
    )
    for i, src in enumerate(sources, 1):
        if isinstance(src, str):
            text, score, meta = src, None, {}
        elif isinstance(src, dict):
            text  = src.get("text", src.get("content", src.get("chunk", "")))
            score = src.get("score", src.get("similarity", src.get("distance", None)))
            meta  = src.get("metadata", {}) or {}
        else:
            text, score, meta = str(src), None, {}

        if score is not None and score > 1.5:
            score = max(0.0, 1.0 - score / 2.0)

        score_html = (
            f'<span style="color:{score_color(score)};font-family:var(--font-mono);font-size:.65rem;">'
            f'{score:.3f} · {score_label(score)}</span>'
            if score is not None else
            '<span style="color:var(--text-ghost);font-size:.65rem;">N/A</span>'
        )

        tags_html = ""
        page = meta.get("page", meta.get("page_number", meta.get("page_num")))
        source_file = meta.get("source", meta.get("file", meta.get("filename")))
        for label, val in [("📄", page), ("📁", source_file)]:
            if val is not None:
                tags_html += f'<span class="src-tag">{label} {val}</span>'
        for k, v in meta.items():
            if k not in ("page","page_number","page_num","source","file","filename") and v:
                tags_html += f'<span class="src-tag">{k}: {v}</span>'

        preview = text[:300] + ("…" if len(text) > 300 else "") if text else "No text available."

        st.markdown(f"""
        <div class="src-card">
            <div class="src-card-head">
                <span class="src-rank">Chunk #{i}</span>
                {score_html}
            </div>
            <div class="src-text">{preview}</div>
            {"<div class='src-tags'>" + tags_html + "</div>" if tags_html else ""}
        </div>
        """, unsafe_allow_html=True)

def render_message(msg: dict, index: int):
    role    = msg["role"]
    content = msg["content"]
    ts      = msg.get("time", "")
    elapsed = msg.get("elapsed")
    sources = msg.get("sources", [])

    if role == "user":
        st.markdown(f"""
        <div class="msg-row user-row">
            <div class="av av-user">🧑</div>
            <div class="bubble-col">
                <div class="bubble bubble-user">{content}</div>
                <div class="msg-meta right">
                    <span class="tag-time">{ts}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        speed_tag = f'<span class="tag-speed">⚡ {elapsed}s</span>' if elapsed else ""
        st.markdown(f"""
        <div class="msg-row">
            <div class="av av-bot">🏢</div>
            <div class="bubble-col">
                <div class="bubble bubble-bot">{content}</div>
                <div class="msg-meta">
                    <span class="tag-time">{ts}</span>
                    {speed_tag}
                    <span class="tag-time">{len(sources)} ref</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if sources:
            with st.expander(f"📎 {len(sources)} source chunk{'s' if len(sources) != 1 else ''}", expanded=False):
                render_source_cards(sources)
        col1, col2 = st.columns([7, 1])
        with col2:
            if st.button("📋", key=f"copy_{index}", help="Copy response"):
                st.session_state["copied"] = content
                st.toast("Copied to clipboard ✓", icon="✅")

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-mark">🏢</div>
        <div class="sb-brand-text">
            <div class="name">PeopleDesk</div>
            <div class="sub">HR Intelligence Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status
    if BACKEND_AVAILABLE and st.session_state.rag_instance is not None:
        st.markdown('<div class="status-row"><div class="dot dot-green"></div><span>Knowledge base connected</span></div>', unsafe_allow_html=True)
    elif not BACKEND_AVAILABLE:
        st.markdown('<div class="status-row"><div class="dot dot-red"></div><span style="color:var(--rose);">Backend unavailable</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-row"><div class="dot dot-amber"></div><span>Connecting…</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="hairline"></div>', unsafe_allow_html=True)

    # Search settings
    st.markdown('<p class="sb-label">⚙ Retrieval Settings</p>', unsafe_allow_html=True)
    top_k = st.slider(
        "Top-K Chunks",
        min_value=1, max_value=15,
        value=st.session_state.top_k,
        help="How many document chunks to retrieve per question.",
    )
    st.session_state.top_k = top_k
    st.markdown(f"""
    <div class="info-panel">
        Fetching <strong>{top_k} chunk{"s" if top_k != 1 else ""}</strong> per query.
        More chunks = richer answers but slightly slower.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hairline"></div>', unsafe_allow_html=True)

    # Model info
    st.markdown('<p class="sb-label">🤖 Stack</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-panel">
        <div class="row">
            <strong>LLM</strong>
            <span class="pill pill-amber">LLaMA 3 · Groq</span>
        </div>
        <div class="row">
            <strong>Embeddings</strong>
            <span class="pill pill-teal">MiniLM-L6-v2</span>
        </div>
        <div class="row">
            <strong>Vector Store</strong>
            <span class="pill pill-slate">FAISS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hairline"></div>', unsafe_allow_html=True)

    # Session
    st.markdown('<p class="sb-label">💬 Session</p>', unsafe_allow_html=True)
    chat_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.markdown(f"""
    <div class="info-panel">
        <strong>{chat_count}</strong> question{"s" if chat_count != 1 else ""} asked this session.
    </div>
    """, unsafe_allow_html=True)
    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div class="hairline"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-panel">
        Ask anything about <strong>policies, benefits, leave, payroll, onboarding</strong>, or any
        company HR topic — answers come straight from your official HR documents.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-col">', unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🏢 Company HR Assistant</div>
    <h1>Ask <span class="accent">People</span><span class="accent2">Desk</span></h1>
    <p class="hero-sub">Instant, accurate answers to your HR questions — sourced directly from company policy</p>
</div>
""", unsafe_allow_html=True)

# Error banners
if not BACKEND_AVAILABLE:
    st.markdown(f"""
    <div class="banner banner-error">
        ⚠️ <strong>Backend unavailable:</strong> {IMPORT_ERROR}<br>
        Ensure <code>src/search.py</code> is accessible and all dependencies are installed.
    </div>
    """, unsafe_allow_html=True)
elif st.session_state.get("load_error"):
    st.markdown(f"""
    <div class="banner banner-error">
        ⚠️ <strong>Failed to initialise:</strong> {st.session_state["load_error"]}
    </div>
    """, unsafe_allow_html=True)

# Chat or empty state
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-wrap">
        <div class="empty-icon-ring">💬</div>
        <h3>No questions yet</h3>
        <p>Type below or pick a suggestion to get started with your HR query.</p>
        <div class="suggestion-grid">
            <div class="suggestion-card">
                <span class="sq-icon">🗓️</span>
                What is the leave policy?
            </div>
            <div class="suggestion-card">
                <span class="sq-icon">⏰</span>
                What are working hours & days?
            </div>
            <div class="suggestion-card">
                <span class="sq-icon">💰</span>
                How does payroll work?
            </div>
            <div class="suggestion-card">
                <span class="sq-icon">🎉</span>
                What benefits am I eligible for?
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for i, msg in enumerate(st.session_state.messages):
        render_message(msg, i)

thinking_placeholder = st.empty()

st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,var(--border-subtle),transparent);margin:.6rem 0;"></div>', unsafe_allow_html=True)

# Input row
q_col, btn_col = st.columns([8, 1])
with q_col:
    query = st.text_input(
        label="question",
        label_visibility="collapsed",
        placeholder="e.g. How many casual leaves do I get per year?",
        key=f"query_input_{st.session_state.input_key}",
    )
with btn_col:
    send_clicked = st.button("Ask →", type="primary", use_container_width=True)

st.markdown(
    '<p class="footer-note">PeopleDesk · Powered by LLaMA 3 via Groq · FAISS + Sentence Transformers</p>',
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  QUERY HANDLER
# ══════════════════════════════════════════════════════════════════════════════
def handle_query(q: str):
    q = q.strip()
    if not q or q == st.session_state.last_query:
        return
    st.session_state.last_query = q

    st.session_state.messages.append({
        "role": "user", "content": q,
        "time": get_timestamp(), "sources": [],
    })

    thinking_placeholder.markdown("""
    <div class="msg-row" style="margin-top:.4rem;">
        <div class="av av-bot">🏢</div>
        <div class="thinking-wrap">
            <div class="thinking-dots"><span></span><span></span><span></span></div>
            Looking through HR documents…
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not BACKEND_AVAILABLE or st.session_state.rag_instance is None:
        thinking_placeholder.empty()
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ **HR knowledge base not available.** Please ensure the backend is running correctly.",
            "time": get_timestamp(), "sources": [], "elapsed": None,
        })
    else:
        try:
            result = run_rag_query(q, st.session_state.top_k)
            thinking_placeholder.empty()
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "time": get_timestamp(),
                "elapsed": result["elapsed"],
            })
        except Exception as exc:
            thinking_placeholder.empty()
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ **Error:** `{exc}`\n\nCheck your backend and Groq API key.",
                "time": get_timestamp(), "sources": [], "elapsed": None,
            })

    st.session_state.input_key += 1
    st.rerun()

if (send_clicked or query) and query:
    handle_query(query)