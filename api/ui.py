"""
api/ui.py
----------
Streamlit UI for the RAG system.
Shows query → sources → answer with latency breakdown.
Run with: streamlit run api/ui.py
"""

import streamlit as st
import requests
import json
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="arXiv RAG System",
    page_icon="🔬",
    layout="wide",
)

# ─────────────────────────────────────────
# Styles
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTextInput > div > div > input { background-color: #1e2130; color: white; }
    .metric-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        border-left: 3px solid #4CAF50;
    }
    .source-card {
        background: #1a1f2e;
        border-radius: 6px;
        padding: 12px;
        margin: 6px 0;
        border-left: 2px solid #2196F3;
        font-size: 0.85em;
    }
    .answer-box {
        background: #1e2130;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.title("🔬 arXiv RAG System")
st.markdown("**Retrieval-Augmented Generation** over research papers | Powered by Mistral-7B + FAISS")

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    stats = requests.get(f"{API_URL}/index-stats", timeout=2).json()
    st.success(f" API Online | **{stats['total_chunks']:,}** chunks indexed | **{stats['total_vectors']:,}** vectors in FAISS")
except:
    st.error(" API is offline. Run: `uvicorn api.main:app --reload`")
    st.stop()

st.divider()

# ─────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────
# Use a separate key to pre-fill the text input via `value=`
# Direct assignment to a widget key after render raises StreamlitAPIException
if "question_prefill" not in st.session_state:
    st.session_state["question_prefill"] = ""

# ─────────────────────────────────────────
# Sidebar: Example Queries
# ─────────────────────────────────────────
# Defined BEFORE the widget so callbacks fire before the widget renders
with st.sidebar:
    st.header("💡 Example Queries")
    examples = [
        "How does retrieval-augmented generation reduce hallucinations?",
        "What are the limitations of transformer-based language models?",
        "Explain chain-of-thought prompting",
        "What is RLHF and how is it used in LLM training?",
        "Compare RAG vs fine-tuning for domain adaptation",
    ]

    def set_example(example_text):
        st.session_state["question_prefill"] = example_text

    for ex in examples:
        st.button(ex, key=ex, on_click=set_example, args=(ex,))

    st.divider()
    st.header("📊 Quick Eval")
    if st.button("Run Mini Evaluation"):
        st.info("Run `python experiments/run_eval.py` in terminal for full evaluation.")

# ─────────────────────────────────────────
# Input
# ─────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_input(
        "Ask a question about the research papers:",
        placeholder="e.g. How does RAG reduce hallucinations in LLMs?",
        key="question_input",
        value=st.session_state["question_prefill"],   # ← safe way to pre-fill
    )

with col2:
    mode = st.selectbox("Mode", ["rag", "baseline", "compare"])

query_btn = st.button("🔍 Run Query", type="primary", use_container_width=True)

# ─────────────────────────────────────────
# Run Query
# ─────────────────────────────────────────
if query_btn and question.strip():
    with st.spinner("Retrieving and generating..."):
        t0 = time.time()

        if mode == "compare":
            resp = requests.post(f"{API_URL}/compare", json={"question": question})
        else:
            resp = requests.post(f"{API_URL}/query", json={"question": question, "mode": mode})

        elapsed = time.time() - t0

    if resp.status_code != 200:
        st.error(f"API Error: {resp.text}")
    else:
        data = resp.json()

        # ── Latency Metrics ──
        st.subheader("⚡ Latency Breakdown")
        if mode == "compare":
            latency = data["rag_latency"]
        else:
            latency = data.get("latency", {})

        lat_cols = st.columns(len(latency))
        for col, (key, val) in zip(lat_cols, latency.items()):
            col.metric(key.replace("_", " ").title(), f"{val} ms")

        st.divider()

        # ── Answer(s) ──
        if mode == "compare":
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🤖 RAG Answer")
                st.markdown(f'<div class="answer-box">{data["rag_answer"]}</div>', unsafe_allow_html=True)
            with c2:
                st.subheader("🧠 Baseline Answer (No Retrieval)")
                st.markdown(f'<div class="answer-box">{data["baseline_answer"]}</div>', unsafe_allow_html=True)
        else:
            st.subheader(f"{'🤖 RAG Answer' if mode == 'rag' else '🧠 Baseline Answer'}")
            st.markdown(f'<div class="answer-box">{data["answer"]}</div>', unsafe_allow_html=True)

        # ── Sources ──
        sources = data.get("sources", [])
        if sources:
            st.divider()
            st.subheader(f"📄 Retrieved Sources (top {len(sources)})")
            for src in sources:
                st.markdown(f"""
                <div class="source-card">
                    <strong>#{src['rank']} {src['paper_title']}</strong><br>
                    <em>Relevance score: {src['score']:.4f}</em><br>
                    {src.get('snippet', '')}
                </div>
                """, unsafe_allow_html=True)

        # ── Token Usage ──
        if "token_usage" in data:
            with st.expander("🔢 Token Usage"):
                st.json(data["token_usage"])

elif query_btn:
    st.warning("Please enter a question.")