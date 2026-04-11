"""Results page – three analysis tabs."""

import time
import streamlit as st

from analysis.topic_modelling import run_topic_modelling
from analysis.sentiment import run_sentiment_analysis
from analysis.summarization import run_summarization


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────
def show_results():
    text = st.session_state.get("input_text", "")

    if not text:
        st.warning("No text found. Please go back and enter some text first.")
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        return

    # ── Top bar ───────────────────────────────────────────────────────────────
    top_left, top_right = st.columns([1, 5])
    with top_left:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back", key="back_btn_top"):
            st.session_state.page = "home"
            st.session_state.analysis_done = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with top_right:
        st.markdown(
            '<h2 style="margin:0;padding-top:.2rem;">NLP Analysis Results</h2>',
            unsafe_allow_html=True,
        )

    # ── Document preview strip ────────────────────────────────────────────────
    preview = text[:150].replace("\n", " ")
    dots = "…" if len(text) > 150 else ""
    word_count = len(text.split())
    import re
    sent_count = len(re.split(r"[.!?]+", text))

    st.markdown(
        f"""
        <div style="background:var(--bg-secondary);border:1px solid var(--border);
                    border-radius:10px;padding:.75rem 1.1rem;margin:.6rem 0 1rem;
                    font-size:.88rem;color:var(--text-secondary);">
            📄 <em>{preview}{dots}</em><br>
            <span style="color:var(--text-primary);">
                ~<strong style="color:var(--accent-light)">{word_count:,}</strong> words
                &nbsp;·&nbsp;
                ~<strong style="color:var(--accent-light)">{sent_count}</strong> sentences
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Run analyses (cached in session state) ────────────────────────────────
    if not st.session_state.get("analysis_done"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        t0 = time.time()
        
        status_text.text("🔬 Step 1/3: Running Topic Modelling...")
        st.session_state.topic_results = run_topic_modelling(text)
        progress_bar.progress(33)
        
        status_text.text("💬 Step 2/3: Running Sentiment Analysis...")
        st.session_state.sentiment_results = run_sentiment_analysis(text)
        progress_bar.progress(66)
        
        status_text.text("📝 Step 3/3: Running Summarization (this may take a moment)...")
        st.session_state.summary_results = run_summarization(text)
        progress_bar.progress(100)
        
        st.session_state.analysis_time = round(time.time() - t0, 1)
        st.session_state.analysis_done = True
        
        progress_bar.empty()
        status_text.empty()

    elapsed = st.session_state.get("analysis_time", "?")
    st.markdown(
        f'<p style="text-align:right;color:var(--text-secondary);'
        f'font-size:.8rem;margin-bottom:.5rem;">⚡ Analysed in {elapsed}s</p>',
        unsafe_allow_html=True,
    )

    # ── Three tabs ────────────────────────────────────────────────────────────
    tab_topics, tab_sentiment, tab_summary = st.tabs(
        ["📊 Topic Modelling", "💬 Sentiment Analysis", "📝 Summary"]
    )

    # ── Tab 1: Topic Modelling ────────────────────────────────────────────────
    with tab_topics:
        from tabs.topic_tab import render_topic_tab
        render_topic_tab(st.session_state.topic_results)

    # ── Tab 2: Sentiment ──────────────────────────────────────────────────────
    with tab_sentiment:
        from tabs.sentiment_tab import render_sentiment_tab
        render_sentiment_tab(st.session_state.sentiment_results)

    # ── Tab 3: Summary ────────────────────────────────────────────────────────
    with tab_summary:
        from tabs.summary_tab import render_summary_tab
        render_summary_tab(st.session_state.summary_results)
