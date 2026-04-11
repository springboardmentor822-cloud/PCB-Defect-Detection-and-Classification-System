"""Home page – text input / file upload."""

import io
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Optional: try to import PDF support (pypdf / pdfminer)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _read_uploaded_file(uploaded_file) -> str:
    """Return plain text from a txt, html, or pdf upload."""
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if name.endswith(".pdf"):
        if HAS_PYPDF:
            reader = PdfReader(io.BytesIO(raw_bytes))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        else:
            return raw_bytes.decode("utf-8", errors="replace")
    else:
        # txt / html – just decode
        return raw_bytes.decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────
def show_home():
    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 2.5rem 0 1.5rem;">
            <div style="font-size:3.5rem; margin-bottom:.4rem;">📚</div>
            <h1 style="font-size:2.6rem; font-weight:700; margin:0;">
                NLP Analysis Suite
            </h1>
            <p style="color:var(--text-secondary); font-size:1.1rem; margin-top:.6rem;">
                Topic Modelling &nbsp;·&nbsp; Sentiment Analysis &nbsp;·&nbsp; Document Summarization
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Input card ───────────────────────────────────────────────────────────
    st.markdown('<div class="nlp-card">', unsafe_allow_html=True)

    st.markdown("#### ✏️ Paste your text")
    text_input = st.text_area(
        label="text_area",
        label_visibility="collapsed",
        placeholder="Paste any article, document, or snippet here…",
        height=280,
        key="ta_input",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### 📂 …or upload a file")
    st.markdown(
        '<p style="color:var(--text-secondary);font-size:.85rem;margin-top:-.5rem;">'
        "Supports <strong>.txt</strong> · <strong>.html</strong> · <strong>.pdf</strong>"
        "</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        label="file_uploader",
        label_visibility="collapsed",
        type=["txt", "html", "htm", "pdf"],
        key="file_upload",
    )
    if uploaded is not None:
        st.success(f"✅ File loaded: **{uploaded.name}**")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Resolve final text ────────────────────────────────────────────────────
    final_text = ""
    source_note = ""

    if uploaded is not None:
        final_text = _read_uploaded_file(uploaded)
        source_note = f"📄 Source: **{uploaded.name}**"
    elif text_input.strip():
        final_text = text_input.strip()
        source_note = "📝 Source: **pasted text**"

    # ── Preview if something is ready ────────────────────────────────────────
    if final_text:
        preview = final_text[:200].replace("\n", " ")
        dots = "…" if len(final_text) > 200 else ""
        word_count = len(final_text.split())
        st.markdown(
            f"""
            <div style="background:var(--bg-secondary);border:1px solid var(--border);
                        border-radius:10px;padding:.9rem 1.1rem;margin-top:.5rem;
                        font-size:.88rem;color:var(--text-secondary);">
                {source_note} &nbsp;·&nbsp;
                ~<strong style="color:var(--accent-light)">{word_count:,}</strong> words<br>
                <span style="font-style:italic;">{preview}{dots}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Analyse button ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    btn_cols = st.columns([1, 2, 1])
    with btn_cols[1]:
        disabled = not bool(final_text.strip())
        if st.button(
            "🔍 Analyse Document →",
            disabled=disabled,
            use_container_width=True,
            key="analyse_btn",
        ):
            st.session_state.input_text = final_text
            st.session_state.analysis_done = False
            st.session_state.page = "results"
            st.rerun()

    if disabled:
        st.markdown(
            '<p style="text-align:center;color:var(--text-secondary);font-size:.85rem;">'
            "Paste text or upload a file to enable analysis."
            "</p>",
            unsafe_allow_html=True,
        )

    # ── Feature cards ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    for col, icon, title, desc in [
        (c1, "🗂️", "Topic Modelling",
         "Pre-trained NMF model reveals the dominant themes in your document."),
        (c2, "💬", "Sentiment Analysis",
         "VADER scores every sentence and shows emotional flow across the text."),
        (c3, "📝", "Summarization",
         "Hybrid extractive + BART abstractive pipeline condenses key ideas."),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="nlp-card" style="text-align:center;">
                    <div style="font-size:2rem;">{icon}</div>
                    <h3 style="margin:.4rem 0 .3rem;font-size:1.05rem;">{title}</h3>
                    <p style="color:var(--text-secondary);font-size:.87rem;margin:0;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
