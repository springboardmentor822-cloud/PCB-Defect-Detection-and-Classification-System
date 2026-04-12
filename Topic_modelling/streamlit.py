"""
NLP Analysis Suite — Main Entry Point


import streamlit as st

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Analysis Suite",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from pages_app.home import show_home
from pages_app.results import show_results

# ── Session state ───────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ── Improved CSS ───────────────────────────────────────────
def inject_global_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --bg-primary:    #0f172a;
            --bg-secondary:  #1e293b;
            --bg-card:       #111827;
            --accent:        #6366f1;
            --accent-light:  #818cf8;
            --text-primary:  #e5e7eb;
            --text-secondary:#9ca3af;
            --border:        #374151;
            --radius:        14px;
            --shadow:        0 8px 30px rgba(0,0,0,0.5);
        }

        /* ── Global ── */
        html, body, .stApp {
            background: linear-gradient(135deg, #0f172a, #020617);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
        }

        #MainMenu, footer, header {visibility: hidden;}
        .block-container {
            padding-top: 2rem;
            padding-left: 4rem;
            padding-right: 4rem;
        }

        /* ── Headings ── */
        h1 {
            text-align: center;
            margin-bottom: 1rem;
        }

        h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
        }

        /* ── Card ── */
        .nlp-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.8rem;
            margin: 1.2rem auto;
            max-width: 900px;
            box-shadow: var(--shadow);
        }

        /* ── Button ── */
        .stButton > button {
            display: block;
            margin: 15px auto;
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            border-radius: 10px;
            padding: 10px 30px;
            font-weight: 600;
            color: white;
            border: none;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99,102,241,0.4);
        }

        /* ── Textarea ── */
        textarea {
            background: var(--bg-secondary) !important;
            color: white !important;
            border-radius: 10px !important;
            border: 1px solid var(--border) !important;
            padding: 12px !important;
        }

        /* ── File uploader ── */
        [data-testid="stFileUploader"] {
            border: 2px dashed var(--border);
            border-radius: 12px;
            padding: 1rem;
            background: var(--bg-secondary);
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
            background: transparent;
            gap: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            background: var(--bg-secondary);
            padding: 8px 20px;
            border-radius: 8px;
            color: var(--text-secondary);
        }

        .stTabs [aria-selected="true"] {
            background: var(--accent);
            color: white;
        }

        /* ── Metrics ── */
        [data-testid="metric-container"] {
            text-align: center;
            border-radius: 10px;
            padding: 10px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
        }

        /* ── Summary Box ── */
        .summary-box {
            background: var(--bg-secondary);
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid var(--border);
            line-height: 1.6;
        }

        /* ── Keyword pills ── */
        .kw-pill {
            display: inline-block;
            padding: 5px 12px;
            margin: 4px;
            border-radius: 20px;
            background: rgba(99,102,241,0.2);
            border: 1px solid #6366f1;
            font-size: 13px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Router ───────────────────────────────────────────────
def main():
    inject_global_css()

    if st.session_state.page == "home":
        show_home()
    elif st.session_state.page == "results":
        show_results()


if __name__ == "__main__":
    main()
