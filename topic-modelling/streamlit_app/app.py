"""
NLP Analysis Suite — Main Entry Point
Dark-themed Streamlit app with session-state routing.
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="NLP Analysis Suite",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Import pages ──────────────────────────────────────────────────────────────
from pages_app.home import show_home
from pages_app.results import show_results

# ── Session state initialisation ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ── Global CSS ────────────────────────────────────────────────────────────────
def inject_global_css():
    st.markdown(
        """
        <style>
        /* ── Google Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        /* ── Root theme tokens ── */
        :root {
            --bg-primary:    #0d1117;
            --bg-secondary:  #161b22;
            --bg-card:       #1c2330;
            --accent:        #7c6af7;
            --accent-glow:   rgba(124,106,247,.25);
            --accent-light:  #a594ff;
            --success:       #22c55e;
            --warning:       #f59e0b;
            --danger:        #ef4444;
            --text-primary:  #e6edf3;
            --text-secondary:#8b949e;
            --border:        #30363d;
            --radius:        12px;
            --radius-lg:     18px;
            --shadow:        0 4px 24px rgba(0,0,0,.45);
        }

        /* ── Global resets ── */
        .stApp, [data-testid="stAppViewContainer"], .main, html, body {
            font-family: 'Inter', sans-serif !important;
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }

        /* Hide default Streamlit chrome */
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 1.5rem !important; }

        /* ── Typography ── */
        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            color: var(--text-primary) !important;
        }

        /* ── Card component ── */
        .nlp-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.75rem 2rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.25rem;
        }

        /* ── Primary button ── */
        .stButton > button {
            background: linear-gradient(135deg, #7c6af7 0%, #a594ff 100%) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: .65rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            letter-spacing: .5px !important;
            transition: opacity .2s, transform .15s !important;
            box-shadow: 0 0 16px var(--accent-glow) !important;
        }
        .stButton > button:hover {
            opacity: .9 !important;
            transform: translateY(-1px) !important;
        }
        .stButton > button:disabled {
            background: #2a2f3a !important;
            color: var(--text-secondary) !important;
            box-shadow: none !important;
        }

        /* ── Text area ── */
        .stTextArea textarea {
            background: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            font-size: .95rem !important;
            padding: .85rem 1rem !important;
        }
        .stTextArea textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px var(--accent-glow) !important;
        }

        /* ── File uploader ── */
        [data-testid="stFileUploader"] {
            background: var(--bg-secondary) !important;
            border: 2px dashed var(--border) !important;
            border-radius: var(--radius) !important;
            padding: 1.2rem !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: var(--accent) !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--bg-secondary) !important;
            border-radius: var(--radius) !important;
            padding: .35rem !important;
            gap: .5rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: var(--text-secondary) !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            padding: .5rem 1.25rem !important;
            border: none !important;
            transition: all .2s !important;
        }
        .stTabs [aria-selected="true"] {
            background: var(--accent) !important;
            color: #fff !important;
        }

        /* ── Metric boxes ── */
        [data-testid="metric-container"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            padding: .9rem 1.1rem !important;
        }
        [data-testid="metric-container"] label {
            color: var(--text-secondary) !important;
            font-size: .8rem !important;
            text-transform: uppercase !important;
            letter-spacing: .8px !important;
        }
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: var(--accent-light) !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.5rem !important;
        }

        /* ── DataFrames ── */
        .stDataFrame { border-radius: var(--radius) !important; overflow: hidden; }

        /* ── Alerts / info boxes ── */
        .stAlert {
            border-radius: var(--radius) !important;
            border-left-width: 4px !important;
        }

        /* ── Divider ── */
        hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

        /* ── Keyword pill ── */
        .kw-pill {
            display: inline-block;
            background: var(--accent-glow);
            border: 1px solid var(--accent);
            color: var(--accent-light);
            border-radius: 99px;
            padding: .2rem .85rem;
            margin: .2rem;
            font-size: .88rem;
            font-weight: 500;
        }

        /* ── Back button (secondary style) ── */
        .back-btn > button {
            background: transparent !important;
            border: 1px solid var(--border) !important;
            color: var(--text-secondary) !important;
            box-shadow: none !important;
        }
        .back-btn > button:hover {
            border-color: var(--accent) !important;
            color: var(--accent-light) !important;
        }

        /* ── Summary card ── */
        .summary-box {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.1rem 1.3rem;
            font-size: .95rem;
            line-height: 1.7;
            color: var(--text-primary);
        }

        /* ── Theme toggle ── */
        .theme-toggle {
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 9999;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── App router ────────────────────────────────────────────────────────────────
def main():
    inject_global_css()



    if st.session_state.page == "home":
        show_home()
    elif st.session_state.page == "results":
        show_results()


if __name__ == "__main__":
    main()
