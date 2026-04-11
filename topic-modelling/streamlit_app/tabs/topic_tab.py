"""Topic Modelling tab renderer."""

import io
import json
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

# ── Shared plot style ─────────────────────────────────────────────────────────
_BG      = "#1c2330"
_BG2     = "#0d1117"
_ACCENT  = "#7c6af7"
_RED     = "#ef4444"
_TEXT    = "#e6edf3"
_BORDER  = "#30363d"

def _apply_dark_style(fig, ax_list=None):
    fig.patch.set_facecolor(_BG2)
    axes = ax_list or fig.axes
    for ax in axes:
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_TEXT, labelsize=8)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        ax.title.set_color(_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(_BORDER)


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    return buf.read()


# ── Render ────────────────────────────────────────────────────────────────────
def render_topic_tab(results: dict):
    dom = results["dominant"]
    top_words = results["top_words"]
    active = results["active"]
    topic_pct = results["topic_pct"]
    topic_dist = results["topic_dist"]
    n_topics = results["n_topics"]
    tww = results["topic_word_weights"]

    # ── Stats ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw tokens", f"{results['raw_tokens']:,}")
    m2.metric("Clean tokens", f"{results['clean_tokens']:,}")
    m3.metric("Active topics", sum(1 for s in topic_dist if s > 1e-5))
    m4.metric("Dominant topic", f"Topic {dom}")

    # ── Dominant topic banner ─────────────────────────────────────────────────
    dom_words = " · ".join(top_words[dom][:6])
    st.markdown(
        f"""
        <div class="nlp-card" style="border-left:4px solid var(--accent); padding: 2rem 2.5rem;">
            <span style="color:var(--text-secondary);font-size:1rem;text-transform:uppercase;letter-spacing:1px;font-weight:600;">
                🏆 Dominant Topic
            </span><br>
            <div style="font-size:2rem;font-weight:700;color:var(--accent-light);margin:0.8rem 0;line-height:1.3;font-family:'Space Grotesk', sans-serif;">
                {dom_words}
            </div>
            <div style="display:inline-block;color:var(--text-primary);font-size:.95rem;background:var(--bg-secondary);padding:0.4rem 0.8rem;border-radius:8px;border:1px solid var(--border);font-weight:500;">
                <span style="color:var(--text-secondary);">Topic identifier:</span> Topic {dom} 
                <span style="margin: 0 0.5rem;color:var(--border);">|</span> 
                <span style="color:#22c55e;">{topic_pct[dom]:.1f}% share</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Topic table ───────────────────────────────────────────────────────────
    st.markdown("#### Topic Rankings")
    rows = []
    for rank, (i, raw, pct) in enumerate(active, 1):
        rows.append(
            {
                "Rank": f"#{rank}",
                "Topic": f"Topic {i}",
                "% Share": f"{pct:.1f}%",
                "Raw Score": f"{raw:.5f}",
                "Top 5 Keywords": " | ".join(top_words[i][:5]),
                "Dominant": "◀" if i == dom else "",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── VIS 1: Raw / % share bars ─────────────────────────────────────────────
    st.markdown("#### 📊 Score Overview")
    a_idx  = [x[0] for x in active]
    a_raw  = [x[1] for x in active]
    a_pct  = [x[2] for x in active]
    a_lbls = [f"T{i}\n{top_words[i][0]}" for i in a_idx]
    bar_col= [_RED if i == dom else _ACCENT for i in a_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    # Before/after
    b0 = axes[0].bar(
        ["Raw", "Clean"],
        [results["raw_tokens"], results["clean_tokens"]],
        color=[_RED, "#22c55e"], edgecolor=_BG2, width=0.4,
    )
    for bar, n in zip(b0, [results["raw_tokens"], results["clean_tokens"]]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
            f"{n:,}", ha="center", fontweight="bold", color=_TEXT, fontsize=9,
        )
    axes[0].set_title("Raw vs Clean Tokens", fontweight="bold")
    axes[0].set_ylabel("Tokens")

    b1 = axes[1].bar(a_lbls, a_raw, color=bar_col, edgecolor=_BG2, width=0.6)
    for bar, s in zip(b1, a_raw):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
            f"{s:.4f}", ha="center", va="bottom", fontsize=6, fontweight="bold", color=_TEXT,
        )
    axes[1].set_title("NMF Raw Scores (red = dominant)", fontweight="bold")
    axes[1].tick_params(axis="x", labelsize=7)

    b2 = axes[2].bar(a_lbls, a_pct, color=bar_col, edgecolor=_BG2, width=0.6)
    for bar, p in zip(b2, a_pct):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
            f"{p:.1f}%", ha="center", va="bottom", fontsize=6, fontweight="bold", color=_TEXT,
        )
    axes[2].set_title("NMF % Share", fontweight="bold")
    axes[2].tick_params(axis="x", labelsize=7)

    fig.suptitle("Topic Overview", fontsize=13, fontweight="bold", color=_TEXT)
    _apply_dark_style(fig)
    plt.tight_layout()
    st.pyplot(fig)
    vis1_bytes = _fig_to_bytes(fig)
    plt.close(fig)

    # ── VIS 2: Pie + Radar ────────────────────────────────────────────────────
    st.markdown("#### 🥧 Topic Distribution")
    top8_idx = a_idx[:8]
    top8_pct = a_pct[:8]
    t8_labels = [f"T{i}: {top_words[i][0]}" for i in top8_idx]
    explode = [0.09 if i == dom else 0 for i in top8_idx]
    pie_col = plt.cm.Set3(np.linspace(0, 1, len(top8_idx)))

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    _, texts, autotexts = axes2[0].pie(
        top8_pct, labels=t8_labels, colors=pie_col,
        autopct="%1.1f%%", startangle=140, explode=explode,
        textprops={"fontsize": 8, "color": _TEXT},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    axes2[0].set_title("Topic % Share — Top 8 (exploded = dominant)", fontweight="bold")

    # Radar
    n_r = len(top8_idx)
    angles = np.linspace(0, 2 * np.pi, n_r, endpoint=False).tolist()
    vals = top8_pct + [top8_pct[0]]
    ang_p = angles + [angles[0]]
    ax_r = fig2.add_subplot(1, 2, 2, polar=True)
    ax_r.fill(ang_p, vals, color=_ACCENT, alpha=0.25)
    ax_r.plot(ang_p, vals, color=_TEXT, linewidth=2)
    ax_r.scatter(angles, top8_pct, color=_RED, s=80, zorder=5)
    ax_r.set_xticks(angles)
    ax_r.set_xticklabels(
        [f"T{i}\n{top_words[i][0]}" for i in top8_idx], fontsize=8, color=_TEXT
    )
    ax_r.tick_params(colors=_TEXT)
    ax_r.set_facecolor(_BG)
    ax_r.set_title("Radar Chart — % Share", fontweight="bold", pad=20, color=_TEXT)

    fig2.patch.set_facecolor(_BG2)
    axes2[0].set_facecolor(_BG2)
    plt.tight_layout()
    st.pyplot(fig2)
    vis2_bytes = _fig_to_bytes(fig2)
    plt.close(fig2)

    # ── VIS 3: All-topic word bars ─────────────────────────────────────────────
    st.markdown("#### 📋 All Topics — Word Weights")
    cols_per_row = 4
    rows_needed  = (n_topics + cols_per_row - 1) // cols_per_row
    fig3, axes3  = plt.subplots(rows_needed, cols_per_row, figsize=(18, rows_needed * 3.2))
    axes3 = axes3.flatten()

    for i in range(n_topics):
        ws = tww[i]["words"]
        sc = tww[i]["weights"]
        cl = _RED if i == dom else _ACCENT
        axes3[i].barh(ws[::-1], sc[::-1], color=cl, edgecolor=_BG2, alpha=0.85)
        pct_s = f"  ({topic_pct[i]:.1f}%)" if topic_dist[i] > 1e-5 else ""
        title_c = _RED if i == dom else _TEXT
        axes3[i].set_title(
            f"Topic {i}{' ◀ dom' if i == dom else ''}{pct_s}",
            fontsize=8, fontweight="bold", color=title_c,
        )
        axes3[i].tick_params(labelsize=6, colors=_TEXT)
        axes3[i].set_xlabel("weight", fontsize=6, color=_TEXT)
        axes3[i].set_facecolor(_BG)
        for spine in axes3[i].spines.values():
            spine.set_edgecolor(_BORDER)

    for j in range(n_topics, len(axes3)):
        axes3[j].set_visible(False)

    fig3.suptitle(
        f"All Topics  |  red = dominant (Topic {dom})",
        fontsize=12, fontweight="bold", y=1.01, color=_TEXT,
    )
    fig3.patch.set_facecolor(_BG2)
    plt.tight_layout()
    st.pyplot(fig3)
    vis3_bytes = _fig_to_bytes(fig3)
    plt.close(fig3)

    # ── Word clouds ───────────────────────────────────────────────────────────
    st.markdown("#### ☁️ Word Clouds")
    wc_cols = st.columns(min(3, n_topics))
    wc_imgs = []
    for idx, (tid, col) in enumerate(zip(top8_idx[:3], wc_cols)):
        wc_text = " ".join(top_words[tid] * 30)
        wc = WordCloud(
            width=400, height=250,
            background_color="#1c2330",
            colormap="plasma" if tid == dom else "Blues",
            max_words=40,
            collocations=False,
        ).generate(wc_text)
        fig_wc, ax_wc = plt.subplots(figsize=(5, 3))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        ax_wc.set_title(
            f"Topic {tid}{' ◀ dominant' if tid == dom else ''}",
            fontsize=9, fontweight="bold", color=_TEXT,
        )
        fig_wc.patch.set_facecolor(_BG2)
        col.pyplot(fig_wc)
        wc_imgs.append(_fig_to_bytes(fig_wc))
        plt.close(fig_wc)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💾 Download Results")
    dl_col1, dl_col2 = st.columns(2)

    # CSV report
    csv_bytes = df.to_csv(index=False).encode()
    dl_col1.download_button(
        "📥 Download Topic Table (.csv)",
        data=csv_bytes,
        file_name="topic_modelling_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ZIP of all charts
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("topic_overview_bars.png", vis1_bytes)
        zf.writestr("topic_pie_radar.png", vis2_bytes)
        zf.writestr("topic_word_weights.png", vis3_bytes)
        for k, img in enumerate(wc_imgs):
            zf.writestr(f"wordcloud_topic_{top8_idx[k]}.png", img)
    buf.seek(0)
    dl_col2.download_button(
        "📥 Download All Charts (.zip)",
        data=buf,
        file_name="topic_charts.zip",
        mime="application/zip",
        use_container_width=True,
    )
