"""Sentiment Analysis tab renderer."""

import io
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

# ── Colours ───────────────────────────────────────────────────────────────────
_BG    = "#1c2330"
_BG2   = "#0d1117"
_TEXT  = "#e6edf3"
_BORDER= "#30363d"
_POS   = "#22c55e"
_NEU   = "#f59e0b"
_NEG   = "#ef4444"

_LABEL_COLORS = {"POSITIVE": _POS, "NEUTRAL": _NEU, "NEGATIVE": _NEG}


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    return buf.read()


def _apply_dark(fig):
    fig.patch.set_facecolor(_BG2)
    for ax in fig.axes:
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_TEXT, labelsize=8)
        for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            item.set_color(_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(_BORDER)


# ── Render ────────────────────────────────────────────────────────────────────
def render_sentiment_tab(results: dict):
    v_scores   = results["v_scores"]
    compound   = v_scores["compound"]
    label      = results["label"]
    sent_rows  = results["sentences"]
    lc         = results["label_counts"]
    top_words  = results["top_words"]
    freq       = results["freq"]

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Overall Sentiment", label)
    m2.metric("Compound Score",    f"{compound:+.4f}")
    m3.metric("Positive %",        f"{v_scores['pos']*100:.1f}%")
    m4.metric("Neutral %",         f"{v_scores['neu']*100:.1f}%")
    m5.metric("Negative %",        f"{v_scores['neg']*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "💡 **Understanding the Scores:** "
        "The **Compound Score** (-1.0 to +1.0) is a normalized metric that estimates the overall sentiment direction. "
        "The **Positive**, **Neutral**, and **Negative percentages** represent the fraction of text falling into each sentiment category according to the VADER lexicon."
    )

    # ── Reliability warning ───────────────────────────────────────────────────
    if results["reliability_warning"]:
        st.warning(
            "⚠️ **High neutral content detected.** This appears to be a technical or formal "
            "document. VADER's compound score may be inflated because it only sums the "
            "sentiment-bearing words and ignores the large neutral portion entirely."
        )

    st.markdown("---")

    # ── VIS 1: Compound gauge + Pie ───────────────────────────────────────────
    st.markdown("#### 🎯 Sentiment Overview")
    col_color = _POS if compound >= 0.05 else (_NEG if compound <= -0.05 else _NEU)

    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 4.5))

    # Gauge bar
    axes1[0].barh([""], [2], left=[-1], color=_BG, height=0.45)
    axes1[0].barh([""], [compound], color=col_color, height=0.45, alpha=0.9)
    axes1[0].axvline(0, color=_TEXT, linewidth=1.5, linestyle="--")
    axes1[0].set_xlim(-1, 1)
    axes1[0].set_yticks([])
    offset = 0.04 if compound >= 0 else -0.04
    ha     = "left" if compound >= 0 else "right"
    axes1[0].text(
        compound + offset, 0, f"{compound:+.3f}",
        va="center", ha=ha, fontweight="bold", color=_TEXT, fontsize=14,
    )
    axes1[0].set_title(
        f"Overall Sentiment: {label}\n(VADER Compound Score)",
        fontweight="bold", fontsize=12,
    )
    axes1[0].set_xlabel("← Negative                       Positive →", fontsize=9)

    # Sentence distribution pie
    labels_pie   = list(lc.keys())
    counts_pie   = list(lc.values())
    pie_colors   = [_LABEL_COLORS.get(l, "#8b949e") for l in labels_pie]
    wedges, texts, autotexts = axes1[1].pie(
        counts_pie, labels=labels_pie, autopct="%1.1f%%",
        colors=pie_colors, startangle=140,
        wedgeprops=dict(edgecolor=_BG2, linewidth=2),
        textprops={"color": _TEXT},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    axes1[1].set_title("Sentence-Level Distribution", fontweight="bold", fontsize=12)

    fig1.suptitle("VADER Sentiment Overview", fontsize=14, fontweight="bold", color=_TEXT)
    _apply_dark(fig1)
    plt.tight_layout()
    st.pyplot(fig1)
    vis1_bytes = _fig_to_bytes(fig1)
    plt.close(fig1)

    # ── VIS 2: Sentiment Flow ─────────────────────────────────────────────────
    if sent_rows:
        st.markdown("#### 📈 Sentiment Flow Across Sentences")
        df_sent = pd.DataFrame(sent_rows)
        x_vals  = df_sent["Sentence #"].values
        c_vals  = df_sent["Compound"].values

        fig2, ax2 = plt.subplots(figsize=(13, 4.5))
        ax2.fill_between(x_vals, c_vals, where=(c_vals >= 0), interpolate=True,
                         color=_POS, alpha=0.35, label="Positive region")
        ax2.fill_between(x_vals, c_vals, where=(c_vals < 0), interpolate=True,
                         color=_NEG, alpha=0.35, label="Negative region")
        ax2.plot(x_vals, c_vals, color=_TEXT, marker="o", markersize=4,
                 linewidth=1.8, label="Compound score")
        ax2.axhline(0,     color="#8b949e", linestyle="--", linewidth=1)
        ax2.axhline( 0.05, color=_POS,     linestyle=":",  linewidth=1, alpha=0.7)
        ax2.axhline(-0.05, color=_NEG,     linestyle=":",  linewidth=1, alpha=0.7)
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_xlabel("Sentence Number", fontsize=10)
        ax2.set_ylabel("VADER Compound Score", fontsize=10)
        ax2.set_title("Sentiment Flow Across Sentences", fontweight="bold", fontsize=12)
        ax2.legend(loc="upper right", fontsize=9,
                   facecolor=_BG, edgecolor=_BORDER, labelcolor=_TEXT)
        _apply_dark(fig2)
        plt.tight_layout()
        st.pyplot(fig2)
        vis2_bytes = _fig_to_bytes(fig2)
        plt.close(fig2)

        # ── VIS 3: Stacked Pos/Neu/Neg bar per sentence ───────────────────────
        st.markdown("#### 📊 Positive / Neutral / Negative Breakdown per Sentence")
        x = np.arange(len(df_sent))
        width = 0.6
        fig3, ax3 = plt.subplots(figsize=(13, 4.5))
        ax3.bar(x, df_sent["Positive"], width, label="Positive", color=_POS)
        ax3.bar(x, df_sent["Neutral"],  width, bottom=df_sent["Positive"],
                label="Neutral", color=_NEU)
        ax3.bar(x, df_sent["Negative"], width,
                bottom=df_sent["Positive"] + df_sent["Neutral"],
                label="Negative", color=_NEG)
        ax3.set_xticks(x)
        ax3.set_xticklabels(
            [f"S{int(n)}" for n in df_sent["Sentence #"]], rotation=45, ha="right",
        )
        ax3.set_ylabel("Proportion")
        ax3.set_ylim(0, 1)
        ax3.set_title("VADER Pos / Neu / Neg Breakdown per Sentence",
                      fontweight="bold", fontsize=12)
        ax3.legend(loc="upper right",
                   facecolor=_BG, edgecolor=_BORDER, labelcolor=_TEXT)
        _apply_dark(fig3)
        plt.tight_layout()
        st.pyplot(fig3)
        vis3_bytes = _fig_to_bytes(fig3)
        plt.close(fig3)
    else:
        vis2_bytes = vis3_bytes = b""

    # ── VIS 4: Top word frequency ─────────────────────────────────────────────
    st.markdown("#### 🔠 Top 20 Word Frequencies (after stopword removal)")
    if top_words:
        words_, counts_ = zip(*top_words)
        palette = plt.cm.plasma(np.linspace(0.2, 0.9, len(words_)))
        fig4, ax4 = plt.subplots(figsize=(12, 4.5))
        bars = ax4.bar(words_, counts_, color=palette, edgecolor=_BG2, linewidth=1)
        ax4.bar_label(bars, fontsize=8, color=_TEXT)
        ax4.set_xlabel("Word", fontsize=10)
        ax4.set_ylabel("Frequency", fontsize=10)
        ax4.set_title("Top 20 Most Frequent Words", fontweight="bold", fontsize=12)
        plt.xticks(rotation=40, ha="right")
        _apply_dark(fig4)
        plt.tight_layout()
        st.pyplot(fig4)
        vis4_bytes = _fig_to_bytes(fig4)
        plt.close(fig4)
    else:
        vis4_bytes = b""

    # ── VIS 5: Word Clouds ────────────────────────────────────────────────────
    st.markdown("#### ☁️ Word Clouds")
    wc_col1, wc_col2, wc_col3 = st.columns(3)
    wc_images = []
    for col, txt, title_, cmap_ in [
        (wc_col1, results["overall_text"],  "All Words",     "viridis"),
        (wc_col2, results["positive_text"], "Positive Words","Greens"),
        (wc_col3, results["negative_text"], "Negative Words","Reds"),
    ]:
        fig_wc, ax_wc = plt.subplots(figsize=(5, 3))
        if txt.strip():
            wc = WordCloud(
                width=500, height=300,
                background_color="#1c2330",
                colormap=cmap_,
                max_words=80,
                collocations=False,
            ).generate(txt)
            ax_wc.imshow(wc, interpolation="bilinear")
        else:
            ax_wc.text(
                0.5, 0.5, "No sentences\nin this category",
                ha="center", va="center", transform=ax_wc.transAxes,
                fontsize=11, color=_TEXT,
            )
        ax_wc.axis("off")
        ax_wc.set_title(title_, fontsize=12, fontweight="bold", color=_TEXT)
        fig_wc.patch.set_facecolor(_BG2)
        col.pyplot(fig_wc)
        wc_images.append(_fig_to_bytes(fig_wc))
        plt.close(fig_wc)

    # ── Sentence-level table ──────────────────────────────────────────────────
    if sent_rows:
        st.markdown("#### 🗒️ Sentence-Level Breakdown")

        def _row_color(row):
            c = row["Compound"]
            if c >= 0.05:
                return ["background-color:#0d2a1a; color:#22c55e"] * len(row)
            if c <= -0.05:
                return ["background-color:#2a0d0d; color:#ef4444"] * len(row)
            return ["background-color:#2a2510; color:#f59e0b"] * len(row)

        df_disp = df_sent[["Sentence #", "Text", "Compound", "Positive", "Neutral", "Negative", "Label"]]
        styled = (
            df_disp.style
            .apply(_row_color, axis=1)
            .format({
                "Compound": "{:+.4f}",
                "Positive": "{:.4f}",
                "Neutral":  "{:.4f}",
                "Negative": "{:.4f}",
            })
        )
        st.dataframe(styled, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💾 Download Results")
    dl1, dl2 = st.columns(2)

    csv_bytes = (
        pd.DataFrame(sent_rows).to_csv(index=False).encode()
        if sent_rows else b""
    )
    dl1.download_button(
        "📥 Download Sentiment Report (.csv)",
        data=csv_bytes,
        file_name="sentiment_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("sentiment_overview.png", vis1_bytes)
        if vis2_bytes:
            zf.writestr("sentiment_flow.png", vis2_bytes)
        if vis3_bytes:
            zf.writestr("sentence_breakdown.png", vis3_bytes)
        if vis4_bytes:
            zf.writestr("word_frequency.png", vis4_bytes)
        for k, img in enumerate(wc_images):
            titles = ["all", "positive", "negative"]
            zf.writestr(f"wordcloud_{titles[k]}.png", img)
    buf.seek(0)
    dl2.download_button(
        "📥 Download All Charts (.zip)",
        data=buf.getvalue(),
        file_name="sentiment_charts.zip",
        mime="application/zip",
        use_container_width=True,
    )
