"""Summary tab renderer."""

import io
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

# ── Colours ───────────────────────────────────────────────────────────────────
_BG   = "#1c2330"
_BG2  = "#0d1117"
_TEXT = "#e6edf3"
_BORDER = "#30363d"
_ACCENT= "#7c6af7"
_A2   = "#22c55e"
_A3   = "#f59e0b"
_A4   = "#ef4444"


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
def render_summary_tab(results: dict):
    wc         = results["word_count"]
    sc         = results["sent_count"]
    rt         = results["read_time"]
    uniq       = results["unique_words"]
    avg_sl     = results["avg_sent_len"]
    kw_pairs   = results["kw_pairs"]
    consensus  = results["consensus"]
    ext_text   = results["extractive_text"]
    abs_sum    = results["abstractive_summary"]
    freq       = results["freq"]
    ac         = results["agree_counts"]
    sentences  = results["sentences"]

    ext_words = len(ext_text.split())
    abs_words = len(abs_sum.split())
    comp_pct  = round(ext_words / wc * 100) if wc else 0

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Words",        f"{wc:,}")
    m2.metric("Sentences",    sc)
    m3.metric("Unique Words", f"{uniq:,}")
    m4.metric("Avg Sent Len", f"{avg_sl} words")
    m5.metric("Read Time",    f"~{rt} min")

    st.markdown("---")

    # ── Summaries (Single Column Layout) ──────────────────────────────────────
    st.markdown("#### 📄 Document Summaries")

    st.markdown(
        f"""
        <h5 style="color:var(--accent-light);">📌 Extractive Summary</h5>
        <p style="color:var(--text-secondary);font-size:.82rem;margin-top:-.4rem;">
            {len(consensus)} sentences · {ext_words} words · {comp_pct}% of original
        </p>
        """,
        unsafe_allow_html=True,
    )
    import html
    ext_html = f'<div class="summary-box" style="display: block; height: auto; min-height: fit-content; overflow: visible; border-top: 3px solid var(--accent); padding: 1.5rem 1.8rem; font-size: 1rem; line-height: 1.8; margin-bottom: 2rem;">'
    for i, sent in enumerate(consensus, 1):
        safe_sent = html.escape(sent)
        ext_html += f'<p style="margin-bottom:0.75rem;"><strong style="color:var(--accent-light);">[{i}]</strong> {safe_sent}</p>'
    ext_html += '</div>'
    st.markdown(ext_html, unsafe_allow_html=True)

    st.markdown(
        f"""
        <h5 style="color:#22c55e;">🤖 Abstractive Summary (BART)</h5>
        <p style="color:var(--text-secondary);font-size:.82rem;margin-top:-.4rem;">
            {abs_words} words
        </p>
        """,
        unsafe_allow_html=True,
    )
    if abs_sum.startswith("[BART unavailable"):
        st.warning(abs_sum)
    else:
        safe_abs = html.escape(abs_sum)
        abs_html = f'<div class="summary-box" style="display: block; height: auto; min-height: fit-content; overflow: visible; border-top: 3px solid #22c55e; padding: 1.5rem 1.8rem; font-size: 1rem; line-height: 1.8;">{safe_abs}</div>'
        st.markdown(abs_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Keywords (pill chips) ─────────────────────────────────────────────────
    st.markdown("#### 🔑 Key Keywords")
    top_kws = [kw for kw, _ in kw_pairs[:15]]
    pills_html = " ".join(f'<span class="kw-pill">{kw}</span>' for kw in top_kws)
    st.markdown(f'<div style="margin-bottom:1rem;">{pills_html}</div>', unsafe_allow_html=True)

    # Compression stat
    st.markdown(
        f"""
        <div style="display:flex;gap:1.5rem;margin:1rem 0;">
            <div style="background:var(--bg-card);border:1px solid var(--border);
                        border-radius:10px;padding:.7rem 1.2rem;text-align:center;">
                <div style="color:var(--text-secondary);font-size:.75rem;text-transform:uppercase;">Original</div>
                <div style="color:var(--accent-light);font-weight:700;font-size:1.3rem;">{wc:,} words</div>
            </div>
            <div style="align-self:center;color:var(--text-secondary);font-size:1.5rem;">→</div>
            <div style="background:var(--bg-card);border:1px solid var(--border);
                        border-radius:10px;padding:.7rem 1.2rem;text-align:center;">
                <div style="color:var(--text-secondary);font-size:.75rem;text-transform:uppercase;">Extractive</div>
                <div style="color:#22c55e;font-weight:700;font-size:1.3rem;">{ext_words} words</div>
            </div>
            <div style="align-self:center;color:var(--text-secondary);font-size:1.5rem;">→</div>
            <div style="background:var(--bg-card);border:1px solid var(--border);
                        border-radius:10px;padding:.7rem 1.2rem;text-align:center;">
                <div style="color:var(--text-secondary);font-size:.75rem;text-transform:uppercase;">Abstractive</div>
                <div style="color:#f59e0b;font-weight:700;font-size:1.3rem;">{abs_words} words</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Dashboard visuals (matches notebook 3x3 grid) ─────────────────────────
    st.markdown("#### 📊 Dashboard Visuals")

    sent_lengths = [len(s.split()) for s in sentences]
    top20 = freq.most_common(20)
    rest  = max(0, wc - ext_words)

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    # 1. Doc statistics
    stats  = {"Words": wc, "Sentences": sc, "Unique\nWords": uniq,
               "Keywords": len(results["keyword_tokens"])}
    bars1 = axes[0][0].bar(
        stats.keys(), stats.values(),
        color=["#3498db", "#22c55e", "#f59e0b", "#9b59b6"],
        edgecolor=_BG2, linewidth=1.5,
    )
    axes[0][0].bar_label(bars1, fmt="{:,.0f}", fontsize=8, fontweight="bold", color=_TEXT)
    axes[0][0].set_title("Document Statistics", fontweight="bold")
    axes[0][0].set_ylabel("Count")

    # 2. Compression ratio
    labels2 = ["Original\nDocument", "Extractive\nSummary", "Abstractive\nSummary"]
    sizes2  = [wc, ext_words, abs_words]
    bars2   = axes[0][1].barh(labels2, sizes2, color=[_A4, _A3, _A2],
                               edgecolor=_BG2, linewidth=1.5)
    axes[0][1].bar_label(bars2, labels=[f"{s} words" for s in sizes2],
                         padding=4, fontsize=8, color=_TEXT)
    axes[0][1].set_title("Words: Original vs Summaries", fontweight="bold")
    axes[0][1].set_xlabel("Word Count")

    # 3. TF-IDF keywords
    if kw_pairs:
        top15 = kw_pairs[:15]
        kw_names  = [k for k, _ in top15]
        kw_scores = [s for _, s in top15]
        palette3  = plt.cm.Blues_r(np.linspace(0.2, 0.8, len(top15)))
        axes[0][2].barh(kw_names[::-1], kw_scores[::-1], color=palette3)
    axes[0][2].set_title("Top 15 Keywords (TF-IDF)", fontweight="bold")
    axes[0][2].set_xlabel("Avg TF-IDF Score")

    # 4. Sentence length distribution
    axes[1][0].hist(sent_lengths, bins=20, color=_ACCENT, edgecolor=_BG2,
                    linewidth=0.8, alpha=0.85)
    mean_len = np.mean(sent_lengths) if sent_lengths else 0
    axes[1][0].axvline(mean_len, color=_A4, linestyle="--", linewidth=2,
                       label=f"Mean: {mean_len:.1f}")
    axes[1][0].set_title("Sentence Length Distribution", fontweight="bold")
    axes[1][0].set_xlabel("Words per Sentence")
    axes[1][0].set_ylabel("Count")
    axes[1][0].legend(fontsize=8, facecolor=_BG, edgecolor=_BORDER, labelcolor=_TEXT)

    # 5. Summariser agreement
    agree_labels = list(ac.keys())
    agree_vals   = list(ac.values())
    bars5 = axes[1][1].bar(agree_labels, agree_vals,
                            color=[_A4, _A3, _A2], edgecolor=_BG2, linewidth=1.5)
    axes[1][1].bar_label(bars5, fmt="{:.0f}", fontsize=10, fontweight="bold", color=_TEXT)
    axes[1][1].set_title("Summariser Agreement\n(sentences per vote count)",
                         fontweight="bold", fontsize=9)
    axes[1][1].set_ylabel("Sentences")

    # 6. Word frequency top-20
    if top20:
        tw, tc = zip(*top20)
        axes[1][2].bar(tw, tc, color=plt.cm.viridis(np.linspace(0, 1, 20)),
                       edgecolor=_BG2)
        axes[1][2].set_title("Top 20 Word Frequencies", fontweight="bold")
        axes[1][2].set_xlabel("Word")
        axes[1][2].set_ylabel("Count")
        plt.setp(axes[1][2].get_xticklabels(), rotation=45, ha="right", fontsize=7)

    # 7. Coverage pie
    axes[2][0].pie(
        [ext_words, abs_words, rest],
        labels=["Extractive\nSummary", "Abstractive\nSummary", "Rest of Doc"],
        colors=[_A3, _A2, _BG],
        autopct="%1.0f%%", startangle=140,
        wedgeprops=dict(edgecolor=_BG2, linewidth=2),
        textprops={"color": _TEXT},
    )
    axes[2][0].set_title("Summary Coverage", fontweight="bold")

    # 8. Document word cloud
    wc_text = " ".join(results["keyword_tokens"])
    if wc_text:
        wc_obj = WordCloud(width=600, height=400,
                           background_color="#1c2330", colormap="Blues",
                           max_words=100, collocations=False).generate(wc_text)
        axes[2][1].imshow(wc_obj, interpolation="bilinear")
    axes[2][1].axis("off")
    axes[2][1].set_title("Document Word Cloud", fontweight="bold")

    # 9. Summary word cloud
    summary_tokens = [
        t for t in ext_text.lower().split() if len(t) > 2
    ]
    wc_sum_text = " ".join(summary_tokens)
    if wc_sum_text:
        wc_obj2 = WordCloud(width=600, height=400,
                            background_color="#1c2330", colormap="Oranges",
                            max_words=80, collocations=False).generate(wc_sum_text)
        axes[2][2].imshow(wc_obj2, interpolation="bilinear")
    axes[2][2].axis("off")
    axes[2][2].set_title("Summary Word Cloud", fontweight="bold")

    fig.suptitle("Document Summary Dashboard", fontsize=18, fontweight="bold",
                 color=_TEXT, y=1.005)
    _apply_dark(fig)
    plt.tight_layout()
    st.pyplot(fig)
    dashboard_bytes = _fig_to_bytes(fig)
    plt.close(fig)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💾 Download Results")
    dl1, dl2 = st.columns(2)

    # Text report
    report = (
        "DOCUMENT SUMMARY REPORT\n"
        + "=" * 60 + "\n"
        + f"Words       : {wc}\n"
        + f"Sentences   : {sc}\n"
        + f"Est. Read   : ~{rt} min\n\n"
        + "TOP KEYWORDS\n" + "-" * 40 + "\n"
        + "\n".join(f"{i+1:2}. {kw} ({sc_:.4f})" for i, (kw, sc_) in enumerate(kw_pairs[:15]))
        + "\n\nEXTRACTIVE SUMMARY (Consensus)\n" + "-" * 40 + "\n"
        + "\n".join(f"[{i+1}] {s}" for i, s in enumerate(consensus))
        + "\n\nABSTRACTIVE SUMMARY (BART)\n" + "-" * 40 + "\n"
        + abs_sum + "\n"
    )
    dl1.download_button(
        "📥 Download Summary Report (.txt)",
        data=report.encode("utf-8"),
        file_name="document_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("summary_dashboard.png", dashboard_bytes)
        zf.writestr("document_summary.txt", report.encode("utf-8"))
        results_df = pd.DataFrame(
            {"Type": ["Extractive"] * len(consensus) + ["Abstractive"],
             "Sentence": list(consensus) + [abs_sum]}
        )
        zf.writestr("summary_results.csv", results_df.to_csv(index=False).encode())
    buf.seek(0)
    dl2.download_button(
        "📥 Download Full Package (.zip)",
        data=buf.getvalue(),
        file_name="summary_package.zip",
        mime="application/zip",
        use_container_width=True,
    )
