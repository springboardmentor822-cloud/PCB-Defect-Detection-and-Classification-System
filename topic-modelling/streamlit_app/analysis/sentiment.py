"""
Sentiment analysis — matches sentiment_analysis.ipynb exactly.
"""

import re
import warnings

import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")
for _r in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
    nltk.download(_r, quiet=True)

_lemmatizer = WordNetLemmatizer()
_STOP_WORDS = set(stopwords.words("english"))
_vader = SentimentIntensityAnalyzer()


# ── Pre-processing pipeline (from notebook) ──────────────────────────────────
def _remove_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()
    clean = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", clean).strip()


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "POSITIVE"
    if compound <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"


def _collect_words(sent_rows: list, label_val: str) -> str:
    tokens = []
    for row in sent_rows:
        if row["Label"] != label_val:
            continue
        toks = word_tokenize(_normalize_text(row["Text"]))
        tokens += [
            _lemmatizer.lemmatize(t)
            for t in toks
            if t.isalpha() and t not in _STOP_WORDS and len(t) > 2
        ]
    return " ".join(tokens)


# ── Public API ────────────────────────────────────────────────────────────────
def run_sentiment_analysis(raw_text: str) -> dict:
    # Step 2a — remove HTML
    html_cleaned = _remove_html(raw_text)

    # Step 2b — normalize
    normalized = _normalize_text(html_cleaned)

    # Step 2c — tokenize
    sentences_raw = sent_tokenize(html_cleaned)
    all_tokens = word_tokenize(normalized)
    filtered_tokens = [
        _lemmatizer.lemmatize(tok)
        for tok in all_tokens
        if tok.isalpha() and tok not in _STOP_WORDS and len(tok) > 2
    ]

    # Step 4a — document-level
    v_scores = _vader.polarity_scores(html_cleaned)

    # Step 4b — sentence-level
    rows = []
    for i, sent in enumerate(sentences_raw, 1):
        if len(sent.strip()) < 5:
            continue
        vs = _vader.polarity_scores(sent)
        rows.append(
            {
                "Sentence #": i,
                "Text": sent.strip(),
                "Compound": round(vs["compound"], 4),
                "Positive": round(vs["pos"], 4),
                "Neutral": round(vs["neu"], 4),
                "Negative": round(vs["neg"], 4),
                "Label": _vader_label(vs["compound"]),
            }
        )

    freq = FreqDist(filtered_tokens)
    label_counts = {}
    for r in rows:
        label_counts[r["Label"]] = label_counts.get(r["Label"], 0) + 1

    # Word texts for word-clouds
    overall_text = " ".join(filtered_tokens)
    positive_text = _collect_words(rows, "POSITIVE")
    negative_text = _collect_words(rows, "NEGATIVE")

    # Reliability flag
    reliability_warning = v_scores["neu"] > 0.70

    return {
        "raw_text": raw_text,
        "html_cleaned": html_cleaned,
        "normalized": normalized,
        "all_tokens": len(all_tokens),
        "filtered_tokens": filtered_tokens,
        "sentences": rows,
        "v_scores": v_scores,
        "label": _vader_label(v_scores["compound"]),
        "label_counts": label_counts,
        "freq": freq,
        "top_words": freq.most_common(20),
        "overall_text": overall_text,
        "positive_text": positive_text,
        "negative_text": negative_text,
        "reliability_warning": reliability_warning,
        # Preprocessing stats
        "char_raw": len(raw_text),
        "char_cleaned": len(html_cleaned),
        "char_normalized": len(normalized),
    }
