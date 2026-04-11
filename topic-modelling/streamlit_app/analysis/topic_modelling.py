"""
NMF topic modelling analysis — exactly as in nmf_classifier.ipynb.
Uses pre-trained joblib models from the project directory.
"""

import os
import re
import warnings
import numpy as np

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")
for _r in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
    nltk.download(_r, quiet=True)

# ── Model directory ──────────────────────────────────────────────────────────
_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Extended stop-word set (from notebook) ───────────────────────────────────
_lemmatizer = WordNetLemmatizer()
try:
    _lemmatizer.lemmatize("dummy")
except Exception:
    pass

_STOP_WORDS = set(stopwords.words("english"))
_STOP_WORDS.update(
    {
        "would", "could", "should", "also", "get", "got", "like", "make",
        "one", "two", "three", "many", "much", "said", "say", "know",
        "think", "use", "used", "using", "may", "might", "well", "back",
        "even", "still", "way", "go", "going", "come", "came", "take",
        "good", "people", "time", "year", "new", "old", "thing", "things",
        "article", "writes", "subject", "document", "edu", "com", "from",
    }
)


# ── Pre-processing (from notebook Cell 3) ────────────────────────────────────
def preprocess(text: str) -> str:
    """HTML → noise → lowercase → lemmatise (exactly as the notebook)."""
    # Kill <script> blocks
    text = re.sub(
        r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE
    )
    # Strip tags
    text = re.sub(r"<[^>]+>", " ", text)
    # HTML entities
    text = re.sub(r"&amp;|&lt;|&gt;|&quot;|&#\d+;|&nbsp;|&[a-z]+;", " ", text)
    # Emojis/non-ASCII
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # URLs
    text = re.sub(r"\S+@\S+", "", text)           # emails
    text = re.sub(r"\d+", "", text)               # digits
    text = re.sub(r"[^a-z\s]", " ", text)         # punctuation
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in word_tokenize(text)
        if t not in _STOP_WORDS and len(t) > 2
    ]
    return " ".join(tokens)


# ── Model loader (cached at module level) ─────────────────────────────────────
_nmf_model = None
_tfidf_vec = None
_shared_vocab = None
_nmf_top_words = None


def _load_models():
    global _nmf_model, _tfidf_vec, _shared_vocab, _nmf_top_words
    if _nmf_model is not None:
        return
    _nmf_model = joblib.load(os.path.join(_MODEL_DIR, "nmf_model.joblib"))
    _tfidf_vec = joblib.load(os.path.join(_MODEL_DIR, "tfidf_vectorizer.joblib"))
    _shared_vocab = joblib.load(os.path.join(_MODEL_DIR, "shared_vocab.joblib"))
    _nmf_top_words = [
        [_shared_vocab[i] for i in row.argsort()[::-1][:10]]
        for row in _nmf_model.components_
    ]


# ── Public API ────────────────────────────────────────────────────────────────
def run_topic_modelling(raw_text: str) -> dict:
    """Return a dict of everything the UI needs."""
    _load_models()

    raw_tokens = raw_text.split()
    cleaned = preprocess(raw_text)
    clean_tokens = cleaned.split()

    # NMF scores
    vec = _tfidf_vec.transform([cleaned])
    topic_dist = _nmf_model.transform(vec)[0]
    total = topic_dist.sum() or 1e-9
    topic_pct = 100 * topic_dist / total
    ranked = topic_dist.argsort()[::-1]
    dom = ranked[0]

    n_topics = _nmf_model.n_components
    active = [
        (i, topic_dist[i], topic_pct[i])
        for i in range(n_topics)
        if topic_dist[i] > 1e-5
    ]
    active.sort(key=lambda x: x[1], reverse=True)

    # Per-topic word weights (for bar charts)
    topic_word_weights = [
        {
            "words": _nmf_top_words[i][:8],
            "weights": sorted(_nmf_model.components_[i], reverse=True)[:8],
        }
        for i in range(n_topics)
    ]

    return {
        "raw_tokens": len(raw_tokens),
        "clean_tokens": len(clean_tokens),
        "n_topics": n_topics,
        "topic_dist": topic_dist.tolist(),
        "topic_pct": topic_pct.tolist(),
        "ranked": ranked.tolist(),
        "dominant": int(dom),
        "active": active,
        "top_words": _nmf_top_words,
        "topic_word_weights": topic_word_weights,
        "cleaned_text": cleaned,
    }
