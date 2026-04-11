"""
Document summarization — matches document_summarization.ipynb exactly.
Extractive: TextRank + LSA + LexRank consensus.
Abstractive: facebook/bart-large-cnn.
"""

import re
import warnings
from collections import Counter

import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

warnings.filterwarnings("ignore")
for _r in ["punkt", "stopwords", "wordnet", "punkt_tab", "averaged_perceptron_tagger"]:
    nltk.download(_r, quiet=True)

_STOP_WORDS = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()
try:
    _lemmatizer.lemmatize("dummy")
except Exception:
    pass
_LANGUAGE = "english"
_SUMMARY_RATIO = 0.30

# ── Pre-processing (from notebook) ───────────────────────────────────────────
def _remove_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "head", "meta", "link", "noscript"]):
        tag.decompose()
    clean = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", clean).strip()


def _light_clean(text: str) -> str:
    """Light clean — preserve sentence structure for summarization."""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _deep_clean_tokens(text: str) -> list:
    """Deep clean for keywords/word-frequency only."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    return [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha() and t not in _STOP_WORDS and len(t) > 2
    ]


# ── BART loader (lazy, cached) ────────────────────────────────────────────────
_bart_model = None
_bart_tokenizer = None


def _load_bart():
    global _bart_model, _bart_tokenizer
    if _bart_model is not None:
        return
    from transformers import BartForConditionalGeneration, BartTokenizer
    _bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    _bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")


def _bart_summarize(text: str, max_length: int = 250, min_length: int = 80) -> str:
    _load_bart()
    inputs = _bart_tokenizer(
        text, max_length=1024, return_tensors="pt", truncation=True
    )
    ids = _bart_model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return _bart_tokenizer.decode(ids[0], skip_special_tokens=True)


def _chunk_text(text: str, max_chars: int = 3500) -> list:
    sents = sent_tokenize(text)
    chunks, current = [], ""
    for s in sents:
        if len(current) + len(s) < max_chars:
            current += " " + s
        else:
            if current:
                chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks


# ── Public API ────────────────────────────────────────────────────────────────
def run_summarization(raw_text: str) -> dict:
    # Pre-process
    html_removed = _remove_html(raw_text)
    clean_text = _light_clean(html_removed)
    keyword_tokens = _deep_clean_tokens(clean_text)

    sentences = sent_tokenize(clean_text)
    words_all = word_tokenize(clean_text)
    word_count = len(words_all)
    sent_count = len(sentences)
    unique_words = len({w.lower() for w in words_all if w.isalpha()})
    avg_sent_len = word_count / sent_count if sent_count else 0
    read_time = max(1, round(word_count / 200))

    # ── Keyword extraction (TF-IDF) ──────────────────────────────────────────
    tfidf = TfidfVectorizer(
        max_features=50, stop_words="english", ngram_range=(1, 2)
    )
    try:
        tfidf_matrix = tfidf.fit_transform(sentences)
        feature_names = tfidf.get_feature_names_out()
        avg_tfidf = tfidf_matrix.mean(axis=0).A1
        kw_pairs = sorted(
            zip(feature_names, avg_tfidf), key=lambda x: x[1], reverse=True
        )[:20]
    except Exception:
        kw_pairs = []

    # ── Extractive summarization ─────────────────────────────────────────────
    n_sentences = max(3, int(sent_count * _SUMMARY_RATIO))
    parser = PlaintextParser.from_string(clean_text, Tokenizer(_LANGUAGE))
    stemmer = Stemmer(_LANGUAGE)

    summarizers = {
        "TextRank": TextRankSummarizer(stemmer),
        "LSA": LsaSummarizer(stemmer),
        "LexRank": LexRankSummarizer(stemmer),
    }
    extractive_summaries = {}
    for name, summarizer in summarizers.items():
        summarizer.stop_words = get_stop_words(_LANGUAGE)
        extracted = summarizer(parser.document, n_sentences)
        extractive_summaries[name] = [str(s) for s in extracted]

    # Consensus — sentences chosen by ≥ 2 summarisers
    all_sents = [s for v in extractive_summaries.values() for s in v]
    vote_counts = Counter(all_sents)

    def _sent_position(sent):
        for i, orig in enumerate(sentences):
            if sent.strip() in orig or orig.strip() in sent:
                return i
        return 9999

    consensus = [s for s, c in vote_counts.items() if c >= 2]
    consensus_ordered = sorted(consensus, key=_sent_position)
    if not consensus_ordered:
        consensus_ordered = extractive_summaries["TextRank"]

    extractive_text = " ".join(consensus_ordered)

    # ── Abstractive summarization (BART) ─────────────────────────────────────
    try:
        chunks = _chunk_text(clean_text)
        chunk_summaries = [_bart_summarize(c) for c in chunks]
        if len(chunk_summaries) > 1:
            combined = " ".join(chunk_summaries)
            abstractive_summary = _bart_summarize(
                combined[:3500], max_length=200, min_length=60
            )
        else:
            abstractive_summary = chunk_summaries[0]
    except Exception as e:
        abstractive_summary = f"[BART unavailable: {e}]"

    freq = FreqDist(keyword_tokens)

    # Summariser agreement counts (for chart)
    agree_counts = {
        "Only 1": sum(1 for v in vote_counts.values() if v == 1),
        "In 2":   sum(1 for v in vote_counts.values() if v == 2),
        "All 3":  sum(1 for v in vote_counts.values() if v == 3),
    }

    return {
        "clean_text": clean_text,
        "word_count": word_count,
        "sent_count": sent_count,
        "unique_words": unique_words,
        "avg_sent_len": round(avg_sent_len, 1),
        "read_time": read_time,
        "keyword_tokens": keyword_tokens,
        "kw_pairs": kw_pairs,
        "extractive_summaries": extractive_summaries,
        "consensus": consensus_ordered,
        "extractive_text": extractive_text,
        "abstractive_summary": abstractive_summary,
        "freq": freq,
        "vote_counts": vote_counts,
        "agree_counts": agree_counts,
        "sentences": sentences,
    }
