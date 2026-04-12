import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import re
import nltk
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from textblob import TextBlob

# =====================================================
# NLTK DOWNLOADS
# =====================================================
@st.cache_resource
def download_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

download_nltk()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="NarrativeNexus", layout="wide")
st.title("📊 NarrativeNexus: Dynamic Text Analysis Platform")
st.caption("Topic modeling powered by NMF (TF-IDF) — selected for higher coherence score (0.695 vs LDA 0.601)")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Select Input Type", ["Upload File", "Enter Text"])
num_topics = st.sidebar.slider("Number of Topics", 2, 30, 30, step=1)
n_top_words = st.sidebar.slider("Top Words per Topic", 5, 15, 10)
st.sidebar.info("💡 For best results: provide 20+ documents or paragraphs. Single sentences will only show sentiment & word cloud.")

# =====================================================
# INPUT HANDLING
# =====================================================
documents = []

if mode == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload CSV / TXT / Excel / JSON",
        type=["csv", "txt", "xlsx", "json"]
    )
    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1]

        if file_type == "csv":
            df_raw = pd.read_csv(uploaded_file)
            text_column = st.selectbox("Select Text Column", df_raw.columns)
            documents = df_raw[text_column].dropna().astype(str).tolist()

        elif file_type == "txt":
            text = uploaded_file.read().decode("utf-8")
            documents = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 20]
            if len(documents) < 5:
                documents = [s.strip() for s in text.split('\n') if len(s.strip()) > 20]
            if len(documents) < 5:
                sentences = sent_tokenize(text)
                documents = [s.strip() for s in sentences if len(s.strip()) > 20]

        elif file_type in ["xlsx", "xls"]:
            df_raw = pd.read_excel(uploaded_file)
            text_column = st.selectbox("Select Text Column", df_raw.columns)
            documents = df_raw[text_column].dropna().astype(str).tolist()

        elif file_type == "json":
            df_raw = pd.read_json(uploaded_file)
            text_column = st.selectbox("Select Text Column", df_raw.columns)
            documents = df_raw[text_column].dropna().astype(str).tolist()

elif mode == "Enter Text":
    text_input = st.text_area("Enter your text here (paste multiple paragraphs for best results)", height=200)
    if text_input:
        documents = [p.strip() for p in text_input.split('\n\n') if len(p.strip()) > 20]
        if not documents:
            documents = [p.strip() for p in text_input.split('\n') if len(p.strip()) > 20]
        if not documents:
            documents = [text_input.strip()]

if not documents:
    st.info("Please provide input to continue.")
    st.stop()

st.success(f"✅ Loaded **{len(documents)}** documents.")

# =====================================================
# PREPROCESSING
# =====================================================
@st.cache_data
def build_stopwords():
    stop_words = set(stopwords.words('english'))
    custom_words = {
        'know', 'think', 'said', 'like', 'really',
        'people', 'make', 'going', 'thing', 'want',
        'does', 'did', 'just', 'good', 'right'
    }
    return stop_words.union(custom_words)

def is_repeated_char(word):
    return len(set(word)) <= 2

@st.cache_data
def preprocess_documents(docs):
    stop_words = build_stopwords()
    lemmatizer = WordNetLemmatizer()
    processed = []
    for text in docs:
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in stop_words]
        tokens = [w for w in tokens if len(w) > 3]
        tokens = [w for w in tokens if not is_repeated_char(w)]
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        processed.append(" ".join(tokens))
    return processed

with st.spinner("🔄 Preprocessing text..."):
    processed_docs = preprocess_documents(documents)

valid_pairs = [(orig, proc) for orig, proc in zip(documents, processed_docs) if len(proc.strip()) > 10]
if not valid_pairs:
    st.error("Not enough content after preprocessing. Please provide more data.")
    st.stop()

documents, processed_docs = zip(*valid_pairs)
documents = list(documents)
processed_docs = list(processed_docs)

# =====================================================
# INPUT PREVIEW
# =====================================================
st.subheader("📄 Input Preview")
preview_df = pd.DataFrame({
    "Original Text": documents[:5],
    "Processed Text": processed_docs[:5]
})
st.dataframe(preview_df, use_container_width=True)

# =====================================================
# CONTENT SIZE CHECK
# =====================================================
n_docs = len(processed_docs)
MIN_DOCS_FOR_TOPICS = 5

if n_docs < MIN_DOCS_FOR_TOPICS:
    st.warning(
        f"⚠️ Only **{n_docs}** document(s) detected. "
        "Topic modeling requires at least 5 documents. "
        "Showing **Sentiment Analysis** and **Word Cloud** only."
    )

    def get_sentiment(text):
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.05:   return "Positive"
        elif polarity < -0.05: return "Negative"
        return "Neutral"

    sentiments = [get_sentiment(d) for d in documents]
    sent_df = pd.DataFrame({"Document": documents, "Sentiment": sentiments})

    st.subheader("😊 Sentiment Analysis")
    sent_counts = sent_df["Sentiment"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]
    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = px.pie(
            sent_counts, names="Sentiment", values="Count",
            color="Sentiment",
            color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
        )
        fig_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        st.dataframe(sent_df, use_container_width=True)

    st.subheader("☁️ Word Cloud")
    all_text = " ".join(processed_docs)
    if all_text.strip():
        wc = WordCloud(width=1000, height=380, background_color='white',
                       colormap='viridis', max_words=100).generate(all_text)
        fig_wc, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig_wc)
        plt.close(fig_wc)

    st.stop()

# =====================================================
# TF-IDF VECTORIZATION
# =====================================================
num_topics = min(num_topics, n_docs - 1) if n_docs <= num_topics else num_topics
num_topics = max(2, num_topics)

if n_docs <= 10:
    min_df_val = 1
    max_df_val = n_docs
elif n_docs <= 50:
    min_df_val = 1
    max_df_val = max(2, int(n_docs * 0.95))
else:
    min_df_val = max(2, int(n_docs * 0.02))
    max_df_val = max(min_df_val + 1, int(n_docs * 0.95))

with st.spinner("📐 Building TF-IDF matrix..."):
    tfidf_vectorizer = TfidfVectorizer(
        max_df=max_df_val,
        min_df=min_df_val,
        stop_words='english'
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)

if tfidf_matrix.shape[1] == 0:
    st.error("No terms found after vectorization. Please provide more or longer text.")
    st.stop()

num_topics = min(num_topics, tfidf_matrix.shape[1])
st.caption(f"📊 TF-IDF matrix: {tfidf_matrix.shape[0]} documents × {tfidf_matrix.shape[1]} terms | Topics: {num_topics}")

# =====================================================
# NMF TOPIC MODELING
# =====================================================
with st.spinner("🧠 Training NMF model..."):
    nmf_model = NMF(n_components=num_topics, random_state=42, max_iter=500)
    nmf_model.fit(tfidf_matrix)
    nmf_doc_topic = nmf_model.transform(tfidf_matrix)

feature_names = tfidf_vectorizer.get_feature_names_out()

n_top_words_safe = min(n_top_words, len(feature_names))
topic_keywords = {}
for i, topic in enumerate(nmf_model.components_):
    top_words = [feature_names[j] for j in topic.argsort()[:-n_top_words_safe - 1:-1]]
    topic_keywords[i] = top_words

nmf_dominant_topic = np.argmax(nmf_doc_topic, axis=1) + 1
nmf_topic_weight   = np.max(nmf_doc_topic, axis=1)

df_results = pd.DataFrame({
    "Document_No":    range(len(documents)),
    "Document":       [d[:80] + "..." if len(d) > 80 else d for d in documents],
    "Dominant_Topic": nmf_dominant_topic,
    "Topic_Weight":   nmf_topic_weight.round(4),
    "Topic_Keywords": [", ".join(topic_keywords[t - 1]) for t in nmf_dominant_topic],
})

# =====================================================
# SENTIMENT ANALYSIS
# =====================================================
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.05:    return "Positive"
    elif polarity < -0.05: return "Negative"
    return "Neutral"

df_results["Sentiment"] = [get_sentiment(d) for d in documents]

# =====================================================
# KPI CARDS
# =====================================================
st.subheader("📊 Analysis Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Documents",    len(documents))
col2.metric("Vocabulary Size",    tfidf_matrix.shape[1])
col3.metric("Topics Modeled",     num_topics)
col4.metric("Dominant Sentiment", df_results["Sentiment"].mode().iloc[0])

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Topic Distribution",
    "🗂️ Topic Keywords",
    "😊 Sentiment",
    "☁️ Word Cloud",
    "📄 Results & Summary"
])

# ─────────────────────────────────────────────
# TAB 1: TOPIC DISTRIBUTION
# ─────────────────────────────────────────────
with tab1:
    st.subheader("NMF Topic Distribution")

    topic_dist = df_results["Dominant_Topic"].value_counts().reset_index()
    topic_dist.columns = ["Topic", "Count"]
    topic_dist["Keywords"] = topic_dist["Topic"].apply(
        lambda x: ", ".join(topic_keywords[x - 1][:5])
    )
    topic_dist["Label"] = "Topic " + topic_dist["Topic"].astype(str) + ": " + topic_dist["Keywords"]
    topic_dist["Percentage"] = (topic_dist["Count"] / topic_dist["Count"].sum() * 100).round(2)
    topic_dist = topic_dist.sort_values("Topic")

    fig = px.bar(
        topic_dist, x="Label", y="Count",
        color="Count", color_continuous_scale="Blues",
        hover_data=["Percentage"],
        title=f"Topic Distribution across {len(documents)} documents"
    )
    fig.update_layout(xaxis_tickangle=-40, xaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(topic_dist[["Topic", "Keywords", "Count", "Percentage"]], use_container_width=True)

# ─────────────────────────────────────────────
# TAB 2: TOPIC KEYWORDS
# ─────────────────────────────────────────────
with tab2:
    st.subheader(f"Top {n_top_words_safe} Keywords per NMF Topic")

    dominant_topic_id = df_results["Dominant_Topic"].mode().iloc[0]
    dominant_share = (
        df_results["Dominant_Topic"].value_counts(normalize=True).max() * 100
    )
    st.success(
        f"🏆 Dominant Topic: **Topic {dominant_topic_id}** — "
        f"{', '.join(topic_keywords[dominant_topic_id - 1])}  |  "
        f"Share: {round(dominant_share, 2)}%"
    )

    # Keywords table — no labelling inputs
    st.subheader("📋 Topic Keywords Table")
    kw_df = pd.DataFrame({
        "Topic":    [f"Topic {i+1}" for i in topic_keywords],
        "Keywords": [", ".join(words) for words in topic_keywords.values()]
    })
    st.dataframe(kw_df, use_container_width=True)

    # Word weight bar charts
    st.subheader("Top Word Weights per Topic")
    n_chart_topics = st.slider("How many topics to visualize", 1, num_topics, min(3, num_topics))
    cols = st.columns(min(n_chart_topics, 3))
    for idx in range(n_chart_topics):
        topic_vec = nmf_model.components_[idx]
        top_idx   = topic_vec.argsort()[:-n_top_words_safe - 1:-1]
        words     = [feature_names[j] for j in top_idx]
        weights   = topic_vec[top_idx]
        fig_w, ax = plt.subplots(figsize=(4, 3))
        ax.barh(words[::-1], weights[::-1], color="#3A8FD4")
        ax.set_xlabel("Word Importance")
        ax.set_title(f"Topic {idx+1}")
        plt.tight_layout()
        with cols[idx % 3]:
            st.pyplot(fig_w)
        plt.close(fig_w)

# ─────────────────────────────────────────────
# TAB 3: SENTIMENT
# ─────────────────────────────────────────────
with tab3:
    sentiment_counts = df_results["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    sentiment_counts["Percentage"] = (
        sentiment_counts["Count"] / sentiment_counts["Count"].sum() * 100
    ).round(2)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = px.pie(
            sentiment_counts, names="Sentiment", values="Count",
            color="Sentiment",
            color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
        )
        fig_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        fig_bar = px.bar(
            sentiment_counts, x="Sentiment", y="Count",
            color="Sentiment",
            color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(sentiment_counts, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 4: WORD CLOUD
# ─────────────────────────────────────────────
with tab4:
    st.subheader("Corpus Word Cloud")
    all_text = " ".join(processed_docs)
    if all_text.strip():
        wc = WordCloud(width=1000, height=380, background_color='white',
                       colormap='viridis', max_words=100).generate(all_text)
        fig_wc, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig_wc)
        plt.close(fig_wc)

    st.subheader("Per-Topic Word Clouds")
    n_wc_topics = min(6, num_topics)
    cols = st.columns(3)
    for i, (topic_id, words) in enumerate(list(topic_keywords.items())[:n_wc_topics]):
        topic_text = " ".join(words * 20)
        wc_t = WordCloud(width=400, height=220, background_color='white',
                         colormap='plasma').generate(topic_text)
        fig_t, ax_t = plt.subplots(figsize=(4, 2.2))
        ax_t.imshow(wc_t, interpolation='bilinear')
        ax_t.axis('off')
        ax_t.set_title(f"Topic {topic_id + 1}", fontsize=9)
        with cols[i % 3]:
            st.pyplot(fig_t)
        plt.close(fig_t)

# ─────────────────────────────────────────────
# TAB 5: DOCUMENT RESULTS + SUMMARY
# ─────────────────────────────────────────────
with tab5:
    st.subheader("Document-Level Results")
    st.caption("Document_No · Dominant_Topic · Topic_Weight · Keywords · Sentiment")
    st.dataframe(df_results, use_container_width=True)

    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv,
        file_name="narrativenexus_results.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("📝 Extractive + Abstractive Summary")
    st.caption("Summarizes the entire corpus. Step 1 — Extractive: picks key sentences using TF-IDF scoring. Step 2 — Abstractive: BART rewrites those into new fluent sentences.")

    # ── STEP 1: EXTRACTIVE SUMMARY of full corpus ──
    def generate_extractive_summary(docs, top_n_sentences=5):
        combined_text = " ".join(docs)
        # Clean text before summarizing
        combined_text = re.sub(r'\S+@\S+', '', combined_text)           # remove emails
        combined_text = re.sub(r'[^a-zA-Z0-9.\s]', ' ', combined_text)  # remove special chars
        combined_text = re.sub(r'\s+', ' ', combined_text)              # remove extra spaces
        sentences = [s.strip() for s in combined_text.split(".")
                     if len(s.strip()) > 30 and not s.strip().isupper()]
        if not sentences:
            return combined_text[:500]
        if len(sentences) <= top_n_sentences:
            return ". ".join(sentences) + "."
        from sklearn.feature_extraction.text import TfidfVectorizer as TV
        vectorizer = TV(stop_words='english')
        tfidf_m = vectorizer.fit_transform(sentences)
        scores = np.array(tfidf_m.sum(axis=1)).flatten()
        top_idx = sorted(scores.argsort()[-top_n_sentences:])
        return ". ".join([sentences[i] for i in top_idx]) + "."

    # ── STEP 2: ABSTRACTIVE SUMMARY via BART ──
    @st.cache_resource
    def load_summarizer():
        from transformers import BartTokenizer, BartForConditionalGeneration
        tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
        return tokenizer, model

    def generate_abstractive_summary(extractive_text):
        if len(extractive_text.split()) < 20:
            return extractive_text
        tokenizer, model = load_summarizer()
        inputs = tokenizer(
            extractive_text[:1024], return_tensors="pt",
            max_length=1024, truncation=True
        )
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=130, min_length=30,
            length_penalty=2.0, num_beams=4,
            early_stopping=True
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    num_sents = st.slider("Number of sentences for extractive summary", 2, 10, 5)

    col_e, col_a = st.columns(2)

    with col_e:
        st.markdown("**Step 1 — Extractive Summary**")
        st.caption("Key sentences from the full corpus, scored by TF-IDF")
        if st.button("📄 Generate Extractive Summary"):
            with st.spinner("Extracting key sentences..."):
                ext_summary = generate_extractive_summary(documents, top_n_sentences=num_sents)
                st.session_state["ext_summary"] = ext_summary
                st.info(ext_summary)

    with col_a:
        st.markdown("**Step 2 — Abstractive Summary (BART)**")
        st.caption("BART rewrites the extractive output into new fluent sentences")
        if st.button("🚀 Generate Abstractive Summary"):
            if "ext_summary" not in st.session_state or not st.session_state["ext_summary"]:
                st.warning("Please run Step 1 first.")
            else:
                with st.spinner("Summarizing with BART..."):
                    try:
                        abs_summary = generate_abstractive_summary(st.session_state["ext_summary"])
                        st.success(abs_summary)
                    except Exception as e:
                        st.error(f"BART failed: {e}")