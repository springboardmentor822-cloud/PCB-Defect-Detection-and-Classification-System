# streamlit_app.py
import streamlit as st
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Dynamic Text Analysis", layout="wide")

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------- SUMMARIZATION ----------------
def summarize_text(text, max_sentences=3):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sentences[:max_sentences]).strip()

# ---------------- SENTIMENT ----------------
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    subjectivity = TextBlob(text).sentiment.subjectivity
    if polarity > 0:
        return "Positive 😊", polarity, subjectivity
    elif polarity < 0:
        return "Negative 😞", polarity, subjectivity
    else:
        return "Neutral 😐", polarity, subjectivity

# ---------------- TOPIC DETECTION ----------------
def detect_topic(text):
    text = text.lower()
    if any(k in text for k in ["achievement", "success", "accomplish", "goal", "milestone", "win", "victory"]):
        return "Achievement"
    if any(k in text for k in ["product", "review", "feedback", "quality", "purchase", "rating"]):
        return "Product Review"
    if any(k in text for k in ["technology", "tech", "innovation", "startup", "ai", "software", "hardware", "gadget"]):
        return "Technology"
    return "General"

def generate_insight(topic, polarity):
    if polarity > 0:
        return {
            "Achievement": "Celebrate this success and share it to motivate others.",
            "Product Review": "Highlight strong product features and collect more positive feedback.",
            "Technology": "Highlight this trending innovation and share it with your audience.",
        }.get(topic, "Use this positive sentiment to guide your next action.")
    elif polarity < 0:
        return {
            "Achievement": "Address the concerns and turn this into a learning opportunity.",
            "Product Review": "Recommend product improvements and investigate customer pain points.",
            "Technology": "Monitor the developing story and improve the technology narrative.",
        }.get(topic, "Address the negative feedback and work on improvements.")
    else:
        return "Use these insights to guide your next action based on topic and sentiment."

# ---------------- TOPIC MODELING ----------------
def create_docs(text):
    cleaned = clean_text(text)
    docs = re.split(r'[.!?]', cleaned)
    docs = [d.strip() for d in docs if len(d.split()) > 5]
    if len(docs) < 3:
        words = cleaned.split()
        chunk_size = max(30, len(words)//3)
        docs = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    if len(docs) < 2:
        docs = [cleaned, cleaned]
    return docs

def topic_model(text):
    docs = create_docs(text)
    count_vec = CountVectorizer(stop_words='english', max_df=1.0, min_df=1)
    tfidf_vec = TfidfVectorizer(stop_words='english', max_df=1.0, min_df=1)
    X_count = count_vec.fit_transform(docs)
    X_tfidf = tfidf_vec.fit_transform(docs)
    n_topics = min(3, len(docs))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X_count)
    nmf = NMF(n_components=n_topics, random_state=42, init='nndsvd', max_iter=400)
    nmf.fit(X_tfidf)
    return lda, nmf, count_vec, tfidf_vec, X_count

def display_topics(model, vectorizer, title):
    words = vectorizer.get_feature_names_out()
    st.subheader(f"📌 {title} Topics")
    for i, topic in enumerate(model.components_):
        top_idx = topic.argsort()[-6:][::-1]
        top_words = [words[j] for j in top_idx]
        st.write(f"👉 Topic {i+1}: {', '.join(top_words)}")

def plot_topics(model, vectorizer, title):
    words = vectorizer.get_feature_names_out()
    n_topics = len(model.components_)
    fig, axes = plt.subplots(n_topics, 1, figsize=(5, 2*n_topics))
    if n_topics == 1:
        axes = [axes]
    for i, topic in enumerate(model.components_):
        top_idx = topic.argsort()[-8:][::-1]
        top_words = [words[j] for j in top_idx]
        weights = topic[top_idx]
        axes[i].barh(top_words, weights)
        axes[i].set_title(f"{title} Topic {i+1}")
        axes[i].tick_params(axis='y', labelsize=8)
    plt.tight_layout(pad=1.0)
    st.pyplot(fig)

# ---------------- STREAMLIT UI ----------------
st.title("🚀 Dynamic Text Analysis Dashboard")

# EMPTY TEXT AREA for user input
text = st.text_area("✍️ Enter your text here...", height=200, value="")

# ---------------- STATS ----------------
if text.strip():
    col1, col2, col3 = st.columns(3)
    col1.metric("Words", len(text.split()))
    col2.metric("Sentences", len(re.findall(r'[.!?]', text)))
    col3.metric("Unique Words", len(set(text.split())))

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🧠 Topic Modeling", "😊 Sentiment Analysis", "📄 Summary"])

# Topic Modeling
with tab1:
    st.subheader("Topic Modeling (LDA + NMF)")
    if st.button("Generate Topics", key="topic_btn"):
        if not text.strip():
            st.warning("Please enter text")
        else:
            lda, nmf, count_vec, tfidf_vec, X = topic_model(text)
            st.markdown("### 📊 Model Scores")
            st.write("🔹 LDA Perplexity:", round(lda.perplexity(X), 2))
            st.write("🔹 NMF Reconstruction Error:", round(nmf.reconstruction_err_, 4))
            display_topics(lda, count_vec, "LDA")
            display_topics(nmf, tfidf_vec, "NMF")
            st.markdown("### 📈 LDA Visualization")
            plot_topics(lda, count_vec, "LDA")
            st.markdown("### 📈 NMF Visualization")
            plot_topics(nmf, tfidf_vec, "NMF")

# Sentiment Analysis
with tab2:
    st.subheader("Sentiment Analysis")
    if st.button("Analyze Sentiment", key="sent_btn"):
        if not text.strip():
            st.warning("Enter text")
        else:
            label, polarity, subjectivity = analyze_sentiment(text)
            topic = detect_topic(text)
            insight = generate_insight(topic, polarity)
            st.markdown(f"**Topic:** {topic}")
            st.subheader(label)
            st.write("Polarity Score:", round(polarity,3))
            st.write("Subjectivity:", round(subjectivity,3))
            st.markdown("**Insight / Recommendation:**")
            st.write(insight)
            st.markdown("**Summary (1 sentence):**")
            st.write(summarize_text(text, max_sentences=1))

# Summary
with tab3:
    st.subheader("Document Summary")
    summary_length = st.slider("Summary length (sentences)", 1, 5, 3)
    if st.button("Generate Summary", key="sum_btn"):
        if not text.strip():
            st.warning("Enter text")
        else:
            summary = summarize_text(text, max_sentences=summary_length)
            st.subheader("📄 Summary")
            st.write(summary)

# Footer
st.markdown("---")
st.markdown("✅ Built with Streamlit | LDA + NMF + Sentiment + Summary")