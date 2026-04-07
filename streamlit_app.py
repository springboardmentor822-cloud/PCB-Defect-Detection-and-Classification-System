# streamlit run streamlit_app.py
import streamlit as st
import re
from textblob import TextBlob

DEFAULT_SUMMARY_TEXT = (
    "Streamlit makes it easy to build interactive web apps in pure Python. "
    "You can paste any text here to summarize, and adjust the output length using the slider. "
    "This app works well for short passages around 30 words up to longer blocks of 100+ words."
)

# ---------- Text Cleaning ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------- Topic Detection ----------
def detect_topic(text):
    text = text.lower()
    if any(keyword in text for keyword in ["achievement", "success", "accomplish", "goal", "milestone", "win", "victory"]):
        return "Achievement"
    if any(keyword in text for keyword in ["product", "review", "feedback", "quality", "purchase", "rating"]):
        return "Product Review"
    if any(keyword in text for keyword in ["technology", "tech", "innovation", "startup", "ai", "software", "hardware", "gadget"]):
        return "Technology"
    return "General"

# ---------- Insight / Recommendation ----------
def generate_insight(topic, polarity):
    if polarity > 0:
        if topic == "Achievement":
            return "Celebrate this success and share it to motivate others."
        elif topic == "Product Review":
            return "Highlight the strong product features and collect more positive feedback."
        elif topic == "Technology":
            return "Highlight this trending innovation and share it with your audience."
        else:
            return "Use this positive sentiment to guide your next action."
    elif polarity < 0:
        if topic == "Product Review":
            return "Recommend product improvements and investigate customer pain points in the review."
        elif topic == "Achievement":
            return "Address the concerns and turn this into a learning opportunity."
        elif topic == "Technology":
            return "Monitor the developing story and look for opportunities to improve the technology narrative."
        else:
            return "Address the negative feedback and work on improvements."
    else:
        return "Use these insights to guide your next action based on topic and sentiment."

# ---------- Extractive Summarization ----------
def summarize_text(text, max_sentences=2):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= max_sentences:
        return ' '.join(sentences).strip()

    words = re.findall(r'\w+', text.lower())
    stopwords = set([
        'the','and','is','in','it','of','to','a','that','this','for','with','on','as','are','was','were','be','by','or','an','at','from','which','but','not','have','has','had','they','you','we','their','its','our','can','will','would','should','could','i','he','she','them','his','her','my','your','so','if','about','what','when','where','who','how','do','does','did'
    ])
    frequencies = {}
    for word in words:
        if word in stopwords:
            continue
        frequencies[word] = frequencies.get(word, 0) + 1

    if not frequencies:
        return ' '.join(sentences[:max_sentences])

    max_freq = max(frequencies.values())
    for word in frequencies:
        frequencies[word] /= max_freq

    sentence_scores = []
    for index, sentence in enumerate(sentences):
        sentence_words = re.findall(r'\w+', sentence.lower())
        score = sum(frequencies.get(word, 0) for word in sentence_words)
        sentence_scores.append((score, index, sentence))

    selected = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:max_sentences]
    selected = sorted(selected, key=lambda x: x[1])
    selected_sentences = [sentence for _, _, sentence in selected]
    return ' '.join(selected_sentences).strip()

# ---------- Streamlit App ----------
def main():
    st.title("📝 NLP Text Analysis Tool")
    st.markdown("Analyze sentiment or summarize your text!")

    # Tabs
    tab1, tab2 = st.tabs(["🎯 Sentiment Analysis", "📄 Text Summarization"])

    # ---------- Sentiment Analysis Tab ----------
    with tab1:
        st.header("Sentiment Analysis")
        user_input = st.text_area("Enter your text:", height=120, placeholder="Type your text here...", key="sentiment_input")

        if st.button("Analyze Sentiment", key="analyze_sentiment"):
            if user_input.strip():
                cleaned_text = clean_text(user_input)
                analysis = TextBlob(user_input)
                polarity = analysis.sentiment.polarity
                subjectivity = analysis.sentiment.subjectivity
                topic = detect_topic(user_input)
                insight = generate_insight(topic, polarity)

                # Sentiment label
                if polarity > 0:
                    sentiment_label = "Positive 😊"
                    color = "green"
                elif polarity < 0:
                    sentiment_label = "Negative 😞"
                    color = "red"
                else:
                    sentiment_label = "Neutral 😐"
                    color = "blue"

                # Summary
                text_summary = summarize_text(user_input, max_sentences=1)

                # Result dictionary
                result = {
                    "Topic": topic,
                    "Sentiment": sentiment_label,
                    "Summary": text_summary,
                    "Recommendation": insight
                }

                # Display Dashboard
                st.subheader("📊 Analysis Results")
                st.markdown("**Topic:**")
                st.write(topic)
                st.markdown("**Sentiment:**")
                st.markdown(f"<span style='color:{color};'>{sentiment_label}</span>", unsafe_allow_html=True)
                st.markdown("**Summary:**")
                st.write(text_summary)
                st.markdown("**Recommendation / Insight:**")
                st.write(insight)

                # Detailed Metrics
                st.markdown("---")
                st.subheader("🎯 Detailed Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Polarity Score", f"{polarity:.3f}")
                with col2:
                    st.metric("Subjectivity", f"{subjectivity:.3f}")

            else:
                st.warning("Please enter some text to analyze!")

    # ---------- Text Summarization Tab ----------
    with tab2:
        st.header("Text Summarization")
        summary_input = st.text_area("Enter your text:", value=DEFAULT_SUMMARY_TEXT, height=200, key="summary_input")
        summary_length = st.slider("Summary length (sentences)", min_value=1, max_value=5, value=3)

        if st.button("Generate Summary", key="generate_summary"):
            if summary_input.strip():
                word_count = len(summary_input.split())
                if word_count < 30:
                    st.warning("Please enter a longer text (at least 30 words) for better summarization.")
                elif word_count > 1024:
                    st.warning("Text is too long. Please shorten it to under 1024 words.")
                else:
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(summary_input, max_sentences=summary_length)
                        st.subheader("📄 Summary")
                        st.write(summary)

                        st.markdown("---")
                        st.subheader("📊 Original Text Stats")
                        st.metric("Word Count", word_count)
                        st.metric("Summary Word Count", len(summary.split()))
            else:
                st.warning("Please enter some text to summarize!")

    st.markdown("---")
    st.markdown("*Built with Streamlit and TextBlob*")

if __name__ == "__main__":
    main()