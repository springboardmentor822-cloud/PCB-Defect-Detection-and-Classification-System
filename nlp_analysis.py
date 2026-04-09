import json
import re
from textblob import TextBlob

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Topic detection using simple keyword rules
def detect_topic(text):
    text = text.lower()
    if any(keyword in text for keyword in ["achievement", "success", "accomplish", "goal", "milestone", "win", "victory"]):
        return "Achievement"
    if any(keyword in text for keyword in ["product", "review", "feedback", "quality", "purchase", "rating"]):
        return "Product Review"
    if any(keyword in text for keyword in ["technology", "tech", "innovation", "startup", "ai", "software", "hardware", "gadget"]):
        return "Technology"
    return "General"

# Simple extractive summarization using sentence scoring
def summarize_text(text, max_sentences=1):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= max_sentences:
        return text.strip()

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
        return sentences[0]

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

# Generate actionable insight based on topic and sentiment
def generate_recommendation(topic, polarity):
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

def analyze_text(text):
    """
    Analyzes the input text and returns a JSON object with Topic, Sentiment, Summary, and Recommendation.

    Args:
        text (str): The input text to analyze

    Returns:
        str: A JSON string containing the analysis results
    """
    if not text or not text.strip():
        return json.dumps({
            "Topic": "General",
            "Sentiment": "Mixed",
            "Summary": "",
            "Recommendation": "Please provide valid text for analysis."
        })

    # Clean the text
    cleaned_text = clean_text(text)

    # Perform sentiment analysis
    analysis = TextBlob(cleaned_text)
    polarity = analysis.sentiment.polarity

    # Determine sentiment label
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Mixed"

    # Detect topic
    topic = detect_topic(text)

    # Generate summary
    summary = summarize_text(text, max_sentences=1)

    # Generate recommendation
    recommendation = generate_recommendation(topic, polarity)

    # Create result dictionary
    result = {
        "Topic": topic,
        "Sentiment": sentiment,
        "Summary": summary,
        "Recommendation": recommendation
    }

    # Return as JSON string
    return json.dumps(result, indent=2, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    sample_text = "I just achieved my goal of running a marathon! It was an amazing success and I'm so proud of myself."
    result = analyze_text(sample_text)
    print(result)