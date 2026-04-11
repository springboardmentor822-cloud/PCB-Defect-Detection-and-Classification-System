# 📚 Topic Modelling & NLP Analysis Pipeline

> A comprehensive NLP pipeline built during the **Infosys Springboard Internship** program, featuring topic modelling, sentiment analysis, and document summarization.

---

## 🗂️ Project Overview

This project implements multiple NLP pipelines for analyzing and understanding large text corpora. It includes:

- **Topic Modelling** using Non-negative Matrix Factorization (NMF)
- **Sentiment Analysis** using VADER with technical document reliability detection
- **Document Summarization** using a hybrid extractive + abstractive approach
- **Interactive Dashboard** built with Streamlit to unify these pipelines into a single web application

---

## 📁 Project Structure

```
topic-modelling/
│
├── nmf_classifier.ipynb          # Main topic modelling notebook using NMF
├── sentiment_analysis.ipynb      # Sentiment analysis pipeline with VADER
├── document_summarization.ipynb  # Hybrid document summarization pipeline
│
├── nlp_preprocessing_1.txt       # Sample document corpus (set 1)
├── nlp_preprocessing_2.txt       # Sample document corpus (set 2)
│
├── nmf_model.joblib              # Trained NMF model (serialized)
├── tfidf_vectorizer.joblib       # Trained TF-IDF vectorizer (serialized)
├── shared_vocab.joblib           # Shared vocabulary file (serialized)
│
├── streamlit_app/                # Interactive Streamlit dashboard
│   ├── app.py                    # Main app entry point
│   ├── pages_app/                # App pages (Home, Results)
│   ├── tabs/                     # Dashboard tabs (Summary, Sentiment)
│   └── analysis/                 # Analysis modules
│
└── README.md                     # Project documentation
```

---

## 🔬 Notebooks

### 1. `nmf_classifier.ipynb` — Topic Modelling with NMF

Performs unsupervised topic discovery on a document corpus (20 newsgroups dataset) using **Non-negative Matrix Factorization (NMF)**.

**Key Features:**

- TF-IDF vectorization of raw text documents
- NMF-based topic extraction and assignment
- Interactive visualizations of topic distributions
- Serialized model artifacts for reuse (`nmf_model.joblib`, `tfidf_vectorizer.joblib`, `shared_vocab.joblib`)

---

### 2. `sentiment_analysis.ipynb` — Sentiment Analysis Pipeline

Analyzes sentiment across documents using the **VADER (Valence Aware Dictionary and sEntiment Reasoner)** lexicon-based model.

**Key Features:**

- Positive / Negative / Neutral / Compound scoring
- Technical document reliability detection (flags overly neutral/ambiguous text)
- Document-level and corpus-level sentiment aggregation
- Detailed visualizations (bar charts, heatmaps, trend plots)

---

### 3. `document_summarization.ipynb` — Hybrid Summarization Pipeline

Generates concise summaries for input documents using a **hybrid extractive + abstractive** strategy.

**Key Features:**

- **Extractive**: Consensus scoring across TextRank, LSA, and LexRank algorithms
- **Abstractive**: Transformer-based summarization using Facebook's **BART** model (`facebook/bart-large-cnn`)
- Keyword extraction using RAKE and TF-IDF
- Multi-panel dashboard visualization for summary reporting

---

## 🖥️ Streamlit Interactive Dashboard

The project includes a beautifully styled, dark-themed Streamlit web application that provides a user-friendly interface for the NLP pipelines.

**Key Features:**

- **Interactive File Upload & Text Input**: Easily test the NLP pipelines on new custom documents.
- **Session-State Routing**: Seamless navigation between the Home and Results pages.
- **Comprehensive Reporting**: The results are split into dedicated tabs (e.g., Summary, Sentiment) for focused analysis.
- **Modern UI Design**: Custom global CSS featuring modern typography (Inter, Space Grotesk) and a consistent dark mode aesthetic.

---

## 🛠️ Technologies Used

| Category          | Libraries / Tools                               |
| ----------------- | ----------------------------------------------- |
| Language          | Python 3.x                                      |
| NLP               | NLTK, spaCy, Gensim, Transformers (HuggingFace) |
| Topic Modelling   | scikit-learn (NMF, TF-IDF)                      |
| Sentiment         | VADER (vaderSentiment)                          |
| Summarization     | sumy, BART (`facebook/bart-large-cnn`)        |
| Visualization     | Matplotlib, Seaborn, WordCloud                  |
| Web Framework     | Streamlit                                       |
| Model Persistence | joblib                                          |
| Notebook          | Jupyter Notebook                                |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with Jupyter Notebook.

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd topic-modelling

# Install required dependencies
pip install numpy pandas scikit-learn nltk gensim transformers
pip install vaderSentiment sumy rake-nltk matplotlib seaborn wordcloud
pip install joblib jupyter streamlit
```

### Running the Notebooks

```bash
# Launch Jupyter Notebook
jupyter notebook
```

Then open any of the three notebooks:

- `nmf_classifier.ipynb`
- `sentiment_analysis.ipynb`
- `document_summarization.ipynb`

### Running the Streamlit App

To launch the interactive dashboard, navigate to the `streamlit_app` folder and run:

```bash
streamlit run streamlit_app/app.py
```

### Using Pre-trained Models

The serialized `.joblib` files allow you to skip retraining:

```python
import joblib

# Load pre-trained artifacts
nmf_model = joblib.load('nmf_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
vocab = joblib.load('shared_vocab.joblib')
```

---

## 📊 Sample Input Format

The `.txt` files (`nlp_preprocessing_1.txt`, `nlp_preprocessing_2.txt`) contain pre-processed text documents used as input to the pipelines. Each line or block represents a separate document.
