# 📰 NarrativeNexus — Dynamic Topic Analysis Platform

> An end-to-end NLP pipeline for topic modelling, sentiment analysis, and document summarization — featuring a fully interactive React dashboard and a Colab-ready Jupyter notebook.

Built during the **Infosys Springboard Internship Program**.

---

## 🗂️ Project Structure

```
NarrativeNexus/
│
├── NarrativeNexus.ipynb           # Main Colab notebook (full pipeline)
│
└── text-analyzer-pro/                   # React web dashboard (Vite + TypeScript)
    ├── src/
    │   ├── pages/
    │   │   ├── Dashboard.tsx            # Overview + coherence curve
    │   │   ├── DataInput.tsx            # File upload, paste, datasets, URL fetch
    │   │   ├── Preprocessing.tsx        # Tokenization + bigram pipeline viewer
    │   │   ├── TopicModeling.tsx        # Topic cards + keyword table
    │   │   ├── Sentiment.tsx            # Sentiment pie + per-topic breakdown
    │   │   ├── Visualizations.tsx       # Word cloud, treemap, scatter, radar
    │   │   ├── Summarization.tsx        # Summaries + actionable insights
    │   │   └── History.tsx              # Session analysis history
    │   ├── lib/
    │   │   ├── analyzeText.ts           # Client-side NLP engine
    │   │   └── types.ts                 # TypeScript interfaces
    │   └── hooks/
    │       └── useAnalysis.tsx          # Global analysis context
    ├── package.json
    └── vite.config.ts
```

---

## 🔬 Notebook Pipeline — `NarrativeNexus.ipynb`

A nine-step Colab notebook that processes the full **20 Newsgroups** corpus (~17,886 documents across 20 categories).

### Pipeline Steps

| Step | Description |
|------|-------------|
| **1** | Install dependencies & import libraries |
| **2** | Load full 20 Newsgroups dataset (train + test) |
| **3** | Text preprocessing — lowercase, URL/email removal, lemmatization, stop word filtering |
| **3A** | Bigram & trigram detection using `gensim.models.Phrases` |
| **3B** | Sub-topic splitting: expands 10 large categories into 2 sub-groups each (20 → 30 groups) |
| **4** | Topic modelling with **LDA** (30 topics) + auto-labelling via **Flan-T5** |
| **4B** | Topic modelling with **NMF** — side-by-side coherence comparison vs LDA |
| **4C** | Dominant topic assignment per document using LDA probabilities |
| **5** | Sentiment analysis using **TextBlob** (polarity + subjectivity per document) |
| **6** | Extractive summarization per category (frequency-weighted sentence scoring) |
| **7** | Visualization dashboard (word clouds, heatmaps, scatter plots, sentiment bars) |
| **8** | Final printed report with dataset stats, topic list, and sentiment ranking |
| **9** | Export results as CSV files + bundled ZIP download |

### Key Features

- **Fully automatic topic labels** — `google/flan-t5-base` generates 2–4 word noun-phrase labels from top keywords. No manual rules, no candidate lists.
- **Bigram/trigram detection** captures compound terms like `space_shuttle`, `gun_control`, `hard_drive`.
- **Sub-topic splitting** uses keyword scoring to expand categories (e.g. `sci.space` → `sci.space.nasa` + `sci.space.astronomy`).
- **LDA vs NMF coherence comparison** with a visual bar chart and highlighted winner table.
- **pyLDAvis** interactive topic browser embedded in the notebook.

### Output Files

| File | Description |
|------|-------------|
| `topic_analysis_results.csv` | Per-document: group, dominant topic, label, probability, sentiment |
| `lda_topic_labels.csv` | LDA topic table with generated labels + top keywords |
| `nmf_topic_labels.csv` | NMF topic table with generated labels + top keywords |
| `topic_analysis_dashboard.png` | 9-panel visualization dashboard |
| `coherence_curve.png` | Coherence score vs number of topics |
| `lda_vs_nmf_coherence.png` | LDA vs NMF comparison bar chart |
| `DynamicTopicAnalysis_outputs.zip` | All outputs bundled for download |

---

## 🖥️ Web Dashboard — `text-analyzer-pro`

A dark-themed, fully client-side React app that brings the same NLP pipeline into an interactive browser UI. No backend required — all analysis runs in the browser.

### Pages

| Page | What it shows |
|------|---------------|
| **Dashboard** | Stat cards, sentiment pie, coherence curve, topic weight bar chart, pipeline overview |
| **Data Input** | File upload (.txt/.csv/.md), paste text, 3 preset datasets, live URL fetcher with HTML stripping, real-time word/bigram/sentiment preview |
| **Preprocessing** | Token counts, stop word stats, before/after text diff, top token chips, bigram bar chart |
| **Topic Modeling** | Per-topic keyword cards with sentiment bars, weight chart, full topic summary table |
| **Sentiment Analysis** | Overall pie chart, per-topic stacked bar, sentiment cards for each topic |
| **Visualizations** | Word frequency bar, treemap, radar chart, bigram chart, cumulative area chart, word-length scatter, word cloud |
| **Summarization** | Overall summary, per-topic text summaries, actionable insights (priority-tagged) |
| **History** | Session history with topic count, sentiment breakdown, reload and delete controls |

### Tech Stack

| Category | Library |
|----------|---------|
| Framework | React 18 + TypeScript |
| Build tool | Vite |
| Styling | Tailwind CSS + shadcn/ui |
| Charts | Recharts |
| Routing | React Router v6 |
| State | React Context + useState |
| Fonts | Inter + JetBrains Mono |
| Animations | Framer Motion |

---

## 🛠️ Technologies Used (Notebook)

| Category | Libraries |
|----------|-----------|
| Language | Python 3.x |
| NLP | NLTK, TextBlob, Gensim |
| Topic Modelling | scikit-learn (NMF, TF-IDF), Gensim (LDA) |
| Auto-labelling | Transformers — `google/flan-t5-base` |
| Sentiment | TextBlob |
| Summarization | NLTK frequency scoring (extractive) |
| Visualization | Matplotlib, Seaborn, WordCloud, pyLDAvis |
| Notebook | Jupyter / Google Colab |

---

## 🚀 Getting Started

### Running the Notebook (Google Colab — recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `NarrativeNexus.ipynb`
3. Run all cells (`Runtime → Run all`)
4. Step 1 installs all dependencies automatically
5. Results download automatically at Step 9

### Running the Notebook Locally

```bash
# Install dependencies
pip install numpy pandas scikit-learn nltk gensim transformers
pip install textblob vaderSentiment wordcloud pyLDAvis
pip install matplotlib seaborn torch sentencepiece jupyter

# Launch
jupyter notebook NarrativeNexus.ipynb
```

> **Note:** Flan-T5 label generation runs on CPU by default. Using a GPU runtime in Colab significantly speeds up Step 4.

### Running the Web Dashboard

```bash
cd text-analyzer-pro

# Install dependencies
npm install

# Start dev server
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173) in your browser.

```bash
# Production build
npm run build
npm run preview
```

---

## 📊 Sample Results

### Dataset Overview
- **Total documents:** ~17,886 (train + test combined)
- **Original categories:** 20
- **Final groups after sub-topic split:** 30
- **Total tokens (after preprocessing):** ~2.5M

### LDA Coherence Scores (c_v)

| Topics | Score |
|--------|-------|
| 10 | 0.539 |
| 20 | 0.570 |
| 30 | 0.563 |
| 40 | 0.531 |

### Example Auto-Generated Topic Labels (Flan-T5)

| Topic | Top Keywords | Generated Label |
|-------|-------------|-----------------|
| 1 | space, shuttle, nasa, orbit | NASA space missions |
| 2 | israel, arab, jewish, state | Israeli Arab conflict |
| 3 | drive, scsi, disk, controller | SCSI disk drives |
| 4 | christian, bible, jesus, church | Christian Bible study |
| 5 | drug, patient, doctor, disease | medical treatment |

---

## 📁 Sub-topic Splitting Rules

10 of the 20 original categories are split into 2 sub-groups each using keyword scoring:

| Original Category | Sub-topic A | Sub-topic B |
|------------------|-------------|-------------|
| `sci.space` | `sci.space.nasa` | `sci.space.astronomy` |
| `talk.politics.guns` | `politics.guns.control` | `politics.guns.rights` |
| `talk.politics.mideast` | `politics.mideast.israel` | `politics.mideast.arab` |
| `comp.graphics` | `comp.graphics.software` | `comp.graphics.hardware` |
| `rec.sport.hockey` | `sport.hockey.nhl` | `sport.hockey.general` |
| `sci.med` | `sci.med.disease` | `sci.med.treatment` |
| `comp.sys.ibm.pc.hardware` | `comp.hardware.cpu` | `comp.hardware.drives` |
| `soc.religion.christian` | `religion.christian.bible` | `religion.christian.church` |
| `rec.autos` | `rec.autos.buying` | `rec.autos.repair` |
| `sci.electronics` | `sci.electronics.circuit` | `sci.electronics.devices` |

---

## 🤝 Acknowledgements

- **Infosys Springboard** — Internship program and mentorship
- **20 Newsgroups dataset** — Ken Lang / scikit-learn
- **google/flan-t5-base** — Google Research via HuggingFace Transformers
- **pyLDAvis** — Interactive LDA visualization
- **shadcn/ui** — React component library

---

## 👩‍💻 Author
Krutika Pawar  
Infosys Springboard Internship Project

---

## 📄 License

This project was built as part of an academic internship. Feel free to reference or extend it for educational purposes.
