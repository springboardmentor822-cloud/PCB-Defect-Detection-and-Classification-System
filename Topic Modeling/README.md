# NarrativeNexus: Dynamic Text Analysis Platform

NarrativeNexus is an internship project built as a notebook-based text analysis platform. It accepts pasted text or uploaded `.txt`, `.csv`, and `.docx` files, then converts the text into themes, topic labels, sentiment results, concise summaries, recommendations, and visual reports.

The main implementation is in `Internship___project.ipynb`. This README is provided outside the notebook so the project can be reviewed, run, and tested without first reading every notebook cell.

## Project Objective

The goal of NarrativeNexus is to help users understand large text inputs quickly. The platform processes articles, reports, customer feedback, social media-style text, or user-generated content and returns:

- Cleaned and preprocessed text.
- Key themes and topics using topic modeling.
- BBC News-trained topic/category signals when the BBC dataset is available.
- Sentiment analysis for each text segment.
- Extractive summaries of the main ideas.
- Actionable recommendations based on the analysis.
- Word cloud, sentiment charts, topic distribution charts, heatmaps, and a downloadable Markdown report.

## Files

| File | Purpose |
| --- | --- |
| `Internship___project.ipynb` | Main notebook containing setup, preprocessing, model training, analysis functions, quick test, and Gradio web interface. |
| `README.md` | External project documentation with run, test, and completion details. |

## Technology Used

- Python 3
- Jupyter Notebook or Google Colab
- Gradio for the web interface
- Pandas and NumPy for data handling
- NLTK for stop words, lemmatization, and VADER sentiment analysis
- Scikit-learn for TF-IDF, NMF, LDA, and Naive Bayes classification
- Matplotlib and WordCloud for visualizations
- python-docx for `.docx` file input
- KaggleHub for optional BBC News dataset download

## How to Run in Google Colab

1. Open Google Colab.
2. Upload `Internship___project.ipynb`.
3. Run the notebook from top to bottom.
4. The first code cell installs the required packages:

   ```python
   !pip -q install gradio wordcloud python-docx matplotlib pandas numpy nltk scikit-learn kagglehub
   ```

5. When the Gradio cell runs, open the generated Gradio link.
6. Paste text or upload a supported file.
7. Click **Analyze** to generate insights, tables, visualizations, and a downloadable report.

## How to Run Locally

1. Open a terminal in the project folder.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install notebook gradio wordcloud python-docx matplotlib pandas numpy nltk scikit-learn kagglehub
   ```

4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `Internship___project.ipynb`.
6. Run all cells in order.
7. Use the Gradio interface that starts from the final cell.

If running locally, the generated report is saved as `narrativenexus_report.md` in the project folder. In Colab, it is saved under `/content/narrativenexus_report.md`.

## Optional BBC News Dataset Setup

The notebook can work without a local BBC News dataset because it includes a fallback topic-modeling path for user input. For the full BBC News-trained behavior, provide the dataset in one of these ways:

- Place a supported BBC News CSV beside the notebook, such as `BBC News Train.csv`, `bbc-news-data.csv`, `bbc_news.csv`, or `bbc-news.csv`.
- Place a BBC News folder structure beside the notebook, such as `bbc` or `BBC News Summary/News Articles`.
- Set the environment variable `BBC_NEWS_DATASET_PATH` to the dataset path.
- Allow the notebook to try downloading the KaggleHub dataset `hgultekin/bbcnewsarchive`.

When the dataset is available, the notebook trains a TF-IDF and NMF model on BBC News articles and optionally trains a Naive Bayes category classifier if labels are present.

## How to Test the Project

### Quick Notebook Test

Run the section named **Quick Notebook Test** in the notebook. It analyzes a sample customer feedback paragraph and displays:

- Topic keywords and distribution.
- Sentiment label and compound score for the text segment.
- BBC category predictions when the BBC model is available.
- Generated insight Markdown with summary and recommendations.

The test is successful if the cell completes without errors and shows tables plus an insight summary.

### Manual Gradio Test

1. Run the final Gradio interface cell.
2. Paste at least 20 words of text into the text box.
3. Click **Analyze**.
4. Confirm that the interface returns:

- Expanded insights.
- Topic table.
- Sentiment table.
- Topic-wise sentiment matrix and statistics.
- Word cloud.
- Sentiment bar chart and pie chart.
- Topic chart.
- Downloadable Markdown report.

### File Upload Test

Test each supported input type:

- Upload a `.txt` file containing plain text.
- Upload a `.csv` file containing at least one text column. If needed, enter the column name in the CSV text column field.
- Upload a `.docx` file containing paragraph text.

Each test should produce analysis outputs similar to the pasted-text test.

### Error Handling Test

Try submitting an empty input or fewer than 20 words. The app should return a clear message asking for enough text to extract meaningful themes.

## How the Internship Project Was Completed

This project completes the internship roadmap by implementing the required platform components inside the notebook:

- **Data collection and input handling:** The Gradio interface accepts pasted text and uploaded `.txt`, `.csv`, and `.docx` files.
- **Data preprocessing:** The pipeline normalizes whitespace, removes noise, tokenizes text, removes stop words, and applies lemmatization.
- **Topic modeling:** The project uses TF-IDF with NMF for topic extraction and also includes LDA comparison topics in the analysis pipeline.
- **Sentiment analysis:** NLTK VADER is used to classify text segments as positive, neutral, or negative and calculate sentiment scores.
- **Summarization:** The notebook includes extractive summarization that selects important sentences from the input.
- **Actionable insights:** The recommendation logic converts topic and sentiment results into practical next steps for the user.
- **Visualization and reporting:** The dashboard includes word clouds, sentiment charts, topic charts, topic-sentiment views, category-topic diagnostics, and a downloadable Markdown report.
- **User interface:** Gradio provides an interactive web interface so non-technical users can run the analysis without editing code.
- **Documentation:** The notebook contains methodology and architecture notes, and this README documents how to run, test, and evaluate the final project.

## Expected Output

After running the platform, the user should receive:

- A concise executive summary.
- Dominant themes and topic keywords.
- Sentiment distribution and interpretation.
- BBC News category signals when available.
- Topic-wise sentiment statistics.
- Actionable recommendations.
- Visual charts and word cloud.
- A Markdown report that can be downloaded and submitted with the project.

## Conclusion

NarrativeNexus delivers a complete dynamic text analysis platform for the internship project. It combines preprocessing, topic modeling, sentiment analysis, summarization, recommendations, visualization, and reporting into one runnable notebook with an interactive Gradio interface.
