## Social Media Sentiment Analysis

An AI-powered Social Media Sentiment Analysis project using **TF-IDF** and **Logistic Regression** to classify social media posts as Positive or Negative.

##  Features

- Sentiment Analysis
- Logistic Regression Model
- TF-IDF Text Vectorization
- Single Text Sentiment Prediction
- CSV Batch Analysis
- Word Cloud
- Model Accuracy & Confusion Matrix
- Interactive Visualizations
- Streamlit Dashboard
- Download Analysis Results

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- WordCloud
- Streamlit
- Joblib
- Jupyter Notebook

## Project Structure

```text
PROJECT 7TH SEM/
│
├── models/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── app.py
├── social media analysis.ipynb
├── sent140_cleaned.csv
├── .gitignore
├── requirements.txt
└── README.md

## model performance

Algorithm: Logistic Regression
Feature Extraction: TF-IDF
Accuracy: 77.35%

▶️ Run the Dashboard

Install the required libraries:
pip install -r requirements.txt

Run the Streamlit application:
python -m streamlit run app.py

📂 Dataset

The project uses the Sentiment140 Twitter sentiment dataset.

The dataset contains approximately 1.6 million tweets with Positive and Negative sentiment labels.