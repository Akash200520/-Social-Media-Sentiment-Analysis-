# 📊 Social Media Sentiment Analysis Using NLP

An end-to-end **Natural Language Processing (NLP)** project that analyzes social media text and automatically classifies it into **Positive** or **Negative** sentiment.

The project uses the **Sentiment140 dataset**, text preprocessing, **TF-IDF vectorization**, and **Logistic Regression** to build a machine learning-based sentiment classification system. It also includes exploratory data analysis, visualization, WordCloud generation, model evaluation, and an interactive **Streamlit dashboard**.

---

## 🚀 Project Overview

Social media platforms generate enormous amounts of textual data every day. Understanding whether users express positive or negative opinions can help organizations monitor customer feedback, identify trends, and make data-driven decisions.

This project applies NLP and machine learning techniques to transform raw social media text into meaningful sentiment insights.

### 🎯 Objective

The main objectives of this project are:

* Clean and preprocess social media text.
* Perform exploratory data analysis (EDA).
* Analyze positive and negative sentiment distribution.
* Convert text into numerical features using TF-IDF.
* Train a Logistic Regression sentiment classifier.
* Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
* Generate WordCloud visualizations.
* Provide an interactive Streamlit dashboard.
* Allow users to predict sentiment for new text.
* Analyze uploaded CSV files.
* Provide downloadable analysis results.

---

## 🧠 Technologies Used

| Technology          | Purpose                         |
| ------------------- | ------------------------------- |
| Python              | Programming Language            |
| Pandas              | Data Processing                 |
| NumPy               | Numerical Computing             |
| Scikit-learn        | Machine Learning                |
| NLTK                | Natural Language Processing     |
| Matplotlib          | Data Visualization              |
| Seaborn             | Statistical Visualization       |
| WordCloud           | Text Visualization              |
| TF-IDF              | Text Feature Extraction         |
| Logistic Regression | Sentiment Classification        |
| Streamlit           | Interactive Web Dashboard       |
| Joblib              | Model Serialization             |
| Jupyter Notebook    | Data Analysis & Experimentation |

---

## 📂 Project Structure

```text
Social-Media-Sentiment-Analysis/
│
├── data/
│   └── sent140_cleaned.csv
│
├── models/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── notebooks/
│   └── social_media_sentiment_analysis.ipynb
│
├── app.py
├── train_model.py
├── preprocess.py
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** Large datasets and generated model files can be excluded from GitHub using `.gitignore` or Git LFS when appropriate.

---

# 📊 Dataset

This project uses the **Sentiment140** Twitter sentiment dataset.

The dataset contains social media posts labeled according to sentiment.

### Dataset Features

Typical information used in the project includes:

* Social media text
* Sentiment label

The cleaned dataset used in the project contains approximately **1.6 million records** before any optional sampling or reduction.

### Sentiment Labels

| Label | Meaning  |
| ----: | -------- |
|     0 | Negative |
|     1 | Positive |

---

# 🔄 Project Workflow

```text
Raw Social Media Data
        ↓
Data Cleaning
        ↓
Text Preprocessing
        ↓
Exploratory Data Analysis
        ↓
TF-IDF Feature Extraction
        ↓
Train/Test Split
        ↓
Logistic Regression
        ↓
Model Evaluation
        ↓
Sentiment Prediction
        ↓
Visualization
        ↓
Streamlit Dashboard
```

---

# 🧹 Data Preprocessing

Social media text contains noise such as URLs, mentions, special characters, and unnecessary symbols.

The preprocessing stage performs operations such as:

* Removing unnecessary characters
* Converting text to lowercase
* Removing URLs
* Removing user mentions
* Removing punctuation
* Removing unnecessary whitespace
* Stopword handling
* Text normalization

Example:

```text
Original:
"I absolutely LOVE this product!!! 😍 https://example.com"

After preprocessing:
"absolutely love product"
```

---

# 🔤 TF-IDF Feature Extraction

Machine learning algorithms cannot directly understand raw text.

Therefore, the project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to transform text into numerical features.

TF-IDF assigns higher importance to words that are useful for distinguishing between documents while reducing the importance of very common words.

This allows the machine learning model to learn relationships between words and sentiment.

---

# 🤖 Machine Learning Model

## Logistic Regression

The project uses **Logistic Regression** as the primary sentiment classification algorithm.

Logistic Regression is suitable for text classification because it performs well with high-dimensional sparse features such as TF-IDF vectors.

### Model Pipeline

```text
Preprocessed Text
        ↓
TF-IDF Vectorization
        ↓
Numerical Feature Matrix
        ↓
Logistic Regression
        ↓
Positive / Negative Prediction
```

---

# 📈 Model Performance

The trained Logistic Regression model achieved approximately:

### **77.35% Accuracy**

This satisfies the project's target of achieving more than **75% classification accuracy**.

Example evaluation results:

| Metric    | Negative | Positive |
| --------- | -------: | -------: |
| Precision |     0.79 |     0.76 |
| Recall    |     0.75 |     0.80 |
| F1-Score  |     0.77 |     0.78 |

### Overall Accuracy

```text
Accuracy: 77.35%
```

The model is therefore capable of effectively distinguishing between positive and negative social media posts.

---

# 📊 Model Evaluation

The project evaluates the model using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

### Confusion Matrix

The confusion matrix helps identify:

* Correctly predicted negative posts
* Incorrectly predicted negative posts
* Correctly predicted positive posts
* Incorrectly predicted positive posts

This provides a more detailed understanding of model performance than accuracy alone.

---

# ☁️ WordCloud Analysis

The project generates WordCloud visualizations to identify frequently occurring words in:

* Positive posts
* Negative posts

This helps understand the vocabulary associated with different sentiment categories.

Example insights:

```text
Positive Sentiment
       ↓
Frequently occurring positive words

Negative Sentiment
       ↓
Frequently occurring negative words
```

---

# 📊 Exploratory Data Analysis

The project includes several visualizations to understand the dataset.

### Visualizations include:

* Sentiment distribution
* Positive vs Negative comparison
* Word frequency analysis
* Positive WordCloud
* Negative WordCloud
* Confusion Matrix
* Model performance metrics

These visualizations make the sentiment patterns easier to understand.

---

# 🌐 Streamlit Dashboard

The project includes an interactive **Streamlit web application**.

The dashboard provides a user-friendly interface for interacting with the trained sentiment analysis model.

### Dashboard Features

#### 🏠 Home

Provides an overview of the project and its objectives.

#### 🔮 Sentiment Predictor

Users can enter their own social media text and receive a sentiment prediction.

Example:

```text
Input:
"I really enjoyed this movie!"

Prediction:
Positive 😊
```

#### 📁 CSV Analysis

Users can upload a CSV file containing social media text and analyze sentiment in bulk.

#### ☁️ WordCloud

Generates word frequency visualizations from the uploaded or analyzed data.

#### 📊 Analytics

Displays sentiment statistics and visual insights.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/your-username/social-media-sentiment-analysis.git
```

Move into the project directory:

```bash
cd social-media-sentiment-analysis
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Jupyter Notebook

Open the notebook in VS Code or Jupyter:

```text
social_media_sentiment_analysis.ipynb
```

Run the cells sequentially.

---

## Train the Model

Run:

```bash
python train_model.py
```

The trained model and vectorizer can be saved inside:

```text
models/
```

For example:

```text
models/
├── sentiment_model.pkl
└── vectorizer.pkl
```

---

# 🌐 Run the Streamlit Dashboard

Start the application using:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local address such as:

```text
http://localhost:8501
```

Open the address in your browser.

---

# 📦 Requirements

Example `requirements.txt`:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
wordcloud
streamlit
joblib
```

Install them with:

```bash
pip install -r requirements.txt
```

---

# 💡 Business Applications

Social media sentiment analysis can be used in several real-world applications.

### 🛍️ Customer Feedback

Companies can analyze customer opinions about their products and services.

### 📢 Brand Monitoring

Organizations can monitor how users perceive their brand.

### 📈 Marketing Analysis

Marketing teams can understand public reactions to campaigns.

### 🎯 Customer Experience

Negative feedback can help organizations identify areas requiring improvement.

### 📰 Social Media Monitoring

Organizations can analyze large volumes of social media posts automatically.

---

# 🔮 Future Enhancements

The project can be extended with:

* ⭐ Multi-class sentiment classification
* 🧠 BERT / Transformer-based models
* 🤖 LSTM-based sentiment classification
* 📌 Topic Modeling using LDA
* 🚀 BERTopic
* 📈 Sentiment trend analysis
* 🔮 Sentiment forecasting
* 💬 AI-powered chat assistant
* 🌍 Multilingual sentiment analysis
* ☁️ Cloud deployment
* 📊 Advanced interactive dashboards
* 🔄 Real-time social media data integration

---

# 📌 Limitations

The current model focuses primarily on **binary sentiment classification**.

Social media language can contain:

* Sarcasm
* Slang
* Emojis
* Abbreviations
* Misspellings
* Context-dependent meanings

These factors can make sentiment classification challenging.

A Transformer-based model such as BERT could potentially improve performance for more complex language patterns.

---

# 👨‍💻 Project Highlights

### ✔ NLP-Based Text Classification

Applied natural language processing techniques to social media text.

### ✔ Machine Learning

Implemented Logistic Regression for sentiment classification.

### ✔ TF-IDF

Converted text into numerical feature vectors.

### ✔ 77%+ Accuracy

Achieved approximately **77.35% accuracy** on the evaluated test set.

### ✔ Data Visualization

Created meaningful charts and WordClouds for sentiment analysis.

### ✔ Interactive Dashboard

Developed a Streamlit interface for real-time sentiment prediction and dataset analysis.

### ✔ End-to-End Pipeline

Built a complete workflow from raw data preprocessing to model deployment.

---

# 📜 License

This project is intended for educational and academic purposes.

---

# ⭐ Acknowledgements

* Sentiment140 dataset
* Python open-source community
* Scikit-learn
* NLTK
* Pandas
* Matplotlib
* Seaborn
* WordCloud
* Streamlit

---

# 👨‍💻 Author

**G. Akash**

### Project

**Social Media Sentiment Analysis Using NLP**

Built as an academic machine learning and NLP project.

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub!

