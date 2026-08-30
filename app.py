import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from collections import Counter
import re
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================
# File paths
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError:
    st.error(
        "Model files not found. Please make sure these files exist:\n\n"
        "models/sentiment_model.pkl\n"
        "models/vectorizer.pkl"
    )
    st.stop()


# =========================================================
# LOAD DATASET
# =========================================================

 HEAD
 HEAD
DATA_FILE = "data/sent140_streamlit.csv"

DATASET_PATH = "data/sent140_streamlit.csv"
 2ba1ff2 (Add Streamlit dataset)

DATASET_PATH = "sent140_streamlit.csv"
 fc091b7 (Fix Streamlit dataset path)

try:
    df = pd.read_csv("data/sent140_streamlit.csv")

    # Make sure text has no missing values
    if "text" in df.columns:
        df["text"] = df["text"].fillna("").astype(str)

except FileNotFoundError:
    st.error(
        "Please ensure sent140_streamlit.csv exists."
    )
    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title("📊 Social Media Sentiment Analysis")

st.write(
    "AI-powered Social Media Sentiment Analysis "
    "using TF-IDF and Logistic Regression."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "🏠 Home",
        "🔍 Sentiment Predictor",
        "📂 CSV Analysis",
        "📈 Model Performance",
        "☁️ Word Cloud",
        "📊 Visualizations"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.header("🏠 Welcome")

    st.write(
        """
        This application performs sentiment analysis on social media
        text using Machine Learning.
        
        The system uses the Sentiment140 dataset and a
        Logistic Regression model with TF-IDF feature extraction.
        """
    )

    st.subheader("📌 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dataset",
            "Sentiment140"
        )

    with col2:
        st.metric(
            "Records",
            f"{len(df):,}"
        )

    with col3:
        st.metric(
            "Algorithm",
            "Logistic Regression"
        )

    with col4:
        st.metric(
            "Feature Extraction",
            "TF-IDF"
        )

    st.divider()

    st.subheader("🚀 Project Features")

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            """
            ✅ Sentiment Prediction  
            ✅ CSV Batch Analysis  
            ✅ Model Accuracy  
            ✅ Confusion Matrix  
            """
        )

    with col2:
        st.write(
            """
            ✅ Classification Report  
            ✅ Word Cloud  
            ✅ Dataset Visualizations  
            ✅ Download Results  
            """
        )

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =========================================================
# SENTIMENT PREDICTOR
# =========================================================

elif page == "🔍 Sentiment Predictor":

    st.header("🔍 Sentiment Predictor")

    st.write(
        "Enter a social media post and the AI model will "
        "predict its sentiment."
    )

    text = st.text_area(
        "Enter your tweet or social media text:",
        height=150,
        placeholder="Example: I really love this product!"
    )

    if st.button("🔍 Analyze Sentiment"):

        if text.strip() == "":
            st.warning("Please enter some text.")

        else:

            text_vector = vectorizer.transform([text])

            prediction = model.predict(text_vector)[0]

            # Handle Sentiment140 labels
            if prediction == 4:
                sentiment = "Positive"
            else:
                sentiment = "Negative"

            st.divider()

            if sentiment == "Positive":

                st.success(
                    "😊 Positive Sentiment"
                )

            else:

                st.error(
                    "😞 Negative Sentiment"
                )

            # Probability if available
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    text_vector
                )[0]

                confidence = max(probabilities) * 100

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )


# =========================================================
# CSV ANALYSIS
# =========================================================

elif page == "📂 CSV Analysis":

    st.header("📂 CSV Batch Analysis")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.success(
            "CSV uploaded successfully!"
        )

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

        # Find text column
        possible_columns = [
            "text",
            "tweet",
            "content",
            "Text",
            "Tweet"
        ]

        text_column = None

        for column in possible_columns:

            if column in data.columns:

                text_column = column
                break

        if text_column is None:

            st.error(
                "No text column found.\n\n"
                "Your CSV must contain a column such as "
                "'text', 'tweet', or 'content'."
            )

        else:

            st.success(
                f"Text column detected: `{text_column}`"
            )

            if st.button("🚀 Analyze CSV"):

                texts = (
                    data[text_column]
                    .fillna("")
                    .astype(str)
                )

                text_vectors = vectorizer.transform(
                    texts
                )

                predictions = model.predict(
                    text_vectors
                )

                data["Prediction"] = predictions

                # Convert labels
                data["Sentiment"] = data[
                    "Prediction"
                ].apply(
                    lambda x:
                    "Positive"
                    if x == 4 or x == 1
                    else "Negative"
                )

                st.subheader(
                    "📊 Analysis Results"
                )

                st.dataframe(
                    data,
                    use_container_width=True
                )

                # Statistics
                positive = (
                    data["Sentiment"]
                    == "Positive"
                ).sum()

                negative = (
                    data["Sentiment"]
                    == "Negative"
                ).sum()

                total = len(data)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Total Records",
                        f"{total:,}"
                    )

                with col2:
                    st.metric(
                        "😊 Positive",
                        f"{positive:,}"
                    )

                with col3:
                    st.metric(
                        "😞 Negative",
                        f"{negative:,}"
                    )

                # Chart
                st.subheader(
                    "📊 Sentiment Distribution"
                )

                counts = (
                    data["Sentiment"]
                    .value_counts()
                )

                st.bar_chart(counts)

                # Download
                csv = data.to_csv(
                    index=False
                )

                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📈 Model Performance":

    st.header("📈 Model Performance")

    st.write(
        "Evaluation of the trained Logistic Regression model."
    )

    # -----------------------------------------------------
    # Prepare evaluation data
    # -----------------------------------------------------

    # Use a sample so Streamlit remains fast
    evaluation_size = min(
        100000,
        len(df)
    )

    evaluation_df = df.sample(
        n=evaluation_size,
        random_state=42
    )

    X_eval = evaluation_df["text"].fillna("").astype(str)

    # IMPORTANT:
    # Original Sentiment140 labels are 0 and 4
    y_eval = evaluation_df["target"]

    X_eval_vector = vectorizer.transform(
        X_eval
    )

    y_pred = model.predict(
        X_eval_vector
    )

    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_eval,
        y_pred
    )

    st.subheader("🎯 Accuracy")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Evaluation Records",
            f"{evaluation_size:,}"
        )

    with col3:

        st.metric(
            "Algorithm",
            "Logistic Regression"
        )

    st.divider()

    # -----------------------------------------------------
    # Confusion Matrix
    # -----------------------------------------------------

    st.subheader("🔢 Confusion Matrix")

    cm = confusion_matrix(
        y_eval,
        y_pred,
        labels=[0, 4]
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.imshow(cm)

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    ax.set_title(
        "Confusion Matrix"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Negative (0)", "Positive (4)"]
    )

    ax.set_yticklabels(
        ["Negative (0)", "Positive (4)"]
    )

    # Display numbers
    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    st.pyplot(fig)

    # -----------------------------------------------------
    # Classification Report
    # -----------------------------------------------------

    st.subheader(
        "📋 Classification Report"
    )

    report = classification_report(
        y_eval,
        y_pred,
        labels=[0, 4],
        target_names=[
            "Negative",
            "Positive"
        ],
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Performance Bar Chart
    # -----------------------------------------------------

    st.subheader(
        "📊 Precision, Recall and F1-Score"
    )

    performance_df = report_df.loc[
        ["Negative", "Positive"],
        ["precision", "recall", "f1-score"]
    ]

    st.bar_chart(
        performance_df
    )


# =========================================================
# WORD CLOUD
# =========================================================

elif page == "☁️ Word Cloud":

    st.header("☁️ Word Cloud Generator")

    st.write(
        "Generate a word cloud from the social media dataset."
    )

    # Sentiment selection
    sentiment_option = st.selectbox(
        "Select Sentiment",
        [
            "All",
            "Positive",
            "Negative"
        ]
    )

    if sentiment_option == "Positive":

        wc_df = df[
            df["target"] == 4
        ]

    elif sentiment_option == "Negative":

        wc_df = df[
            df["target"] == 0
        ]

    else:

        wc_df = df

    # Sample for performance
    sample_size = min(
        100000,
        len(wc_df)
    )

    if sample_size == 0:

        st.warning(
            "No records found for this sentiment."
        )

    else:

        sample_df = wc_df.sample(
            n=sample_size,
            random_state=42
        )

        text_data = " ".join(
            sample_df["text"]
            .fillna("")
            .astype(str)
        )

        if text_data.strip():

            wc = WordCloud(
                width=1200,
                height=600,
                background_color="white",
                max_words=150
            ).generate(text_data)

            fig, ax = plt.subplots(
                figsize=(14, 7)
            )

            ax.imshow(
                wc,
                interpolation="bilinear"
            )

            ax.axis("off")

            st.pyplot(fig)

        else:

            st.warning(
                "The text data is empty."
            )


# =========================================================
# VISUALIZATIONS
# =========================================================

elif page == "📊 Visualizations":

    st.header("📊 Dataset Visualizations")

    st.write(
        "Explore different visualizations of the Sentiment140 dataset."
    )

    # -----------------------------------------------------
    # 1. Sentiment Distribution
    # -----------------------------------------------------

    st.subheader(
        "1️⃣ Sentiment Distribution"
    )

    sentiment_counts = (
        df["target"]
        .map({
            0: "Negative",
            4: "Positive"
        })
        .value_counts()
    )

    col1, col2 = st.columns(2)

    with col1:

        st.bar_chart(
            sentiment_counts
        )

    with col2:

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct="%1.1f%%"
        )

        ax.set_title(
            "Sentiment Percentage"
        )

        st.pyplot(fig)

    st.divider()

    # -----------------------------------------------------
    # 2. Dataset Size
    # -----------------------------------------------------

    st.subheader(
        "2️⃣ Dataset Statistics"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Tweets",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Negative Tweets",
            f"{(df['target'] == 0).sum():,}"
        )

    with col3:

        st.metric(
            "Positive Tweets",
            f"{(df['target'] == 4).sum():,}"
        )

    st.divider()

    # -----------------------------------------------------
    # 3. Tweet Length Distribution
    # -----------------------------------------------------

    st.subheader(
        "3️⃣ Tweet Length Distribution"
    )

    # Sample for speed
    length_sample = df.sample(
        n=min(100000, len(df)),
        random_state=42
    ).copy()

    length_sample["tweet_length"] = (
        length_sample["text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.hist(
        length_sample["tweet_length"],
        bins=50
    )

    ax.set_xlabel(
        "Tweet Length"
    )

    ax.set_ylabel(
        "Number of Tweets"
    )

    ax.set_title(
        "Tweet Length Distribution"
    )

    st.pyplot(fig)

    st.divider()

    # -----------------------------------------------------
    # 4. Most Common Words
    # -----------------------------------------------------

    st.subheader(
        "4️⃣ Most Common Words"
    )

    text_sample = df.sample(
        n=min(100000, len(df)),
        random_state=42
    )

    all_text = " ".join(
        text_sample["text"]
        .fillna("")
        .astype(str)
    ).lower()

    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        all_text
    )

    # Remove common words
    stop_words = {
        "the",
        "and",
        "for",
        "you",
        "that",
        "this",
        "with",
        "have",
        "was",
        "are",
        "but",
        "not",
        "from",
        "your",
        "just",
        "they",
        "will",
        "what",
        "about",
        "like",
        "http",
        "www"
    }

    filtered_words = [
        word
        for word in words
        if word not in stop_words
    ]

    word_counts = Counter(
        filtered_words
    ).most_common(20)

    if word_counts:

        words_df = pd.DataFrame(
            word_counts,
            columns=[
                "Word",
                "Count"
            ]
        )

        words_df = words_df.set_index(
            "Word"
        )

        st.bar_chart(
            words_df
        )

    st.divider()

    # -----------------------------------------------------
    # 5. Positive vs Negative Word Counts
    # -----------------------------------------------------

    st.subheader(
        "5️⃣ Positive vs Negative Word Frequency"
    )

    positive_df = df[
        df["target"] == 4
    ].sample(
        n=min(
            50000,
            (df["target"] == 4).sum()
        ),
        random_state=42
    )

    negative_df = df[
        df["target"] == 0
    ].sample(
        n=min(
            50000,
            (df["target"] == 0).sum()
        ),
        random_state=42
    )

    positive_text = " ".join(
        positive_df["text"]
        .fillna("")
        .astype(str)
    ).lower()

    negative_text = " ".join(
        negative_df["text"]
        .fillna("")
        .astype(str)
    ).lower()

    positive_words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        positive_text
    )

    negative_words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        negative_text
    )

    positive_counts = Counter(
        word
        for word in positive_words
        if word not in stop_words
    )

    negative_counts = Counter(
        word
        for word in negative_words
        if word not in stop_words
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "😊 Top Positive Words"
        )

        positive_top = (
            pd.DataFrame(
                positive_counts.most_common(15),
                columns=[
                    "Word",
                    "Count"
                ]
            )
            .set_index("Word")
        )

        st.bar_chart(
            positive_top
        )

    with col2:

        st.write(
            "😞 Top Negative Words"
        )

        negative_top = (
            pd.DataFrame(
                negative_counts.most_common(15),
                columns=[
                    "Word",
                    "Count"
                ]
            )
            .set_index("Word")
        )

        st.bar_chart(
            negative_top
        )

    st.divider()

    # -----------------------------------------------------
    # 6. Dataset Sample
    # -----------------------------------------------------

    st.subheader(
        "6️⃣ Random Dataset Sample"
    )

    st.dataframe(
        df.sample(
            min(20, len(df)),
            random_state=42
        ),
        use_container_width=True
    )
# ============================================
# CUSTOM COLORFUL DESIGN
# ============================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e1b4b, #312e81);
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

/* Information card */
.info-card {
    background: linear-gradient(
        135deg,
        rgba(59,130,246,0.25),
        rgba(168,85,247,0.25)
    );

    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;

    padding: 30px;
    margin: 20px 0;

    box-shadow: 0 10px 35px rgba(0,0,0,0.35);
}

/* Card heading */
.info-card h2 {
    color: #f8fafc;
    font-size: 28px;
}

/* Card paragraph */
.info-card p {
    color: #cbd5e1;
    font-size: 17px;
    line-height: 1.7;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        rgba(59,130,246,0.20),
        rgba(139,92,246,0.20)
    );

    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;

    padding: 20px;

    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #38bdf8;
    font-weight: 800;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827,
        #1e1b4b,
        #312e81
    );
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #f8fafc;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );

    color: white;
    border: none;
    border-radius: 12px;

    padding: 10px 25px;

    font-weight: 700;

    transition: 0.3s;
}

/* Button hover */
.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #7c3aed,
        #db2777
    );

    transform: scale(1.03);
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(
        90deg,
        #059669,
        #0d9488
    );

    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
}

/* Headers */
h1, h2, h3 {
    color: #f8fafc;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 10px;
}

/* Success message */
.stAlert {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

            
