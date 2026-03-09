import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# --- Page configuration ---
st.set_page_config(
    page_title="Disaster Tweets Analysis",
    page_icon="🌪️",
    layout="wide"
)

# --- App title ---
st.title("🌪️ Disaster Tweets Analysis & Prediction")
st.markdown("""
This application performs an exploratory data analysis (EDA) on disaster-related tweets 
and allows you to predict whether a new tweet refers to a real disaster or not.
""")

# --- Data and model loading ---
@st.cache_data
def load_data():
    """Load the DataFrame from CSV file"""
    df = pd.read_csv("dataset_cleaned.csv")
    return df

@st.cache_resource
def load_model_and_vectorizer():
    """Load the model and TF-IDF vectorizer"""
    model = joblib.load("disaster_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

# Load data and model
df = load_data()
model, vectorizer = load_model_and_vectorizer()

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["📊 Interactive EDA", "🤖 Tweet Prediction"])

# --- Sidebar filters (only for EDA page) ---
if page == "📊 Interactive EDA":
    st.sidebar.header("Global Filters")
    selected_classes = st.sidebar.multiselect(
        "Select classes to analyze:",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda x: "Disaster (1)" if x == 1 else "Non-disaster (0)"
    )

# --- PAGE 1: INTERACTIVE EDA ---
if page == "📊 Interactive EDA":
    st.header("Interactive Exploratory Data Analysis (EDA)")
    
    # Filter data based on selection
    df_filtered = df[df['target'].isin(selected_classes)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Class Distribution")
        fig = px.bar(
            df_filtered['target'].value_counts().reset_index(),
            x='target', y='count',
            labels={'target': 'Class', 'count': 'Count'},
            color='target',
            color_discrete_map={0: 'blue', 1: 'red'},
            title="Number of Tweets by Class"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Tweet Length by Class")
        fig = px.box(
            df_filtered, x='target', y='tweet_length',
            color='target',
            labels={'target': 'Class', 'tweet_length': 'Tweet Length'},
            title="Tweet Length Distribution",
            color_discrete_map={0: 'blue', 1: 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature analysis
    st.subheader("Feature Analysis")
    feature_cols = ['hashtag_count', 'url_count', 'sentiment_score', 'exclamation_count']
    feature_names = ['Hashtag Count', 'URL Count', 'Sentiment Score', 'Exclamation Count']
    
    for i in range(0, len(feature_cols), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(feature_cols):
                with cols[j]:
                    fig = px.box(
                        df_filtered, 
                        x='target', 
                        y=feature_cols[i+j],
                        color='target',
                        labels={'target': 'Class', feature_cols[i+j]: feature_names[i+j]},
                        title=f"Distribution of {feature_names[i+j]}",
                        color_discrete_map={0: 'blue', 1: 'red'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Word clouds by class
    st.subheader("Word Clouds by Class")
    wc_col1, wc_col2 = st.columns(2)
    
    with wc_col1:
        st.markdown("**Non-disaster (0)**")
        if 0 in selected_classes:
            text_non_disaster = " ".join(df[df['target'] == 0]['text'])
            wordcloud_non = WordCloud(width=400, height=200, background_color='white').generate(text_non_disaster)
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.imshow(wordcloud_non, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Class 0 not selected")
    
    with wc_col2:
        st.markdown("**Disaster (1)**")
        if 1 in selected_classes:
            text_disaster = " ".join(df[df['target'] == 1]['text'])
            wordcloud_disaster = WordCloud(width=400, height=200, background_color='white').generate(text_disaster)
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.imshow(wordcloud_disaster, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Class 1 not selected")

    # Sentiment analysis
    st.subheader("Sentiment Analysis")
    fig = px.histogram(
        df_filtered, x='sentiment_score', color='target',
        nbins=50, barmode='overlay',
        labels={'sentiment_score': 'Sentiment Score', 'count': 'Count'},
        title="Sentiment Score Distribution by Class",
        color_discrete_map={0: 'blue', 1: 'red'}
    )
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: TWEET PREDICTION ---
elif page == "🤖 Tweet Prediction":
    st.header("Tweet Disaster Prediction")
    st.markdown("""
    Enter a tweet in the text area below. The model (Logistic Regression with TF-IDF) 
    will predict whether it refers to a real disaster (1) or not (0).
    """)
    
    # Simple text cleaning function (match your notebook preprocessing)
    def clean_text(text):
        """Basic text cleaning - adjust based on your notebook"""
        text = text.lower()
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)     # Remove mentions
        text = re.sub(r'#\w+', '', text)      # Remove hashtags (optional)
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
        return text
    
    # Input text
    user_input = st.text_area(
        "Enter the tweet:",
        height=150,
        placeholder="Example: We just got a new earthquake warning system in California..."
    )
    
    # Prediction button
    if st.button("🔍 Predict", type="primary"):
        if user_input:
            with st.spinner("Analyzing the tweet..."):
                # Clean the text using the same preprocessing as in training
                cleaned_text = clean_text(user_input)
                
                # IMPORTANT FIX: Vectorize using the same vectorizer
                # The vectorizer will transform the text into the same 5000-feature space
                input_vectorized = vectorizer.transform([cleaned_text])
                
                # Make prediction
                prediction = model.predict(input_vectorized)[0]
                probabilities = model.predict_proba(input_vectorized)[0]
                
                # Show results
                st.subheader("Result:")
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 **Alert!** This tweet appears to refer to a **real disaster**.")
                    else:
                        st.success("✅ This tweet **does not appear** to refer to a real disaster.")
                
                with col2:
                    st.metric("Confidence (Class 0 - Non-disaster)", f"{probabilities[0]:.2%}")
                    st.metric("Confidence (Class 1 - Disaster)", f"{probabilities[1]:.2%}")
                
                # Probability chart
                prob_df = pd.DataFrame({
                    'Class': ['Non-disaster (0)', 'Disaster (1)'],
                    'Probability': probabilities
                })
                fig = px.bar(
                    prob_df, x='Class', y='Probability',
                    color='Class',
                    color_discrete_map={'Non-disaster (0)': 'blue', 'Disaster (1)': 'red'},
                    title="Prediction Probabilities",
                    text_auto='.2%'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional tweet analysis
                st.subheader("Additional tweet analysis:")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Length", len(user_input))
                with col_b:
                    st.metric("Hashtags", user_input.count('#'))
                with col_c:
                    sentiment = analyzer.polarity_scores(user_input)['compound']
                    st.metric("Sentiment", f"{sentiment:.2f}")
        else:
            st.warning("⚠️ Please enter a tweet to analyze.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("""
    **Model used:** Logistic Regression with TF-IDF  
    **Dataset:** Disaster Tweets  
    **Developed by:** Mishell Guano
""")