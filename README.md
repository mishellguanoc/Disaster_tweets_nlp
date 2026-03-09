# 🌪️ Disaster Tweets Analysis & Prediction App

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.11.9-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**An interactive web application for analyzing and predicting disaster-related tweets using Natural Language Processing (NLP) and Machine Learning.**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [📦 Installation](#-installation)
- [🚀 Running Locally](#-running-locally)
- [☁️ Deployment](#️-deployment)
- [📁 Project Structure](#-project-structure)
- [🧠 Model Details](#-model-details)
- [📊 Dataset](#-dataset)
- [🎯 Usage Examples](#-usage-examples)
- [⚠️ Troubleshooting](#️-troubleshooting)
- [👩‍💻 Author](#-author)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📊 Interactive EDA** | Visualize class distribution, tweet length, sentiment scores, hashtag/URL counts, and word clouds for both disaster and non-disaster tweets |
| **😊 Sentiment Analysis** | Analyze the emotional tone of tweets using VADER sentiment analyzer |
| **🤖 Tweet Prediction** | Enter any tweet and get a real-time prediction with confidence scores using a Logistic Regression model trained with TF-IDF features |

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|--------------|
| **Core** | Python 3.11.9, Streamlit |
| **ML/DL** | Scikit-learn, Sentence-Transformers, UMAP |
| **NLP** | spaCy, NLTK, VADER, LangDetect |
| **Data** | Pandas, NumPy, Joblib |
| **Visualization** | Plotly, Matplotlib, Seaborn, WordCloud, Folium |
| **Geospatial** | GeoPy, Folium |

</div>

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/mishellguanoc/Disaster_tweets_nlp.git
cd Disaster_tweets_nlp
```
### 2. Create a virtual environment

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
pip install -r requirements.txt
🚀 Running Locally
```bash
streamlit run app.py
```
The app will open automatically in your browser at http://localhost:8501


##  Deployment

You can deploy this Streamlit app easily on Streamlit Cloud or Heroku. Make sure your disaster_model.pkl and tfidf_vectorizer.pkl are included in the repo for predictions to work.

## Project Structure
Disaster_tweets_nlp

app.py                  - Main Streamlit app
data_loader.py          - Functions to load dataset
disaster_model.pkl      - Trained Logistic Regression model
tfidf_vectorizer.pkl    - TF-IDF vectorizer
requirements.txt        - Project dependencies
README.md               - Project documentation



## Model Details

Algorithm: Logistic Regression

Features: TF-IDF vectors (max 5000 features)

Purpose: Predict whether a tweet is disaster-related (1) or not (0)

## 📊 Dataset

Source: https://www.kaggle.com/datasets/vstepanenko/disaster-tweets

Format: CSV with columns like id, text, target, language, normalized_text, etc.

Cleaned for PII (emails, phone numbers) and normalized text for prediction

## 🎯 Usage Examples

Select language(s) and tweet target filters in the sidebar

Explore dataset visualizations and statistics

Input your own tweet in the prediction box to get a disaster classification with confidence

## ⚠️ Troubleshooting

Ensure disaster_model.pkl and tfidf_vectorizer.pkl exist in the project folder

Activate your virtual environment before running Streamlit

If predictions fail, check that the loaded vectorizer matches the trained model

## 👩‍💻 Author

Mishell Guano
