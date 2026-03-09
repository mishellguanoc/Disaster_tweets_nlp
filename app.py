# app.py
import streamlit as st
import joblib
from data_loader import load_data

# --- Load dataset ---
df = load_data()

# Load model + vectorizer guardados
model = joblib.load("disaster_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Sidebar filters ---
st.sidebar.header("Filters")

# Language filter
languages = df['language'].unique()
selected_languages = st.sidebar.multiselect(
    "Select language(s):",
    options=languages,
    default=list(languages)
)

# Target filter
targets = df['target'].unique()
selected_targets = st.sidebar.multiselect(
    "Select target(s):",
    options=targets,
    default=list(targets)
)

# Location filter
locations = df['location'].dropna().unique()
selected_locations = st.sidebar.multiselect(
    "Select location(s):",
    options=locations,
    default=list(locations)
)

# --- Apply filters ---
filtered_df = df[
    df['language'].isin(selected_languages) &
    df['target'].isin(selected_targets) &
    df['location'].isin(selected_locations)
]

# --- Dataset Overview ---
st.header("Dataset Overview")
st.write(f"Showing {len(filtered_df)} out of {len(df)} rows")
st.dataframe(filtered_df[['id','text','masked_text','target','language','location']])

# --- Statistics ---
st.header("Statistics")
st.subheader("Target distribution")
st.bar_chart(filtered_df['target'].value_counts())

st.subheader("Language distribution")
st.bar_chart(filtered_df['language'].value_counts())

# --- Predictions ---
st.header("Predictions")

num_to_predict = st.slider("Number of tweets to predict", min_value=1, max_value=50, value=10)
tweets_to_predict = filtered_df.head(num_to_predict)

# Vectorize and Predict
X_input = vectorizer.transform(tweets_to_predict['normalized_text'])
preds = model.predict(X_input)

# Show results
results_df = tweets_to_predict.copy()
results_df['prediction'] = preds
st.dataframe(results_df[['masked_text','target','prediction']])