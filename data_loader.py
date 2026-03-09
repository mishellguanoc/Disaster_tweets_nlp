import pandas as pd
import streamlit as st

@st.cache_data
def load_data(path="dataset_cleaned.csv"):
    """
    Load the cleaned and feature-enriched dataset.
    Returns a pandas DataFrame.
    """
    df = pd.read_csv(path)
    return df