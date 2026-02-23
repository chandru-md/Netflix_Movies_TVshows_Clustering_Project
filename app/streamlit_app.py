import streamlit as st
import pandas as pd
import joblib

st.title("Netflix Content Clustering")

df = pd.read_csv("data/raw/netflix.csv")

content_type = st.selectbox("Select Type", ["Movie", "TV Show"])

filtered = df[df["type"] == content_type]

st.write(filtered.head())