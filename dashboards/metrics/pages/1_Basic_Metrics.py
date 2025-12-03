import streamlit as st
import pandas as pd
from utils.metrics import compute_basic_metrics

st.title("📊 Basic Accuracy & Speed Metrics")

df = pd.read_csv("data/verify_df_fixed.csv")
df = compute_basic_metrics(df)

st.subheader("Metrics Overview")
st.dataframe(df.head())
