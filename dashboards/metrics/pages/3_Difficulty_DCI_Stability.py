import streamlit as st
from utils.metrics import load_data_with_upload, compute_difficulty_df

st.title("Difficulty, Pass-Rate, Stability & DCI")

df = load_data_with_upload()

if df is None or df.empty:
    st.warning("Upload data to continue.")
    st.stop()

# ------------------------------------------------
# 🔍 Filters
# ------------------------------------------------
test_filter = st.sidebar.multiselect(
    "Filter by Test",
    options=sorted(df["name"].unique())
)

if test_filter:
    df = df[df["name"].isin(test_filter)]

difficulty_df = compute_difficulty_df(df)

st.subheader("Per-Test Difficulty & DCI Metrics")
st.dataframe(difficulty_df, use_container_width=True)

csv = difficulty_df.to_csv(index=False)
st.download_button("Download Difficulty & DCI CSV", csv, "difficulty_dci.csv")
