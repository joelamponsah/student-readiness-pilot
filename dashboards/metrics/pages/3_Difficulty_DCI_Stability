import streamlit as st
import pandas as pd

st.title("⚙️ Difficulty, DCI & Test Stability")

df = pd.read_csv("data/verify_df_fixed.csv")

difficulty = df.groupby("test_id").agg(
    pass_rate=("correct_answers", lambda x: (x > 0).mean()),
    avg_accuracy=("accuracy", "mean")
).reset_index()

difficulty["difficulty_score"] = 1 - difficulty["pass_rate"]

st.subheader("Difficulty Table")
st.dataframe(difficulty)
