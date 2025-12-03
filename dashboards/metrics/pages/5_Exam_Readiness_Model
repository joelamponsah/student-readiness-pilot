import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("🎯 Exam Readiness Prediction")

df = pd.read_csv("data/verify_df_fixed.csv")

features = ["accuracy", "speed_acc_raw", "efficiency_ratio"]
df = df.dropna(subset=features)

# Assume pass/fail exists
df["passed"] = (df["final_score"] >= 0.5).astype(int) if "final_score" in df else 0

X = df[features]
y = df["passed"]

model = LogisticRegression()
model.fit(X, y)

st.subheader("Feature Coefficients")
st.write(dict(zip(features, model.coef_[0])))

st.success("Model trained successfully.")
