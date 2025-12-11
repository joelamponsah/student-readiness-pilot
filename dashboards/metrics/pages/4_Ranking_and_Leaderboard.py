import streamlit as st
import pandas as pd
from utils.metrics import load_data_from_disk_or_session, compute_basic_metrics2

st.title("Ranking & Leaderboards ")
st.subheader("Weighted Accuracy - Speed Ratio (WASR)")
st.write("Using an adjusted weighting system to rank users on the platform gives a flixible option in deciding who is leading in a test, or global test outcomes ")
st.write("Adjust the weights below to define how accuracy and speed contribute to ranking.")
st.info("accuracy weight + speed weight = 1")

# ---------------------------------------------
# Load data
# ---------------------------------------------
df = load_data_from_disk_or_session()
if df is None:
    st.warning("No dataset loaded. Upload in sidebar or add data/verify_df_fixed.csv.")
    st.stop()

df = compute_basic_metrics2(df)

# Optional percent score
df["percent_score"] = df["accuracy_total"] * 100
df.rename(columns={'name': 'Test'}, inplace=True)

# ---------------------------------------------
# WEIGHT SLIDERS
# ---------------------------------------------
st.subheader("Adjust Leaderboard Weights")

w_accuracy = st.slider("Accuracy Weight", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
w_speed = st.slider("Speed Weight", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

# Auto-normalize
total = w_accuracy + w_speed
if total == 0:
    w_accuracy, w_speed = 0.7, 0.3
else:
    w_accuracy /= total
    w_speed /= total

st.info(f"Normalized Weights → Accuracy: **{w_accuracy:.2f}**, Speed: **{w_speed:.2f}**")

# ---------------------------------------------
# Ensure safe speed scaling
# ---------------------------------------------
max_speed = df["time_consumed"].max()
df["speed_norm"] = df["time_consumed"] / max_speed if max_speed > 0 else 0 #avoid division by zero

# ---------------------------------------------
# LEADERBOARD SCORE
# ---------------------------------------------
df["leaderboard_score"] = (
    (df["accuracy_total"] * w_accuracy) +
    (df["speed_norm"] * w_speed)
)

# ---------------------------------------------
# SORT AND DISPLAY
# ---------------------------------------------
leaderboard_df = df.sort_values("leaderboard_score", ascending=False).reset_index(drop=True)

st.subheader("Global Leaderboard Results")
st.dataframe(leaderboard_df[[
    "user_id", "Test", "accuracy_total", "time_consumed", 
    "speed_norm", "leaderboard_score"
]])

st.subheader("Test Leaderboard Results")
st.dataframe(leaderboard_df.groupby("Test")[[
    "user_id", "Test", "accuracy_total", "time_consumed", 
    "speed_norm", "leaderboard_score"
]])
