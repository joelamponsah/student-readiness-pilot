import streamlit as st
import pandas as pd
from utils.metrics import load_data_from_disk_or_session, compute_basic_metrics2

st.title(" Ranking & Leaderboard")

st.write("Ranking users by test based on weighted test accuracy (0.7) and accurate speed (0.3)")
# try to get df from session or disk
df = load_data_from_disk_or_session()
if df is None:
    st.warning("No dataset loaded. Upload in sidebar or add data/verify_df_fixed.csv.")
else:
    df = compute_basic_metrics2(df)

# ===============================================
# 3. FEATURE ENGINEERING
# ===============================================

# Raw speed (questions per minute)
#df["speed_raw"] = df["attempted_questions"] / df["time_taken"]

# Accuracy
#df["accuracy"] = df["correct_answers"] / df["no_of_questions"]

# Adjusted speed = correct answers per minute
#df["speed_acc_raw"] = df["correct_answers"] / df["time_taken"]

# Optional percent score
df["percent_score"] = df["accuracy_total"] * 100

# ===============================================
# 4. LEADERBOARD SCORE
# ===============================================
# Weighted score (you can tune weights)
W_ACCURACY = 0.7
W_SPEED = 0.3

df["leaderboard_score"] = (
    (df["accuracy_total"] * W_ACCURACY) +
    (df["adj_speed"] / df["adj_speed"].max() * W_SPEED)
)

# ===============================================
# 5. SORT LEADERBOARD
# ===============================================
leaderboard_df = df.sort_values("leaderboard_score", ascending=False).reset_index(drop=True)

leaderboard_df
