#import streamlit as st
#mport pandas as pd
#from utils.metrics import compute_SAB

#st.title("🏅 SAB Index & Leaderboard")

#df = pd.read_csv("data/verify_df_fixed.csv")
#sab = compute_SAB(df)

#sab["robust_SAB_scaled"] = 100 * (
 #   sab["robust_SAB_index"] - sab["robust_SAB_index"].min()
#) / (
#    sab["robust_SAB_index"].max() - sab["robust_SAB_index"].min()
#)

#st.subheader("Leaderboard (Top 20)")
#st.dataframe(sab.sort_values("robust_SAB_scaled", ascending=False).head(20))

import streamlit as st

from utils.metrics import (
    load_data_from_disk_or_session,
    compute_sab_behavioral,
    compute_test_analytics
)

st.title("Advanced Metrics ")

st.subheader("(Speed - Accuracy Behavior (SAB)")

"This is a more advanced form of our Accuracy to Speed ratio (ASR)."
"However, this takes into accoun that Accuracy and Speed might carry different weights or importance"
"For exampls, getting answers right are more import than finishing fast"
"Therefore, we assume that a user's speed consistency over time is also of great importance"

st.subheader("Speed & Time Consistency") 

"Speed consistency describes how uniformly students complete a test in terms of time."

"It measures behavioral variability — do most test-takers take roughly the same amount of time, or are there big differences?"



df = load_data_from_disk_or_session()
if df is None or df.empty:
    st.warning("Upload file to begin.")
    st.stop()

# ------------------------------------------------
# 🔍 Filters Sidebar
# ------------------------------------------------
st.sidebar.header("Filters")

user_ids = st.sidebar.multiselect(
    "Filter by User ID",
    options=sorted(df["user_id"].unique())
)

test_ids = st.sidebar.multiselect(
    "Filter by Test",
    options=sorted(df["name"].unique())
)

if user_ids:
    df = df[df["user_id"].isin(user_ids)]

if test_ids:
    df = df[df["name"].isin(test_ids)]

# ------------------------------------------------
# SAB PER USER
# ------------------------------------------------
st.subheader("Per-User Advanced Metrics")

sab = compute_sab_behavioral(df)
st.dataframe(sab, use_container_width=True)

csv_sab = sab.to_csv(index=False)
st.download_button("Download SAB User Metrics CSV", csv_sab, "sab_user_metrics.csv")

# ------------------------------------------------
# PER TEST
# ------------------------------------------------
st.subheader("Per-Test Advanced Metrics")

test_analytics = compute_test_analytics(df)
st.dataframe(test_analytics, use_container_width=True)

csv_test = test_analytics.to_csv(index=False)
st.download_button("Download Test Analytics CSV", csv_test, "test_advanced_metrics.csv")

