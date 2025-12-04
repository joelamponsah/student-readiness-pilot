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
    load_data_with_upload,
    compute_sab_behavioral,
    compute_test_analytics
)

st.title("Advanced Behavioral Metrics (SAB, Consistency, Robust Index)")

df = load_data_with_upload()
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
    "Filter by Test ID",
    options=sorted(df["test_id"].unique())
)

if user_ids:
    df = df[df["user_id"].isin(user_ids)]

if test_ids:
    df = df[df["test_id"].isin(test_ids)]

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

