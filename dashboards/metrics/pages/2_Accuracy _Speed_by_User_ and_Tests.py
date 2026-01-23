import streamlit as st
import pandas as pd

from utils.metrics import (
    load_data_from_disk_or_session,
    compute_basic_metrics2,
)

st.title("Basic Speed & Accuracy Metrics")

# ---- Load data ----
df = load_data_from_disk_or_session()

if df is None or df.empty:
    st.warning("No data loaded. Upload verify_df_fixed.csv.")
    st.stop()

df = compute_basic_metrics(df)

# ------------------------------------------------
# 🔍 Filters Sidebar
# ------------------------------------------------
st.sidebar.header("Filters")

user_filter = st.sidebar.multiselect(
    "Filter by User ID",
    options=sorted(df["user_id"].unique()),
)

test_filter = st.sidebar.multiselect(
    "Filter by Test ",
    options=sorted(df["name"].unique()),
)

if user_filter:
    df = df[df["user_id"].isin(user_filter)]

if test_filter:
    df = df[df["name"].isin(test_filter)]

# ------------------------------------------------
# PER-USER AGGREGATED
# ------------------------------------------------
st.subheader("Per-User Basic Metrics")

user_metrics = df.groupby(["user_id", "l_name"]).agg(
    avg_speed=("speed_raw", "mean"),
    avg_accurate_speed=("adj_speed", "mean"),
    avg_speed_marks=("speed", "mean"),
    avg_accuracy=("accuracy", "mean"),
    avg_efficiency=("efficiency", "mean"),
    avg_speed_norm=("speed_norm", "mean"),
    avg_time_consumed=("time_used", "mean"),
    attempts=("test_id", "count")
).reset_index()

st.dataframe(user_metrics, use_container_width=True)

csv_user = user_metrics.to_csv(index=False)
st.download_button("Download User Metrics CSV", csv_user, "user_basic_metrics.csv")

# ------------------------------------------------
# PER-TEST AGGREGATED
# ------------------------------------------------
st.subheader("Per-Test Basic Metrics")

test_metrics = df.groupby(["test_id", "name"]).agg(
    mean_speed=("speed_raw", "mean"),
    adj_speed=("adj_speed", "mean"),
    speed_marks=("speed_marks", "mean"),
    accuracy=("accuracy_total", "mean"),
    efficiency=("efficiency", "mean"),
    speed_norm=("speed_norm", "mean"),
    time_consumed=("time_used", "mean"),
    takers=("user_id", "nunique")
).reset_index()

st.dataframe(test_metrics, use_container_width=True)

csv_test = test_metrics.to_csv(index=False)
st.download_button("Download Test Metrics CSV", csv_test, "test_basic_metrics.csv")
