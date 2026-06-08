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

df = compute_basic_metrics2(df)

# ------------------------------------------------
# Showcase hotfix: normalize expected columns
# ------------------------------------------------

# Test name column
if "name" not in df.columns:
    if "test_name" in df.columns:
        df["name"] = df["test_name"]
    elif "title" in df.columns:
        df["name"] = df["title"]
    else:
        st.error("Missing test name column. Expected one of: name, test_name, title.")
        st.write("Available columns:", df.columns.tolist())
        st.stop()

# Learner/user column
if "user_id" not in df.columns:
    possible_user_cols = [
        "learner_id",
        "student_id",
        "student_user_id",
        "test_taker_user_id",
        "test_taker_id",
    ]
    matched_user_col = next((col for col in possible_user_cols if col in df.columns), None)

    if matched_user_col:
        df["user_id"] = df[matched_user_col]
    else:
        st.error("Missing learner/user column. Expected user_id or learner/student equivalent.")
        st.write("Available columns:", df.columns.tolist())
        st.stop()

# Learner name column
if "l_name" not in df.columns:
    if "learner_name" in df.columns:
        df["l_name"] = df["learner_name"]
    elif "student_name" in df.columns:
        df["l_name"] = df["student_name"]
    elif "full_name" in df.columns:
        df["l_name"] = df["full_name"]
    else:
        df["l_name"] = df["user_id"].astype(str)

# Required metric columns
required_cols = [
    "user_id",
    "l_name",
    "test_id",
    "name",
    "speed_raw",
    "adj_speed",
    "accuracy_total",
    "efficiency_ratio",
    "speed_norm",
    "time_consumed",
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing required columns for this page: {missing_cols}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# Optional metric used only in per-test table
if "speed_marks" not in df.columns:
    df["speed_marks"] = pd.NA

# Clean display/filter values
df = df.copy()
df["name"] = df["name"].fillna("Unknown Test")
df["l_name"] = df["l_name"].fillna(df["user_id"].astype(str))

# ------------------------------------------------
# Filters Sidebar
# ------------------------------------------------
st.sidebar.header("Filters")

user_options = sorted(df["user_id"].dropna().unique())
test_options = sorted(df["name"].dropna().unique())

user_filter = st.sidebar.multiselect(
    "Filter by User ID",
    options=user_options,
)

test_filter = st.sidebar.multiselect(
    "Filter by Test",
    options=test_options,
)

if user_filter:
    df = df[df["user_id"].isin(user_filter)]

if test_filter:
    df = df[df["name"].isin(test_filter)]

if df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# ------------------------------------------------
# PER-USER AGGREGATED
# ------------------------------------------------
st.subheader("Per-User Basic Metrics")

user_metrics = df.groupby(["user_id", "l_name"], dropna=False).agg(
    avg_speed=("speed_raw", "mean"),
    avg_accurate_speed=("adj_speed", "mean"),
    avg_accuracy=("accuracy_total", "mean"),
    avg_efficiency=("efficiency_ratio", "mean"),
    avg_speed_norm=("speed_norm", "mean"),
    avg_time_consumed=("time_consumed", "mean"),
    attempts=("test_id", "count"),
).reset_index()

st.dataframe(user_metrics, use_container_width=True)

csv_user = user_metrics.to_csv(index=False)
st.download_button(
    "Download User Metrics CSV",
    csv_user,
    "user_basic_metrics.csv",
    mime="text/csv",
)

# ------------------------------------------------
# PER-TEST AGGREGATED
# ------------------------------------------------
st.subheader("Per-Test Basic Metrics")

test_metrics = df.groupby(["test_id", "name"], dropna=False).agg(
    mean_speed=("speed_raw", "mean"),
    adj_speed=("adj_speed", "mean"),
    speed_marks=("speed_marks", "mean"),
    accuracy=("accuracy_total", "mean"),
    efficiency=("efficiency_ratio", "mean"),
    speed_norm=("speed_norm", "mean"),
    time_consumed=("time_consumed", "mean"),
    takers=("user_id", "nunique"),
).reset_index()

st.dataframe(test_metrics, use_container_width=True)

csv_test = test_metrics.to_csv(index=False)
st.download_button(
    "Download Test Metrics CSV",
    csv_test,
    "test_basic_metrics.csv",
    mime="text/csv",
)
