import streamlit as st
import pandas as pd
import plotly.express as px

from utils.insights import apply_insight_engine
from utils.metrics import (
    load_data_from_disk_or_session,
    compute_basic_metrics2,
    compute_sab_behavioral,
    compute_test_analytics,
)

from utils.institute_standardization import standardize_institute  # uncomment and ensure exists

st.set_page_config(page_title="Institute Performance", layout="wide")
st.title("Institute Performance Summary")

# ---------------------------------------------------
# Load
# ---------------------------------------------------
df = load_data_from_disk_or_session()
if df is None or df.empty:
    st.warning("Upload data to continue.")
    st.stop()

# ---------------------------------------------------
# Standardize Institute (DO THIS EARLY)
# ---------------------------------------------------
df = standardize_institute(
    df=df,
    column="institute",                 # raw column name
    mapping_path="data/mapping.csv"     # your mapping file
)

st.write(df["institute_std"].value_counts().head(10))
st.write("Unknown rate:", (df["institute_std"] == "Unknown").mean())

# Create one canonical name used across the app
# (If your standardizer outputs institute_std)
if "institute_std" not in df.columns:
    st.error("Standardization failed: missing `institute_std`.")
    st.stop()

# Hard guarantees for stability
df["institute_std"] = df["institute_std"].fillna("Unknown").astype(str)

# ---------------------------------------------------
# Compute Metrics
# ---------------------------------------------------
df = compute_basic_metrics2(df)

# IMPORTANT: sab_df should be user-level; ensure institute_std survives or is merged in
sab_df = compute_sab_behavioral(df)
if "institute_std" not in sab_df.columns:
    # If compute_sab_behavioral drops it, merge back from df at user_id level
    user_inst = df[["user_id", "institute_std"]].drop_duplicates("user_id")
    sab_df = sab_df.merge(user_inst, on="user_id", how="left")

sab_df["institute_std"] = sab_df["institute_std"].fillna("Unknown").astype(str)

sab_df = apply_insight_engine(sab_df)

test_df = compute_test_analytics(df)

# ---------------------------------------------------
# Institute Selector
# ---------------------------------------------------
institutes = sorted(sab_df["institute_std"].unique().tolist())
institute = st.selectbox("Select Institute", institutes)

# User-level slice
sab_inst_users = sab_df[sab_df["institute_std"] == institute].copy()

# Attempt-level slice (this is the one for attempts / tests taken)
df_inst_attempts = df[df["user_id"].isin(sab_inst_users["user_id"])].copy()

# ---------------------------------------------------
# KPI METRICS (use separate rows)
# ---------------------------------------------------
row1 = st.columns(3)
row1[0].metric("Learners", sab_inst_users["user_id"].nunique())
row1[1].metric("Unique Tests", df_inst_attempts["test_id"].nunique())
row1[2].metric("Total Attempts", len(df_inst_attempts))

row2 = st.columns(3)
row2[0].metric("Avg Accuracy", f"{df_inst_attempts['accuracy_total'].mean():.2f}")
row2[1].metric("Avg Speed", f"{df_inst_attempts['speed_raw'].mean():.2f}")
row2[2].metric("Avg Readiness (Robust SAB)", f"{sab_inst_users['robust_SAB_scaled'].mean():.1f}")

at_risk = sab_inst_users[sab_inst_users["exam_status"] == "Not Eligible"]
non_risk = sab_inst_users[sab_inst_users["exam_status"] == "Conditionally Eligible"]
ready = sab_inst_users[sab_inst_users["exam_status"] == "Eligible"]

row3 = st.columns(3)
row3[0].metric("At-Risk Learners", len(at_risk))
row3[1].metric("Non-risk Learners", len(non_risk))
row3[2].metric("Ready Learners", len(ready))

st.divider()

# ---------------------------------------------------
# Institute Readiness Summary
# ---------------------------------------------------
st.subheader("Institute Readiness Summary")

eligible_pct = (sab_inst_users["exam_status"] == "Eligible").mean() * 100
near_ready_pct = (sab_inst_users["insight_code"] == "NEAR_READY").mean() * 100
at_risk_pct = (sab_inst_users["exam_status"] == "Not Eligible").mean() * 100

st.markdown(
    f"""
    **{eligible_pct:.1f}%** of learners meet exam eligibility criteria.  
    **{near_ready_pct:.1f}%** are approaching readiness with targeted support.  
    **{at_risk_pct:.1f}%** require foundational intervention before exam attempts.
    """
)

st.divider()

# ---------------------------------------------------
# Readiness Distribution
# ---------------------------------------------------
st.subheader("Readiness Distribution")

insight_dist = (
    sab_inst_users["insight_code"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "Insight", "insight_code": "Learners"})
)

st.bar_chart(insight_dist.set_index("Insight"))

# ---------------------------------------------------
# Top Performers
# ---------------------------------------------------
st.subheader("Top Performers")

st.dataframe(
    ready[["user_id", "mean_accuracy", "mean_speed", "test_count", "robust_SAB_scaled"]],
    use_container_width=True
)

# ---------------------------------------------------
# Filter learners by exam_status
# ---------------------------------------------------
st.subheader("Learners (Filtered)")

selected_status = st.multiselect(
    "Filter by Exam Status",
    sab_inst_users["exam_status"].dropna().unique().tolist(),
    default=sab_inst_users["exam_status"].dropna().unique().tolist()
)

filtered = sab_inst_users[sab_inst_users["exam_status"].isin(selected_status)]

if filtered.empty:
    st.success("No learners detected")
else:
    st.dataframe(
        filtered[[
            "user_id", "test_count", "speed_consistency",
            "accuracy_consistency", "robust_SAB_scaled", "exam_status"
        ]],
        use_container_width=True
    )

# ---------------------------------------------------
# Test Stability & Difficulty
# ---------------------------------------------------
st.subheader("Test Stability & Difficulty")

inst_tests = test_df[test_df["test_id"].isin(df_inst_attempts["test_id"])]

fig = px.scatter(
    inst_tests,
    x="mean_accuracy",
    y="speed_consistency",
    size="taker_count",
    color="time_consistency",
    hover_data=["test_id"],
    title="Test Stability Map"
)

st.plotly_chart(fig, use_container_width=True)
