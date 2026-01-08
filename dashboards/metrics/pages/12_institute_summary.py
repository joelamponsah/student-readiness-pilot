import streamlit as st
import pandas as pd
import plotly.express as px

from utils.metrics import (
    load_data_from_disk_or_session,
    compute_basic_metrics2,
    compute_sab_behavioral,
    compute_test_analytics,
    compute_difficulty_df
)

st.set_page_config(page_title="Institute Performance", layout="wide")
st.title("Institute Performance Summary")

# ---------------------------------------------------
# Load & Compute
# ---------------------------------------------------
df = load_data_from_disk_or_session()
if df is None or df.empty:
    st.warning("Upload data to continue.")
    st.stop()

df = compute_basic_metrics2(df)
sab_df = compute_sab_behavioral(df)
test_df = compute_test_analytics(df)

if "institute" not in df.columns:
    st.error("Missing `institute_name` column.")
    st.stop()

# ---------------------------------------------------
# Institute Selector
# ---------------------------------------------------
institutes = sorted(df["institute"].dropna().unique())
institute = st.selectbox("Select Institute", institutes)

inst_df = df[df["institute"] == institute]
inst_users = sab_df[sab_df["user_id"].isin(inst_df["user_id"])]

# ---------------------------------------------------
# KPI METRICS
# ---------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("👥 Learners", inst_users["user_id"].nunique())
col2.metric("🧪 Unique Tests", inst_df["test_id"].nunique())
col3.metric("📊 Total Attempts", len(inst_df))
col4.metric("🎯 Avg Accuracy", f"{inst_df['accuracy_total'].mean():.2f}")
col5.metric("🧠 Avg Readiness (Robust SAB)", f"{inst_users['robust_SAB_scaled'].mean():.1f}")

st.divider()

# ---------------------------------------------------
# TOP PERFORMERS
# ---------------------------------------------------
st.subheader("Top Performers")

top_users = inst_users.sort_values("robust_SAB_scaled", ascending=False).head(10)

st.dataframe(
    top_users[[
        "user_id", "mean_accuracy", "mean_speed",
        "test_count", "robust_SAB_scaled"
    ]],
    use_container_width=True
)

# ---------------------------------------------------
# AT-RISK USERS
# ---------------------------------------------------
st.subheader("🚩 At-Risk Learners")

at_risk = inst_users[
    (inst_users["robust_SAB_scaled"] < 40) &
    (inst_users["test_count"] >= 5)
].sort_values("robust_SAB_scaled")

if at_risk.empty:
    st.success("No at-risk learners detected 🎉")
else:
    st.dataframe(
        at_risk[[
            "user_id", "mean_accuracy",
            "speed_consistency", "accuracy_consistency",
            "test_count", "robust_SAB_scaled"
        ]],
        use_container_width=True
    )

# ---------------------------------------------------
# TEST STABILITY INSIGHTS
# ---------------------------------------------------
st.subheader("Test Stability & Difficulty")

inst_tests = test_df[test_df["test_id"].isin(inst_df["test_id"])]

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
