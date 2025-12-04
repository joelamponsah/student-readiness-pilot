import streamlit as st
import plotly.express as px

from utils.metrics import (
    load_data_from_disk_or_session,
    compute_basic_metrics2,
    compute_sab_behavioral,
    compute_test_analytics
)

# ---------------------------
# Page Title
# ---------------------------
st.title("🧑‍🎓 User Performance Profile")

# ---------------------------
# Load Data
# ---------------------------
df = load_data_from_disk_or_session()
if df is None or df.empty:
    st.warning("Upload data to continue.")
    st.stop()

# Compute metrics
df = compute_basic_metrics2(df)
sab_df = compute_sab_behavioral(df)
test_df = compute_test_analytics(df)

# ---------------------------
# Select User
# ---------------------------
user_id = st.selectbox("Select User ID", sorted(df["user_id"].unique()))
user_basic = df[df["user_id"] == user_id]
user_sab = sab_df[sab_df["user_id"] == user_id]
#user_tests = test_df[test_df["user_id"] == user_id]
user_tests = df[df["user_id"] == user_id].copy()
user_tests = user_tests.merge(test_df, on="test_id", how="left")

st.subheader(f"📌 Profile Summary for User {user_id}")

# ---------------------------
# Basic Stats
# ---------------------------
if not user_basic.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{user_basic['accuracy_total'].mean():.2f}")
    col2.metric("Speed (SA Raw)", f"{user_basic['adj_speed'].mean():.2f}")
    col3.metric("Efficiency Ratio", f"{user_basic['efficiency_ratio'].mean():.2f}")
else:
    st.info("No basic metric data for this user.")

# ---------------------------
# SAB Behavioral Metrics
# ---------------------------
st.subheader("🧭 SAB Behavioral Metrics")

if not user_sab.empty:
    st.write(user_sab)

    fig_sab = px.bar(
        user_sab.melt(id_vars="user_id", var_name="Metric", value_name="Value"),
        x="Metric", y="Value", title="SAB Behavioral Metrics Breakdown",
        text_auto=True
    )
    st.plotly_chart(fig_sab, use_container_width=True)

else:
    st.info("No SAB metrics for this user.")

# ---------------------------
# Test Performance Over Time
# ---------------------------
st.subheader("📈 Test Performance Trends")

if not user_tests.empty:
    fig_test = px.line(
        user_tests,
        x="test_date",
        y="accuracy_total",
        title="Accuracy Over Time",
        markers=True
    )
    st.plotly_chart(fig_test, use_container_width=True)

    fig_speed = px.line(
        user_tests,
        x="test_date",
        y="speed_acc_raw",
        title="Speed Over Time",
        markers=True
    )
    st.plotly_chart(fig_speed, use_container_width=True)

    fig_eff = px.line(
        user_tests,
        x="test_date",
        y="efficiency_ratio",
        title="Efficiency Over Time",
        markers=True
    )
    st.plotly_chart(fig_eff, use_container_width=True)

else:
    st.info("No test trend data available for this user.")

# ---------------------------
# Test Difficulty Breakdown
# ---------------------------
st.subheader("📊 Test Difficulty & Stability")

if not user_tests.empty:

    fig_diff = px.bar(
        user_tests,
        x="test_id",
        y="difficulty_score",
        title="Difficulty Score per Test",
        text_auto=True
    )
    st.plotly_chart(fig_diff, use_container_width=True)

    fig_stable = px.bar(
        user_tests,
        x="test_id",
        y="stability_score",
        title="Stability Score per Test",
        text_auto=True
    )
    st.plotly_chart(fig_stable, use_container_width=True)

else:
    st.info("No test-level difficulty or stability data for this user.")
