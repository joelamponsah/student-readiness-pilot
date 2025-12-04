import streamlit as st
import pandas as pd
import plotly.express as px

from utils.metrics import (
    load_data_from_session_or_disk,
    compute_basic_metrics2,
    compute_difficulty_df,
    compute_test_analytics
)

st.title("Tests Overview")

df = load_data_from_session_or_disk()
if df is None or df.empty:
    st.warning("No data loaded.")
    st.stop()

df = compute_basic_metrics2(df)
difficulty_df = compute_difficulty_df(df)
test_stats = compute_test_analytics(df)

# ---------------------------
# Select Test
# ---------------------------
test_id = st.selectbox("Select Test ID", sorted(df["test_id"].unique()))

if test_id is None:
    st.info("Select a test to view drilldown.")
    st.stop()

# ---------------------------
# Extract Test-Level Metrics
# ---------------------------
test_summary = test_stats[test_stats["test_id"] == test_id]
difficulty_summary = difficulty_df[difficulty_df["test_id"] == test_id]

st.subheader("📊 Test Summary")
st.write(test_summary.merge(difficulty_summary, on="test_id", how="left"))

# ---------------------------
# User-Level Performance for this Test
# ---------------------------
test_users = df[df["test_id"] == test_id][[
    "user_id",
    "name",
    "accuracy_total",
    "speed_raw",
    "adj_speed",
    "speed_marks",
    "marks",
    "time_taken",
    "time_consumed",
]]

st.subheader("Users Who Took This Test")
st.dataframe(test_users, use_container_width=True)

# ---------------------------
# Charts
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(test_users, x="accuracy_total", nbins=20, title="Accuracy Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(test_users, x="speed_raw", nbins=20, title="Speed Distribution")
    st.plotly_chart(fig, use_container_width=True)

fig = px.histogram(test_users, x="marks", nbins=20, title="Marks Distribution")
st.plotly_chart(fig, use_container_width=True)
