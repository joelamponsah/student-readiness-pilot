import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from utils.metrics import (
    load_data_from_disk_or_session,
    compute_basic_metrics2,
    compute_difficulty_df,
    compute_test_analytics
)

st.title("Tests Overview")

# ---------------------------
# Load data
# ---------------------------
df = load_data_from_disk_or_session()
if df is None or df.empty:
    st.warning("No data loaded.")
    st.stop()

df = compute_basic_metrics2(df)
difficulty_df = compute_difficulty_df(df)
test_stats = compute_test_analytics(df)

# ---------------------------
# Normalize test name column
# ---------------------------
if "name" not in df.columns:
    if "test_name" in df.columns:
        df["name"] = df["test_name"]
    elif "title" in df.columns:
        df["name"] = df["title"]
    else:
        df["name"] = df["test_id"].astype(str)

df["name"] = df["name"].fillna("Unknown Test").astype(str)

# Parse dates if available
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# ---------------------------
# Select Test by Name
# ---------------------------
st.subheader("Select Test")

test_names = sorted(df["name"].dropna().unique().tolist())

selected_test_name = st.selectbox(
    "Search / Select Test Name",
    test_names
)

if not selected_test_name:
    st.info("Select a test to view drilldown.")
    st.stop()

# Filter selected test attempts by name
test_attempts = df[df["name"] == selected_test_name].copy()

if test_attempts.empty:
    st.warning("No records found for the selected test.")
    st.stop()

selected_test_ids = test_attempts["test_id"].dropna().unique().tolist()

if len(selected_test_ids) > 1:
    st.info(
        f"This test name appears under {len(selected_test_ids)} test IDs. "
        "The summary below combines them."
    )

# ---------------------------
# Extract Test-Level Metrics
# ---------------------------
test_summary = test_stats[test_stats["test_id"].isin(selected_test_ids)].copy()
difficulty_summary = difficulty_df[difficulty_df["test_id"].isin(selected_test_ids)].copy()

# ---------------------------
# Key Metrics
# ---------------------------
st.subheader(f"📊 Test Summary: {selected_test_name}")

total_attempts = len(test_attempts)
unique_learners = test_attempts["user_id"].nunique() if "user_id" in test_attempts.columns else np.nan
unique_test_ids = test_attempts["test_id"].nunique() if "test_id" in test_attempts.columns else np.nan

marks = pd.to_numeric(test_attempts["marks"], errors="coerce") if "marks" in test_attempts.columns else pd.Series(dtype=float)

highest_score = marks.max() if not marks.empty else np.nan
lowest_score = marks.min() if not marks.empty else np.nan
avg_score = marks.mean() if not marks.empty else np.nan

avg_accuracy = test_attempts["accuracy_total"].mean() if "accuracy_total" in test_attempts.columns else np.nan
avg_speed = test_attempts["speed_raw"].mean() if "speed_raw" in test_attempts.columns else np.nan
avg_time = test_attempts["time_taken"].mean() if "time_taken" in test_attempts.columns else np.nan

# Pass rate if pass_mark exists
if "pass_mark" in test_attempts.columns and "marks" in test_attempts.columns:
    pass_mark = pd.to_numeric(test_attempts["pass_mark"], errors="coerce")
    test_attempts["passed"] = np.where(
        pass_mark.notna(),
        marks >= pass_mark,
        np.nan
    )
    pass_rate = test_attempts["passed"].mean()
else:
    pass_rate = np.nan

# Activity window
if "created_at" in test_attempts.columns and test_attempts["created_at"].notna().any():
    first_attempt = test_attempts["created_at"].min()
    last_attempt = test_attempts["created_at"].max()
    activity_window = f"{first_attempt.date()} → {last_attempt.date()}"
    last_taken = str(last_attempt.date())
else:
    activity_window = "N/A"
    last_taken = "N/A"

# KPI Row 1
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total attempts", f"{total_attempts:,}")
c2.metric("Unique learners", f"{unique_learners:,}" if pd.notna(unique_learners) else "N/A")
c3.metric("Test IDs", f"{unique_test_ids:,}" if pd.notna(unique_test_ids) else "N/A")
c4.metric("Last test taken", last_taken)

# KPI Row 2
c5, c6, c7, c8 = st.columns(4)
c5.metric("Highest score", f"{highest_score:.1f}" if pd.notna(highest_score) else "N/A")
c6.metric("Lowest score", f"{lowest_score:.1f}" if pd.notna(lowest_score) else "N/A")
c7.metric("Average score", f"{avg_score:.2f}" if pd.notna(avg_score) else "N/A")
c8.metric("Pass rate", f"{pass_rate * 100:.1f}%" if pd.notna(pass_rate) else "N/A")

# KPI Row 3
c9, c10, c11 = st.columns(3)
c9.metric("Average accuracy", f"{avg_accuracy * 100:.1f}%" if pd.notna(avg_accuracy) else "N/A")
c10.metric("Average speed", f"{avg_speed:.2f} q/min" if pd.notna(avg_speed) else "N/A")
c11.metric("Activity window", activity_window)

st.divider()

# ---------------------------
# Detailed Test-Level Metrics
# ---------------------------
st.subheader("Detailed Test Analytics")

if not test_summary.empty:
    detailed_summary = test_summary.merge(
        difficulty_summary,
        on="test_id",
        how="left"
    )
    st.dataframe(detailed_summary, use_container_width=True)
else:
    st.info("No test analytics available for this test.")

# ---------------------------
# User-Level Performance for this Test
# ---------------------------
st.subheader("Users Who Took This Test")

cols_to_show = [
    "user_id",
    "name",
    "accuracy_total",
    "speed_raw",
    "adj_speed",
    "marks",
    "time_taken",
    "time_consumed"
]

if "created_at" in test_attempts.columns:
    cols_to_show.append("created_at")

if "passed" in test_attempts.columns:
    cols_to_show.append("passed")

available_cols = [c for c in cols_to_show if c in test_attempts.columns]

test_users = test_attempts[available_cols].copy()

# Cleaner display formatting
if "accuracy_total" in test_users.columns:
    test_users["accuracy_pct"] = (test_users["accuracy_total"] * 100).round(1)

display_cols = test_users.columns.tolist()
if "accuracy_pct" in display_cols:
    display_cols = [c for c in display_cols if c != "accuracy_total"]
    display_cols.insert(2, "accuracy_pct")
# Remove duplicate columns before display to avoid Streamlit / PyArrow error
test_users = test_users.loc[:, ~test_users.columns.duplicated()].copy()
display_cols = [c for c in display_cols if c in test_users.columns]
display_cols = list(dict.fromkeys(display_cols))

st.dataframe(test_users[display_cols], use_container_width=True)

# ---------------------------
# Charts
# ---------------------------
st.subheader("Performance Distributions")

col1, col2 = st.columns(2)

with col1:
    if "accuracy_total" in test_attempts.columns and test_attempts["accuracy_total"].notna().any():
        fig = px.histogram(
            test_attempts,
            x="accuracy_total",
            nbins=20,
            title="Accuracy Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No accuracy data available.")

with col2:
    if "speed_raw" in test_attempts.columns and test_attempts["speed_raw"].notna().any():
        fig = px.histogram(
            test_attempts,
            x="speed_raw",
            nbins=20,
            title="Speed Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No speed data available.")

if "marks" in test_attempts.columns and test_attempts["marks"].notna().any():
    fig = px.histogram(
        test_attempts,
        x="marks",
        nbins=20,
        title="Marks Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No marks data available.")

# ---------------------------
# Optional trend if dates exist
# ---------------------------
if "created_at" in test_attempts.columns and test_attempts["created_at"].notna().any():
    st.subheader("Activity Over Time")

    trend_df = test_attempts.dropna(subset=["created_at"]).copy()
    trend_df["week"] = trend_df["created_at"].dt.to_period("W").dt.start_time

    weekly = trend_df.groupby("week").agg(
        attempts=("test_id", "count"),
        avg_accuracy=("accuracy_total", "mean") if "accuracy_total" in trend_df.columns else ("test_id", "count"),
        avg_score=("marks", "mean") if "marks" in trend_df.columns else ("test_id", "count")
    ).reset_index()

    fig_weekly_attempts = px.line(
        weekly,
        x="week",
        y="attempts",
        markers=True,
        title="Weekly Attempts"
    )
    st.plotly_chart(fig_weekly_attempts, use_container_width=True)

    if "avg_accuracy" in weekly.columns:
        weekly["avg_accuracy_pct"] = weekly["avg_accuracy"] * 100
        fig_weekly_acc = px.line(
            weekly,
            x="week",
            y="avg_accuracy_pct",
            markers=True,
            title="Weekly Average Accuracy (%)"
        )
        st.plotly_chart(fig_weekly_acc, use_container_width=True)
