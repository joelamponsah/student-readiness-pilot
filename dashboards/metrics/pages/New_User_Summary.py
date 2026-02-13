import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from utils.insights import apply_insight_engine
from utils.metrics import (
    load_data_from_disk_or_session,
    compute_basic_metrics2,
    compute_sab_behavioral,
    compute_test_analytics,
    compute_difficulty_df,
    compute_user_pass_features
)

st.title("User Performance Profile")

# ---------------------------
# Load Data
# ---------------------------
df = load_data_from_disk_or_session()
if df is None or df.empty:
    st.warning("Upload data to continue.")
    st.stop()

# Ensure username exists
if "username" not in df.columns:
    st.error("Missing 'username' column in dataset.")
    st.stop()

# Parse dates if available
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# Compute metrics
df = compute_basic_metrics2(df)

# SAB + pass features
sab_df = compute_sab_behavioral(df)
pass_df = compute_user_pass_features(df)
sab_df = sab_df.merge(pass_df, on="user_id", how="left")

# Apply insight engine (now adds risk_band + probability + plan)
sab_df = apply_insight_engine(sab_df)

# Test and difficulty analytics
test_df = compute_test_analytics(df)
diff_df = compute_difficulty_df(df)

# ---------------------------
# Select User by username
# ---------------------------
usernames = sorted(df["username"].dropna().unique())
username = st.selectbox("Select Username", usernames)

# Map username -> user_id (unique by your statement)
user_id = df.loc[df["username"] == username, "user_id"].iloc[0]

user_tests = df[df["user_id"] == user_id].copy()
user_sab = sab_df[sab_df["user_id"] == user_id].copy()

st.subheader(f"Profile Summary for {username}")

if user_tests.empty:
    st.info("No performance records found for this user.")
    st.stop()

# ---------------------------
# Activity Context
# ---------------------------
attempts = len(user_tests)
tests_unique = user_tests["test_id"].nunique()

c1, c2, c3 = st.columns(3)
c1.metric("Total Attempts", f"{attempts:,}")
c2.metric("Unique Tests", f"{tests_unique:,}")
if "created_at" in user_tests.columns and user_tests["created_at"].notna().any():
    c3.metric(
        "Activity Window",
        f"{user_tests['created_at'].min().date()} → {user_tests['created_at'].max().date()}"
    )
else:
    c3.metric("Activity Window", "N/A")

st.divider()

st.write("DEBUG — efficiency inputs for selected user")
st.write({
    "rows": len(user_tests),
    "accuracy_total_notna": int(user_tests["accuracy_total"].notna().sum()) if "accuracy_total" in user_tests.columns else None,
    "time_consumed_notna": int(user_tests["time_consumed"].notna().sum()) if "time_consumed" in user_tests.columns else None,
    "time_consumed_zero": int((pd.to_numeric(user_tests["time_consumed"], errors="coerce") == 0).sum()) if "time_consumed" in user_tests.columns else None,
    "efficiency_notna": int(user_tests["efficiency_ratio"].notna().sum()) if "efficiency_ratio" in user_tests.columns else None,
    "duration_notna": int(user_tests["duration"].notna().sum()) if "duration" in user_tests.columns else None,
    "time_taken_notna": int(user_tests["time_taken"].notna().sum()) if "time_taken" in user_tests.columns else None,
})
st.dataframe(user_tests[["time_taken","duration","time_consumed","accuracy_total","efficiency_ratio"]].head(15), use_container_width=True)

# ---------------------------
# Basic Stats
# ---------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Avg Score Rate (marks/q)", f"{user_tests['accuracy_total'].mean():.2f}")
col2.metric("Avg Correct Speed (correct/time)", f"{user_tests['adj_speed'].mean():.2f}")
eff_ratio = user_tests['efficiency_ratio']
eff_per_min = user_tests['efficiency_per_min']

eff_ratio_val = float(eff_ratio.mean()) if eff_ratio.notna().any() else None
eff_per_min_val = float(eff_per_min.mean()) if eff_per_min.notna().any() else None

if eff_ratio_val is not None:
    col3.metric("Learning Efficiency (accuracy / time used)", f"{eff_ratio_val:.2f}")
elif eff_per_min_val is not None:
    col3.metric("Learning Efficiency (accuracy per minute)", f"{eff_per_min_val:.2f}")
else:
    col3.metric("Learning Efficiency", "N/A")

st.divider()

# ---------------------------
# Readiness Insight (Minister-friendly)
# ---------------------------
st.subheader("🧠 Exam Readiness Insight")

if user_sab.empty:
    st.warning("No readiness record available for this user (likely insufficient valid attempts).")
else:
    r = user_sab.iloc[0]

    top1, top2, top3 = st.columns(3)
    top1.metric("Exam Status", str(r.get("exam_status", "Unknown")))
    top2.metric("Risk Band", str(r.get("risk_band", "Unknown")))
    top3.metric("Readiness Probability", f"{float(r.get('readiness_probability_pct', 0)):.1f}%")

    status = str(r.get("exam_status", "Unknown"))
    msg = str(r.get("insight_message", ""))

    if status.lower() == "eligible":
        st.success(msg)
    elif status.lower() in ["not eligible", "at risk", "needs support"]:
        st.error(msg)
    else:
        st.info(msg)

    st.info("📘 Instructor / Stakeholder Summary")
    st.write(str(r.get("stakeholder_insight", "")))

    st.success("🎯 Coach Feedback")
    st.write(str(r.get("coach_feedback", "")))

    st.info(f"👉 Recommended Action: {str(r.get('recommended_action',''))}")
    st.caption(f"Insight Code: {str(r.get('insight_code',''))}")

    # Redemption arc milestones
    st.subheader("🛠️ Redemption Arc Plan (Milestones)")
    plan = r.get("redemption_plan", [])
    if isinstance(plan, list) and plan:
        for i, step in enumerate(plan, 1):
            st.write(f"{i}. {step}")
    else:
        st.write("No plan available.")

st.divider()

# ---------------------------
# SAB Metrics (Numeric only)
# ---------------------------
st.subheader("Speed–Accuracy–Behavior (SAB) Metrics")

if not user_sab.empty:
    numeric_cols = [c for c in user_sab.columns if c not in ['user_id'] and pd.api.types.is_numeric_dtype(user_sab[c])]
    show_cols = ["user_id"] + numeric_cols
    st.dataframe(user_sab[show_cols], use_container_width=True)

    melt_df = user_sab[show_cols].melt(id_vars="user_id", var_name="Metric", value_name="Value")
    fig_sab = px.bar(melt_df, x="Metric", y="Value", title="SAB Metrics Breakdown", text_auto=True)
    st.plotly_chart(fig_sab, use_container_width=True)

st.divider()

# ---------------------------
# Performance Trends
# ---------------------------
st.subheader("📈 Performance Trends Over Time")

if "created_at" in user_tests.columns and user_tests["created_at"].notna().any():
    user_tests = user_tests.sort_values("created_at")

    fig_acc = px.line(user_tests, x="created_at", y="accuracy_total", title="Score Rate Over Time", markers=True)
    st.plotly_chart(fig_acc, use_container_width=True)

    if "time_taken" in user_tests.columns:
        fig_time = px.line(user_tests, x="created_at", y="time_taken", title="Time Taken Over Time", markers=True)
        st.plotly_chart(fig_time, use_container_width=True)

    fig_eff = px.line(user_tests, x="created_at", y="efficiency_ratio", title="Efficiency Over Time", markers=True)
    st.plotly_chart(fig_eff, use_container_width=True)
else:
    st.info("No valid timestamps (created_at) available for trend plots.")

st.divider()

# ---------------------------
# Difficulty & Stability (test-level)
# ---------------------------
st.subheader("📚 Test Difficulty & Stability")

user_diff = user_tests[['test_id']].drop_duplicates().merge(diff_df, on="test_id", how="left")

if user_diff.empty or user_diff.get("difficulty", pd.Series(dtype=float)).isna().all():
    st.info("No difficulty/stability data available for this user's tests.")
else:
    fig_diff = px.bar(user_diff, x="test_id", y="difficulty", title="Difficulty per Test", text_auto=True)
    st.plotly_chart(fig_diff, use_container_width=True)

    if "stability" in user_diff.columns:
        fig_stable = px.bar(user_diff, x="test_id", y="stability", title="Stability per Test", text_auto=True)
        st.plotly_chart(fig_stable, use_container_width=True)
