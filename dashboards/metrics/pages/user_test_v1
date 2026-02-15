import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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

# Guards
required = ["user_id", "username", "test_id", "marks", "time_taken"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# Parse datetime
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# Compute attempt-level metrics (includes efficiency_pct and efficiency_per_min)
df = compute_basic_metrics2(df)

# Compute SAB + pass features and merge
sab_df = compute_sab_behavioral(df)
pass_df = compute_user_pass_features(df)
sab_df = sab_df.merge(pass_df, on="user_id", how="left")

# Apply insight engine (adds exam_status, messages, etc.)
sab_df = apply_insight_engine(sab_df)

# Test and difficulty analytics
test_df = compute_test_analytics(df)
diff_df = compute_difficulty_df(df)

# ---------------------------
# Build a user list for selection + sorting
# ---------------------------
# Map user_id -> username
user_map = df[["user_id", "username"]].drop_duplicates("user_id")
user_list = sab_df.merge(user_map, on="user_id", how="left")

# Add attempts (some pipelines treat test_count as attempts; here it's attempts with time_taken>0 already)
if "test_count" not in user_list.columns:
    # fallback
    user_list["test_count"] = df.groupby("user_id")["test_id"].count().values

# Selector controls
st.subheader("Select Learner")
left, mid, right = st.columns(3)
sort_mode = left.selectbox(
    "Sort users by",
    ["Attempts (ascending)", "Pass rate (descending)", "Robust SAB (descending)"]
)
min_attempts = mid.number_input("Min attempts filter", min_value=0, value=0, step=1)
show_n = right.number_input("Show top N in selector", min_value=10, value=200, step=10)

# Apply attempts filter
user_list_f = user_list[user_list["test_count"].fillna(0) >= min_attempts].copy()

# Sorting
if sort_mode == "Attempts (ascending)":
    user_list_f = user_list_f.sort_values(["test_count", "pass_rate", "robust_SAB_scaled"], ascending=[True, False, False])
elif sort_mode == "Pass rate (descending)":
    user_list_f = user_list_f.sort_values(["pass_rate", "test_count", "robust_SAB_scaled"], ascending=[False, False, False])
else:
    user_list_f = user_list_f.sort_values(["robust_SAB_scaled", "test_count", "pass_rate"], ascending=[False, False, False])

user_list_f = user_list_f.head(int(show_n))

# Create label for selection (makes minister demo readable)
user_list_f["selector_label"] = user_list_f.apply(
    lambda r: f"{r['username']}  | attempts={int(r['test_count'])} | pass={0 if pd.isna(r.get('pass_rate_pct')) else r.get('pass_rate_pct')}% | SAB={r.get('robust_SAB_scaled', np.nan):.1f}",
    axis=1
)

selected_label = st.selectbox("Choose learner", user_list_f["selector_label"].tolist())
selected_row = user_list_f[user_list_f["selector_label"] == selected_label].iloc[0]
user_id = selected_row["user_id"]
username = selected_row["username"]

# User slices
user_tests = df[df["user_id"] == user_id].copy()
user_sab = sab_df[sab_df["user_id"] == user_id].copy()

st.divider()
st.subheader(f"Profile Summary: {username}")

if user_tests.empty:
    st.info("No performance records found for this learner.")
    st.stop()

# ---------------------------
# KPI Columns requested (top summary)
# ---------------------------
# Score stats
marks_series = pd.to_numeric(user_tests["marks"], errors="coerce")
highest_score = marks_series.max()
lowest_score = marks_series.min()
avg_score = marks_series.mean()

# Average accuracy (explicit)
avg_accuracy = user_tests["accuracy_total"].mean()  # marks/no_of_questions

# Pass stats (from merged pass_df if available)
tests_passed = float(selected_row.get("tests_passed", 0) or 0)
tests_failed = float(selected_row.get("tests_failed", 0) or 0)
pass_rate_pct = selected_row.get("pass_rate_pct", np.nan)
pass_ratio_pct = selected_row.get("avg_pass_ratio_pct", np.nan)

# Efficiency %: prefer duration-based if any exists, else fallback to score/min displayed as "efficiency proxy"
eff_pct = user_tests["efficiency_pct"]
eff_pct_val = float(eff_pct.mean()) if eff_pct.notna().any() else None
eff_pm = user_tests["efficiency_per_min"]
eff_pm_val = float(eff_pm.mean()) if eff_pm.notna().any() else None

# Activity window
wcol1, wcol2, wcol3, wcol4 = st.columns(4)
wcol1.metric("Highest score", f"{highest_score:.1f}" if pd.notna(highest_score) else "N/A")
wcol2.metric("Lowest score", f"{lowest_score:.1f}" if pd.notna(lowest_score) else "N/A")
wcol3.metric("Average score", f"{avg_score:.2f}" if pd.notna(avg_score) else "N/A")
wcol4.metric("Avg accuracy (marks/q)", f"{avg_accuracy:.2f}" if pd.notna(avg_accuracy) else "N/A")

colA, colB, colC, colD = st.columns(4)

# Efficiency display
if eff_pct_val is not None:
    colA.metric("Efficiency (%)", f"{eff_pct_val:.1f}%")
elif eff_pm_val is not None:
    colA.metric("Efficiency (score/min)", f"{eff_pm_val:.2f}")
else:
    colA.metric("Efficiency", "N/A")

colB.metric("Tests passed", f"{int(tests_passed)}")
colC.metric("Tests failed", f"{int(tests_failed)}")
colD.metric("Pass rate (%)", f"{pass_rate_pct:.1f}%" if pd.notna(pass_rate_pct) else "N/A")

# Pass ratio
st.caption(f"Pass ratio (avg marks/pass_mark): {pass_ratio_pct:.1f}%" if pd.notna(pass_ratio_pct) else "Pass ratio: N/A")

st.divider()

# ---------------------------
# Readiness Insight block
# ---------------------------
st.subheader("🧠 Exam Readiness Insight")

if user_sab.empty:
    st.warning("No readiness record available for this learner (likely insufficient valid attempts).")
else:
    r = user_sab.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Exam status", str(r.get("exam_status", "Unknown")))
    c2.metric("Robust SAB (0–100)", f"{float(r.get('robust_SAB_scaled', 0)):.1f}")
    # if your insight engine added probability, show it; else omit
    if "readiness_probability_pct" in user_sab.columns:
        c3.metric("Readiness probability", f"{float(r.get('readiness_probability_pct', 0)):.1f}%")
    else:
        c3.metric("Readiness probability", "N/A")

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

    st.info(f"👉 Recommended action: {str(r.get('recommended_action',''))}")
    st.caption(f"Insight code: {str(r.get('insight_code',''))}")

    # If your insight engine adds redemption_plan list, show it
    if "redemption_plan" in user_sab.columns:
        st.subheader("🛠️ Redemption Arc Plan")
        plan = r.get("redemption_plan", [])
        if isinstance(plan, list) and plan:
            for i, step in enumerate(plan, 1):
                st.write(f"{i}. {step}")

st.divider()
st.subheader("🧪 Readiness by Test (using test name)")

# Ensure 'name' exists for labeling
if "name" not in df.columns:
    st.info("No 'name' column found for test labels. Can't build per-test readiness.")
else:
    # Ensure pass flag exists (pass_mark required)
    if "pass_mark" in user_tests.columns and user_tests["pass_mark"].notna().any():
        user_tests["passed"] = np.where(
            pd.to_numeric(user_tests["pass_mark"], errors="coerce").notna(),
            (pd.to_numeric(user_tests["marks"], errors="coerce") >= pd.to_numeric(user_tests["pass_mark"], errors="coerce")).astype(int),
            np.nan
        )
    else:
        user_tests["passed"] = np.nan

    # Aggregate learner performance per test
    per_test = user_tests.groupby("test_id").agg(
        attempts=("test_id", "count"),
        avg_accuracy=("accuracy_total", "mean"),     # marks/no_of_questions (0..1)
        std_accuracy=("accuracy_total", "std"),
        avg_speed_qpm=("speed_acc_raw", "mean"),     # questions per min (since time_taken is minutes)
        pass_rate=("passed", "mean"),
        avg_marks=("marks", "mean"),
    ).reset_index()

    per_test["std_accuracy"] = per_test["std_accuracy"].fillna(0)
    per_test["avg_accuracy_pct"] = (per_test["avg_accuracy"] * 100).round(1)
    per_test["pass_rate_pct"] = (per_test["pass_rate"] * 100).round(1)

    # Map test_id -> name
    name_map = df[["test_id", "name"]].dropna().drop_duplicates("test_id")
    per_test = per_test.merge(name_map, on="test_id", how="left")
    per_test["test_label"] = per_test["name"].fillna(per_test["test_id"].astype(str))

    # -------- Simple per-test "Work Habits" proxy (within this learner's tests) --------
    # Idea: accuracy + (stability) + evidence, speed lightly weighted (to avoid speed cheating)
    # This is NOT the global Work Habits Score; it's "within-learner per-test readiness signal".
    # Normalize speed within learner tests to avoid huge absolute differences.
    sp = per_test["avg_speed_qpm"].replace([np.inf, -np.inf], np.nan).fillna(per_test["avg_speed_qpm"].median())
    sp_min, sp_max = sp.min(), sp.max()
    sp_norm = (sp - sp_min) / ((sp_max - sp_min) if sp_max > sp_min else 1)

    stability = 1 / (1 + per_test["std_accuracy"])     # higher = more stable
    evidence = per_test["attempts"] / (per_test["attempts"] + 3)  # saturates quickly

    per_test["test_work_habits_index"] = (
        (0.65 * per_test["avg_accuracy"].fillna(0)) +
        (0.15 * sp_norm.fillna(0)) +
        (0.20 * stability.fillna(0))
    ) * evidence.fillna(0)

    # Scale to 0–100 within the learner's tests
    per_test["test_work_habits_score"] = (per_test["test_work_habits_index"].rank(pct=True) * 100).round(1)

    # -------- Simple per-test status (V1) --------
    # Low evidence -> needs more attempts
    # Otherwise: use pass rate + accuracy as primary gates
    def _test_status(row):
        if row["attempts"] < 2:
            return "Low evidence"
        if pd.notna(row["pass_rate"]) and row["pass_rate"] >= 0.7 and row["avg_accuracy"] >= 0.6:
            return "On track"
        if pd.notna(row["pass_rate"]) and row["pass_rate"] < 0.5:
            return "At risk"
        if row["avg_accuracy"] < 0.5:
            return "At risk"
        return "Improving"

    per_test["test_status"] = per_test.apply(_test_status, axis=1)

    # Show table
    show_cols = ["test_label", "attempts", "avg_accuracy_pct", "avg_speed_qpm", "pass_rate_pct", "test_work_habits_score", "test_status"]
    table = per_test[show_cols].rename(columns={
        "test_label": "Test",
        "attempts": "Attempts",
        "avg_accuracy_pct": "Avg accuracy (%)",
        "avg_speed_qpm": "Avg speed (q/min)",
        "pass_rate_pct": "Pass rate (%)",
        "test_work_habits_score": "Work Habits Score (test)",
        "test_status": "Status"
    }).sort_values(["Pass rate (%)", "Avg accuracy (%)"], ascending=False)

    st.dataframe(table, use_container_width=True)

    # Charts
    fig_acc = px.bar(per_test, x="test_label", y="avg_accuracy_pct", title="Avg accuracy (%) by test", text_auto=True)
    fig_acc.update_layout(xaxis_title="Test", xaxis_tickangle=-30)
    st.plotly_chart(fig_acc, use_container_width=True)

    fig_speed = px.bar(per_test, x="test_label", y="avg_speed_qpm", title="Avg speed (q/min) by test", text_auto=True)
    fig_speed.update_layout(xaxis_title="Test", xaxis_tickangle=-30)
    st.plotly_chart(fig_speed, use_container_width=True)

    fig_pass = px.bar(per_test, x="test_label", y="pass_rate_pct", title="Pass rate (%) by test", text_auto=True)
    fig_pass.update_layout(xaxis_title="Test", xaxis_tickangle=-30)
    st.plotly_chart(fig_pass, use_container_width=True)

    st.caption(
        "Note: This is readiness *per test*. It highlights where a learner is strong/weak across different tests. "
        "Low evidence means we need more attempts on that specific test to be confident."
    )

# ---------------------------
# Trends: Accuracy by week, Speed by week
# ---------------------------
st.subheader("📈 Trends (Weekly)")

if "created_at" in user_tests.columns and user_tests["created_at"].notna().any():
    ut = user_tests.dropna(subset=["created_at"]).copy()
    ut["week"] = ut["created_at"].dt.to_period("W").dt.start_time

    weekly = ut.groupby("week").agg(
        weekly_accuracy=("accuracy_total", "mean"),
        weekly_speed=("adj_speed", "mean"),
        weekly_pass_ratio=("pass_ratio", "mean") if "pass_ratio" in ut.columns else ("accuracy_total", "mean"),
        attempts=("test_id", "count")
    ).reset_index()

    fig_w_acc = px.line(weekly, x="week", y="weekly_accuracy", title="Accuracy by week (avg marks/q)", markers=True)
    st.plotly_chart(fig_w_acc, use_container_width=True)

    fig_w_spd = px.line(weekly, x="week", y="weekly_speed", title="Speed by week (avg correct/time)", markers=True)
    st.plotly_chart(fig_w_spd, use_container_width=True)
else:
    st.info("No valid timestamps (created_at) available for weekly trend plots.")

st.divider()

# ---------------------------
# Pass ratio by test
# ---------------------------
st.subheader("✅ Pass ratio by test")

# Ensure pass_ratio exists (from pass_mark)
if "pass_mark" in user_tests.columns and user_tests["pass_mark"].notna().any():
    user_tests["pass_ratio"] = (pd.to_numeric(user_tests["marks"], errors="coerce") / pd.to_numeric(user_tests["pass_mark"], errors="coerce")).replace([np.inf, -np.inf], np.nan)
    user_tests["pass_ratio_pct"] = (user_tests["pass_ratio"] * 100).clip(0, 200)

    pr_test = user_tests.groupby("test_id").agg(
        pass_ratio_pct=("pass_ratio_pct", "mean"),
        attempts=("test_id", "count")
    ).reset_index()

    fig_pr = px.bar(pr_test, x="test_id", y="pass_ratio_pct", title="Pass ratio (%) by test", text_auto=True)
    st.plotly_chart(fig_pr, use_container_width=True)
else:
    st.info("pass_mark is missing/empty for this learner, so pass ratio by test can't be computed.")

st.divider()

# ---------------------------
# Top % among others in same window (cohort percentile)
# ---------------------------
st.subheader("🏅 Standing among peers (same activity window)")

if "created_at" in df.columns and df["created_at"].notna().any() and "created_at" in user_tests.columns and user_tests["created_at"].notna().any():
    start = user_tests["created_at"].min()
    end = user_tests["created_at"].max()

    cohort_df = df[(df["created_at"] >= start) & (df["created_at"] <= end)].copy()
    cohort_df = compute_basic_metrics2(cohort_df)

    # cohort pass rate
    cohort_pass = compute_user_pass_features(cohort_df)

    # cohort SAB
    cohort_sab = compute_sab_behavioral(cohort_df).merge(cohort_pass, on="user_id", how="left")

    # percentiles
    cohort_sab["sab_pct"] = cohort_sab["robust_SAB_scaled"].rank(pct=True) * 100
    cohort_sab["pass_rate_pctile"] = cohort_sab["pass_rate"].rank(pct=True) * 100

    urow = cohort_sab[cohort_sab["user_id"] == user_id]
    if not urow.empty:
        urow = urow.iloc[0]
        p1, p2, p3 = st.columns(3)
        p1.metric("SAB percentile", f"{float(urow.get('sab_pct', 0)):.1f}%")
        p2.metric("Pass rate percentile", f"{float(urow.get('pass_rate_pctile', 0)):.1f}%")
        p3.metric("Cohort window", f"{start.date()} → {end.date()}")
    else:
        st.info("Unable to compute cohort standing for this learner in the selected window.")
else:
    st.info("Cohort standing needs created_at timestamps for both cohort and learner window.")

st.divider()

# ---------------------------
# Speed–Accuracy correlation map
# ---------------------------
st.subheader("🧭 Speed–Accuracy correlation map")

# correlation matrix on numeric features
corr_cols = [c for c in ["accuracy_total", "adj_speed", "time_taken", "pass_ratio", "efficiency_per_min"] if c in user_tests.columns]
corr_df = user_tests[corr_cols].copy()

# ensure numeric
for c in corr_df.columns:
    corr_df[c] = pd.to_numeric(corr_df[c], errors="coerce")

if corr_df.dropna().shape[0] >= 2 and len(corr_cols) >= 2:
    corr = corr_df.corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation heatmap (within learner attempts)")
    st.plotly_chart(fig_corr, use_container_width=True)

    # scatter for intuition: speed vs accuracy
    fig_scatter = px.scatter(
        user_tests,
        x="adj_speed",
        y="accuracy_total",
        title="Speed vs Accuracy (each attempt)",
        hover_data=["test_id"] if "test_id" in user_tests.columns else None
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Not enough numeric data points to compute a correlation map for this learner.")

st.divider()

# ---------------------------
# Difficulty & Stability (optional keep)
# ---------------------------
st.subheader("📚 Test Difficulty & Stability")

user_diff = user_tests[['test_id']].drop_duplicates().merge(diff_df, on="test_id", how="left")
if user_diff.empty or user_diff.get("difficulty", pd.Series(dtype=float)).isna().all():
    st.info("No difficulty/stability data available for this learner's tests.")
else:
    fig_diff = px.bar(user_diff, x="test_id", y="difficulty", title="Difficulty per test", text_auto=True)
    st.plotly_chart(fig_diff, use_container_width=True)

    if "stability" in user_diff.columns:
        fig_stable = px.bar(user_diff, x="test_id", y="stability", title="Stability per test", text_auto=True)
        st.plotly_chart(fig_stable, use_container_width=True)
