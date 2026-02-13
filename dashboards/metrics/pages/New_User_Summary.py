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

# Required columns
req = ["user_id", "username", "test_id", "marks", "time_taken"]
missing = [c for c in req if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# Parse created_at if present
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# Compute attempt-level metrics (speed_acc_raw, accuracy_total, efficiency, etc.)
df = compute_basic_metrics2(df)

# Compute pass features (tests passed/failed, pass_rate, pass_ratio)
pass_df = compute_user_pass_features(df)

# SAB + merge pass features
sab_df = compute_sab_behavioral(df).merge(pass_df, on="user_id", how="left")
sab_df = apply_insight_engine(sab_df)

# Test and difficulty analytics
test_df = compute_test_analytics(df)
diff_df = compute_difficulty_df(df)

# ---------------------------
# Build selector table (user-level) + institute option
# ---------------------------
user_map = df[["user_id", "username"]].drop_duplicates("user_id")

# institute mapping (optional)
institute_col = None
for cand in ["institute_std", "institute", "Institute", "school", "institution"]:
    if cand in df.columns:
        institute_col = cand
        break

if institute_col:
    # map each user to their most frequent institute
    user_inst = (
        df[["user_id", institute_col]]
        .dropna()
        .groupby("user_id")[institute_col]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={institute_col: "institute"})
    )
else:
    user_inst = pd.DataFrame({"user_id": [], "institute": []})

user_list = sab_df.merge(user_map, on="user_id", how="left").merge(user_inst, on="user_id", how="left")

# ensure attempts exist
if "test_count" not in user_list.columns:
    user_list["test_count"] = df.groupby("user_id")["test_id"].count().values

st.subheader("Select Learner")

# Institute filter (top 11 only)
selected_institutes = []
if "institute" in user_list.columns and user_list["institute"].notna().any():
    top11 = (
        user_list["institute"]
        .dropna()
        .value_counts()
        .head(11)
        .index
        .tolist()
    )
    selected_institutes = st.multiselect(
        "Filter by Institute (top 11)",
        options=top11,
        default=[]
    )

# Sorting controls
c1, c2, c3, c4 = st.columns(4)
sort_field = c1.selectbox("Sort by", ["Attempts", "Pass rate", "Robust SAB"])
sort_order = c2.selectbox("Order", ["Ascending", "Descending"])
min_attempts = c3.number_input("Min attempts", min_value=0, value=0, step=1)
show_n = c4.number_input("Show top N", min_value=10, value=200, step=10)

u = user_list.copy()

# Apply institute filter only if selected
if selected_institutes:
    u = u[u["institute"].isin(selected_institutes)]

# Filter by attempts
u = u[u["test_count"].fillna(0) >= min_attempts]

# Sort
ascending = (sort_order == "Ascending")
if sort_field == "Attempts":
    u = u.sort_values(["test_count", "pass_rate", "robust_SAB_scaled"], ascending=[ascending, False, False])
elif sort_field == "Pass rate":
    # pass_rate ascending/descending, then attempts, then SAB
    u = u.sort_values(["pass_rate", "test_count", "robust_SAB_scaled"], ascending=[ascending, False, False])
else:
    u = u.sort_values(["robust_SAB_scaled", "test_count", "pass_rate"], ascending=[ascending, False, False])

u = u.head(int(show_n)).copy()

# selector label
u["selector_label"] = u.apply(
    lambda r: (
        f"{r['username']}"
        + (f" | {r['institute']}" if pd.notna(r.get("institute")) else "")
        + f" | attempts={int(r.get('test_count', 0) or 0)}"
        + f" | pass={0 if pd.isna(r.get('pass_rate_pct')) else float(r.get('pass_rate_pct')):.1f}%"
        + f" | SAB={float(r.get('robust_SAB_scaled', 0)):.1f}"
    ),
    axis=1
)

if u.empty:
    st.warning("No learners match the current filters.")
    st.stop()

selected_label = st.selectbox("Choose learner", u["selector_label"].tolist())
sel = u[u["selector_label"] == selected_label].iloc[0]
user_id = sel["user_id"]
username = sel["username"]

user_tests = df[df["user_id"] == user_id].copy()
user_sab = sab_df[sab_df["user_id"] == user_id].copy()

st.divider()
st.subheader(f"Profile Summary: {username}")

if user_tests.empty:
    st.info("No performance records found for this learner.")
    st.stop()

# ---------------------------
# KPI Row 1: attempts, unique tests, activity window
# ---------------------------
attempts = len(user_tests)
unique_tests = user_tests["test_id"].nunique()

r1c1, r1c2, r1c3 = st.columns(3)
r1c1.metric("Total attempts", f"{attempts:,}")
r1c2.metric("Unique tests", f"{unique_tests:,}")

if "created_at" in user_tests.columns and user_tests["created_at"].notna().any():
    r1c3.metric("Activity window", f"{user_tests['created_at'].min().date()} → {user_tests['created_at'].max().date()}")
else:
    r1c3.metric("Activity window", "N/A")

# ---------------------------
# KPI Row 2: highest/lowest/avg score (marks)
# ---------------------------
marks_series = pd.to_numeric(user_tests["marks"], errors="coerce")
highest_score = marks_series.max()
lowest_score = marks_series.min()
avg_score = marks_series.mean()

r2c1, r2c2, r2c3 = st.columns(3)
r2c1.metric("Highest score", f"{highest_score:.1f}" if pd.notna(highest_score) else "N/A")
r2c2.metric("Lowest score", f"{lowest_score:.1f}" if pd.notna(lowest_score) else "N/A")
r2c3.metric("Avg score", f"{avg_score:.2f}" if pd.notna(avg_score) else "N/A")

# ---------------------------
# KPI Row 3: avg accuracy %, avg speed (q/min via speed_acc_raw), efficiency
# ---------------------------
avg_accuracy_pct = float(user_tests["accuracy_total"].mean() * 100) if user_tests["accuracy_total"].notna().any() else None
avg_speed_qpm = float(user_tests["speed_acc_raw"].mean()) if "speed_acc_raw" in user_tests.columns and user_tests["speed_acc_raw"].notna().any() else None

eff_pct = user_tests["efficiency_pct"] if "efficiency_pct" in user_tests.columns else pd.Series(dtype=float)
eff_pct_val = float(eff_pct.mean()) if eff_pct.notna().any() else None
eff_pm = user_tests["efficiency_per_min"] if "efficiency_per_min" in user_tests.columns else pd.Series(dtype=float)
eff_pm_val = float(eff_pm.mean()) if eff_pm.notna().any() else None

r3c1, r3c2, r3c3 = st.columns(3)
r3c1.metric("Avg accuracy", f"{avg_accuracy_pct:.1f}%" if avg_accuracy_pct is not None else "N/A")
r3c2.metric("Avg speed", f"{avg_speed_qpm:.1f} q/min" if avg_speed_qpm is not None else "N/A")

if eff_pct_val is not None:
    r3c3.metric("Learner efficiency", f"{eff_pct_val:.1f}%")
elif eff_pm_val is not None:
    r3c3.metric("Learner efficiency", f"{eff_pm_val:.2f} score/min")
else:
    r3c3.metric("Learner efficiency", "N/A")

st.divider()

# ---------------------------
# Readiness Insight
# ---------------------------
st.subheader("🧠 Exam Readiness Insight")

if user_sab.empty:
    st.warning("No readiness record available for this learner (likely insufficient valid attempts).")
else:
    r = user_sab.iloc[0]
    a, b, c = st.columns(3)
    a.metric("Exam status", str(r.get("exam_status", "Unknown")))
    b.metric("Robust SAB (0–100)", f"{float(r.get('robust_SAB_scaled', 0)):.1f}")
    if "readiness_probability_pct" in user_sab.columns:
        c.metric("Readiness probability", f"{float(r.get('readiness_probability_pct', 0)):.1f}%")
    else:
        c.metric("Readiness probability", "N/A")

    status = str(r.get("exam_status", "Unknown"))
    msg = str(r.get("insight_message", ""))

    if status.lower() == "eligible":
        st.success(msg)
    else:
        st.warning(msg)

    st.info("📘 Instructor / Stakeholder Summary")
    st.write(str(r.get("stakeholder_insight", "")))

    st.success("🎯 Coach Feedback")
    st.write(str(r.get("coach_feedback", "")))

    st.info(f"👉 Recommended action: {str(r.get('recommended_action',''))}")
    st.caption(f"Insight code: {str(r.get('insight_code',''))}")

    if "redemption_plan" in user_sab.columns:
        st.subheader("🛠️ Redemption Arc Plan")
        plan = r.get("redemption_plan", [])
        if isinstance(plan, list) and plan:
            for i, step in enumerate(plan, 1):
                st.write(f"{i}. {step}")

st.divider()

# ---------------------------
# Weekly trends: include scores too
# ---------------------------
st.subheader("📈 Trends (Weekly)")

if "created_at" in user_tests.columns and user_tests["created_at"].notna().any():
    ut = user_tests.dropna(subset=["created_at"]).copy()
    ut["week"] = ut["created_at"].dt.to_period("W").dt.start_time

    # pass ratio per attempt if possible
    if "pass_mark" in ut.columns and ut["pass_mark"].notna().any():
        ut["pass_ratio"] = (pd.to_numeric(ut["marks"], errors="coerce") / pd.to_numeric(ut["pass_mark"], errors="coerce")).replace([np.inf, -np.inf], np.nan)

    weekly = ut.groupby("week").agg(
        weekly_score=("marks", "mean"),
        weekly_accuracy=("accuracy_total", "mean"),
        weekly_speed_qpm=("speed_acc_raw", "mean"),
        weekly_pass_ratio=("pass_ratio", "mean") if "pass_ratio" in ut.columns else ("accuracy_total", "mean"),
        attempts=("test_id", "count")
    ).reset_index()

    fig_w_score = px.line(weekly, x="week", y="weekly_score", title="Average score by week", markers=True)
    st.plotly_chart(fig_w_score, use_container_width=True)

    fig_w_acc = px.line(weekly, x="week", y="weekly_accuracy", title="Accuracy by week (marks/q)", markers=True)
    st.plotly_chart(fig_w_acc, use_container_width=True)

    fig_w_spd = px.line(weekly, x="week", y="weekly_speed_qpm", title="Speed by week (questions/min)", markers=True)
    st.plotly_chart(fig_w_spd, use_container_width=True)
else:
    st.info("No valid timestamps (created_at) available for weekly trend plots.")

st.divider()

# ---------------------------
# Pass ratio by test (use test name labels)
# ---------------------------
st.subheader("✅ Pass ratio by test")

# Build test label mapping: test_id -> name
test_name_col = None
for cand in ["name", "test_name", "title"]:
    if cand in df.columns:
        test_name_col = cand
        break

if "pass_mark" in user_tests.columns and user_tests["pass_mark"].notna().any():
    user_tests["pass_ratio"] = (pd.to_numeric(user_tests["marks"], errors="coerce") / pd.to_numeric(user_tests["pass_mark"], errors="coerce")).replace([np.inf, -np.inf], np.nan)
    user_tests["pass_ratio_pct"] = (user_tests["pass_ratio"] * 100).clip(0, 200)

    pr_test = user_tests.groupby("test_id").agg(
        pass_ratio_pct=("pass_ratio_pct", "mean"),
        attempts=("test_id", "count")
    ).reset_index()

    if test_name_col:
        name_map = df[["test_id", test_name_col]].dropna().drop_duplicates("test_id").rename(columns={test_name_col: "test_name"})
        pr_test = pr_test.merge(name_map, on="test_id", how="left")
        pr_test["label"] = pr_test["test_name"].fillna(pr_test["test_id"].astype(str))
    else:
        pr_test["label"] = pr_test["test_id"].astype(str)

    fig_pr = px.bar(pr_test, x="label", y="pass_ratio_pct", title="Pass ratio (%) by test", text_auto=True)
    fig_pr.update_layout(xaxis_title="Test", xaxis_tickangle=-30)
    st.plotly_chart(fig_pr, use_container_width=True)
else:
    st.info("pass_mark is missing/empty for this learner, so pass ratio by test can't be computed.")

st.divider()

# ---------------------------
# Standing among peers: add plain-English interpretation
# ---------------------------
st.subheader("🏅 Standing among peers (same activity window)")

st.caption(
    "Interpretation: We compare this learner to all other learners who attempted tests in the SAME date window. "
    "Percentile answers: 'What percentage of peers is this learner performing better than?' "
    "Example: 80th percentile = better than ~80% of peers in that window."
)

if "created_at" in df.columns and df["created_at"].notna().any() and "created_at" in user_tests.columns and user_tests["created_at"].notna().any():
    start = user_tests["created_at"].min()
    end = user_tests["created_at"].max()

    cohort = df[(df["created_at"] >= start) & (df["created_at"] <= end)].copy()
    cohort = compute_basic_metrics2(cohort)

    cohort_pass = compute_user_pass_features(cohort)
    cohort_sab = compute_sab_behavioral(cohort).merge(cohort_pass, on="user_id", how="left")

    cohort_sab["sab_percentile"] = cohort_sab["robust_SAB_scaled"].rank(pct=True) * 100
    cohort_sab["pass_rate_percentile"] = cohort_sab["pass_rate"].rank(pct=True) * 100

    row = cohort_sab[cohort_sab["user_id"] == user_id]
    if not row.empty:
        row = row.iloc[0]
        p1, p2, p3 = st.columns(3)
        p1.metric("SAB percentile", f"{float(row.get('sab_percentile', 0)):.1f}%")
        p2.metric("Pass-rate percentile", f"{float(row.get('pass_rate_percentile', 0)):.1f}%")
        p3.metric("Comparison window", f"{start.date()} → {end.date()}")

        st.info(
            f"In this window, {username} is at the **{float(row.get('sab_percentile', 0)):.1f}th percentile** for readiness behavior (SAB) "
            f"and **{float(row.get('pass_rate_percentile', 0)):.1f}th percentile** for pass outcomes."
        )
    else:
        st.info("Unable to compute peer standing for this learner in the selected window.")
else:
    st.info("Peer standing requires valid created_at timestamps for both the learner and the cohort.")

st.divider()

# ---------------------------
# Speed vs Accuracy scatter only (no heatmap)
# ---------------------------
st.subheader("🎯 Speed vs Accuracy (each attempt)")

if "adj_speed" in user_tests.columns and "accuracy_total" in user_tests.columns and user_tests[["adj_speed","accuracy_total"]].dropna().shape[0] >= 2:
    fig_scatter = px.scatter(
        user_tests,
        x="adj_speed",
        y="accuracy_total",
        title="Speed vs Accuracy (each attempt)",
        hover_data=["test_id"] if "test_id" in user_tests.columns else None
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Not enough attempt data to plot speed vs accuracy for this learner.")

st.divider()

# ---------------------------
# Difficulty summary: how many easy/moderate/hard + stability
# ---------------------------
st.subheader("📚 Test difficulty summary (counts + stability)")

user_diff = user_tests[['test_id']].drop_duplicates().merge(diff_df, on="test_id", how="left")

if user_diff.empty or user_diff.get("difficulty_label", pd.Series(dtype=object)).isna().all():
    st.info("No difficulty classification available for this learner's tests.")
else:
    # counts by difficulty label
    counts = user_diff["difficulty_label"].value_counts(dropna=True).rename_axis("difficulty").reset_index(name="tests")
    st.dataframe(counts, use_container_width=True)

    # stability summary by difficulty label (if stability exists)
    if "stability" in user_diff.columns and user_diff["stability"].notna().any():
        stab = user_diff.groupby("difficulty_label").agg(
            avg_stability=("stability", "mean"),
            n_tests=("test_id", "nunique")
        ).reset_index().sort_values("difficulty_label")
        st.caption("Stability interpretation: higher stability = more consistent cohort performance on that test.")
        st.dataframe(stab, use_container_width=True)

    # Optional: keep the existing graphs
    fig_diff = px.bar(user_diff, x="test_id", y="difficulty", title="Difficulty score per test", text_auto=True)
    st.plotly_chart(fig_diff, use_container_width=True)

    if "stability" in user_diff.columns:
        fig_stable = px.bar(user_diff, x="test_id", y="stability", title="Stability score per test", text_auto=True)
        st.plotly_chart(fig_stable, use_container_width=True)
