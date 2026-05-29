# 7_User_Summary.py  (drop-in replacement, cleaned + consistent)

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
    compute_user_pass_features,
    safe_accuracy_series,
)
from utils.dq_policy import apply_dq_gate
from utils.dq_profiles import learner_diagnostic_config
from utils.dq_reporting import render_dq_summary
from utils.dq_controls import dq_sidebar_controls
from utils.institute_standardization import standardize_institute
from utils.learn_smarter_v13 import add_test_exercise_readiness_fields


st.set_page_config(page_title="User Performance Profile", layout="wide")
st.title("User Performance Profile")

# ---------------------------
# Load Data
# ---------------------------
df_raw = load_data_from_disk_or_session()

# Sidebar controls (authoritative)
config = dq_sidebar_controls()
if config is None:
    config = learner_diagnostic_config()

df_clean, dq_report, df_exclusions = apply_dq_gate(df_raw, config=config)

proxy_config = learner_diagnostic_config()
proxy_config.dedupe_best_attempt = False
proxy_config.export_artifacts = False
df_proxy, proxy_dq_report, _ = apply_dq_gate(df_raw, config=proxy_config)

if config.include_incomplete_if_has_evidence:
    salvage = dq_report.get("salvage_stats", {})
    st.caption(
        "Diagnostic mode: incomplete attempts with positive activity evidence may be included. "
        f"Partial-evidence attempts in raw data: {salvage.get('incomplete_usable_count_raw', 0):,}."
    )

# Compute metrics only on gated data
df = compute_basic_metrics2(df_clean)

if "institute" in df.columns:
    df = standardize_institute(df=df, column="institute", mapping_path="data/mapping.csv")

if df is None or df.empty:
    st.warning("Upload data to continue.")
    st.stop()

# Required columns (time_taken optional because we may salvage marks-only attempts)
req = ["user_id", "test_id", "marks"]
missing = [c for c in req if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# Ensure username exists (fallback to user_id)
if "username" not in df.columns:
    df["username"] = df["user_id"].astype(str)

# Parse created_at if present
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
if df_raw is not None and "created_at" in df_raw.columns:
    df_raw = df_raw.copy()
    df_raw["created_at"] = pd.to_datetime(df_raw["created_at"], errors="coerce")

# ---------------------------
# Build username map FROM RAW (stable)
# ---------------------------
if df_raw is not None and not df_raw.empty and "user_id" in df_raw.columns:
    if "username" in df_raw.columns:
        umap = df_raw[["user_id", "username"]].copy()
    else:
        umap = df_raw[["user_id"]].assign(username=np.nan)

    umap["username"] = umap["username"].astype("string").fillna("").str.strip()

    user_map_raw = (
        umap.groupby("user_id")["username"]
        .agg(lambda x: x[x != ""].value_counts().index[0] if (x != "").any() else "")
        .reset_index()
    )
else:
    user_map_raw = df[["user_id"]].drop_duplicates().assign(username="")

user_map_raw.loc[user_map_raw["username"] == "", "username"] = user_map_raw["user_id"].astype(str)

# ---------------------------
# Pass KPIs (user-level) excluding ambiguous pass_mark (strict)
# ---------------------------
df_pass = df_clean.copy()
if config.strict_pass_mark and "pass_mark_ambiguous" in df_pass.columns:
    df_pass = df_pass[~df_pass["pass_mark_ambiguous"]].copy()

pass_user = compute_user_pass_features(df_pass)  # user-level

# ---------------------------
# SAB + insight engine
# ---------------------------
sab_df = compute_sab_behavioral(df).merge(pass_user, on="user_id", how="left")
sab_df = apply_insight_engine(sab_df)

# Test and difficulty analytics (uses df: gated + metrics)
test_df = compute_test_analytics(df)
diff_df = compute_difficulty_df(df)

# ---------------------------
# Build selector table (user-level) + institute option
# ---------------------------
# Institute mapping (optional)
institute_col = None
for cand in ["institute_std", "institute_standardized", "institute", "Institute", "school", "institution"]:
    if cand in df.columns:
        institute_col = cand
        break

if institute_col:
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

# Merge selector base
user_list = (
    sab_df.merge(user_map_raw, on="user_id", how="left")
    .merge(user_inst, on="user_id", how="left")
)

# Ensure username exists
user_list["username"] = user_list.get("username", user_list["user_id"].astype(str))
user_list["username"] = user_list["username"].astype("string").fillna("").str.strip()
user_list.loc[user_list["username"] == "", "username"] = user_list["user_id"].astype(str)

# Ensure attempts exist (use SAB's test_count)
if "test_count" not in user_list.columns:
    user_list["test_count"] = 0
user_list["test_count"] = pd.to_numeric(user_list["test_count"], errors="coerce").fillna(0)

st.subheader("Select Learner")

# Institute filter (top 11 only)
selected_institutes = []
if "institute" in user_list.columns and user_list["institute"].notna().any():
    top11 = user_list["institute"].dropna().value_counts().head(11).index.tolist()
    selected_institutes = st.multiselect("Filter by Institute (top 11)", options=top11, default=[])

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
for col in ["pass_rate", "robust_SAB_scaled"]:
    if col not in u.columns:
        u[col] = np.nan

if sort_field == "Attempts":
    u = u.sort_values(["test_count", "pass_rate", "robust_SAB_scaled"], ascending=[ascending, False, False])
elif sort_field == "Pass rate":
    u = u.sort_values(["pass_rate", "test_count", "robust_SAB_scaled"], ascending=[ascending, False, False])
else:
    u = u.sort_values(["robust_SAB_scaled", "test_count", "pass_rate"], ascending=[ascending, False, False])

u = u.head(int(show_n)).copy()

if u.empty:
    st.warning("No learners match the current filters.")
    st.stop()

# Username dropdown (requested). Disambiguate duplicates by picking most active.
username_options = sorted(u["username"].astype("string").fillna("").unique().tolist())
selected_username = st.selectbox("Choose learner (username)", username_options)

cand = u[u["username"] == selected_username].copy()
if cand["user_id"].nunique() > 1:
    cand = cand.sort_values(["test_count", "robust_SAB_scaled"], ascending=[False, False])
    st.info(f"Multiple learners share username '{selected_username}'. Showing the most active record.")
sel = cand.iloc[0]
user_id = sel["user_id"]
username = sel["username"]

# ---------------------------
# Slice learner data
# ---------------------------
user_tests = df[df["user_id"] == user_id].copy()
user_sab = sab_df[sab_df["user_id"] == user_id].copy()

st.subheader("Data Quality for this learner")
st.caption("DQ summary shown for the selected learner only. Dataset-wide DQ is available on the Data Quality page.")
# Raw slice for learner (use df_raw)
raw_user = (
    df_raw[df_raw["user_id"] == user_id].copy()
    if df_raw is not None and "user_id" in df_raw.columns
    else pd.DataFrame()
)

# Exclusions slice for learner (use df_exclusions)
ex_user = (
    df_exclusions[df_exclusions["user_id"] == user_id].copy()
    if df_exclusions is not None and not df_exclusions.empty and "user_id" in df_exclusions.columns
    else pd.DataFrame()
)

eligible_user = user_tests.copy()

raw_n = len(raw_user)
eligible_n = len(eligible_user)
excluded_n = len(ex_user)

c1, c2, c3 = st.columns(3)
c1.metric("Attempts (raw)", f"{raw_n:,}")
c2.metric("Eligible attempts", f"{eligible_n:,}")
c3.metric("Excluded attempts", f"{excluded_n:,}")

# Key learner-level DQ rates
def _rate(series_bool):
    return float(series_bool.mean() * 100) if series_bool is not None and len(series_bool) else 0.0

dq1, dq2, dq3, dq4 = st.columns(4)

if "is_incomplete" in raw_user.columns:
    dq1.metric("Incomplete (raw)", f"{_rate(raw_user['is_incomplete']):.1f}%")
elif "finished_at" in raw_user.columns:
    dq1.metric("Incomplete (raw)", f"{_rate(raw_user['finished_at'].isna()):.1f}%")
else:
    dq1.metric("Incomplete (raw)", "N/A")

if "incomplete_but_usable" in raw_user.columns:
    dq2.metric("Incomplete but usable", f"{_rate(raw_user['incomplete_but_usable']):.1f}%")
else:
    # fallback heuristic: finished_at missing AND (marks or time_taken present)
    if "finished_at" in raw_user.columns:
        mk = pd.to_numeric(raw_user.get("marks", np.nan), errors="coerce")
        tt = pd.to_numeric(raw_user.get("time_taken", np.nan), errors="coerce")
        usable = raw_user["finished_at"].isna() & (mk.notna() | (tt.notna() & (tt > 0)))
        dq2.metric("Incomplete but usable", f"{_rate(usable):.1f}%")
    else:
        dq2.metric("Incomplete but usable", "N/A")

dq3.metric("Pass mark ambiguous (eligible)", f"{_rate(eligible_user.get('pass_mark_ambiguous', pd.Series([], dtype=bool))):.1f}%")
dq4.metric("Missing Q-level support (eligible)", f"{_rate(eligible_user.get('missing_question_level_support', pd.Series([], dtype=bool))):.1f}%")

# Exclusion reasons for this learner
if not ex_user.empty and "exclusion_reason" in ex_user.columns:
    st.markdown("**Exclusions for this learner (reasons)**")
    reason_tbl = (
        ex_user["exclusion_reason"].value_counts()
        .rename_axis("reason").reset_index(name="count")
    )
    reason_tbl["pct_of_raw"] = (reason_tbl["count"] / raw_n * 100).round(1) if raw_n else 0
    st.dataframe(reason_tbl, use_container_width=True)
st.divider()
st.subheader(f"Profile Summary: {username}")

if user_tests.empty:
    st.info("No performance records found for this learner.")
    st.stop()

# ---------------------------
# Safe accuracy
# ---------------------------
# Exclude inactive rows from the average so zero-attempt records do not drag the
# learner's accuracy down.
user_tests["accuracy_safe"] = safe_accuracy_series(user_tests)
acc_cov = float(user_tests["accuracy_safe"].notna().mean() * 100) if len(user_tests) else 0.0
avg_accuracy_pct = float(user_tests["accuracy_safe"].mean() * 100) if user_tests["accuracy_safe"].notna().any() else None

st.caption(
    "Accuracy coverage (safe): "
    f"{acc_cov:.1f}% of eligible attempts. "
    "Inactive zero-attempt rows are excluded from the average."
)

# Activity window (eligible, based on created_at if available)
activity_window = "N/A"
if "created_at" in user_tests.columns and user_tests["created_at"].notna().any():
    activity_window = f"{user_tests['created_at'].min().date()} -> {user_tests['created_at'].max().date()}"
    
# ---------------------------
# KPI Row 1: raw vs eligible attempts + unique tests
# ---------------------------
raw_user = (
    df_raw[df_raw["user_id"] == user_id].copy()
    if df_raw is not None and not df_raw.empty and "user_id" in df_raw.columns
    else pd.DataFrame()
)
raw_attempts = int(len(raw_user)) if not raw_user.empty else 0
raw_completed = (
    int(raw_user["finished_at"].notna().sum())
    if ("finished_at" in raw_user.columns and not raw_user.empty)
    else np.nan
)

eligible_attempts = int(len(user_tests))
eligible_unique_tests = int(user_tests["test_id"].nunique())

r1c1, r1c2, r1c3, r1c4 = st.columns(4)
r1c1.metric("Attempts (raw)", f"{raw_attempts:,}")
eligible_label = "Attempts (eligible, deduped)" if config.dedupe_best_attempt else "Attempts (eligible)"
r1c2.metric(eligible_label, f"{eligible_attempts:,}")
r1c3.metric("Unique tests (eligible)", f"{eligible_unique_tests:,}")
r1c4.metric("Activity window", activity_window)

if not np.isnan(raw_completed):
    st.caption(f"Completed attempts (raw): {raw_completed:,}")

# ---------------------------
# KPI Row 4: pass KPIs + coverage
# ---------------------------
tests_passed = int(sel.get("tests_passed", 0) or 0)
tests_failed = int(sel.get("tests_failed", 0) or 0)
graded_attempts = int(sel.get("graded_attempts", 0) or 0)
pass_rate_pct = sel.get("pass_rate_pct", np.nan)

r4c1, r4c2, r4c3, r4c4 = st.columns(4)
r4c1.metric("Tests passed", f"{tests_passed}")
r4c2.metric("Tests failed", f"{tests_failed}")
r4c3.metric("Pass rate (%)", f"{float(pass_rate_pct):.1f}%" if pd.notna(pass_rate_pct) else "N/A")

coverage = (graded_attempts / eligible_attempts * 100) if eligible_attempts > 0 else np.nan
r4c4.metric("Pass-mark coverage", f"{coverage:.1f}%" if pd.notna(coverage) else "N/A")
st.caption(f"Pass KPIs computed on {graded_attempts} graded attempts (ambiguous pass_mark excluded).")

# ---------------------------
# KPI Row 2: marks stats
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
# KPI Row 3: avg accuracy (safe), avg speed (q/min), efficiency
# ---------------------------
speed_base = user_tests.copy()
if "speed_eligible" in speed_base.columns:
    speed_base = speed_base[speed_base["speed_eligible"]].copy()

avg_speed_qpm = (
    float(speed_base["speed_acc_raw"].mean())
    if ("speed_acc_raw" in speed_base.columns and speed_base["speed_acc_raw"].notna().any())
    else None
)

# True marks per minute (score/min)
marks_per_min = None
if "time_taken" in user_tests.columns and user_tests["time_taken"].notna().any():
    m = pd.to_numeric(user_tests["marks"], errors="coerce")
    t = pd.to_numeric(user_tests["time_taken"], errors="coerce")
    mp = (m / t).replace([np.inf, -np.inf], np.nan)
    marks_per_min = float(mp.mean()) if mp.notna().any() else None

eff_pct_val = None
if "efficiency_pct" in user_tests.columns and user_tests["efficiency_pct"].notna().any():
    eff_pct_val = float(user_tests["efficiency_pct"].mean())

r3c1, r3c2, r3c3 = st.columns(3)
r3c1.metric("Avg accuracy (safe)", f"{avg_accuracy_pct:.1f}%" if avg_accuracy_pct is not None else "N/A")
r3c2.metric("Avg speed", f"{avg_speed_qpm:.1f} q/min" if avg_speed_qpm is not None else "N/A")

if eff_pct_val is not None:
    r3c3.metric("Learner efficiency", f"{eff_pct_val:.1f}%")
elif marks_per_min is not None:
    r3c3.metric("Learner efficiency", f"{marks_per_min:.2f} marks/min")
else:
    r3c3.metric("Learner efficiency", "N/A")

st.divider()
st.subheader("Learn Smarter v1.3 proxy summary")

proxy_user_tests = df_proxy[df_proxy["user_id"] == user_id].copy()
user_readiness = add_test_exercise_readiness_fields(proxy_user_tests.copy())


def _mean_pct(series):
    values = pd.to_numeric(series, errors="coerce")
    values = values.dropna()
    return float(values.mean()) if not values.empty else np.nan


proxy_rows = user_readiness.copy()
inferred_n = int(proxy_rows["is_inferred_bls_proxy"].sum()) if "is_inferred_bls_proxy" in proxy_rows.columns else 0
current_n = int(proxy_rows["is_current_als_proxy"].sum()) if "is_current_als_proxy" in proxy_rows.columns else 0
potential_n = int(proxy_rows["is_potential_als_proxy"].sum()) if "is_potential_als_proxy" in proxy_rows.columns else 0
repeated_groups = (
    int(proxy_rows.loc[proxy_rows["v13_attempt_count_user_test"].ge(2), ["user_id", "test_id"]].drop_duplicates().shape[0])
    if "v13_attempt_count_user_test" in proxy_rows.columns
    else 0
)
high_n = int((proxy_rows["proxy_evidence_band"] == "high").sum()) if "proxy_evidence_band" in proxy_rows.columns else 0
med_n = int((proxy_rows["proxy_evidence_band"] == "medium").sum()) if "proxy_evidence_band" in proxy_rows.columns else 0
low_n = int((proxy_rows["proxy_evidence_band"] == "low").sum()) if "proxy_evidence_band" in proxy_rows.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Inferred BLS Proxy rows", f"{inferred_n:,}")
c2.metric("Current ALS Proxy rows", f"{current_n:,}")
c3.metric("Potential ALS Proxy rows", f"{potential_n:,}")
c4.metric("Repeated test groups", f"{repeated_groups:,}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("High evidence rows", f"{high_n:,}")
c6.metric("Medium evidence rows", f"{med_n:,}")
c7.metric("Low evidence rows", f"{low_n:,}")
c8.metric("CAS Proxy rows", f"{int(proxy_rows['cas_proxy_test_avg_score_pct'].notna().sum()) if 'cas_proxy_test_avg_score_pct' in proxy_rows.columns else 0:,}")

c9, c10, c11, c12 = st.columns(4)
c9.metric("Mean inferred score", f"{_mean_pct(proxy_rows.get('inferred_bls_proxy_score_pct', pd.Series(dtype=float))):.1f}%" if inferred_n else "N/A")
c10.metric("Mean current ALS score", f"{_mean_pct(proxy_rows.get('current_als_proxy_score_pct', pd.Series(dtype=float))):.1f}%" if current_n else "N/A")
c11.metric("Mean potential ALS score", f"{_mean_pct(proxy_rows.get('potential_als_proxy_score_pct', pd.Series(dtype=float))):.1f}%" if potential_n else "N/A")
c12.metric("Mean gain proxy", f"{_mean_pct(proxy_rows.get('learning_gain_proxy_pct', pd.Series(dtype=float))):.1f}%" if current_n else "N/A")

st.caption(
    "Proxy-only v1.3 view. Current ALS Proxy and gain metrics depend on repeated eligible attempts; "
    "question-pool comparability remains unknown without question IDs."
)

summary_note = proxy_rows.get("learn_smarter_mapping_note")
if isinstance(summary_note, pd.Series) and not summary_note.empty:
    st.info(str(summary_note.iloc[0]))

band_tbl = (
    proxy_rows["proxy_evidence_band"].value_counts(dropna=False).rename_axis("evidence_band").reset_index(name="rows")
    if "proxy_evidence_band" in proxy_rows.columns
    else pd.DataFrame()
)
if not band_tbl.empty:
    st.dataframe(band_tbl, use_container_width=True, hide_index=True)

# ---------------------------
# Readiness Insight
# ---------------------------
st.subheader("Overall Exam Readiness Insight")

if user_sab.empty:
    st.warning("No readiness record available for this learner (likely insufficient valid attempts).")
else:
    r = user_sab.iloc[0]
    a, b, c = st.columns(3)
    a.metric("Exam status", str(r.get("exam_status", "Unknown")))
    b.metric("Work Habits Score (0-100)", f"{float(r.get('robust_SAB_scaled', 0)):.1f}")
    st.caption("Work Habits Score reflects consistency of accuracy + pace across attempts (more evidence = more reliable).")

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

    if bool(r.get("is_blocked", False)):
        st.warning(
            f"**Why not eligible:** {r.get('blocking_reason','')}  \n"
            f"(Insight code: `{r.get('insight_code','')}`)"
        )
    st.info("Instructor / Stakeholder Summary")
    st.write(str(r.get("stakeholder_insight", "")))

    st.success("Coach Feedback")
    st.write(str(r.get("coach_feedback", "")))

    st.info(f"Recommended action: {str(r.get('recommended_action',''))}")
    st.caption(f"Insight code: {str(r.get('insight_code',''))}")

    if "redemption_plan" in user_sab.columns:
        plan_title = "Staying on Track" if str(r.get("exam_status", "")).lower() == "eligible" else "Redemption Arc Plan"
        st.subheader(f"{plan_title}")

        plan = r.get("redemption_plan", [])
        if isinstance(plan, list) and plan:
            for i, step in enumerate(plan, 1):
                st.write(f"{i}. {step}")

# ---------------------------
# Readiness Breakdown by Test
# ---------------------------
st.divider()
st.subheader("Readiness Breakdown by Subject / Test")
st.info(f"Let's see how {username} is performing by Subject / Test")

if "name" not in df.columns:
    st.info("No 'name' column found for test labels. Can't build per-test readiness.")
else:
    # Pass calculations should exclude ambiguous pass marks
    ut_pass = user_tests.copy()
    if config.strict_pass_mark and "pass_mark_ambiguous" in ut_pass.columns:
        ut_pass = ut_pass[~ut_pass["pass_mark_ambiguous"]].copy()

    if "pass_mark" in ut_pass.columns and ut_pass["pass_mark"].notna().any():
        ut_pass["passed"] = np.where(
            pd.to_numeric(ut_pass["pass_mark"], errors="coerce").notna(),
            (pd.to_numeric(ut_pass["marks"], errors="coerce") >= pd.to_numeric(ut_pass["pass_mark"], errors="coerce")).astype(int),
            np.nan,
        )
    else:
        ut_pass["passed"] = np.nan

    per_test = user_tests.groupby("test_id").agg(
        attempts=("test_id", "count"),
        avg_accuracy=("accuracy_safe", "mean"),
        std_accuracy=("accuracy_safe", "std"),
        avg_speed_qpm=("speed_acc_raw", "mean"),
        avg_marks=("marks", "mean"),
    ).reset_index()

    # Pass rate from filtered pass dataset
    pass_rate_tbl = ut_pass.groupby("test_id").agg(pass_rate=("passed", "mean")).reset_index()
    per_test = per_test.merge(pass_rate_tbl, on="test_id", how="left")

    per_test["std_accuracy"] = per_test["std_accuracy"].fillna(0)
    per_test["avg_accuracy_pct"] = (per_test["avg_accuracy"] * 100).round(1)
    per_test["pass_rate_pct"] = (per_test["pass_rate"] * 100).round(1)

    # Map test_id -> name
    name_map = df[["test_id", "name"]].dropna().drop_duplicates("test_id")
    per_test = per_test.merge(name_map, on="test_id", how="left")
    per_test["test_label"] = per_test["name"].fillna(per_test["test_id"].astype(str))

    # Work-habits proxy within learner tests
    sp = per_test["avg_speed_qpm"].replace([np.inf, -np.inf], np.nan).fillna(per_test["avg_speed_qpm"].median())
    sp_min, sp_max = sp.min(), sp.max()
    sp_norm = (sp - sp_min) / ((sp_max - sp_min) if sp_max > sp_min else 1)

    stability = 1 / (1 + per_test["std_accuracy"])
    evidence = per_test["attempts"] / (per_test["attempts"] + 3)

    per_test["test_work_habits_index"] = (
        (0.65 * per_test["avg_accuracy"].fillna(0)) +
        (0.15 * sp_norm.fillna(0)) +
        (0.20 * stability.fillna(0))
    ) * evidence.fillna(0)

    per_test["test_work_habits_score"] = (per_test["test_work_habits_index"].rank(pct=True) * 100).round(1)

    def _test_status(row):
        if row["attempts"] < 2:
            return "Low evidence"
        if pd.notna(row["pass_rate"]) and row["pass_rate"] >= 0.7 and (row["avg_accuracy"] or 0) >= 0.6:
            return "On track"
        if pd.notna(row["pass_rate"]) and row["pass_rate"] < 0.5:
            return "At risk"
        if (row["avg_accuracy"] or 0) < 0.5:
            return "At risk"
        return "Improving"

    per_test["test_status"] = per_test.apply(_test_status, axis=1)

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
        "Note: This is readiness per test. Low evidence means we need more attempts on that specific test to be confident. "
        "Pass rate excludes ambiguous pass_mark tests when strict pass_mark is ON."
    )

st.subheader("Per-test performance table")

# Base table uses all eligible attempts for this user
ut = user_tests.copy()

# Preferred test label
label_col = "name" if "name" in df.columns else None
if label_col:
    name_map = df[["test_id", label_col]].dropna().drop_duplicates("test_id").rename(columns={label_col: "test_name"})
    ut = ut.merge(name_map, on="test_id", how="left")
else:
    ut["test_name"] = ut["test_id"].astype(str)
ut["test_name"] = ut.get("test_name", ut["test_id"].astype(str)).fillna(ut["test_id"].astype(str))

# Total marks candidate per test (best effort). Full-test totals should come
# from COUNT(test_questions), exposed as max_marks_db. no_of_questions is raw DQ
# evidence only because the audit found impossible values.
ut["total_marks_candidate"] = np.nan
if "max_marks_db" in ut.columns:
    ut["total_marks_candidate"] = ut["total_marks_candidate"].fillna(pd.to_numeric(ut["max_marks_db"], errors="coerce"))
if "total_questions" in ut.columns:
    ut["total_marks_candidate"] = ut["total_marks_candidate"].fillna(pd.to_numeric(ut["total_questions"], errors="coerce"))
if "question_limit" in ut.columns:
    ut["total_marks_candidate"] = ut["total_marks_candidate"].fillna(pd.to_numeric(ut["question_limit"], errors="coerce"))

# Pass/fail only on non-ambiguous pass marks (when strict)
ut_pass = ut.copy()
if config.strict_pass_mark and "pass_mark_ambiguous" in ut_pass.columns:
    ut_pass = ut_pass[~ut_pass["pass_mark_ambiguous"]].copy()

if "pass_mark" in ut_pass.columns and ut_pass["pass_mark"].notna().any():
    ut_pass["passed"] = (
        pd.to_numeric(ut_pass["marks"], errors="coerce") >= pd.to_numeric(ut_pass["pass_mark"], errors="coerce")
    ).astype(int)
else:
    ut_pass["passed"] = np.nan

# Speed only where speed_eligible
ut_speed = ut.copy()
if "speed_eligible" in ut_speed.columns:
    ut_speed = ut_speed[ut_speed["speed_eligible"]].copy()

# Aggregate core table
per_test = ut.groupby("test_id").agg(
    Test=("test_name", "first"),
    Attempts=("test_id", "count"),
    Avg_Accuracy=("accuracy_safe", "mean"),
    Avg_Speed_qpm=("speed_acc_raw", "mean"),
    Highest_Score=("marks", "max"),
    Lowest_Score=("marks", "min"),
    Avg_Score=("marks", "mean"),
    Total_Test_Mark=("total_marks_candidate", "max"),
).reset_index()

# Replace Avg_Speed_qpm with speed-only aggregation if you want strict speed eligibility
spd = ut_speed.groupby("test_id").agg(Avg_Speed_qpm=("speed_acc_raw", "mean"))
per_test = per_test.merge(spd, on="test_id", how="left", suffixes=("", "_spd"))
per_test["Avg_Speed_qpm"] = per_test["Avg_Speed_qpm_spd"].fillna(per_test["Avg_Speed_qpm"])
per_test.drop(columns=["Avg_Speed_qpm_spd"], inplace=True, errors="ignore")

# Pass/fail counts
pf = ut_pass.groupby("test_id").agg(
    Passes=("passed", lambda x: int((x == 1).sum()) if x.notna().any() else 0),
    Fails=("passed", lambda x: int((x == 0).sum()) if x.notna().any() else 0),
    Graded_Attempts=("passed", lambda x: int(x.notna().sum())),
).reset_index()
per_test = per_test.merge(pf, on="test_id", how="left")

per_test["Passes"] = per_test["Passes"].fillna(0).astype(int)
per_test["Fails"] = per_test["Fails"].fillna(0).astype(int)
per_test["Graded_Attempts"] = per_test["Graded_Attempts"].fillna(0).astype(int)

# Formatting
per_test["Avg_Accuracy_%"] = (per_test["Avg_Accuracy"] * 100).round(1)
per_test["Avg_Speed_qpm"] = per_test["Avg_Speed_qpm"].round(2)
per_test["Avg_Score"] = per_test["Avg_Score"].round(2)

# Coverage note
coverage = (per_test["Graded_Attempts"].sum() / per_test["Attempts"].sum() * 100) if per_test["Attempts"].sum() else 0
st.caption(f"Pass/Fail computed on {coverage:.1f}% of attempts (ambiguous pass_mark excluded when strict).")

show_cols = [
    "Test", "Attempts",
    "Avg_Accuracy_%", "Avg_Speed_qpm",
    "Highest_Score", "Lowest_Score", "Avg_Score",
    "Total_Test_Mark",
    "Passes", "Fails",
]
st.dataframe(per_test[show_cols].sort_values(["Avg_Accuracy_%", "Attempts"], ascending=[False, False]), use_container_width=True)

# ---------------------------
# Weekly trends
# ---------------------------
st.divider()
st.subheader("Trends (Weekly)")

if "created_at" in user_tests.columns and user_tests["created_at"].notna().any():
    ut = user_tests.dropna(subset=["created_at"]).copy()
    ut["week"] = ut["created_at"].dt.to_period("W").dt.start_time

    # pass_ratio uses strict pass set
    if "pass_mark" in ut.columns and ut["pass_mark"].notna().any():
        # exclude ambiguous
        if config.strict_pass_mark and "pass_mark_ambiguous" in ut.columns:
            utp = ut[~ut["pass_mark_ambiguous"]].copy()
        else:
            utp = ut.copy()
        utp["pass_ratio"] = (
            pd.to_numeric(utp["marks"], errors="coerce") / pd.to_numeric(utp["pass_mark"], errors="coerce")
        ).replace([np.inf, -np.inf], np.nan)
        ut = ut.merge(utp[["test_id", "created_at", "pass_ratio"]], on=["test_id", "created_at"], how="left")

    weekly = ut.groupby("week").agg(
        weekly_score=("marks", "mean"),
        weekly_accuracy=("accuracy_safe", "mean"),
        weekly_speed_qpm=("speed_acc_raw", "mean"),
        weekly_pass_ratio=("pass_ratio", "mean") if "pass_ratio" in ut.columns else ("accuracy_safe", "mean"),
        attempts=("test_id", "count"),
    ).reset_index()

    fig_w_score = px.line(weekly, x="week", y="weekly_score", title="Average score by week", markers=True)
    st.plotly_chart(fig_w_score, use_container_width=True)

    fig_w_acc = px.line(weekly, x="week", y="weekly_accuracy", title="Accuracy by week (safe)", markers=True)
    st.plotly_chart(fig_w_acc, use_container_width=True)

    fig_w_spd = px.line(weekly, x="week", y="weekly_speed_qpm", title="Speed by week (q/min)", markers=True)
    st.plotly_chart(fig_w_spd, use_container_width=True)
else:
    st.info("No valid timestamps (created_at) available for weekly trend plots.")

# ---------------------------
# Learner vs peers (accuracy distribution by test)
# ---------------------------
st.divider()
st.subheader("Learner vs peers (accuracy distribution by test)")

test_name_col = None
for cand in ["name", "test_name", "title"]:
    if cand in df.columns:
        test_name_col = cand
        break

learner_test_ids = user_tests["test_id"].dropna().unique().tolist()
if not learner_test_ids:
    st.info("No tests found for this learner.")
else:
    selected_test_id = st.selectbox("Select a test to compare (accuracy)", learner_test_ids)

    test_label = str(selected_test_id)
    if test_name_col:
        nm = df.loc[df["test_id"] == selected_test_id, test_name_col].dropna()
        if not nm.empty:
            test_label = nm.iloc[0]

    peers = df[df["test_id"] == selected_test_id].copy()
    if "accuracy_safe" not in peers.columns:
        peers = compute_basic_metrics2(peers)
        # fallback
        peers["accuracy_safe"] = peers.get("accuracy_total", np.nan)

    peers_acc = pd.to_numeric(peers["accuracy_safe"], errors="coerce").dropna()
    learner_acc = pd.to_numeric(user_tests[user_tests["test_id"] == selected_test_id]["accuracy_safe"], errors="coerce").dropna()

    if peers_acc.shape[0] < 5 or learner_acc.empty:
        st.info("Not enough peer data or learner accuracy available for this test.")
    else:
        fig_hist = px.histogram(
            peers_acc.to_frame(name="accuracy_safe"),
            x="accuracy_safe",
            nbins=15,
            title=f"Accuracy distribution for '{test_label}' (peers) with learner overlay",
        )
        for a in learner_acc.tolist():
            fig_hist.add_vline(x=float(a), line_width=3)

        st.plotly_chart(fig_hist, use_container_width=True)

        peer_vals = peers_acc.values
        learner_med = float(np.median(learner_acc.values))
        pct = float((peer_vals < learner_med).mean() * 100)
        st.caption(f"Learner median accuracy is approximately at the {pct:.1f}th percentile among peers for this test.")

# ---------------------------
# Speed vs Accuracy scatter
# ---------------------------
st.divider()
st.subheader("Speed vs Accuracy (each attempt)")

if "speed_acc_raw" in user_tests.columns and "accuracy_safe" in user_tests.columns and user_tests[["speed_acc_raw", "accuracy_safe"]].dropna().shape[0] >= 2:
    fig_scatter = px.scatter(
        user_tests,
        x="speed_acc_raw",
        y="accuracy_safe",
        title="Speed (q/min) vs Accuracy (safe)",
        hover_data=["test_id"] if "test_id" in user_tests.columns else None,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Not enough attempt data to plot speed vs accuracy for this learner.")

# ---------------------------
# Difficulty summary
# ---------------------------
st.divider()
st.subheader("Test difficulty summary")
st.info("Easy -> difficulty 0.00-0.59")
st.info("Moderate -> difficulty 0.60-0.89")
st.info("Hard -> difficulty 0.90-1.00")

user_diff = user_tests[["test_id"]].drop_duplicates().merge(diff_df, on="test_id", how="left")

if user_diff.empty or user_diff.get("difficulty_label", pd.Series(dtype=object)).isna().all():
    st.info("No difficulty classification available for this learner's tests.")
else:
    test_name_col = None
    for cand in ["name", "test_name", "title"]:
        if cand in df.columns:
            test_name_col = cand
            break

    if test_name_col:
        name_map = df[["test_id", test_name_col]].dropna().drop_duplicates("test_id").rename(columns={test_name_col: "test_name"})
        user_diff = user_diff.merge(name_map, on="test_id", how="left")
        user_diff["label"] = user_diff["test_name"].fillna(user_diff["test_id"].astype(str))
    else:
        user_diff["label"] = user_diff["test_id"].astype(str)

    cols_to_show = ["label", "difficulty_label"]
    if "test_stability" in user_diff.columns:
        cols_to_show.append("test_stability")
    if "difficulty" in user_diff.columns:
        cols_to_show.append("difficulty")
    if "stability" in user_diff.columns:
        cols_to_show.append("stability")

    st.dataframe(user_diff[cols_to_show].sort_values("difficulty_label"), use_container_width=True)

    counts = user_diff["difficulty_label"].value_counts(dropna=True).rename_axis("difficulty").reset_index(name="tests")
    fig_counts = px.bar(counts, x="difficulty", y="tests", title="Difficulty distribution (count of tests)", text_auto=True)
    st.plotly_chart(fig_counts, use_container_width=True)

    if "difficulty" in user_diff.columns and user_diff["difficulty"].notna().any():
        fig_diff = px.bar(user_diff, x="label", y="difficulty", title="Difficulty score per test", text_auto=True)
        fig_diff.update_layout(xaxis_title="Test", xaxis_tickangle=-30)
        st.plotly_chart(fig_diff, use_container_width=True)
