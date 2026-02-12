import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.insights import apply_insight_engine
from utils.metrics import (
    load_data_from_disk_or_session,
    compute_basic_metrics2,
    compute_sab_behavioral,
    compute_test_analytics,
)

from utils.institute_standardization import standardize_institute  # ensure exists

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Institute Performance", layout="wide")
st.title("Institute Performance Summary")

# ----------------------------
# Helpers
# ----------------------------
def fmt_pct(x, decimals=0):
    """Format fraction (0..1) as percentage string."""
    try:
        if pd.isna(x):
            return "—"
        return f"{x*100:.{decimals}f}%"
    except Exception:
        return "—"

def fmt_pct_from_0_100(x, decimals=0):
    """Format 0..100 as percentage string."""
    try:
        if pd.isna(x):
            return "—"
        return f"{x:.{decimals}f}%"
    except Exception:
        return "—"

def fmt_num(x, decimals=1):
    try:
        if pd.isna(x):
            return "—"
        return f"{x:.{decimals}f}"
    except Exception:
        return "—"
        
def fmt_pct_guard(p, n, decimals=0):
    """
    p: proportion 0..1
    n: count corresponding to p
    If rounding to 0 decimals would show 0% but n>0 and p<1%, show '<1%'.
    """
    if pd.isna(p):
        return "—"
    if decimals == 0 and n > 0 and p < 0.01:
        return "<1%"
    return f"{p*100:.{decimals}f}%"

def detect_school_id_column(df: pd.DataFrame):
    """
    Detect the School ID / username column.
    Prefer explicit school identifiers, then username-like fields.
    """
    candidates = [
        "school_id", "school_user_id", "student_id", "student_code",
        "username", "user_name", "login", "account_username",
        "candidate_id", "index_number"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def detect_datetime_column(df: pd.DataFrame):
    """
    Detect attempt timestamp column for trends.
    """
    candidates = [
        "attempt_time", "attempted_at", "created_at", "timestamp",
        "submitted_at", "date", "datetime"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce", utc=False)

def compute_pass_fields(df_attempts: pd.DataFrame,
                        marks_col: str = "marks",
                        pass_mark_col: str = "pass_mark",
                        default_pass_mark: float | None = None) -> pd.DataFrame:
    """
    Adds:
      - pass_ratio = marks / pass_mark
      - is_pass = marks >= pass_mark
    Supports:
      - per-row pass_mark column, OR
      - a global default_pass_mark (if pass_mark column missing)
    """
    out = df_attempts.copy()

    if marks_col not in out.columns:
        st.error(f"Missing required column: `{marks_col}`")
        st.stop()

    if pass_mark_col in out.columns:
        out[pass_mark_col] = pd.to_numeric(out[pass_mark_col], errors="coerce")
    else:
        if default_pass_mark is None:
            st.error(f"Missing `{pass_mark_col}` and no default_pass_mark provided.")
            st.stop()
        out[pass_mark_col] = float(default_pass_mark)

    out[marks_col] = pd.to_numeric(out[marks_col], errors="coerce")

    # Avoid divide-by-zero
    out["pass_ratio"] = np.where(
        out[pass_mark_col] > 0,
        out[marks_col] / out[pass_mark_col],
        np.nan
    )

    out["is_pass"] = (out[marks_col] >= out[pass_mark_col]).fillna(False)

    return out

# ----------------------------
# Load
# ----------------------------
df = load_data_from_disk_or_session()
if df is None or df.empty:
    st.warning("Upload data to continue.")
    st.stop()

# ----------------------------
# Standardize Institute (DO THIS EARLY)
# ----------------------------
df = standardize_institute(
    df=df,
    column="institute",
    mapping_path="data/mapping.csv"
)

if "institute_std" not in df.columns:
    st.error("Standardization failed: missing `institute_std`.")
    st.stop()

df["institute_std"] = df["institute_std"].fillna("Unknown").astype(str)

unknown_rate = (df["institute_std"] == "Unknown").mean()
with st.expander("Data quality checks"):
    st.write("Institute 'Unknown' rate:", fmt_pct(unknown_rate, decimals=1))
    st.write("Top standardized institutes:")
    st.dataframe(df["institute_std"].value_counts().head(15).reset_index().rename(
        columns={"index": "Institute", "institute_std": "Rows"}
    ), use_container_width=True)

# ----------------------------
# Compute Metrics
# ----------------------------
df = compute_basic_metrics2(df)              # attempt-level enrichments
sab_df = compute_sab_behavioral(df)          # user-level metrics

# ----------------------------
# Attach Institute + SchoolID to sab_df robustly
# ----------------------------
school_id_col = detect_school_id_column(df)

# canonical per-user mapping
cols = ["user_id", "institute_std"]
if school_id_col:
    cols.append(school_id_col)

user_map = df[cols].drop_duplicates("user_id").copy()

# ensure sab_df has institute_std
if "institute_std" in sab_df.columns:
    sab_df = sab_df.drop(columns=["institute_std"])
sab_df = sab_df.merge(user_map, on="user_id", how="left")
sab_df["institute_std"] = sab_df["institute_std"].fillna("Unknown").astype(str)

# create a display identifier column for UI tables
if school_id_col:
    sab_df["learner_id_display"] = sab_df[school_id_col].astype(str)
else:
    # fallback: still show user_id if no school id exists
    sab_df["learner_id_display"] = sab_df["user_id"].astype(str)

# apply insight engine (your provided logic maps insight_code -> exam_status/message/action)
sab_df = apply_insight_engine(sab_df)

# test analytics (test-level)
test_df = compute_test_analytics(df)

# ----------------------------
# View Toggle (Head of School first, Minister optional)
# ----------------------------
view_mode = st.radio(
    "View mode",
    ["Head of School", "High-level"],
    horizontal=True
)

# ----------------------------
# Institute Selector
# Build from attempt-level df to avoid missing institutes
# ----------------------------
#institutes = sorted(df["institute_std"].unique().tolist())
#institute = st.selectbox("Select Institute", institutes)

# ----------------------------
# Institute Selector (Mapped-only, with search)
# ----------------------------
MAPPING_PATH = "data/mapping.csv"

mapping_df = pd.read_csv(MAPPING_PATH)

# Canonical mapped institute names from your mapping scheme
mapped_institutes = (
    mapping_df["institute_standardized"]
    .dropna()
    .astype(str)
    .str.strip()
)

# Remove placeholders / non-real names
mapped_institutes = mapped_institutes[
    ~mapped_institutes.isin(["", "Other", "Unmapped", "N/A", "NA"])
]

# Only show mapped institutes that actually appear in the dataset
institutes = sorted(
    set(mapped_institutes).intersection(set(df["institute_std"].dropna().astype(str)))
)

if not institutes:
    st.warning("No mapped institutes found in the current dataset.")
    st.stop()

search = st.text_input("Search institute", value="")
if search.strip():
    q = search.strip().lower()
    filtered_institutes = [i for i in institutes if q in i.lower()]
else:
    filtered_institutes = institutes

if not filtered_institutes:
    st.warning("No matches found. Try a different search term.")
    st.stop()

st.caption(f"Showing {len(filtered_institutes)} of {len(institutes)} mapped institutes")

institute = st.selectbox("Select Institute", filtered_institutes)


# slice
sab_inst_users = sab_df[sab_df["institute_std"] == institute].copy()
df_inst_attempts = df[df["institute_std"] == institute].copy()

# ----------------------------
# Pass/Fail fields
# ----------------------------
# If pass_mark is stored per attempt, keep default_pass_mark=None
# If you don't have a pass_mark column, set a default like 50 (or 50% depending on scale)
df_inst_attempts = compute_pass_fields(
    df_inst_attempts,
    marks_col="marks",
    pass_mark_col="pass_mark",
    default_pass_mark=None
)


if sab_inst_users.empty:
    st.info("No learner-level records found for this institute (SAB table empty).")
    st.stop()

# ----------------------------
# Friendly Segmentation (consistent)
# Use exam_status only (READY / BORDERLINE / AT-RISK mapped via your engine)
# ----------------------------
status_counts = sab_inst_users["exam_status"].value_counts(dropna=False)
n_learners = sab_inst_users["user_id"].nunique()

eligible_n = int(status_counts.get("Eligible", 0))
cond_n = int(status_counts.get("Conditionally Eligible", 0))
not_eligible_n = int(status_counts.get("Not Eligible", 0))

eligible_pct = eligible_n / n_learners if n_learners else np.nan
cond_pct = cond_n / n_learners if n_learners else np.nan
risk_pct = not_eligible_n / n_learners if n_learners else np.nan

# ----------------------------
# KPI Metrics (user-friendly labels)
# ----------------------------


row1 = st.columns(3)
row1[0].metric("Learners", f"{n_learners}")
row1[1].metric("Unique Tests Taken", f"{df_inst_attempts['test_id'].nunique() if 'test_id' in df_inst_attempts.columns else 0}")
row1[2].metric("Total Attempts", f"{len(df_inst_attempts)}")
#row1[3].metric("Institute Data Quality (Unknown Institute Rate)", fmt_pct(unknown_rate, 1))

# accuracy_total is typically 0..1
avg_acc = df_inst_attempts["accuracy_total"].mean() if "accuracy_total" in df_inst_attempts.columns else np.nan
avg_speed = df_inst_attempts["speed_raw"].mean() if "speed_raw" in df_inst_attempts.columns else np.nan
avg_sab = sab_inst_users["robust_SAB_scaled"].mean() if "robust_SAB_scaled" in sab_inst_users.columns else np.nan

row2 = st.columns(3)
row2[0].metric("Average Accuracy", fmt_pct(avg_acc, 0))
row2[1].metric("Average Speed ", f"{fmt_num(avg_speed, 2)}")
row2[2].metric("Average Readiness Score (0–100)", f"{fmt_num(avg_sab, 1)}")

row3 = st.columns(3)
row3[0].metric("At-risk learners", f"{not_eligible_n}")
row3[1].metric("Almost ready learners", f"{cond_n}")
row3[2].metric("Exam-ready learners", f"{eligible_n}")

st.divider()
# ----------------------------
# Institute Summary Narrative
# ----------------------------
# ----------------------------
# Institute Summary Narrative (Dynamic, non-generic)
# Place this RIGHT AFTER segmentation counts/pcts and BEFORE KPIs
# ----------------------------
#st.subheader("What this means (plain English)")

# 1) Always show the three-group snapshot with the <1% guard
st.markdown(
    f"""
- **Exam-ready:** {fmt_pct_guard(eligible_pct, eligible_n, 0)} (**{eligible_n} learner{'s' if eligible_n != 1 else ''}**)  
- **Almost ready:** {fmt_pct_guard(cond_pct, cond_n, 0)} (**{cond_n} learners**)  
- **At risk:** {fmt_pct_guard(risk_pct, not_eligible_n, 0)} (**{not_eligible_n} learners**)  
"""
)

# 2) Add message lines that only appear if the group exists
bullets = []

if not_eligible_n > 0:
    bullets.append(f"**At risk:** {not_eligible_n} learners need foundational support before exam attempts.")
else:
    bullets.append("**At risk:** None detected in current data (good sign).")

if cond_n > 0:
    bullets.append(f"**Almost ready:** {cond_n} learners are close—targeted support can move them into exam-ready.")
else:
    bullets.append("**Almost ready:** None detected in current data.")

if eligible_n > 0:
    bullets.append(f"**Exam-ready:** {eligible_n} learners can proceed to mock exams to maintain readiness.")
else:
    bullets.append("**Exam-ready:** None detected yet—focus on building consistency through more practice.")

st.markdown("\n".join([f"- {b}" for b in bullets]))

# 3) Policy signal vs recommended approach:
#    - Minister view: one policy sentence, but only if relevant
#    - Head of School: weekly action plan, but only if there is something to act on
if view_mode == "High-level":
    if not_eligible_n > 0 or cond_n > 0:
        st.markdown(
            f"**Policy signal:** Focus resources on **{not_eligible_n} at-risk** learners and "
            f"move **{cond_n} almost-ready** learners into exam-ready status to lift overall performance."
        )
    else:
        st.markdown("**Policy signal:** No immediate risk pockets detected in this institute’s current data.")
else:
    # Head of School weekly plan (only show steps if needed)
    st.markdown("**Recommended actions this week (based on your data):**")
    action_lines = []
    if not_eligible_n > 0:
        action_lines.append(f"- Prioritize **{not_eligible_n} at-risk** learners for remediation sessions.")
    if cond_n > 0:
        action_lines.append(f"- Push **{cond_n} almost-ready** learners to close gaps and complete 2–3 more tests.")
    if eligible_n > 0:
        action_lines.append(f"- Keep **{eligible_n} exam-ready** learners on mock exams to maintain stability.")

    if action_lines:
        st.markdown("\n".join(action_lines))
    else:
        st.markdown("- No action items detected from current data (no learners in any group).")

st.divider()


st.divider()

st.subheader("Pass / fail overview")

passes = int(df_inst_attempts["is_pass"].sum())
fails = int((~df_inst_attempts["is_pass"]).sum())
overall_pass_rate = df_inst_attempts["is_pass"].mean() if len(df_inst_attempts) else np.nan
avg_pass_ratio = df_inst_attempts["pass_ratio"].mean() if len(df_inst_attempts) else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall pass rate", fmt_pct(overall_pass_rate, 1))
c2.metric("Passes (attempts)", f"{passes}")
c3.metric("Fails (attempts)", f"{fails}")
c4.metric("Avg pass ratio (marks ÷ pass mark)", fmt_pct(avg_pass_ratio, 2))

st.subheader("Pass rate per test taken")

if "name" not in df_inst_attempts.columns:
    st.info("No test_id column found — cannot compute pass rate per test.")
else:
    test_pass = (
        df_inst_attempts.groupby("name", as_index=False)
        .agg(
            Attempts=("is_pass", "size"),
            PassRate=("is_pass", "mean"),
            AvgPassRatio=("pass_ratio", "mean")
        )
    )
    test_pass["PassRatePct"] = test_pass["PassRate"] * 100

    # Hardest tests: low pass rate + meaningful attempts
    min_attempts = st.slider("Minimum attempts to rank a test", 5, 100, 10)
    hardest = test_pass[test_pass["Attempts"] >= min_attempts].sort_values(
        ["PassRate", "Attempts"], ascending=[True, False]
    )

    st.markdown("**Hardest tests (priority for review / intervention):**")
    st.dataframe(
        hardest.rename(columns={
            "name": "Test",
            #"name": "Test",
            "Attempts": "Attempts",
            "PassRatePct": "Pass rate (%)",
            "AvgPassRatio": "Avg pass ratio"
        })[["Test", "Attempts", "Pass rate (%)", "Avg pass ratio"]].head(30),
        use_container_width=True
    )

    fig_test_pass = px.bar(
        hardest.head(30),
        x="name",
        y="PassRatePct",
        text="Attempts",
        title="Pass rate by test (filtered to meaningful attempt counts)"
    )
    st.plotly_chart(fig_test_pass, use_container_width=True)

st.divider()
# ----------------------------
# Readiness Breakdown (visual)
# ----------------------------
st.subheader("Readiness breakdown")

seg_df = pd.DataFrame({
    "Group": ["Exam-ready", "Almost ready", "At risk"],
    "Learners": [eligible_n, cond_n, not_eligible_n]
})

fig_seg = px.bar(
    seg_df,
    x="Group",
    y="Learners",
    text="Learners",
    title="Learners by readiness group"
)
st.plotly_chart(fig_seg, use_container_width=True)

# Insight code distribution (diagnostic)
with st.expander("Detailed diagnostic breakdown (risk types)"):
    vc = sab_inst_users["insight_code"].value_counts(dropna=False).reset_index()
    vc.columns = ["Insight type", "Learners"]
    st.dataframe(vc, use_container_width=True)

    fig_vc = px.bar(vc, x="Insight type", y="Learners", text="Learners", title="Learners by insight type")
    st.plotly_chart(fig_vc, use_container_width=True)

st.divider()

# ----------------------------
# "What to do this week" (action counts)
# ----------------------------
st.subheader("Recommended actions (this week)")

action_counts = (
    sab_inst_users["recommended_action"]
    .fillna("No action generated")
    .value_counts()
    .reset_index()
)
action_counts.columns = ["Recommended action", "Learners"]

st.dataframe(action_counts, use_container_width=True)

fig_actions = px.bar(action_counts, x="Recommended action", y="Learners", text="Learners", title="Action plan volume")
st.plotly_chart(fig_actions, use_container_width=True)

st.divider()

# ----------------------------
# Priority lists (non-technical, decision-ready)
# ----------------------------
st.subheader("Priority intervention list (at-risk learners)")

at_risk_df = sab_inst_users[sab_inst_users["exam_status"] == "Not Eligible"].copy()
# sort: lowest readiness first, then lowest test_count
sort_cols = [c for c in ["robust_SAB_scaled", "test_count"] if c in at_risk_df.columns]
if sort_cols:
    at_risk_df = at_risk_df.sort_values(sort_cols, ascending=True)

cols_show = ["learner_id_display"]
for c in ["test_count", "mean_accuracy", "mean_speed", "robust_SAB_scaled", "insight_code", "recommended_action"]:
    if c in at_risk_df.columns:
        cols_show.append(c)

# convert mean_accuracy to %
if "mean_accuracy" in at_risk_df.columns:
    at_risk_df["mean_accuracy_pct"] = at_risk_df["mean_accuracy"].apply(lambda x: x*100 if pd.notna(x) else np.nan)
    cols_show = [c for c in cols_show if c != "mean_accuracy"]
    cols_show.insert(2, "mean_accuracy_pct")

# nicer column names
rename_map = {
    "learner_id_display": "Learner (School ID)",
    "test_count": "Tests taken",
    "mean_accuracy_pct": "Avg accuracy (%)",
    "mean_speed": "Avg speed (time/item)",
    "robust_SAB_scaled": "Readiness (0–100)",
    "insight_code": "Main issue",
    "recommended_action": "Recommended action"
}

show_table = at_risk_df[cols_show].rename(columns=rename_map)

# limit for demo cleanliness
st.dataframe(show_table.head(50), use_container_width=True)

st.subheader("Almost ready (quick wins)")
near_df = sab_inst_users[sab_inst_users["exam_status"] == "Conditionally Eligible"].copy()
if "robust_SAB_scaled" in near_df.columns:
    near_df = near_df.sort_values("robust_SAB_scaled", ascending=True)

cols_show2 = ["learner_id_display"]
for c in ["test_count", "mean_accuracy", "mean_speed", "robust_SAB_scaled", "recommended_action"]:
    if c in near_df.columns:
        cols_show2.append(c)

if "mean_accuracy" in near_df.columns:
    near_df["mean_accuracy_pct"] = near_df["mean_accuracy"].apply(lambda x: x*100 if pd.notna(x) else np.nan)
    cols_show2 = [c for c in cols_show2 if c != "mean_accuracy"]
    cols_show2.insert(2, "mean_accuracy_pct")

show_table2 = near_df[cols_show2].rename(columns=rename_map)
st.dataframe(show_table2.head(50), use_container_width=True)

st.subheader("Top learners (by pass performance)")

# attempt-level summary per learner
if "username" in df_inst_attempts.columns:
    learner_pass = (
        df_inst_attempts.groupby("username", as_index=False)
        .agg(
            Attempts=("is_pass", "size"),
            Passes=("is_pass", "sum"),
            PassRate=("is_pass", "mean"),
            AvgPassRatio=("pass_ratio", "mean")
        )
    )
    learner_pass["PassRatePct"] = learner_pass["PassRate"] * 100
    learner_pass = learner_pass.sort_values(
        ["PassRate", "Attempts", "AvgPassRatio"],
        ascending=[False, False, False]
    )

    st.dataframe(
        learner_pass.rename(columns={
            "username": "Learner (School ID)",
            "Attempts": "Attempts",
            "Passes": "Passes",
            "PassRatePct": "Pass rate (%)",
            "AvgPassRatio": "Avg pass ratio"
        })[["Learner (School ID)", "Attempts", "Passes", "Pass rate (%)", "Avg pass ratio"]].head(30),
        use_container_width=True
    )
else:
    st.info("Missing `username` column — cannot rank learners by passes.")

st.divider()

# ----------------------------
# Learner Explorer (filters)
# ----------------------------
st.subheader("Learners (filter & review)")

status_options = [s for s in ["Eligible", "Conditionally Eligible", "Not Eligible"] if s in sab_inst_users["exam_status"].unique()]
selected_status = st.multiselect(
    "Filter by readiness group",
    status_options,
    default=status_options
)

filtered = sab_inst_users[sab_inst_users["exam_status"].isin(selected_status)].copy()

# add stakeholder-friendly columns
cols_f = ["learner_id_display"]
for c in ["exam_status", "insight_message", "recommended_action", "test_count", "robust_SAB_scaled"]:
    if c in filtered.columns:
        cols_f.append(c)

rename_map2 = {
    "learner_id_display": "Learner (School ID)",
    "exam_status": "Readiness group",
    "insight_message": "What we see",
    "recommended_action": "What to do next",
    "test_count": "Tests taken",
    "robust_SAB_scaled": "Readiness (0–100)",
}

st.dataframe(filtered[cols_f].rename(columns=rename_map2), use_container_width=True)

st.divider()

# ----------------------------
# Trends over time (if timestamp exists)
# ----------------------------
st.subheader("Trends")

dt_col = detect_datetime_column(df_inst_attempts)
if dt_col:
    tmp = df_inst_attempts.copy()
    tmp[dt_col] = safe_to_datetime(tmp[dt_col])
    tmp = tmp.dropna(subset=[dt_col])

    if not tmp.empty:
        # Weekly attempts & accuracy
        tmp["week"] = tmp[dt_col].dt.to_period("W").astype(str)
        wk = tmp.groupby("week").agg(
            Attempts=("user_id", "size"),
            AvgAccuracy=("accuracy_total", "mean") if "accuracy_total" in tmp.columns else ("user_id", "size"),
            AvgSpeed=("speed_raw", "mean") if "speed_raw" in tmp.columns else ("user_id", "size")
        ).reset_index()

        if "AvgAccuracy" in wk.columns:
            wk["AvgAccuracyPct"] = wk["AvgAccuracy"] * 100

        fig_wk1 = px.line(wk, x="week", y="Attempts", title="Weekly platform activity (attempts)")
        st.plotly_chart(fig_wk1, use_container_width=True)

        if "AvgAccuracyPct" in wk.columns:
            fig_wk2 = px.line(wk, x="week", y="AvgAccuracyPct", title="Weekly average accuracy (%)")
            st.plotly_chart(fig_wk2, use_container_width=True)
    else:
        st.info("Attempt date column detected, but no valid dates found to show trends.")
else:
    st.info("No attempt date column detected (add one like created_at/timestamp to enable trends).")

st.divider()

# ----------------------------
# Test stability & difficulty (reframed)
# ----------------------------
st.subheader("Assessment quality check (are tests stable and fair?)")

if "test_id" in df_inst_attempts.columns and "test_id" in test_df.columns:
    inst_tests = test_df[test_df["test_id"].isin(df_inst_attempts["test_id"])].copy()

    if not inst_tests.empty:
        # Rename axes for clarity if columns exist
        x_col = "mean_accuracy" if "mean_accuracy" in inst_tests.columns else None
        y_col = "speed_consistency" if "speed_consistency" in inst_tests.columns else None

        if x_col and y_col:
            fig = px.scatter(
                inst_tests,
                x=x_col,
                y=y_col,
                size="taker_count" if "taker_count" in inst_tests.columns else None,
                color="time_consistency" if "time_consistency" in inst_tests.columns else None,
                hover_data=["test_id"],
                title="Which tests are reliable? (higher consistency = more stable measurement)"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
**How to read this chart (plain English):**
- Each dot is a test.
- Bigger dots = more learners attempted it (more evidence).
- Higher consistency suggests the test measures performance more reliably.
- Use this to flag assessments that may need review (too inconsistent or unstable).
"""
            )
        else:
            st.info("Test analytics table exists but missing required columns for the stability map.")
    else:
        st.info("No test analytics records found for this institute.")
else:
    st.info("Missing test_id in attempts or test analytics; cannot show assessment quality.")

