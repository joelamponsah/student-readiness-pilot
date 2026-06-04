import numpy as np
import pandas as pd
import streamlit as st

from utils.metrics import get_v13_artifacts


st.set_page_config(page_title="Class Summary", layout="wide")
st.title("Class Summary")
st.caption("v1.3 Test / Exercise Readiness by class. BLS/ALS/CAS are proxy signals.")

raw_df, artifacts = get_v13_artifacts()
if raw_df is None or not artifacts:
    st.warning("No raw_attempts.csv input or shared v1.3 artifacts are available.")
    st.stop()

user_test_summary = artifacts.get("user_test_summary")
group_summary = artifacts.get("group_summary")
readiness_user = artifacts.get("readiness_user")
proxy_sequence = artifacts.get("proxy_sequence")
dq_attempts = artifacts.get("dq_attempts")
smoke_report = artifacts.get("smoke_report")

required_frames = [user_test_summary, group_summary, readiness_user, proxy_sequence, dq_attempts, smoke_report]
if any(frame is None for frame in required_frames):
    st.warning("The shared artifact bundle is incomplete for class summary inspection.")
    st.stop()

if user_test_summary.empty or "class_id_std" not in user_test_summary.columns:
    st.warning("class_id_std is missing from user_test_summary, so class summaries cannot be built yet.")
    st.stop()

class_df = user_test_summary.loc[user_test_summary["class_id_std"].notna()].copy()
if class_df.empty:
    st.warning("No class_id_std values are available in user_test_summary.")
    st.stop()

class_df["class_id_std"] = class_df["class_id_std"].astype("string").str.strip()
class_df = class_df[class_df["class_id_std"] != ""].copy()
if class_df.empty:
    st.warning("No usable class_id_std values are available in user_test_summary.")
    st.stop()

st.markdown(
    """
### Boundary
- BLS / ALS / CAS are proxies only in v1.3.
- CAS Proxy is not true CAS yet.
- Difficulty / DCI is context, not score correction.
- No topic / subject / year-group inference is performed here.
- No final unified readiness score is created.
"""
)

class_meta = class_df.groupby("class_id_std", dropna=False).agg(
    learner_count=("user_id", "nunique"),
    test_count=("test_id", "nunique"),
).reset_index()

class_meta["class_label"] = class_meta.apply(
    lambda r: f"{r['class_id_std']} (learners: {int(r['learner_count'])} | tests: {int(r['test_count'])})",
    axis=1,
)
class_meta = class_meta.sort_values(["learner_count", "test_count", "class_id_std"], ascending=[False, False, True], kind="mergesort")

selected_class_label = st.selectbox("Select class", class_meta["class_label"].tolist())
selected_class_id = class_meta.loc[class_meta["class_label"] == selected_class_label, "class_id_std"].iloc[0]

class_rows = class_df[class_df["class_id_std"] == selected_class_id].copy()
class_learners = class_rows["user_id"].dropna().astype(str).unique().tolist()

readiness_class = readiness_user[readiness_user["user_id"].isin(class_rows["user_id"].unique())].copy()
proxy_class = proxy_sequence[proxy_sequence["user_id"].isin(class_rows["user_id"].unique())].copy()
dq_class = dq_attempts[dq_attempts["user_id"].isin(class_rows["user_id"].unique())].copy()

st.subheader(f"Selected Class: {selected_class_id}")
st.caption("Custom selected-test groups will be added after source-backed class summaries are validated.")

# ---------------------------
# Overview KPIs
# ---------------------------
st.subheader("Class Overview")
overview = st.columns(10)
overview[0].metric("Learners", f"{class_rows['user_id'].nunique():,}")
overview[1].metric("Tests/exercises", f"{class_rows['test_id'].nunique():,}")
overview[2].metric("User-test groups", f"{len(class_rows):,}")
overview[3].metric("BLS rows", f"{int(class_rows['bls_score_pct'].notna().sum()):,}" if "bls_score_pct" in class_rows.columns else "N/A")
overview[4].metric("Current ALS rows", f"{int(class_rows['current_als_score_pct'].notna().sum()):,}" if "current_als_score_pct" in class_rows.columns else "N/A")
overview[5].metric("Potential ALS rows", f"{int(class_rows['potential_als_score_pct'].notna().sum()):,}" if "potential_als_score_pct" in class_rows.columns else "N/A")
overview[6].metric("Avg BLS %", f"{class_rows['bls_score_pct'].mean():.2f}" if "bls_score_pct" in class_rows.columns and class_rows["bls_score_pct"].notna().any() else "N/A")
overview[7].metric("Avg Current ALS %", f"{class_rows['current_als_score_pct'].mean():.2f}" if "current_als_score_pct" in class_rows.columns and class_rows["current_als_score_pct"].notna().any() else "N/A")
overview[8].metric("Avg learning gain %", f"{class_rows['learning_gain_pct'].mean():.2f}" if "learning_gain_pct" in class_rows.columns and class_rows["learning_gain_pct"].notna().any() else "N/A")
overview[9].metric("Avg CAS proxy %", f"{class_rows['cas_proxy_score_pct'].mean():.2f}" if "cas_proxy_score_pct" in class_rows.columns and class_rows["cas_proxy_score_pct"].notna().any() else "N/A")

overview2 = st.columns(2)
overview2[0].metric(
    "Avg readiness probability %",
    f"{readiness_class['readiness_probability_pct'].mean():.2f}" if "readiness_probability_pct" in readiness_class.columns and readiness_class["readiness_probability_pct"].notna().any() else "N/A",
)
overview2[1].metric(
    "Avg robust_SAB_scaled",
    f"{readiness_class['robust_SAB_scaled'].mean():.2f}" if "robust_SAB_scaled" in readiness_class.columns and readiness_class["robust_SAB_scaled"].notna().any() else "N/A",
)

# ---------------------------
# Readiness distribution
# ---------------------------
st.subheader("Class Readiness Distribution")
dist_frames = []
for column in ["exam_status", "risk_band", "coverage_risk"]:
    if column in readiness_class.columns:
        dist = readiness_class[column].fillna("UNKNOWN").value_counts(dropna=False).rename_axis(column).reset_index(name="rows")
        dist.insert(0, "metric", column)
        dist_frames.append(dist.rename(columns={column: "value"}))

if dist_frames:
    st.dataframe(pd.concat(dist_frames, ignore_index=True, sort=False), use_container_width=True)
else:
    st.info("No readiness distribution fields are available for this class.")

# ---------------------------
# Class test / exercise table
# ---------------------------
st.subheader("Class Test / Exercise Table")
agg_map = {
    "learner_count": ("user_id", "nunique"),
    "mean_bls_score_pct": ("bls_score_pct", "mean"),
    "mean_current_als_score_pct": ("current_als_score_pct", "mean"),
    "mean_potential_als_score_pct": ("potential_als_score_pct", "mean"),
    "mean_learning_gain_pct": ("learning_gain_pct", "mean"),
    "cas_proxy_score_pct": ("cas_proxy_score_pct", "mean"),
    "repeated_group_count": ("attempt_count", lambda s: int((pd.to_numeric(s, errors="coerce") >= 2).sum())),
}
group_cols = ["test_id"]
if "test_name" in class_rows.columns:
    group_cols.append("test_name")
test_table = class_rows.groupby(group_cols, dropna=False).agg(**agg_map).reset_index()

if "proxy_evidence_band" in class_rows.columns:
    evidence = (
        class_rows.groupby(group_cols, dropna=False)["proxy_evidence_band"]
        .agg(
            high_rate=lambda s: float((s == "high").mean()),
            medium_rate=lambda s: float((s == "medium").mean()),
            low_rate=lambda s: float((s == "low").mean()),
        )
        .reset_index()
    )
    test_table = test_table.merge(evidence, on=group_cols, how="left")
else:
    test_table["high_rate"] = np.nan
    test_table["medium_rate"] = np.nan
    test_table["low_rate"] = np.nan

for col in ["difficulty_label", "DCI", "test_stability"]:
    if col in class_rows.columns:
        if col == "difficulty_label":
            test_table = test_table.merge(
                class_rows[group_cols + [col]].drop_duplicates(group_cols, keep="first"),
                on=group_cols,
                how="left",
            )
        else:
            test_table = test_table.merge(
                class_rows[group_cols + [col]].drop_duplicates(group_cols, keep="first"),
                on=group_cols,
                how="left",
            )

display_cols = [c for c in ["test_id", "test_name", "learner_count", "mean_bls_score_pct", "mean_current_als_score_pct", "mean_potential_als_score_pct", "mean_learning_gain_pct", "cas_proxy_score_pct", "repeated_group_count", "high_rate", "medium_rate", "low_rate", "difficulty_label", "DCI", "test_stability"] if c in test_table.columns]
st.dataframe(test_table[display_cols].sort_values(["learner_count", "test_id"], ascending=[False, True], kind="mergesort"), use_container_width=True)

# ---------------------------
# Learner table
# ---------------------------
st.subheader("Learner Table")
learner_summary = class_rows.groupby(["user_id"], dropna=False).agg(
    learner_id_display=("learner_id_display", "first") if "learner_id_display" in class_rows.columns else ("user_id", "first"),
    readiness_probability_pct=("readiness_probability_pct", "first") if "readiness_probability_pct" in readiness_class.columns else ("user_id", "first"),
    exam_status=("exam_status", "first") if "exam_status" in readiness_class.columns else ("user_id", "first"),
    risk_band=("risk_band", "first") if "risk_band" in readiness_class.columns else ("user_id", "first"),
    robust_SAB_scaled=("robust_SAB_scaled", "first") if "robust_SAB_scaled" in readiness_class.columns else ("user_id", "first"),
    number_of_tests=("test_id", "nunique"),
    avg_bls_score_pct=("bls_score_pct", "mean"),
    avg_current_als_score_pct=("current_als_score_pct", "mean"),
    avg_learning_gain_pct=("learning_gain_pct", "mean"),
).reset_index()

dq_learner = pd.DataFrame({"user_id": class_rows["user_id"].drop_duplicates().tolist()})
if not dq_class.empty:
    if "finished_at" in dq_class.columns:
        finished_missing = dq_class.groupby("user_id", dropna=False)["finished_at"].apply(lambda s: int(s.isna().sum())).reset_index(name="missing_finished_at_count")
        dq_learner = dq_learner.merge(finished_missing, on="user_id", how="left")
    if "completion_status" in dq_class.columns:
        unknown_usable = dq_class.groupby("user_id", dropna=False)["completion_status"].apply(lambda s: int((s == "unknown_but_usable").sum())).reset_index(name="unknown_but_usable_count")
        dq_learner = dq_learner.merge(unknown_usable, on="user_id", how="left")

learner_table = learner_summary.merge(dq_learner, on="user_id", how="left")
learner_display_cols = [c for c in ["learner_id_display", "user_id", "readiness_probability_pct", "exam_status", "risk_band", "robust_SAB_scaled", "number_of_tests", "avg_bls_score_pct", "avg_current_als_score_pct", "avg_learning_gain_pct", "missing_finished_at_count", "unknown_but_usable_count"] if c in learner_table.columns]
st.dataframe(learner_table[learner_display_cols].sort_values(["number_of_tests", "avg_current_als_score_pct"], ascending=[False, False], kind="mergesort"), use_container_width=True)

# ---------------------------
# Attempt trace
# ---------------------------
with st.expander("Attempt trace for selected class"):
    trace = proxy_class.copy()
    trace_cols = [
        col
        for col in [
            "learner_id_display",
            "user_id",
            "test_id",
            "test_name",
            "created_at",
            "marks",
            "accuracy_total",
            "v13_score_pct",
            "completion_status",
            "dq_included",
            "dq_bucket",
            "dq_eligible_published",
            "dq_eligible_proxy_sequence",
        ]
        if col in trace.columns
    ]
    if trace_cols:
        st.dataframe(trace[trace_cols].sort_values([c for c in ["user_id", "test_id", "created_at"] if c in trace.columns], kind="mergesort"), use_container_width=True)
    else:
        st.info("No trace fields are available for this class.")

# ---------------------------
# DQ caveats
# ---------------------------
st.subheader("DQ Caveats")
dq_class_total = len(dq_class)
dq_included = int(dq_class["dq_included"].fillna(False).sum()) if "dq_included" in dq_class.columns else 0
dq_excluded = int((~dq_class["dq_included"].fillna(False)).sum()) if "dq_included" in dq_class.columns else 0
missing_finished = int(dq_class["finished_at"].isna().sum()) if "finished_at" in dq_class.columns else 0
unknown_but_usable = int((dq_class["completion_status"] == "unknown_but_usable").sum()) if "completion_status" in dq_class.columns else 0

dq_cols = st.columns(4)
dq_cols[0].metric("Total rows", f"{dq_class_total:,}")
dq_cols[1].metric("Included rows", f"{dq_included:,}")
dq_cols[2].metric("Excluded rows", f"{dq_excluded:,}")
dq_cols[3].metric("Missing finished_at", f"{missing_finished:,}")

dq_summary = pd.DataFrame(
    [
        {"measure": "unknown_but_usable rows", "value": unknown_but_usable},
        {"measure": "missing finished_at rows", "value": missing_finished},
    ]
)
st.dataframe(dq_summary, use_container_width=True)

if "exclusion_reason" in dq_class.columns:
    exclusion_counts = dq_class.loc[dq_class.get("dq_bucket", pd.Series(dtype="object")).eq("excluded"), "exclusion_reason"].value_counts(dropna=False)
    if not exclusion_counts.empty:
        st.subheader("Exclusion Reason Counts")
        st.dataframe(exclusion_counts.rename_axis("reason").reset_index(name="rows"), use_container_width=True)

# ---------------------------
# Smoke report
# ---------------------------
with st.expander("Smoke report"):
    st.dataframe(smoke_report.T, use_container_width=True)

st.caption("Custom selected-test groups will be added after source-backed class summaries are validated.")
