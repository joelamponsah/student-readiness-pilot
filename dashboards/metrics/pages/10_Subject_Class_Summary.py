import numpy as np
import pandas as pd
import streamlit as st

from utils.metrics import get_v13_artifacts


st.set_page_config(page_title="Subject / Class Summary", layout="wide")
st.title("Subject / Class Summary")
st.caption(
    "v1.3 interpretation: eCampus class_id is treated as a subject/class enrolment. "
    "Tests are treated as topic-test proxies because topic_id/topic_name is not yet available."
)


# ---------------------------
# Helpers
# ---------------------------
def _normalise_string_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_metric_mean(df: pd.DataFrame, col: str | None, decimals: int = 2) -> str:
    if col is None or df is None or df.empty or col not in df.columns:
        return "N/A"
    vals = pd.to_numeric(df[col], errors="coerce")
    if not vals.notna().any():
        return "N/A"
    return f"{vals.mean():.{decimals}f}"


def _safe_metric_count_notna(df: pd.DataFrame, col: str) -> str:
    if df is None or df.empty or col not in df.columns:
        return "N/A"
    return f"{int(df[col].notna().sum()):,}"


def _safe_nunique(df: pd.DataFrame, col: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 0
    return int(df[col].nunique(dropna=True))


def _filter_by_subject_or_fallback_users(
    frame: pd.DataFrame | None,
    selected_class_id: str,
    selected_user_keys: set[str],
    frame_name: str,
) -> tuple[pd.DataFrame, bool]:
    """
    Prefer strict class_id_std filtering.
    In v1.3, class_id_std is interpreted as subject/class enrolment.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(), False

    df = frame.copy()

    if "class_id_std" in df.columns:
        class_key = _normalise_string_series(df["class_id_std"])
        return df.loc[class_key.eq(selected_class_id)].copy(), False

    if "user_id" in df.columns:
        st.warning(
            f"{frame_name} has no class_id_std column. Falling back to learner-based filtering; "
            "this may include tests outside the selected subject/class."
        )
        user_key = df["user_id"].astype(str)
        return df.loc[user_key.isin(selected_user_keys)].copy(), True

    st.warning(f"{frame_name} has neither class_id_std nor user_id, so it cannot be filtered.")
    return pd.DataFrame(), False


def _value_counts_table(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame()
    out = (
        df[col]
        .fillna("UNKNOWN")
        .value_counts(dropna=False)
        .rename_axis("value")
        .reset_index(name="rows")
    )
    out.insert(0, "metric", label)
    return out


def _build_institute_context(df: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    """
    Build cautious institute context from selected subject/class source rows.
    Institute is contextual evidence only because institute coverage is incomplete.
    """
    if df is None or df.empty or "institute_std" not in df.columns:
        empty = pd.DataFrame(columns=["institute_std", "rows", "share"])
        return "N/A", "missing", empty

    inst = df["institute_std"].fillna("Unknown").astype(str).str.strip()
    inst = inst.replace("", "Unknown")

    counts = inst.value_counts(dropna=False).rename_axis("institute_std").reset_index(name="rows")
    total_rows = int(counts["rows"].sum()) if not counts.empty else 0
    counts["share"] = counts["rows"] / total_rows if total_rows > 0 else np.nan
    counts = counts.sort_values(["rows", "institute_std"], ascending=[False, True], kind="mergesort")

    known = counts[~counts["institute_std"].str.lower().eq("unknown")].copy()
    if known.empty:
        return "Unknown", "low_no_known_institute", counts

    top = known.iloc[0]
    top_label = str(top["institute_std"])
    top_share = float(top["share"])
    known_count = int(len(known))

    if top_share >= 0.70 and known_count == 1:
        return top_label, "high_single_institute", counts
    if top_share >= 0.70:
        return f"{top_label} (dominant, mixed)", "medium_dominant_mixed", counts

    return "Mixed / unclear", "low_mixed_institutes", counts


def _build_subscription_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a first-activity proxy for subscription/intake timing.
    This is NOT true subscription date. It uses first attempt date per subscriber_id.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if "subscriber_id" not in df.columns or "created_at" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["created_at"] = pd.to_datetime(work["created_at"], errors="coerce")
    work = work.dropna(subset=["subscriber_id", "created_at"])

    if work.empty:
        return pd.DataFrame()

    group_cols = ["subscriber_id"]
    if "user_id" in work.columns:
        group_cols.append("user_id")

    out = (
        work.groupby(group_cols, dropna=False)
        .agg(
            first_activity_at=("created_at", "min"),
            last_activity_at=("created_at", "max"),
            tests_attempted=("test_id", "nunique") if "test_id" in work.columns else ("subscriber_id", "size"),
            rows=("subscriber_id", "size"),
        )
        .reset_index()
    )

    out["first_activity_month"] = out["first_activity_at"].dt.to_period("M").astype(str)
    return out


# ---------------------------
# Load shared artifacts
# ---------------------------
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

required_frames = {
    "user_test_summary": user_test_summary,
    "group_summary": group_summary,
    "readiness_user": readiness_user,
    "proxy_sequence": proxy_sequence,
    "dq_attempts": dq_attempts,
    "smoke_report": smoke_report,
}

missing_frames = [name for name, frame in required_frames.items() if frame is None]
if missing_frames:
    st.warning(f"The shared artifact bundle is incomplete. Missing: {missing_frames}")
    st.stop()

if user_test_summary.empty or "class_id_std" not in user_test_summary.columns:
    st.warning("class_id_std is missing from user_test_summary, so subject/class summaries cannot be built yet.")
    st.stop()

subject_df = user_test_summary.loc[user_test_summary["class_id_std"].notna()].copy()
subject_df["class_id_std"] = _normalise_string_series(subject_df["class_id_std"])
subject_df = subject_df[subject_df["class_id_std"] != ""].copy()

if subject_df.empty:
    st.warning("No usable class_id_std values are available.")
    st.stop()

sab_col = _first_existing_column(
    readiness_user,
    ["robust_SAB_scaled", "robust_sab_scaled", "robust_SAB_index", "robust_SAB"],
)


# ---------------------------
# Methodology
# ---------------------------
with st.expander("v1.3 methodology: how this page defines subject/class and topic-tests", expanded=True):
    st.markdown(
        """
**Current v1.3 interpretation**

- `class_id_std` is treated as the eCampus **subject/class enrolment**.
- `subscriber_id` is treated as a learner's subscription/enrolment record into that subject/class.
- `test_id` / `test_name` is treated as a **topic-test proxy**, because true `topic_id` / `topic_name` is not available in the current dataset.
- Current topic-level averages should be read as **TAS proxy**: Topic Average Score proxy.
- A future true **CAS** should aggregate topic-level TAS values into the subject/class level.

**Important limitation**

The current data does not yet prove official subject → topic → test relationships.  
This page is a v1.3 proof-of-concept using the best available fields.
"""
    )


# ---------------------------
# Subject/Class selector
# ---------------------------
subject_meta = (
    subject_df.groupby("class_id_std", dropna=False)
    .agg(
        learner_count=("user_id", "nunique"),
        source_test_count=("test_id", "nunique"),
    )
    .reset_index()
)

subject_meta["subject_label"] = subject_meta.apply(
    lambda r: (
        f"{r['class_id_std']} "
        f"(learners: {int(r['learner_count'])} | source topic-tests: {int(r['source_test_count'])})"
    ),
    axis=1,
)

subject_meta = subject_meta.sort_values(
    ["learner_count", "source_test_count", "class_id_std"],
    ascending=[False, False, True],
    kind="mergesort",
)

selected_subject_label = st.selectbox("Select subject/class", subject_meta["subject_label"].tolist())
selected_class_id = subject_meta.loc[
    subject_meta["subject_label"] == selected_subject_label,
    "class_id_std",
].iloc[0]

st.caption(
    "Subject/Class Summary uses class_id_std as the selected subject/class enrolment. "
    "The dropdown source topic-test count is before the temporary topic-test confidence filter."
)


# ---------------------------
# Subject/Class rows + heuristic topic-test filter
# ---------------------------
subject_rows_all = subject_df[subject_df["class_id_std"].eq(selected_class_id)].copy()

selected_user_keys_all = set(subject_rows_all["user_id"].dropna().astype(str).unique().tolist())
selected_subject_learner_count = len(selected_user_keys_all)

selected_institute_label, institute_confidence, institute_context = _build_institute_context(subject_rows_all)
subscription_proxy = _build_subscription_proxy(subject_rows_all)

if {"test_id", "user_id"}.issubset(subject_rows_all.columns) and selected_subject_learner_count > 0:
    topic_test_confidence = (
        subject_rows_all.groupby("test_id", dropna=False)
        .agg(
            subject_test_learner_count=("user_id", "nunique"),
            subject_test_row_count=("user_id", "size"),
        )
        .reset_index()
    )

    topic_test_confidence["subject_test_learner_share"] = (
        topic_test_confidence["subject_test_learner_count"] / selected_subject_learner_count
    )

    st.sidebar.subheader("Topic-test filter")
    min_share = st.sidebar.slider(
        "Minimum learner share",
        min_value=0.00,
        max_value=0.50,
        value=0.20,
        step=0.05,
        help="A topic-test is kept if at least this share of selected subject/class learners attempted it.",
    )

    use_min_count = st.sidebar.checkbox(
        "Also allow minimum learner count fallback",
        value=False,
        help="Use with caution. This can keep low-confidence extra tests in large subject/classes.",
    )

    min_count = st.sidebar.number_input(
        "Minimum learners fallback",
        min_value=1,
        max_value=100,
        value=3,
        step=1,
        disabled=not use_min_count,
    )

    topic_test_confidence["topic_test_included"] = (
        topic_test_confidence["subject_test_learner_share"] >= min_share
    )

    if use_min_count:
        topic_test_confidence["topic_test_included"] = (
            topic_test_confidence["topic_test_included"]
            | (topic_test_confidence["subject_test_learner_count"] >= int(min_count))
        )

    included_test_ids = set(
        topic_test_confidence.loc[topic_test_confidence["topic_test_included"], "test_id"].tolist()
    )

    subject_rows = subject_rows_all[subject_rows_all["test_id"].isin(included_test_ids)].copy()

    excluded_test_count = int((~topic_test_confidence["topic_test_included"]).sum())
    excluded_row_count = int(len(subject_rows_all) - len(subject_rows))

    st.info(
        "Temporary topic-test confidence filter is active. "
        "This helps suppress tests that appear because of broad subscription/class joins. "
        "It is not official topic assignment logic."
    )

    st.caption(
        f"Filtered out {excluded_test_count:,} low-confidence topic-test(s) "
        f"and {excluded_row_count:,} learner-test row(s)."
    )

    with st.expander("Low-confidence topic-tests filtered out"):
        filtered_out_tests = topic_test_confidence.loc[
            ~topic_test_confidence["topic_test_included"]
        ].sort_values(
            ["subject_test_learner_count", "subject_test_learner_share"],
            ascending=[False, False],
            kind="mergesort",
        )
        st.dataframe(filtered_out_tests, use_container_width=True)

else:
    subject_rows = subject_rows_all.copy()
    topic_test_confidence = pd.DataFrame()
    included_test_ids = set(subject_rows["test_id"].dropna().tolist()) if "test_id" in subject_rows.columns else set()
    st.warning(
        "Could not apply topic-test confidence filter because test_id/user_id is missing "
        "or selected subject/class learner count is zero."
    )

if subject_rows.empty:
    st.warning("No rows remain after the topic-test confidence filter.")
    st.stop()

selected_user_keys = set(subject_rows["user_id"].dropna().astype(str).unique().tolist())
included_test_ids = set(subject_rows["test_id"].dropna().tolist()) if "test_id" in subject_rows.columns else set()

# Learner-level readiness.
readiness_user_work = readiness_user.copy()
if "user_id" in readiness_user_work.columns:
    readiness_user_work["_user_id_key"] = readiness_user_work["user_id"].astype(str)
    readiness_subject = readiness_user_work[readiness_user_work["_user_id_key"].isin(selected_user_keys)].copy()
else:
    readiness_subject = pd.DataFrame()

# Class-level attempt/DQ traces, interpreted as subject/class traces.
proxy_subject_all, proxy_fallback_used = _filter_by_subject_or_fallback_users(
    proxy_sequence,
    selected_class_id,
    selected_user_keys_all,
    "proxy_sequence",
)

dq_subject_all, dq_fallback_used = _filter_by_subject_or_fallback_users(
    dq_attempts,
    selected_class_id,
    selected_user_keys_all,
    "dq_attempts",
)

if included_test_ids and "test_id" in proxy_subject_all.columns:
    proxy_subject = proxy_subject_all[proxy_subject_all["test_id"].isin(included_test_ids)].copy()
else:
    proxy_subject = proxy_subject_all.copy()

if included_test_ids and "test_id" in dq_subject_all.columns:
    dq_subject = dq_subject_all[dq_subject_all["test_id"].isin(included_test_ids)].copy()
else:
    dq_subject = dq_subject_all.copy()


# ---------------------------
# Selected subject/class context
# ---------------------------
st.subheader(f"Selected Subject/Class: {selected_class_id}")

source_test_count = _safe_nunique(subject_rows_all, "test_id")
filtered_test_count = _safe_nunique(subject_rows, "test_id")
source_learner_count = _safe_nunique(subject_rows_all, "user_id")
filtered_learner_count = _safe_nunique(subject_rows, "user_id")
subscriber_count = _safe_nunique(subject_rows_all, "subscriber_id")

st.caption(
    f"Source view: {source_learner_count:,} learners | {source_test_count:,} source topic-tests | "
    f"{subscriber_count:,} subscriptions. "
    f"Filtered view: {filtered_learner_count:,} learners | {filtered_test_count:,} topic-tests."
)

if proxy_fallback_used or dq_fallback_used:
    st.info(
        "One or more artifacts used learner-based fallback filtering because class_id_std was unavailable. "
        "Validate these results carefully."
    )


# ---------------------------
# Institute Context
# ---------------------------
st.subheader("Institute Context")

if not institute_context.empty:
    known_mask = ~institute_context["institute_std"].str.lower().eq("unknown")
    known_institute_count = int(known_mask.sum())
    known_coverage = float(institute_context.loc[known_mask, "share"].sum()) * 100
else:
    known_institute_count = 0
    known_coverage = np.nan

inst_cols = st.columns(4)
inst_cols[0].metric("Institute", selected_institute_label)
inst_cols[1].metric("Institute confidence", institute_confidence)
inst_cols[2].metric("Known institutes", f"{known_institute_count:,}" if not institute_context.empty else "N/A")
inst_cols[3].metric("Known institute coverage", f"{known_coverage:.1f}%" if pd.notna(known_coverage) else "N/A")

with st.expander("Institute distribution for selected subject/class"):
    if not institute_context.empty:
        st.dataframe(institute_context, use_container_width=True)
    else:
        st.info("No institute context is available.")


# ---------------------------
# Subscription / Intake Proxy
# ---------------------------
st.subheader("Subscription / Intake Proxy")

if not subscription_proxy.empty:
    intake_summary = (
        subscription_proxy.groupby("first_activity_month", dropna=False)
        .agg(
            subscriptions=("subscriber_id", "nunique"),
            learners=("user_id", "nunique") if "user_id" in subscription_proxy.columns else ("subscriber_id", "nunique"),
            avg_tests_attempted=("tests_attempted", "mean"),
        )
        .reset_index()
        .sort_values("first_activity_month", kind="mergesort")
    )

    st.caption(
        "This uses first attempt month per subscriber_id as a proxy for subscription/intake timing. "
        "It is not true subscription start date."
    )
    st.dataframe(intake_summary, use_container_width=True)
else:
    st.info("No subscriber_id/created_at combination is available to derive intake proxy.")


# ---------------------------
# Overview KPIs
# ---------------------------
st.subheader("Subject/Class Overview")

overview1 = st.columns(4)
overview1[0].metric("Institute", selected_institute_label)
overview1[1].metric("Learners", f"{_safe_nunique(subject_rows, 'user_id'):,}")
overview1[2].metric("Topic-tests", f"{_safe_nunique(subject_rows, 'test_id'):,}")
overview1[3].metric("Subscriptions", f"{_safe_nunique(subject_rows, 'subscriber_id'):,}")

overview2 = st.columns(4)
overview2[0].metric("Avg readiness probability %", _safe_metric_mean(readiness_subject, "readiness_probability_pct"))
overview2[1].metric("BLS rows", _safe_metric_count_notna(subject_rows, "bls_score_pct"))
overview2[2].metric("Current ALS rows", _safe_metric_count_notna(subject_rows, "current_als_score_pct"))
overview2[3].metric("Avg robust SAB", _safe_metric_mean(readiness_subject, sab_col) if sab_col else "N/A")

overview3 = st.columns(4)
overview3[0].metric("Potential ALS rows", _safe_metric_count_notna(subject_rows, "potential_als_score_pct"))
overview3[1].metric("Avg BLS %", _safe_metric_mean(subject_rows, "bls_score_pct"))
overview3[2].metric("Avg Current ALS %", _safe_metric_mean(subject_rows, "current_als_score_pct"))
overview3[3].metric("Avg learning gain %", _safe_metric_mean(subject_rows, "learning_gain_pct"))

overview4 = st.columns(4)
overview4[0].metric(
    "TAS proxy %",
    _safe_metric_mean(subject_rows, "cas_proxy_score_pct"),
    help="In v1.3 this is better interpreted as Topic Average Score proxy, not final CAS.",
)

if sab_col:
    st.caption(f"Robust SAB source column: `{sab_col}` from readiness_user.")
else:
    st.caption("Robust SAB source column was not found in readiness_user.")


# ---------------------------
# Readiness Distribution
# ---------------------------
st.subheader("Readiness Distribution")
dist_frames = []

for column in ["exam_status", "risk_band", "coverage_risk"]:
    dist = _value_counts_table(readiness_subject, column, column)
    if not dist.empty:
        dist_frames.append(dist)

if dist_frames:
    st.dataframe(pd.concat(dist_frames, ignore_index=True, sort=False), use_container_width=True)
else:
    st.info("No readiness distribution fields are available for this subject/class.")

st.caption("Readiness distribution uses learner-level overall readiness for learners in this selected subject/class.")


# ---------------------------
# Topic-Test Summary
# ---------------------------
st.subheader("Topic-Test Summary")
st.caption("Each row is a test_id/test_name used as a topic-test proxy in v1.3.")

group_cols = ["test_id"]
if "test_name" in subject_rows.columns:
    group_cols.append("test_name")

topic_test_table = (
    subject_rows.groupby(group_cols, dropna=False)
    .agg(
        learner_count=("user_id", "nunique"),
    )
    .reset_index()
)

mean_cols = {
    "mean_bls_score_pct": "bls_score_pct",
    "mean_current_als_score_pct": "current_als_score_pct",
    "mean_potential_als_score_pct": "potential_als_score_pct",
    "mean_learning_gain_pct": "learning_gain_pct",
    "tas_proxy_pct": "cas_proxy_score_pct",
}

for out_col, src_col in mean_cols.items():
    if src_col in subject_rows.columns:
        tmp = (
            subject_rows.groupby(group_cols, dropna=False)[src_col]
            .mean()
            .reset_index(name=out_col)
        )
        topic_test_table = topic_test_table.merge(tmp, on=group_cols, how="left")
    else:
        topic_test_table[out_col] = np.nan

if "attempt_count" in subject_rows.columns:
    repeated = (
        subject_rows.groupby(group_cols, dropna=False)["attempt_count"]
        .apply(lambda s: int((pd.to_numeric(s, errors="coerce") >= 2).sum()))
        .reset_index(name="repeated_group_count")
    )
    topic_test_table = topic_test_table.merge(repeated, on=group_cols, how="left")
else:
    topic_test_table["repeated_group_count"] = np.nan

if "proxy_evidence_band" in subject_rows.columns:
    evidence = (
        subject_rows.groupby(group_cols, dropna=False)["proxy_evidence_band"]
        .agg(
            high_rate=lambda s: float((_normalise_string_series(s).str.lower() == "high").mean()),
            medium_rate=lambda s: float((_normalise_string_series(s).str.lower() == "medium").mean()),
            low_rate=lambda s: float((_normalise_string_series(s).str.lower() == "low").mean()),
        )
        .reset_index()
    )
    topic_test_table = topic_test_table.merge(evidence, on=group_cols, how="left")

for col in ["difficulty_label", "DCI", "test_stability"]:
    if col in subject_rows.columns:
        context = subject_rows[group_cols + [col]].drop_duplicates(group_cols, keep="first")
        topic_test_table = topic_test_table.merge(context, on=group_cols, how="left")

if "institute_std" in subject_rows.columns:
    institute_context_by_test = (
        subject_rows[group_cols + ["institute_std"]]
        .dropna(subset=["institute_std"])
        .drop_duplicates(group_cols, keep="first")
    )
    topic_test_table = topic_test_table.merge(institute_context_by_test, on=group_cols, how="left")

topic_display_cols = [
    c for c in [
        "test_id",
        "test_name",
        "institute_std",
        "learner_count",
        "mean_bls_score_pct",
        "mean_current_als_score_pct",
        "mean_potential_als_score_pct",
        "mean_learning_gain_pct",
        "tas_proxy_pct",
        "repeated_group_count",
        "high_rate",
        "medium_rate",
        "low_rate",
        "difficulty_label",
        "DCI",
        "test_stability",
    ]
    if c in topic_test_table.columns
]

st.dataframe(
    topic_test_table[topic_display_cols].sort_values(
        ["learner_count", "test_id"],
        ascending=[False, True],
        kind="mergesort",
    ),
    use_container_width=True,
)


# ---------------------------
# Learner Table
# ---------------------------
st.subheader("Learner Table")

learner_base_agg = {
    "number_of_topic_tests": ("test_id", "nunique"),
}

if "learner_id_display" in subject_rows.columns:
    learner_base_agg["learner_id_display"] = ("learner_id_display", "first")
else:
    learner_base_agg["learner_id_display"] = ("user_id", "first")

if "institute_std" in subject_rows.columns:
    learner_base_agg["institute_std"] = ("institute_std", "first")

if "subscriber_id" in subject_rows.columns:
    learner_base_agg["subscriptions"] = ("subscriber_id", "nunique")

learner_table = (
    subject_rows.groupby("user_id", dropna=False)
    .agg(**learner_base_agg)
    .reset_index()
)

learner_mean_cols = {
    "avg_bls_score_pct": "bls_score_pct",
    "avg_current_als_score_pct": "current_als_score_pct",
    "avg_potential_als_score_pct": "potential_als_score_pct",
    "avg_learning_gain_pct": "learning_gain_pct",
    "avg_tas_proxy_pct": "cas_proxy_score_pct",
}

for out_col, src_col in learner_mean_cols.items():
    if src_col in subject_rows.columns:
        tmp = (
            subject_rows.groupby("user_id", dropna=False)[src_col]
            .mean()
            .reset_index(name=out_col)
        )
        learner_table = learner_table.merge(tmp, on="user_id", how="left")

readiness_merge_cols = ["user_id"]
for col in ["readiness_probability_pct", "exam_status", "risk_band"]:
    if col in readiness_subject.columns:
        readiness_merge_cols.append(col)

if sab_col and sab_col in readiness_subject.columns:
    readiness_merge_cols.append(sab_col)

if len(readiness_merge_cols) > 1:
    readiness_for_merge = readiness_subject[readiness_merge_cols].drop_duplicates("user_id", keep="first")
    learner_table = learner_table.merge(readiness_for_merge, on="user_id", how="left")

if sab_col and sab_col in learner_table.columns:
    learner_table = learner_table.rename(columns={sab_col: "robust_sab"})

dq_learner = pd.DataFrame({"user_id": subject_rows["user_id"].drop_duplicates().tolist()})

if not dq_subject.empty and "user_id" in dq_subject.columns:
    if "finished_at" in dq_subject.columns:
        finished_missing = (
            dq_subject.groupby("user_id", dropna=False)["finished_at"]
            .apply(lambda s: int(s.isna().sum()))
            .reset_index(name="missing_finished_at_count")
        )
        dq_learner = dq_learner.merge(finished_missing, on="user_id", how="left")

    if "completion_status" in dq_subject.columns:
        unknown_usable = (
            dq_subject.groupby("user_id", dropna=False)["completion_status"]
            .apply(lambda s: int((s == "unknown_but_usable").sum()))
            .reset_index(name="unknown_but_usable_count")
        )
        dq_learner = dq_learner.merge(unknown_usable, on="user_id", how="left")

learner_table = learner_table.merge(dq_learner, on="user_id", how="left")

for col in ["missing_finished_at_count", "unknown_but_usable_count"]:
    if col in learner_table.columns:
        learner_table[col] = learner_table[col].fillna(0).astype(int)

learner_display_cols = [
    c for c in [
        "learner_id_display",
        "institute_std",
        "user_id",
        "subscriptions",
        "readiness_probability_pct",
        "exam_status",
        "risk_band",
        "robust_sab",
        "number_of_topic_tests",
        "avg_bls_score_pct",
        "avg_current_als_score_pct",
        "avg_potential_als_score_pct",
        "avg_learning_gain_pct",
        "avg_tas_proxy_pct",
        "missing_finished_at_count",
        "unknown_but_usable_count",
    ]
    if c in learner_table.columns
]

sort_cols = [c for c in ["number_of_topic_tests", "avg_current_als_score_pct"] if c in learner_table.columns]
if sort_cols:
    learner_table_display = learner_table[learner_display_cols].sort_values(
        sort_cols,
        ascending=[False] * len(sort_cols),
        kind="mergesort",
    )
else:
    learner_table_display = learner_table[learner_display_cols]

st.dataframe(learner_table_display, use_container_width=True)


# ---------------------------
# Learner × Topic-Test Comparison
# ---------------------------
st.subheader("Learner × Topic-Test Comparison")

learner_test_cols = [
    c for c in [
        "learner_id_display",
        "institute_std",
        "subscriber_id",
        "user_id",
        "test_id",
        "test_name",
        "bls_score_pct",
        "current_als_score_pct",
        "potential_als_score_pct",
        "learning_gain_pct",
        "potential_gain_pct",
        "cas_proxy_score_pct",
        "proxy_evidence_band",
        "completion_status_mix",
        "difficulty_label",
        "DCI",
        "test_stability",
    ]
    if c in subject_rows.columns
]

if learner_test_cols:
    learner_test_table = subject_rows[learner_test_cols].copy()
    if "cas_proxy_score_pct" in learner_test_table.columns:
        learner_test_table = learner_test_table.rename(columns={"cas_proxy_score_pct": "tas_proxy_pct"})

    sort_cols = [c for c in ["learner_id_display", "user_id", "test_id"] if c in learner_test_table.columns]
    if sort_cols:
        learner_test_table = learner_test_table.sort_values(sort_cols, kind="mergesort")

    st.dataframe(learner_test_table, use_container_width=True)
else:
    st.info("No learner-topic-test comparison columns are available.")


# ---------------------------
# Attempt Trace
# ---------------------------
with st.expander("Attempt trace for selected subject/class"):
    trace = proxy_subject.copy()
    trace_candidate_cols = [
        "learner_id_display",
        "institute_std",
        "subscriber_id",
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

    trace_cols = list(dict.fromkeys(col for col in trace_candidate_cols if col in trace.columns))

    if trace_cols:
        sort_cols = [c for c in ["user_id", "test_id", "created_at"] if c in trace.columns]
        trace_display = trace[trace_cols].copy()
        if sort_cols:
            trace_display = trace_display.sort_values(sort_cols, kind="mergesort")
        st.dataframe(trace_display, use_container_width=True)
    else:
        st.info("No trace fields are available.")


# ---------------------------
# DQ Caveats
# ---------------------------
st.subheader("DQ Caveats")

dq_subject_total = len(dq_subject)
dq_included = int(dq_subject["dq_included"].fillna(False).sum()) if not dq_subject.empty and "dq_included" in dq_subject.columns else 0
dq_excluded = int((~dq_subject["dq_included"].fillna(False)).sum()) if not dq_subject.empty and "dq_included" in dq_subject.columns else 0
missing_finished = int(dq_subject["finished_at"].isna().sum()) if not dq_subject.empty and "finished_at" in dq_subject.columns else 0
unknown_but_usable = int((dq_subject["completion_status"] == "unknown_but_usable").sum()) if not dq_subject.empty and "completion_status" in dq_subject.columns else 0

dq_cols = st.columns(4)
dq_cols[0].metric("Total rows", f"{dq_subject_total:,}")
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

if not dq_subject.empty and "exclusion_reason" in dq_subject.columns:
    exclusion_filter = (
        dq_subject["dq_bucket"].eq("excluded")
        if "dq_bucket" in dq_subject.columns
        else ~dq_subject["dq_included"].fillna(False)
    )

    exclusion_counts = (
        dq_subject.loc[exclusion_filter, "exclusion_reason"]
        .value_counts(dropna=False)
        .rename_axis("reason")
        .reset_index(name="rows")
    )

    if not exclusion_counts.empty:
        st.subheader("Exclusion Reasons")
        st.dataframe(exclusion_counts, use_container_width=True)


# ---------------------------
# Smoke / Group Preview
# ---------------------------
with st.expander("Smoke report"):
    if smoke_report is not None and not smoke_report.empty:
        st.dataframe(smoke_report.T, use_container_width=True)
    else:
        st.info("No smoke_report is available.")

with st.expander("Group summary preview"):
    if group_summary is not None and not group_summary.empty:
        group_preview = group_summary.copy()

        if "class_id_std" in group_preview.columns:
            group_preview["_class_id_key"] = _normalise_string_series(group_preview["class_id_std"])
            group_preview = group_preview[group_preview["_class_id_key"].eq(selected_class_id)].drop(columns=["_class_id_key"])

        if included_test_ids and "test_id" in group_preview.columns:
            group_preview = group_preview[group_preview["test_id"].isin(included_test_ids)]

        preferred_group_cols = [
            "group_level",
            "group_value",
            "class_id_std",
            "institute_std",
            "subscriber_id",
            "test_id",
            "learner_count",
            "repeated_group_count",
            "mean_bls_score_pct",
            "mean_current_als_score_pct",
            "mean_potential_als_score_pct",
            "mean_learning_gain_pct",
            "cas_proxy_score_pct",
            "formula_readiness_avg",
            "robust_SAB_avg",
            "high_evidence_rate",
            "medium_evidence_rate",
            "low_evidence_rate",
            "difficulty_label",
            "DCI",
            "test_stability",
        ]

        display_group_cols = [c for c in preferred_group_cols if c in group_preview.columns]
        if display_group_cols:
            group_display = group_preview[display_group_cols].copy()
            if "cas_proxy_score_pct" in group_display.columns:
                group_display = group_display.rename(columns={"cas_proxy_score_pct": "tas_proxy_pct"})
            st.dataframe(group_display.head(100), use_container_width=True)
        else:
            st.dataframe(group_preview.head(100), use_container_width=True)
    else:
        st.info("No group_summary is available.")

st.caption(
    "This page consumes shared v1.3 artifacts only. "
    "Terminology is updated for the current eCampus model: class_id = subject/class; test_id = topic-test proxy."
)
