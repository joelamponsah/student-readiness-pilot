import streamlit as st

from utils.metrics import get_v13_artifacts


st.set_page_config(page_title="Metrics", layout="wide")
st.title("Metrics")
st.caption("Core metrics, the math behind them, and v1.3 proxy outputs. DQ gating is shown on DQ Monitors.")

st.markdown(
    """
### What this page is
This page inspects the shared v1.3 artifact bundle built from `raw_attempts.csv`.
It does not recompute DQ, readiness, difficulty, or proxy logic locally.

### v1.3 boundary
- v1.3 is Test / Exercise Readiness only.
- `published_kpi` is the stricter published readiness slice.
- `proxy_sequence` preserves repeated attempts for proxy inspection.
- BLS / ALS / CAS remain proxies in v1.3.
- Difficulty / DCI is context only, not a score correction.

### Schema boundary
- The current dataset does not provide `topic_id`, `subject_id`, or `year_group`.
- `class_id` is available and is the preferred cohort key alongside `subscriber_id` and `created_at`.
- `test_name` may support a derived assessment theme only if naming is consistent enough.

### Math used in the analysis
The canonical formulas are built upstream in the shared pipeline:
- `accuracy_total = marks / denominator`
- `speed_raw = attempted_questions / time_taken`
- `adj_speed = correct_answers / time_taken`
- `efficiency_ratio = accuracy_total / time_consumed`
- `time_consumed = time_taken / duration`
"""
)

view_mode = st.sidebar.radio(
    "Dataset view",
    ["Published KPI data", "Proxy sequence data"],
    index=0,
)

raw_df, artifacts = get_v13_artifacts()

if raw_df is None or not artifacts:
    st.warning("No raw_attempts.csv input or shared v1.3 artifacts are available. Upload raw_attempts.csv or place it at data/raw_attempts.csv.")
    st.stop()

required_keys = [
    "dq_attempts",
    "published_kpi",
    "proxy_sequence",
    "readiness_user",
    "difficulty_df",
    "user_test_summary",
    "group_summary",
    "smoke_report",
]
missing_keys = [key for key in required_keys if key not in artifacts]
if missing_keys:
    st.warning(f"Shared artifact bundle is incomplete: missing {missing_keys}")

published_kpi = artifacts.get("published_kpi")
proxy_sequence = artifacts.get("proxy_sequence")
readiness_user = artifacts.get("readiness_user")
difficulty_df = artifacts.get("difficulty_df")
user_test_summary = artifacts.get("user_test_summary")
group_summary = artifacts.get("group_summary")
smoke_report = artifacts.get("smoke_report")

dataset = published_kpi if view_mode == "Published KPI data" else proxy_sequence
if dataset is None or dataset.empty:
    st.warning(f"{view_mode} is unavailable in the current artifact bundle.")
    st.stop()

st.caption(
    "Published KPI data is the strict published readiness view. Proxy sequence data preserves repeated eligible attempts for BLS/ALS/CAS inspection."
)

st.subheader("Bundle Summary")
summary_cols = st.columns(6)
summary_cols[0].metric("Raw rows", f"{len(raw_df):,}")
summary_cols[1].metric("Raw users", f"{raw_df['user_id'].nunique():,}" if "user_id" in raw_df.columns else "N/A")
summary_cols[2].metric("Raw tests", f"{raw_df['test_id'].nunique():,}" if "test_id" in raw_df.columns else "N/A")
summary_cols[3].metric("Published rows", f"{len(published_kpi):,}" if published_kpi is not None else "N/A")
summary_cols[4].metric("Proxy rows", f"{len(proxy_sequence):,}" if proxy_sequence is not None else "N/A")
summary_cols[5].metric("Summary rows", f"{len(user_test_summary):,}" if user_test_summary is not None else "N/A")

st.subheader("Smoke Report")
if smoke_report is not None and not smoke_report.empty:
    st.dataframe(smoke_report.T, use_container_width=True)
else:
    st.warning("No smoke_report found in the shared artifact bundle.")

st.subheader("Artifact Counts")
counts_cols = st.columns(4)
counts_cols[0].metric(
    "Published KPI",
    f"{len(published_kpi):,} rows | {published_kpi['user_id'].nunique():,} users | {published_kpi['test_id'].nunique():,} tests"
    if published_kpi is not None and not published_kpi.empty and {"user_id", "test_id"}.issubset(published_kpi.columns)
    else "N/A",
)
counts_cols[1].metric(
    "Proxy sequence",
    f"{len(proxy_sequence):,} rows | {proxy_sequence['user_id'].nunique():,} users | {proxy_sequence['test_id'].nunique():,} tests"
    if proxy_sequence is not None and not proxy_sequence.empty and {"user_id", "test_id"}.issubset(proxy_sequence.columns)
    else "N/A",
)
counts_cols[2].metric(
    "User-test summary",
    f"{len(user_test_summary):,} rows | {user_test_summary['user_id'].nunique():,} users | {user_test_summary['test_id'].nunique():,} tests"
    if user_test_summary is not None and not user_test_summary.empty and {"user_id", "test_id"}.issubset(user_test_summary.columns)
    else "N/A",
)
counts_cols[3].metric(
    "Readiness user",
    f"{len(readiness_user):,} rows | {readiness_user['user_id'].nunique():,} users"
    if readiness_user is not None and not readiness_user.empty and "user_id" in readiness_user.columns
    else "N/A",
)

secondary_cols = st.columns(4)
secondary_cols[0].metric(
    "Difficulty/DCI",
    f"{len(difficulty_df):,} rows | {difficulty_df['test_id'].nunique():,} tests"
    if difficulty_df is not None and not difficulty_df.empty and "test_id" in difficulty_df.columns
    else "N/A",
)
secondary_cols[1].metric(
    "BLS rows",
    f"{int(user_test_summary['bls_score_pct'].notna().sum()):,}"
    if user_test_summary is not None and not user_test_summary.empty and "bls_score_pct" in user_test_summary.columns
    else "N/A",
)
secondary_cols[2].metric(
    "Current ALS rows",
    f"{int(user_test_summary['current_als_score_pct'].notna().sum()):,}"
    if user_test_summary is not None and not user_test_summary.empty and "current_als_score_pct" in user_test_summary.columns
    else "N/A",
)
secondary_cols[3].metric(
    "Potential ALS rows",
    f"{int(user_test_summary['potential_als_score_pct'].notna().sum()):,}"
    if user_test_summary is not None and not user_test_summary.empty and "potential_als_score_pct" in user_test_summary.columns
    else "N/A",
)

tertiary_cols = st.columns(2)
tertiary_cols[0].metric(
    "CAS proxy rows",
    f"{int(user_test_summary['cas_proxy_score_pct'].notna().sum()):,}"
    if user_test_summary is not None and not user_test_summary.empty and "cas_proxy_score_pct" in user_test_summary.columns
    else "N/A",
)
tertiary_cols[1].metric(
    "Readiness probability non-null users",
    f"{int(readiness_user['readiness_probability_pct'].notna().sum()):,}"
    if readiness_user is not None and not readiness_user.empty and "readiness_probability_pct" in readiness_user.columns
    else "N/A",
)

tertiary_cols2 = st.columns(1)
tertiary_cols2[0].metric(
    "robust_SAB_scaled non-null users",
    f"{int(readiness_user['robust_SAB_scaled'].notna().sum()):,}"
    if readiness_user is not None and not readiness_user.empty and "robust_SAB_scaled" in readiness_user.columns
    else "N/A",
)

st.subheader("Current View Preview")
preview_cols = [
    col for col in [
        "user_id",
        "test_id",
        "test_name",
        "marks",
        "accuracy_total",
        "accuracy_total_safe",
        "accuracy_denominator",
        "accuracy_denominator_source",
        "v13_score_pct",
        "completion_status",
        "proxy_evidence_band",
        "learning_gain_proxy_pct",
    ]
    if col in dataset.columns
]
st.dataframe(dataset[preview_cols].head(20), use_container_width=True)

st.subheader("Shared Artifact Preview")
if view_mode == "Published KPI data":
    st.dataframe(published_kpi.head(20), use_container_width=True)
else:
    st.dataframe(proxy_sequence.head(20), use_container_width=True)

st.caption("The page above consumes the shared artifact bundle only. No local DQ, readiness, difficulty, or proxy recomputation happens here.")
