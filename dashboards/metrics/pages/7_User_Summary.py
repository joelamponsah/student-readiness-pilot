import numpy as np
import pandas as pd
import streamlit as st

from utils.metrics import get_v13_artifacts


st.set_page_config(page_title="User Performance Profile", layout="wide")
st.title("User Performance Profile")
st.caption(
    "v1.3 Test / Exercise Readiness. Readiness is shown from shared artifacts only. "
    "BLS / ALS / CAS are proxies in v1.3, not final Learn Smarter outputs."
)

raw_df, artifacts = get_v13_artifacts()
if raw_df is None or not artifacts:
    st.warning("No raw_attempts.csv input or shared v1.3 artifacts are available.")
    st.stop()

readiness_user = artifacts.get("readiness_user")
user_test_summary = artifacts.get("user_test_summary")
proxy_sequence = artifacts.get("proxy_sequence")
published_kpi = artifacts.get("published_kpi")
difficulty_df = artifacts.get("difficulty_df")
dq_attempts = artifacts.get("dq_attempts")
smoke_report = artifacts.get("smoke_report")

if readiness_user is None or readiness_user.empty or user_test_summary is None or user_test_summary.empty or proxy_sequence is None or proxy_sequence.empty:
    st.warning("Shared artifact bundle is incomplete for user summary inspection.")
    st.stop()

st.markdown(
    """
### What this page shows
- Existing readiness signal from `readiness_user`
- Learn Smarter proxy signal from `user_test_summary`
- Attempt trace from `proxy_sequence`
- DQ caveats from `dq_attempts`

### Boundary
- BLS / ALS / CAS are proxies only in v1.3.
- Difficulty / DCI is context only, not a score correction.
- No local DQ, readiness, SAB, or proxy logic is recalculated here.
"""
)

# ---------------------------
# Learner selector
# ---------------------------
selector_df = readiness_user.copy()
if "learner_id_display" not in selector_df.columns:
    selector_df["learner_id_display"] = selector_df["user_id"].astype(str)
selector_df["learner_label"] = selector_df["learner_id_display"].astype("string").fillna("").str.strip()
selector_df.loc[selector_df["learner_label"] == "", "learner_label"] = selector_df["user_id"].astype(str)

selector_df = selector_df.sort_values(
    ["learner_label", "user_id"],
    kind="mergesort",
)

learner_options = selector_df[["learner_label", "user_id"]].drop_duplicates()
learner_choice = st.selectbox(
    "Choose learner",
    learner_options["learner_label"].tolist(),
)

selected_user_id = learner_options.loc[learner_options["learner_label"] == learner_choice, "user_id"].iloc[0]

readiness_row = readiness_user.loc[readiness_user["user_id"] == selected_user_id].copy()
summary_rows = user_test_summary.loc[user_test_summary["user_id"] == selected_user_id].copy()
proxy_rows = proxy_sequence.loc[proxy_sequence["user_id"] == selected_user_id].copy()
raw_user_rows = raw_df.loc[raw_df["user_id"] == selected_user_id].copy() if "user_id" in raw_df.columns else pd.DataFrame()
dq_user_rows = dq_attempts.loc[dq_attempts["user_id"] == selected_user_id].copy() if dq_attempts is not None and "user_id" in dq_attempts.columns else pd.DataFrame()
exclusion_rows = dq_user_rows.loc[dq_user_rows.get("dq_bucket", pd.Series(dtype="object")).eq("excluded")] if not dq_user_rows.empty and "dq_bucket" in dq_user_rows.columns else pd.DataFrame()

if readiness_row.empty:
    st.warning("No readiness row found for the selected learner.")
    st.stop()

readiness_row = readiness_row.iloc[0]

st.subheader(f"Selected Learner: {learner_choice}")
st.caption(f"user_id: {selected_user_id}")

# ---------------------------
# Existing readiness signal
# ---------------------------
st.subheader("Existing Readiness Signal")
ready_cols = st.columns(4)
ready_cols[0].metric("Readiness probability (%)", f"{readiness_row.get('readiness_probability_pct', np.nan):.1f}" if pd.notna(readiness_row.get("readiness_probability_pct", np.nan)) else "N/A")
ready_cols[1].metric("Exam status", str(readiness_row.get("exam_status", "N/A")))
ready_cols[2].metric("Risk band", str(readiness_row.get("risk_band", "N/A")))
ready_cols[3].metric("robust_SAB_scaled", f"{readiness_row.get('robust_SAB_scaled', np.nan):.2f}" if pd.notna(readiness_row.get("robust_SAB_scaled", np.nan)) else "N/A")

ready_cols2 = st.columns(4)
ready_cols2[0].metric("Mean accuracy", f"{readiness_row.get('mean_accuracy', np.nan):.3f}" if pd.notna(readiness_row.get("mean_accuracy", np.nan)) else "N/A")
ready_cols2[1].metric("Mean speed", f"{readiness_row.get('mean_speed', np.nan):.3f}" if pd.notna(readiness_row.get("mean_speed", np.nan)) else "N/A")
ready_cols2[2].metric("Pass rate", f"{readiness_row.get('pass_rate', np.nan):.3f}" if pd.notna(readiness_row.get("pass_rate", np.nan)) else "N/A")
ready_cols2[3].metric("Coverage risk", str(readiness_row.get("coverage_risk", "N/A")))

st.write(f"Insight code: {readiness_row.get('insight_code', 'N/A')}")
st.write(f"Stakeholder insight: {readiness_row.get('stakeholder_insight', 'N/A')}")
st.write(f"Coach feedback: {readiness_row.get('coach_feedback', 'N/A')}")
if "redemption_plan" in readiness_row.index:
    st.write(f"Redemption plan: {readiness_row.get('redemption_plan', 'N/A')}")

# ---------------------------
# Learn Smarter proxy signal
# ---------------------------
st.subheader("Learn Smarter Proxy Signal")
proxy_display_cols = [
    col
    for col in [
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
        "question_pool_comparability",
        "difficulty_label",
        "DCI",
        "test_stability",
    ]
    if col in summary_rows.columns
]
st.dataframe(summary_rows[proxy_display_cols].sort_values("test_id", kind="mergesort"), use_container_width=True)

# ---------------------------
# Attempt trace
# ---------------------------
st.subheader("Attempt Trace")
trace_candidate_cols = [
    col
    for col in [
        "attempt_id",
        "test_taker_id",
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
        "completion_source",
        "completion_status",
        "exclusion_reason",
    ]
]
trace_cols = list(dict.fromkeys(col for col in trace_candidate_cols if col in proxy_rows.columns))
if trace_cols:
    st.dataframe(proxy_rows[trace_cols].sort_values([c for c in ["created_at", "attempt_id", "test_taker_id"] if c in proxy_rows.columns], kind="mergesort"), use_container_width=True)
else:
    st.info("No trace fields available in proxy_sequence for this learner.")

# ---------------------------
# DQ caveats
# ---------------------------
st.subheader("DQ Caveats")
raw_total = len(raw_user_rows)
included_total = len(dq_user_rows)
excluded_total = len(exclusion_rows)
missing_finished = int(raw_user_rows["finished_at"].isna().sum()) if not raw_user_rows.empty and "finished_at" in raw_user_rows.columns else 0
unknown_usable = int(dq_user_rows["completion_status"].eq("unknown_but_usable").sum()) if not dq_user_rows.empty and "completion_status" in dq_user_rows.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw rows", f"{raw_total:,}")
c2.metric("Included rows", f"{included_total:,}")
c3.metric("Excluded rows", f"{excluded_total:,}")
c4.metric("Missing finished_at", f"{missing_finished:,}")

dq_summary = pd.DataFrame(
    [
        {"measure": "unknown_but_usable rows", "value": unknown_usable},
        {"measure": "missing finished_at rows", "value": missing_finished},
    ]
)
st.dataframe(dq_summary, use_container_width=True)

if not exclusion_rows.empty and "exclusion_reason" in exclusion_rows.columns:
    st.subheader("Exclusion Reasons")
    st.dataframe(
        exclusion_rows["exclusion_reason"].value_counts(dropna=False).rename_axis("reason").reset_index(name="rows"),
        use_container_width=True,
    )

# ---------------------------
# Shared artifact preview
# ---------------------------
st.subheader("Shared Artifacts")
if smoke_report is not None and not smoke_report.empty:
    st.dataframe(smoke_report.T, use_container_width=True)

preview_cols = [col for col in ["user_id", "test_id", "test_name", "bls_score_pct", "current_als_score_pct", "potential_als_score_pct", "learning_gain_pct"] if col in summary_rows.columns]
if preview_cols:
    st.dataframe(summary_rows[preview_cols].head(20), use_container_width=True)

st.caption("This page consumes shared artifacts only. No local readiness or DQ logic is recalculated here.")
