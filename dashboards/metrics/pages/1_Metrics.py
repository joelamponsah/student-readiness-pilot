import streamlit as st
import pandas as pd

from utils.dq_policy import apply_dq_gate
from utils.dq_profiles import learner_diagnostic_config, published_performance_config
from utils.dq_reporting import render_dq_summary
from utils.learn_smarter_v13 import add_test_exercise_readiness_fields
from utils.metrics import compute_basic_metrics2, load_data_from_disk_or_session


st.set_page_config(page_title="Metrics", layout="wide")
st.title("Metrics")
st.caption("Core metrics, the math behind them, and v1.3 proxy outputs. DQ gating is shown on DQ Monitors.")

st.markdown(
    """
### What this page is
This page shows the metrics used in the v1.3 Test / Exercise Readiness analysis.
It does not define DQ policy. DQ policy is applied on the DQ Monitors page; this page
shows the resulting published KPI data and the proxy-sequence preview.

### Math used in the analysis
- `accuracy_total = marks / max_marks_effective`
- `max_marks_effective` prefers `max_marks_effective`, then `total_questions`, then `no_of_questions`
- `speed_raw = attempted_questions / time_taken`
- `adj_speed = correct_answers / time_taken`
- `efficiency_ratio = accuracy_total / time_consumed`
- `time_consumed = time_taken / duration`

### Interpretation
- Published KPI data uses best-attempt dedupe for rollups, rankings, and summaries.
- Proxy-sequence data preserves repeated eligible attempts for the v1.3 proxy outputs only.
- Proxy values are partial readiness signals, not full Learn Smarter BLS/ALS/CAS results.
"""
)

view_mode = st.sidebar.radio(
    "Dataset view",
    ["Published KPI data", "Proxy sequence data"],
    index=0,
)

published_config = published_performance_config()
proxy_config = learner_diagnostic_config()
proxy_config.dedupe_best_attempt = False
proxy_config.export_artifacts = False

df_raw = load_data_from_disk_or_session()
df_clean, _, _ = apply_dq_gate(df_raw, config=published_config)
proxy_df, _, _ = apply_dq_gate(df_raw, config=proxy_config)

if df_clean is None or df_clean.empty:
    st.warning("No dataset loaded. Upload in the sidebar or add data/verify_df_fixed.csv.")
    st.stop()

dataset = proxy_df if view_mode == "Proxy sequence data" and proxy_df is not None and not proxy_df.empty else df_clean
df = compute_basic_metrics2(dataset)

active_mask = None
if "attempted_questions_raw" in df.columns:
    active_mask = pd.to_numeric(df["attempted_questions_raw"], errors="coerce").fillna(0) > 0
elif "attempted_questions" in df.columns:
    active_mask = pd.to_numeric(df["attempted_questions"], errors="coerce").fillna(0) > 0
else:
    active_mask = pd.Series(True, index=df.index)

df["accuracy_safe"] = df["accuracy_total"].where(active_mask)

proxy_source = proxy_df if proxy_df is not None and not proxy_df.empty else df
proxy_metrics = compute_basic_metrics2(proxy_source)
readiness_df = add_test_exercise_readiness_fields(proxy_metrics)

if view_mode == "Proxy sequence data":
    st.caption(
        "Proxy sequence data keeps repeated eligible attempts visible so the v1.3 proxy outputs can be inspected."
    )
else:
    st.caption(
        "Published KPI data uses the deduped rollup view that powers rankings and published summaries."
    )

st.subheader("Core Metrics")
top = st.columns(5)
top[0].metric("Rows", f"{len(df):,}")
top[1].metric("Users", f"{df['user_id'].nunique():,}" if "user_id" in df.columns else "N/A")
top[2].metric("Tests", f"{df['test_id'].nunique():,}" if "test_id" in df.columns else "N/A")
top[3].metric("Mean accuracy", f"{df['accuracy_safe'].mean():.3f}" if df["accuracy_safe"].notna().any() else "N/A")
top[4].metric("Mean efficiency", f"{df['efficiency_ratio'].mean():.3f}" if df["efficiency_ratio"].notna().any() else "N/A")

st.dataframe(
    df[[
        "user_id",
        "test_id",
        "attempted_questions_raw" if "attempted_questions_raw" in df.columns else "attempted_questions",
        "accuracy_attempt",
        "accuracy_total",
        "accuracy_safe",
        "time_consumed",
        "speed_raw",
        "adj_speed",
        "efficiency_ratio",
    ]].head(20),
    use_container_width=True,
)

st.subheader("Learn Smarter Proxy Metrics")
st.caption(
    "These are v1.3 proxy signals only. They use repeated eligible attempts and do not claim full Learn Smarter coverage."
)
proxy_cols = st.columns(5)
proxy_cols[0].metric("Inferred BLS Proxy", f"{int(readiness_df['inferred_bls_proxy_score_pct'].notna().sum()):,}")
proxy_cols[1].metric("Current ALS Proxy", f"{int(readiness_df['current_als_proxy_score_pct'].notna().sum()):,}")
proxy_cols[2].metric("Potential ALS Proxy", f"{int(readiness_df['potential_als_proxy_score_pct'].notna().sum()):,}")
proxy_cols[3].metric("High evidence rows", f"{int((readiness_df['proxy_evidence_band'] == 'high').sum()):,}")
proxy_cols[4].metric(
    "CAS Proxy coverage",
    f"{readiness_df['cas_proxy_test_avg_score_pct'].notna().mean() * 100:.1f}%",
)

gain_cols = st.columns(4)
gain_cols[0].metric("Mean inferred BLS", f"{readiness_df['inferred_bls_proxy_score_pct'].mean():.2f}")
gain_cols[1].metric("Mean current ALS", f"{readiness_df['current_als_proxy_score_pct'].mean():.2f}")
gain_cols[2].metric("Mean potential ALS", f"{readiness_df['potential_als_proxy_score_pct'].mean():.2f}")
gain_cols[3].metric("Mean gain proxy", f"{readiness_df['learning_gain_proxy_pct'].mean():.2f}")

st.dataframe(
    readiness_df[[
        "user_id",
        "test_id",
        "inferred_bls_proxy_score_pct",
        "current_als_proxy_score_pct",
        "potential_als_proxy_score_pct",
        "learning_gain_proxy_pct",
        "proxy_evidence_band",
        "proxy_evidence_note",
    ]].head(20),
    use_container_width=True,
)
