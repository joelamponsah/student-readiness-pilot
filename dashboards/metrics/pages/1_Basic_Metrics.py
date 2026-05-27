import streamlit as st

from utils.dq_policy import apply_dq_gate
from utils.dq_profiles import learner_diagnostic_config, published_performance_config
from utils.dq_reporting import render_dq_summary
from utils.learn_smarter_v13 import add_test_exercise_readiness_fields
from utils.metrics import compute_basic_metrics2, load_data_from_disk_or_session


st.title("Basic Metrics")
st.caption("DQ-gated core metrics plus v1.3 Test / Exercise Readiness proxy metrics.")

view_mode = st.sidebar.radio(
    "Metric view",
    ["Published", "Diagnostic preview"],
    index=0,
)

published_config = published_performance_config()
proxy_config = learner_diagnostic_config()
proxy_config.dedupe_best_attempt = False
proxy_config.export_artifacts = False

df_raw = load_data_from_disk_or_session()
df_clean, dq_report, df_exclusions = apply_dq_gate(df_raw, config=published_config)
proxy_df, proxy_dq_report, _ = apply_dq_gate(df_raw, config=proxy_config)

render_dq_summary(dq_report if view_mode == "Published" else proxy_dq_report)

if df_clean is None or df_clean.empty:
    st.warning("No dataset loaded. Upload in sidebar or add data/verify_df_fixed.csv.")
    st.stop()

df = compute_basic_metrics2(df_clean)
proxy_metrics = compute_basic_metrics2(proxy_df) if proxy_df is not None and not proxy_df.empty else proxy_df
readiness_df = add_test_exercise_readiness_fields(proxy_metrics if proxy_metrics is not None else df.copy())

if view_mode == "Diagnostic preview":
    st.caption(
        "Diagnostic preview keeps repeated eligible attempts visible so proxy outputs can be inspected. "
        "Published mode remains the default for leaderboard and rollup logic."
    )

st.subheader("Core Metrics")
top = st.columns(5)
top[0].metric("Rows", f"{len(df):,}")
top[1].metric("Users", f"{df['user_id'].nunique():,}" if "user_id" in df.columns else "N/A")
top[2].metric("Tests", f"{df['test_id'].nunique():,}" if "test_id" in df.columns else "N/A")
top[3].metric("Mean accuracy", f"{df['accuracy_total'].mean():.3f}")
top[4].metric("Mean efficiency", f"{df['efficiency_ratio'].mean():.3f}")

st.dataframe(
    df[[
        "user_id",
        "test_id",
        "accuracy_attempt",
        "accuracy_total",
        "time_consumed",
        "speed_raw",
        "adj_speed",
        "efficiency_ratio",
    ]].head(20),
    use_container_width=True,
)

st.subheader("Learn Smarter Proxy Metrics")
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
