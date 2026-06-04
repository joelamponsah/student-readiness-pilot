import pandas as pd
import streamlit as st

from utils.metrics import get_v13_artifacts


st.title("DQ Monitors")
st.caption("DQ monitoring now reads the shared v1.3 artifact bundle built from raw_attempts.csv.")

raw_df, artifacts = get_v13_artifacts()
if raw_df is None or not artifacts:
    st.warning("No raw_attempts.csv input or shared v1.3 artifacts are available.")
    st.stop()

dq_attempts = artifacts.get("dq_attempts")
published_kpi = artifacts.get("published_kpi")
proxy_sequence = artifacts.get("proxy_sequence")
smoke_report = artifacts.get("smoke_report")

if dq_attempts is None or dq_attempts.empty:
    st.warning("dq_attempts is unavailable in the current artifact bundle.")
    st.stop()

st.markdown(
    """
### DQ focus
- completion status
- exclusions
- published vs proxy coverage
- denominator readiness
- missing `finished_at` caveat
"""
)

st.subheader("Bundle Summary")
summary_cols = st.columns(4)
summary_cols[0].metric("Raw rows", f"{len(raw_df):,}")
summary_cols[1].metric("Raw users", f"{raw_df['user_id'].nunique():,}" if "user_id" in raw_df.columns else "N/A")
summary_cols[2].metric("Raw tests", f"{raw_df['test_id'].nunique():,}" if "test_id" in raw_df.columns else "N/A")
summary_cols[3].metric("DQ rows", f"{len(dq_attempts):,}")

coverage_cols = st.columns(3)
coverage_cols[0].metric(
    "Published KPI",
    f"{len(published_kpi):,} rows | {published_kpi['user_id'].nunique():,} users | {published_kpi['test_id'].nunique():,} tests"
    if published_kpi is not None and not published_kpi.empty and {"user_id", "test_id"}.issubset(published_kpi.columns)
    else "N/A",
)
coverage_cols[1].metric(
    "Proxy sequence",
    f"{len(proxy_sequence):,} rows | {proxy_sequence['user_id'].nunique():,} users | {proxy_sequence['test_id'].nunique():,} tests"
    if proxy_sequence is not None and not proxy_sequence.empty and {"user_id", "test_id"}.issubset(proxy_sequence.columns)
    else "N/A",
)
coverage_cols[2].metric(
    "Smoke report rows",
    f"{len(smoke_report):,}" if smoke_report is not None and not smoke_report.empty else "N/A",
)

st.subheader("Smoke Report")
if smoke_report is not None and not smoke_report.empty:
    st.dataframe(smoke_report.T, use_container_width=True)
else:
    st.warning("No smoke_report found in the shared artifact bundle.")

st.subheader("DQ Status Counts")
if "completion_status" in dq_attempts.columns:
    st.dataframe(
        dq_attempts["completion_status"].value_counts(dropna=False).rename_axis("completion_status").reset_index(name="rows"),
        use_container_width=True,
    )
else:
    st.warning("completion_status is not present in dq_attempts.")

status_cols = st.columns(3)
status_cols[0].metric(
    "dq_included true",
    f"{int(dq_attempts['dq_included'].fillna(False).sum()):,}" if "dq_included" in dq_attempts.columns else "N/A",
)
status_cols[1].metric(
    "dq_bucket counts",
    f"{dq_attempts['dq_bucket'].nunique(dropna=True):,}" if "dq_bucket" in dq_attempts.columns else "N/A",
)
status_cols[2].metric(
    "missing finished_at",
    f"{int(dq_attempts['finished_at'].isna().sum()):,}" if "finished_at" in dq_attempts.columns else "N/A",
)

st.subheader("DQ Included / Excluded")
if "dq_bucket" in dq_attempts.columns:
    st.dataframe(dq_attempts["dq_bucket"].value_counts(dropna=False).rename_axis("dq_bucket").reset_index(name="rows"), use_container_width=True)
else:
    st.warning("dq_bucket is not present in dq_attempts.")

if "exclusion_reason" in dq_attempts.columns:
    st.subheader("Exclusion Reasons")
    exclusion_counts = dq_attempts.loc[dq_attempts["dq_bucket"].eq("excluded"), "exclusion_reason"].value_counts(dropna=False)
    if not exclusion_counts.empty:
        st.dataframe(exclusion_counts.rename_axis("reason").reset_index(name="rows"), use_container_width=True)
    else:
        st.info("No exclusion_reason values were found in the excluded rows.")

if "accuracy_denominator_source" in dq_attempts.columns:
    st.subheader("Denominator Source Counts")
    st.dataframe(
        dq_attempts["accuracy_denominator_source"].value_counts(dropna=False).rename_axis("source").reset_index(name="rows"),
        use_container_width=True,
    )

if "finished_at" in dq_attempts.columns:
    st.subheader("finished_at Summary")
    fin = pd.DataFrame(
        [
            {"status": "present", "rows": int(dq_attempts["finished_at"].notna().sum())},
            {"status": "missing", "rows": int(dq_attempts["finished_at"].isna().sum())},
        ]
    )
    st.dataframe(fin, use_container_width=True)

st.subheader("Published vs Proxy Coverage")
coverage = pd.DataFrame(
    [
        {
            "dataset": "published_kpi",
            "rows": len(published_kpi) if published_kpi is not None else 0,
            "users": published_kpi["user_id"].nunique() if published_kpi is not None and not published_kpi.empty and "user_id" in published_kpi.columns else 0,
            "tests": published_kpi["test_id"].nunique() if published_kpi is not None and not published_kpi.empty and "test_id" in published_kpi.columns else 0,
        },
        {
            "dataset": "proxy_sequence",
            "rows": len(proxy_sequence) if proxy_sequence is not None else 0,
            "users": proxy_sequence["user_id"].nunique() if proxy_sequence is not None and not proxy_sequence.empty and "user_id" in proxy_sequence.columns else 0,
            "tests": proxy_sequence["test_id"].nunique() if proxy_sequence is not None and not proxy_sequence.empty and "test_id" in proxy_sequence.columns else 0,
        },
    ]
)
st.dataframe(coverage, use_container_width=True)

st.caption("The page above consumes shared artifacts only. No local DQ gate is rebuilt here.")
