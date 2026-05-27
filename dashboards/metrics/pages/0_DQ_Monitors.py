import pandas as pd
import streamlit as st

from utils.dq_policy import apply_dq_gate
from utils.dq_profiles import dq_monitor_config
from utils.dq_reporting import render_dq_summary
from utils.metrics import load_data_from_disk_or_session


st.title("DQ Monitors")

df_raw = load_data_from_disk_or_session()
if df_raw is None or df_raw.empty:
    st.warning("No dataset loaded. Upload in sidebar or add data/verify_df_fixed.csv.")
    st.stop()

config = dq_monitor_config()

df_eligible, dq_report, df_exclusions = apply_dq_gate(df_raw, config=config)
render_dq_summary(dq_report)

if df_eligible.empty:
    st.warning("No rows are eligible under the active DQ policy. Monitor raw exclusions before publishing KPI views.")

st.subheader("Coverage")
coverage = dq_report.get("coverage_rates_on_included", {})
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Institute coverage", f"{coverage.get('institute_coverage_rate', 0) * 100:.1f}%")
c2.metric("City coverage", f"{coverage.get('city_coverage_rate', 0) * 100:.1f}%")
c3.metric("Country coverage", f"{coverage.get('country_coverage_rate', 0) * 100:.1f}%")
c4.metric("Question support", f"{coverage.get('question_level_support_rate', 0) * 100:.1f}%")
c5.metric("Usable pass marks", f"{coverage.get('strict_pass_mark_coverage_rate', 0) * 100:.1f}%")

st.subheader("Exclusions")
if df_exclusions.empty:
    st.success("No excluded rows under the active DQ policy.")
else:
    st.dataframe(
        df_exclusions["exclusion_reason"]
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="rows"),
        use_container_width=True,
    )

st.subheader("Unmapped Segmentation Values")

def _top_missing_values(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "rows"])
    values = df[column].astype("string").fillna("").str.strip()
    values = values.replace({"": "UNKNOWN", "-": "UNKNOWN"})
    return values.value_counts().head(25).rename_axis(column).reset_index(name="rows")

for label, column in [
    ("Institute", "institute"),
    ("City", "city"),
    ("Country", "country"),
]:
    with st.expander(label):
        st.dataframe(_top_missing_values(df_raw, column), use_container_width=True)

st.subheader("Trend Checks")
if "created_at" in df_raw.columns:
    trend = df_raw.copy()
    trend["created_at"] = pd.to_datetime(trend["created_at"], errors="coerce")
    trend = trend[trend["created_at"].notna()].copy()
    if not trend.empty:
        trend["week"] = trend["created_at"].dt.to_period("W").dt.start_time
        trend["finished_at_missing"] = pd.to_datetime(
            trend.get("finished_at", pd.Series(pd.NaT, index=trend.index)),
            errors="coerce",
        ).isna()
        weekly = trend.groupby("week").agg(
            attempts=("week", "size"),
            finished_at_missing_rate=("finished_at_missing", "mean"),
        ).reset_index()
        st.line_chart(weekly, x="week", y="finished_at_missing_rate")
    else:
        st.info("created_at exists, but no parseable timestamps were found.")
else:
    st.info("No created_at column available for trend checks.")
