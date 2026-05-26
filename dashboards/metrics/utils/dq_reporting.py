# utils/dq_reporting.py
from __future__ import annotations

from typing import Dict
import pandas as pd

def render_dq_summary(dq_report: Dict):
    """
    Streamlit-only renderer.
    Call near the top of every page after apply_dq_gate().
    """
    import streamlit as st

    if not dq_report:
        st.warning("DQ report unavailable.")
        return

    policy = dq_report.get("policy", {})
    rows_raw = dq_report.get("rows_raw", 0)
    rows_included = dq_report.get("rows_included", 0)
    rows_excluded = dq_report.get("rows_excluded", 0)
    included_rate = dq_report.get("included_rate", 0.0)

    st.subheader("Data Quality Gate (DQ)")

    schema_warnings = dq_report.get("schema_warnings", {})
    if schema_warnings.get("missing_finished_at_column"):
        st.error("Missing `finished_at` column. Completed-only DQ policy treats all rows as ineligible for published KPIs.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (raw)", f"{rows_raw:,}")
    c2.metric("Rows (eligible)", f"{rows_included:,}", f"{included_rate*100:.1f}%")
    c3.metric("Rows (excluded)", f"{rows_excluded:,}")
    c4.metric("Completed-only", "ON" if policy.get("completed_only", True) else "OFF")

    # Key flag KPIs on included set
    flag_rates = dq_report.get("flag_rates_on_included", {})
    st.caption("Flag rates are computed on the eligible dataset used for KPIs.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pass mark ambiguous", f"{flag_rates.get('pass_mark_ambiguous_rate', 0)*100:.1f}%")
    k2.metric("Missing Q-level support", f"{flag_rates.get('missing_question_level_support_rate', 0)*100:.1f}%")
    k3.metric("time_taken outliers", f"{flag_rates.get('time_taken_outlier_rate', 0)*100:.1f}%")
    k4.metric("no_of_questions suspect", f"{flag_rates.get('no_of_questions_suspect_rate', 0)*100:.1f}%")

    salv = dq_report.get("salvage_stats", {})
    if salv:
        s1, s2, s3 = st.columns(3)
        s1.metric("Incomplete (raw)", f"{salv.get('incomplete_rate_raw',0)*100:.1f}%")
        s2.metric("Incomplete but usable (raw)", f"{salv.get('incomplete_usable_rate_raw',0)*100:.1f}%")
        s3.metric("Usable incomplete rows", f"{salv.get('incomplete_usable_count_raw',0):,}")
    # Exclusion breakdown
    reasons = dq_report.get("exclusion_reasons", {})
    if reasons:
        st.markdown("#### Exclusion reasons")
        df = pd.DataFrame(
            [{"reason": k, "count": v, "pct_of_raw": (v / rows_raw) if rows_raw else 0.0} for k, v in reasons.items()]
        ).sort_values("count", ascending=False)
        df["pct_of_raw"] = (df["pct_of_raw"] * 100).round(2)
        st.dataframe(df, use_container_width=True)

    # Policy display (collapsed)
    with st.expander("DQ policy (active)"):
        st.json(policy)
