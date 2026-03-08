import streamlit as st
from utils.dq_policy import DQConfig

def dq_sidebar_controls() -> DQConfig:
    st.sidebar.markdown("## DQ Controls")

    completed_only = st.sidebar.toggle("Completed-only", value=True)
    dedupe = st.sidebar.toggle("Dedupe best attempt per user/test", value=True)
    strict_pass_mark = st.sidebar.toggle("Strict pass_mark (exclude ambiguous in pass KPIs)", value=True)

    st.sidebar.markdown("---")
    show_incomplete = st.sidebar.toggle("Show incomplete attempts (exploration)", value=False)
    export_artifacts = st.sidebar.toggle("Export DQ artifacts", value=True)

    return DQConfig(
        completed_only=completed_only,
        dedupe_best_attempt=dedupe,
        strict_pass_mark=strict_pass_mark,
        show_incomplete=show_incomplete,
        export_artifacts=export_artifacts,
    )
