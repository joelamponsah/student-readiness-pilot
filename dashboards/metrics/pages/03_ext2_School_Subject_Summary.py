import streamlit as st

from utils.artifact_loader import load_required_artifact
from utils.ui_helpers import min_numeric_filter, optional_filter


st.set_page_config(page_title="School / Subject Summary", layout="wide")
st.title("School / Subject Summary")
st.caption("School-Subject CAS Proxy. CAS Proxy is not true final CAS.")

df = load_required_artifact("school_subject_cas_proxy")

with st.sidebar:
    df = optional_filter(df, "institute_std", "School")
    df = optional_filter(df, "class_name", "Class")
    df = optional_filter(df, "content_provider_name", "Content provider")
    df = optional_filter(df, "evidence_level", "Evidence level")
    df = min_numeric_filter(df, "learner_count", "Minimum learners")
    df = min_numeric_filter(df, "attempt_count", "Minimum attempts")
    if "multi_class_attempt_count" in df.columns:
        include_multi = st.toggle("Include groups with multi-class effects", value=True)
        if not include_multi:
            df = df[df["multi_class_attempt_count"].fillna(0) == 0]

cols = st.columns(4)
cols[0].metric("Rows", len(df))
cols[1].metric("Learners", int(df["learner_count"].sum()) if "learner_count" in df.columns else "N/A")
cols[2].metric("Attempts", int(df["attempt_count"].sum()) if "attempt_count" in df.columns else "N/A")
cols[3].metric("Avg CAS Proxy", f"{df['cas_proxy_score_pct'].mean():.1f}%" if "cas_proxy_score_pct" in df.columns and not df.empty else "N/A")

show_cols = [
    "institute_std", "class_id", "class_name", "content_provider_name",
    "learner_count", "attempt_count", "test_count", "avg_score_pct",
    "avg_accuracy_expected_pct", "avg_completion_pct", "cas_proxy_score_pct",
    "evidence_score", "evidence_level", "multi_class_attempt_count",
    "excluded_multi_class_attempt_count", "first_attempt_at", "latest_attempt_at",
    "dq_warning_count",
]
st.dataframe(df[[c for c in show_cols if c in df.columns]], use_container_width=True)

