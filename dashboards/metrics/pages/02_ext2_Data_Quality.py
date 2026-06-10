import streamlit as st

from utils.artifact_loader import get_artifact_status, load_artifact
from utils.ui_helpers import fmt_pct


st.set_page_config(page_title="Data Quality", layout="wide")
st.title("Data Quality")
st.caption("Data trust checks and artifact warnings for v1.3-ext2.")

status = get_artifact_status()
dq = load_artifact("dq_summary")
build = load_artifact("build_summary")
raw = load_artifact("raw_attempts")

cols = st.columns(4)
cols[0].metric("Missing artifacts", int((~status["exists"]).sum()))
cols[1].metric("Invalid artifacts", int((status["exists"] & ~status["valid"]).sum()))
cols[2].metric("Artifacts loaded", int(status["exists"].sum()))
cols[3].metric("Artifacts expected", len(status))

st.subheader("Artifact Status")
st.dataframe(status[["artifact", "filename", "exists", "valid", "rows", "missing_required"]], use_container_width=True)

if raw is not None:
    st.subheader("Raw Attempt DQ Signals")
    c = st.columns(4)
    c[0].metric("Missing institute rate", fmt_pct(raw.get("missing_institute_flag", []).mean() * 100) if "missing_institute_flag" in raw.columns else "N/A")
    c[1].metric("Generic institute rate", fmt_pct(raw.get("generic_institute_flag", []).mean() * 100) if "generic_institute_flag" in raw.columns else "N/A")
    c[2].metric("Multi-class rows", int(raw["multi_class_mapping_flag"].fillna(False).sum()) if "multi_class_mapping_flag" in raw.columns else "N/A")
    c[3].metric("Low repeat evidence rows", int(raw["low_repeat_evidence_flag"].fillna(False).sum()) if "low_repeat_evidence_flag" in raw.columns else "N/A")

    for column in ["question_denominator_source", "dq_status", "attempt_status", "subscription_status"]:
        if column in raw.columns:
            st.write(f"{column} distribution")
            st.dataframe(raw[column].value_counts(dropna=False).rename_axis(column).reset_index(name="rows"), use_container_width=True)

if dq is not None:
    st.subheader("DQ Summary Artifact")
    st.dataframe(dq, use_container_width=True)
else:
    st.warning("v13_dq_summary.csv is missing.")

if build is not None:
    st.subheader("Build Summary")
    st.dataframe(build, use_container_width=True)

