import streamlit as st

from utils.artifact_loader import get_artifact_status, load_artifact, resolve_artifact_dir
from utils.ui_helpers import fmt_count


st.set_page_config(page_title="v1.3-ext2 Overview", layout="wide")
st.title("eCampus Learner Readiness v1.3-ext2")
st.subheader("Test / Exercise Readiness Evidence Layer")

st.markdown(
    """
### v1.3 scope
This version displays prebuilt Test / Exercise Readiness artifacts. It covers
school, class, content, learner, cohort, data-quality, and raw evidence views.

### Not covered
This is not the full Learn Smarter v2 model. It does not include read/listen/watch
activity weighting, final predictions, true TAS/CAS calibration, or live database
queries.

CAS, TAS, BLS, ALS, work habits, and readiness are proxy/evidence signals.
"""
)

status = get_artifact_status()
raw = load_artifact("raw_attempts")
school = load_artifact("school_readiness_summary")

st.caption(f"Artifact directory: `{resolve_artifact_dir()}`")

cols = st.columns(6)
cols[0].metric("Artifacts loaded", fmt_count(status["exists"].sum()))
cols[1].metric("Artifacts expected", fmt_count(len(status)))
cols[2].metric("Attempts", fmt_count(len(raw)) if raw is not None else "N/A")
cols[3].metric("Learners", fmt_count(raw["user_id"].nunique()) if raw is not None and "user_id" in raw.columns else "N/A")
cols[4].metric("Schools", fmt_count(raw["institute_std"].nunique()) if raw is not None and "institute_std" in raw.columns else "N/A")
cols[5].metric("Classes", fmt_count(raw["class_id"].nunique()) if raw is not None and "class_id" in raw.columns else "N/A")

if raw is not None:
    more = st.columns(2)
    more[0].metric("Tests", fmt_count(raw["test_id"].nunique()) if "test_id" in raw.columns else "N/A")
    more[1].metric("School summary rows", fmt_count(len(school)) if school is not None else "N/A")

st.subheader("Artifact Build Status")
st.dataframe(status[["artifact", "filename", "exists", "valid", "rows", "missing_required"]], use_container_width=True)

st.subheader("Navigation Guide")
st.markdown(
    """
- Definitions and Assumptions: formulas, caveats, and proxy status
- Data Quality: trust checks, missing artifacts, and warning indicators
- School / Subject Summary: School-Subject CAS Proxy
- Test / Topic Proxy Summary: Content/Topic TAS Proxy
- Learner Summary: learner readiness evidence and learning gain
- Cohort Context: subscription window context and class cohorts
- Raw Data Explorer: technical artifact inspection
"""
)

