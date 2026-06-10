import streamlit as st

from utils.artifact_loader import load_artifact
from utils.ui_helpers import optional_filter


st.set_page_config(page_title="Cohort Context", layout="wide")
st.title("Cohort Context")
st.caption("Subscription cohorts are context and drilldown only in v1.3. They do not hard-filter readiness.")

cohort = load_artifact("cohort_context")
subs = load_artifact("user_subscription_base")

if cohort is None and subs is None:
    st.warning("Missing cohort context artifacts.")
    st.stop()

df = cohort if cohort is not None else subs
with st.sidebar:
    for col in ["institute_std", "class_id", "class_name", "subscription_start_month", "subscription_window_status"]:
        df = optional_filter(df, col, col)

st.subheader("Cohort Context")
st.dataframe(df, use_container_width=True)

if subs is not None:
    st.subheader("User Subscription Base")
    st.dataframe(subs, use_container_width=True)

