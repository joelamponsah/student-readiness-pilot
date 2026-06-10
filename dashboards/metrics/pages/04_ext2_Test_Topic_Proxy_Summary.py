import streamlit as st

from utils.artifact_loader import load_artifact
from utils.ui_helpers import optional_filter


st.set_page_config(page_title="Test / Topic Proxy Summary", layout="wide")
st.title("Test / Topic Proxy Summary")
st.caption("Content/Topic TAS Proxy. This is not final true TAS.")

tests = load_artifact("test_readiness_summary")
tas = load_artifact("content_topic_tas_proxy")
question_map = load_artifact("content_question_map")

if tas is None and tests is None:
    st.warning("Missing v13_test_readiness_summary.csv and v13_content_topic_tas_proxy.csv.")
    st.stop()

df = tas if tas is not None else tests
with st.sidebar:
    for col in ["institute_std", "class_name", "test_id", "content_title", "evidence_level"]:
        df = optional_filter(df, col, col)

st.subheader("Content/Topic TAS Proxy")
st.dataframe(df, use_container_width=True)

if tests is not None:
    st.subheader("Test Readiness Summary")
    st.dataframe(tests, use_container_width=True)

if question_map is not None:
    with st.expander("Content Question Map"):
        st.dataframe(question_map, use_container_width=True)

