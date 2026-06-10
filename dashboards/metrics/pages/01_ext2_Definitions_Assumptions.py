import pandas as pd
import streamlit as st

from utils.artifact_loader import load_artifact, validate_artifact_columns
from utils.ui_helpers import show_missing_columns


st.set_page_config(page_title="Definitions and Assumptions", layout="wide")
st.title("Definitions and Assumptions")

defs = load_artifact("metric_definitions")
dictionary = load_artifact("data_dictionary")

if defs is not None:
    validation = validate_artifact_columns("metric_definitions", defs)
    show_missing_columns("v13_metric_definitions.csv", validation["missing_required"])
    st.subheader("Metric Definitions")
    st.dataframe(defs, use_container_width=True)
else:
    st.warning("Missing v13_metric_definitions.csv. Showing built-in v1.3-ext2 caveats.")

st.subheader("Core Assumptions")
assumptions = pd.DataFrame(
    [
        ("score_pct", "Score percentage from the artifact denominator policy."),
        ("accuracy_attempted_pct", "Accuracy over attempted questions."),
        ("accuracy_expected_pct", "Accuracy over expected delivered questions."),
        ("completion_pct", "Attempt completion evidence."),
        ("question_denominator_source", "Source used for denominator confidence."),
        ("CAS Proxy", "School-Subject CAS Proxy; not true final CAS."),
        ("Content/Topic TAS Proxy", "Content/topic evidence proxy; not true final TAS."),
        ("BLS proxy", "Baseline-like first-attempt evidence when available."),
        ("ALS proxy", "Later-attempt evidence when repeat attempts exist."),
        ("readiness bands", "Evidence-based readiness bands from generated artifacts."),
        ("work habits score", "Artifact-provided work habits signal."),
        ("finished_at rule", "complete, incomplete_but_usable, incomplete_unusable."),
        ("subscriptions", "Active/expired/failed subscription windows are context only in v1.3."),
        ("institutes", "Null/generic/reverse-mapping institute flags are DQ context."),
        ("multi-class mapping", "Multi-class rows are flagged; default CAS excludes them where artifact logic does so."),
    ],
    columns=["term", "definition"],
)
st.dataframe(assumptions, use_container_width=True)

if dictionary is not None:
    st.subheader("Data Dictionary")
    st.dataframe(dictionary, use_container_width=True)

