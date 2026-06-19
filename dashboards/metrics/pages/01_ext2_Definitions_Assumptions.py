import pandas as pd
import streamlit as st

from utils.artifact_loader import load_artifact, validate_artifact_columns
from utils.ui_helpers import show_missing_columns


st.set_page_config(page_title="Definitions and Assumptions", layout="wide")
st.title("Definitions and Assumptions")

st.caption(
    "v1.3-ext2 separates learner readiness from class/topic CAS. "
    "The new CAS Definition Comparison page uses cALS as the primary teacher progression signal "
    "and pALS as a diagnostic/recovery signal only."
)

metric_definitions = load_artifact("metric_definitions")
metric_summary = load_artifact("metric_summary")
data_definitions = load_artifact("data_definitions")

if metric_definitions is not None:
    validation = validate_artifact_columns("metric_definitions", metric_definitions)
    show_missing_columns("v13_metric_definitions.csv", validation["missing_required"])
    st.subheader("Metric Definitions")
    st.dataframe(metric_definitions, use_container_width=True)
else:
    st.warning("Missing v13_metric_definitions.csv. Showing built-in v1.3-ext2 caveats.")

if metric_summary is not None:
    validation = validate_artifact_columns("metric_summary", metric_summary)
    show_missing_columns("v13_metric_summary.csv", validation["missing_required"])
    st.subheader("Metric Summary")
    st.dataframe(metric_summary, use_container_width=True)
else:
    st.info("v13_metric_summary.csv is not available yet. Run Build 99 in the notebook to refresh it.")

st.subheader("Core Assumptions")
assumptions = pd.DataFrame(
    [
        ("score_pct", "Score percentage from the artifact denominator policy."),
        ("accuracy_attempted_pct", "Accuracy over attempted questions."),
        ("accuracy_expected_pct", "Accuracy over expected delivered questions."),
        ("completion_pct", "Attempt completion evidence."),
        ("question_denominator_source", "Source used for denominator confidence."),
        (
            "Learner Readiness",
            "Individual learner-level evidence. It may use score, accuracy, coverage, work habits, and readiness bands.",
        ),
        (
            "CAS",
            "Class/topic-level understanding signal. CAS is not learner readiness and must not use speed, efficiency, SAB, readiness probability, or readiness confidence.",
        ),
        (
            "Legacy CAS Proxy",
            "Old school_subject_cas_proxy artifact retained as reference only; use CAS Definition Comparison for v1.3-ext2 cALS/pALS CAS.",
        ),
        ("BLS", "Before Lesson Score proxy: first valid completed attempt for user_id + institute_std + class_id + test_id."),
        ("cALS", "Current ALS proxy: first valid completed attempt after BLS. This is the primary CAS teacher progression signal."),
        ("pALS", "Potential ALS proxy: highest valid completed attempt after BLS. This is diagnostic only, not a proceed signal."),
        ("cas_cals_threshold_pct", "Primary CAS: percentage of learners with cALS score_pct >= fixed ALS threshold."),
        ("cas_pals_threshold_pct", "Potential/recovery CAS: percentage of learners with pALS score_pct >= fixed ALS threshold; diagnostic only."),
        ("cals_pals_gap_pct", "Gap between potential CAS and current CAS: cas_pals_threshold_pct - cas_cals_threshold_pct."),
        ("Content/Topic TAS Proxy", "Content/topic evidence proxy; not true final TAS."),
        ("readiness bands", "Evidence-based readiness bands from generated artifacts."),
        ("work habits score", "Artifact-provided work habits signal."),
        ("finished_at rule", "complete, incomplete_but_usable, incomplete_unusable."),
        ("subscriptions", "Active/expired/failed subscription windows are context only in v1.3."),
        ("institutes", "Null/generic/reverse-mapping institute flags are DQ context."),
        ("institute_group", "Grouping field for school/institute analysis. Build 99/DQ checks should flag blanks or conflicting mappings."),
        ("multi-class mapping", "Multi-class rows are flagged; default CAS excludes them where artifact logic does so."),
    ],
    columns=["term", "definition"],
)
st.dataframe(assumptions, use_container_width=True)

if data_definitions is not None:
    validation = validate_artifact_columns("data_definitions", data_definitions)
    show_missing_columns("v13_data_definitions.csv", validation["missing_required"])
    st.subheader("Data Definitions")
    st.dataframe(data_definitions, use_container_width=True)
else:
    st.info("v13_data_definitions.csv is not available yet. Run Build 99 in the notebook to refresh it.")
