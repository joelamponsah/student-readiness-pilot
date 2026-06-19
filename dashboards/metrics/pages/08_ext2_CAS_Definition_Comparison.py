import pandas as pd
import streamlit as st

from utils.artifact_loader import load_artifact, load_required_artifact
from utils.ui_helpers import dataframe_with_download, fmt_count, fmt_pct, min_numeric_filter, optional_filter


st.set_page_config(page_title="CAS Definition Comparison", layout="wide")
st.title("CAS Definition Comparison")
st.caption(
    "v1.3-ext2 CAS uses cALS as the primary class/topic progression signal. "
    "pALS is diagnostic only. CAS is not learner readiness."
)

cas_df = load_required_artifact("cas_definition_comparison")
sequence_df = load_artifact("user_bls_cals_pals_sequence")

REQUIRED_COLUMNS = [
    "institute_std",
    "institute_group",
    "class_id",
    "class_name",
    "test_id",
    "test_name",
    "learner_count",
    "cals_available_learner_count",
    "pals_available_learner_count",
    "cals_coverage_pct",
    "pals_coverage_pct",
    "cas_cals_threshold_pct",
    "cas_pals_threshold_pct",
    "cals_pals_gap_pct",
    "cals_cas_signal",
    "pals_potential_signal",
    "primary_evidence_level",
]

missing = [col for col in REQUIRED_COLUMNS if col not in cas_df.columns]
if missing:
    st.warning(f"CAS comparison artifact is missing expected dashboard columns: {missing}")

df = cas_df.copy()

with st.sidebar:
    st.header("Filters")
    df = optional_filter(df, "institute_group", "Institute group")
    df = optional_filter(df, "institute_std", "Institute")
    df = optional_filter(df, "class_name", "Class / topic")
    df = optional_filter(df, "test_name", "Test")
    df = optional_filter(df, "primary_evidence_level", "Primary evidence level")
    df = optional_filter(df, "cals_cas_signal", "cALS CAS signal")
    df = optional_filter(df, "cals_pals_gap_status", "cALS / pALS gap status")
    df = min_numeric_filter(df, "learner_count", "Minimum learners")

    hide_insufficient = st.toggle("Hide insufficient primary evidence", value=True)
    if hide_insufficient and "primary_evidence_level" in df.columns:
        df = df[df["primary_evidence_level"].astype(str) != "INSUFFICIENT"].copy()

    show_legacy_warning = st.toggle("Show CAS caveat", value=True)


if show_legacy_warning:
    st.info(
        "CAS Definition Comparison is class/topic-level. "
        "Primary CAS = cas_cals_threshold_pct. "
        "Potential CAS = cas_pals_threshold_pct and should not be used as a proceed signal. "
        "BLS, cALS, and pALS are v1.3 proxy values derived from attempt sequencing."
    )

# -----------------------------
# KPI cards
# -----------------------------
kpi_cols = st.columns(6)

avg_current_cas = df["cas_cals_threshold_pct"].mean() if "cas_cals_threshold_pct" in df.columns and not df.empty else pd.NA
avg_potential_cas = df["cas_pals_threshold_pct"].mean() if "cas_pals_threshold_pct" in df.columns and not df.empty else pd.NA
avg_gap = df["cals_pals_gap_pct"].mean() if "cals_pals_gap_pct" in df.columns and not df.empty else pd.NA
avg_cals = df["avg_cals_score_pct"].mean() if "avg_cals_score_pct" in df.columns and not df.empty else pd.NA
avg_pals = df["avg_pals_score_pct"].mean() if "avg_pals_score_pct" in df.columns and not df.empty else pd.NA
low_evidence_count = (
    int(df["low_evidence_flag"].fillna(False).astype(bool).sum())
    if "low_evidence_flag" in df.columns and not df.empty
    else 0
)

kpi_cols[0].metric("Groups", fmt_count(len(df)))
kpi_cols[1].metric("Avg Current CAS", fmt_pct(avg_current_cas))
kpi_cols[2].metric("Avg Potential CAS", fmt_pct(avg_potential_cas))
kpi_cols[3].metric("Avg CAS Gap", fmt_pct(avg_gap))
kpi_cols[4].metric("Avg cALS Score", fmt_pct(avg_cals))
kpi_cols[5].metric("Low Evidence Groups", fmt_count(low_evidence_count))

# -----------------------------
# Distributions
# -----------------------------
st.subheader("CAS signal distribution")

dist_cols = st.columns(3)
if "cals_cas_signal" in df.columns:
    dist_cols[0].write("cALS CAS signal")
    dist_cols[0].dataframe(
        df["cals_cas_signal"].value_counts(dropna=False).rename_axis("signal").reset_index(name="count"),
        use_container_width=True,
    )

if "pals_potential_signal" in df.columns:
    dist_cols[1].write("pALS potential signal")
    dist_cols[1].dataframe(
        df["pals_potential_signal"].value_counts(dropna=False).rename_axis("signal").reset_index(name="count"),
        use_container_width=True,
    )

if "cals_pals_gap_status" in df.columns:
    dist_cols[2].write("cALS / pALS gap status")
    dist_cols[2].dataframe(
        df["cals_pals_gap_status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count"),
        use_container_width=True,
    )

# -----------------------------
# Main table
# -----------------------------
st.subheader("Class/topic CAS comparison")

default_sort = [
    col
    for col in ["primary_evidence_level", "cas_cals_threshold_pct", "learner_count"]
    if col in df.columns
]
if default_sort:
    ascending = [True if col != "learner_count" else False for col in default_sort]
    df = df.sort_values(default_sort, ascending=ascending, na_position="last").copy()

show_cols = [
    "institute_std",
    "institute_group",
    "class_id",
    "class_name",
    "test_id",
    "test_name",
    "learner_count",
    "attempt_count",
    "bls_available_learner_count",
    "cals_available_learner_count",
    "pals_available_learner_count",
    "bls_coverage_pct",
    "cals_coverage_pct",
    "pals_coverage_pct",
    "avg_bls_score_pct",
    "avg_cals_score_pct",
    "median_cals_score_pct",
    "avg_pals_score_pct",
    "median_pals_score_pct",
    "avg_cals_learning_gain_pct",
    "avg_pals_learning_gain_pct",
    "cas_cals_threshold_pct",
    "cas_pals_threshold_pct",
    "cas_cals_pass_mark_pct",
    "cas_pals_pass_mark_pct",
    "cals_pals_gap_pct",
    "cals_pals_gap_status",
    "cals_cas_signal",
    "pals_potential_signal",
    "primary_evidence_level",
    "pass_mark_valid_flag",
    "threshold_vs_passmark_status",
    "caveat_note",
]
visible_cols = [col for col in show_cols if col in df.columns]
dataframe_with_download(
    df[visible_cols] if visible_cols else df,
    "Download filtered CAS comparison",
    "filtered_cas_definition_comparison.csv",
)

# -----------------------------
# Optional learner sequence drill-down
# -----------------------------
with st.expander("Learner BLS / cALS / pALS proxy sequence drill-down", expanded=False):
    if sequence_df is None:
        st.warning("Optional artifact missing: v13_user_bls_cals_pals_sequence.csv")
    else:
        seq = sequence_df.copy()

        if "institute_std" in df.columns and "institute_std" in seq.columns:
            selected_institutes = df["institute_std"].dropna().astype(str).unique()
            seq = seq[seq["institute_std"].astype(str).isin(selected_institutes)].copy()

        if "class_id" in df.columns and "class_id" in seq.columns:
            selected_classes = df["class_id"].dropna().astype(str).unique()
            seq = seq[seq["class_id"].astype(str).isin(selected_classes)].copy()

        if "test_id" in df.columns and "test_id" in seq.columns:
            selected_tests = df["test_id"].dropna().astype(str).unique()
            seq = seq[seq["test_id"].astype(str).isin(selected_tests)].copy()

        learner_cols = [
            "user_id",
            "institute_std",
            "institute_group",
            "class_id",
            "class_name",
            "test_id",
            "test_name",
            "learner_attempt_count",
            "bls_score_display",
            "bls_score_pct",
            "cals_score_display",
            "cals_score_pct",
            "pals_score_display",
            "pals_score_pct",
            "cals_learning_gain_pct",
            "pals_learning_gain_pct",
            "has_bls",
            "has_cals",
            "has_pals",
            "sequence_caveat_note",
        ]
        learner_cols = [col for col in learner_cols if col in seq.columns]
        st.caption("Filtered to the class/test groups currently visible in the CAS comparison table.")
        dataframe_with_download(
            seq[learner_cols] if learner_cols else seq,
            "Download filtered learner sequence",
            "filtered_user_bls_cals_pals_sequence.csv",
        )
