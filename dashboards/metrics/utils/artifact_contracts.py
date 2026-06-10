"""Artifact contracts for the v1.3-ext2 dashboard.

The ext2 dashboard is intentionally artifact-based. It validates and displays
CSV outputs from the build notebook, but does not rebuild source joins or core
readiness metrics inside Streamlit.
"""

from __future__ import annotations


ARTIFACTS = {
    "attempt_question_metrics": {
        "filename": "v13_attempt_question_metrics.csv",
        "description": "Attempt-question evidence used for denominator and question support checks.",
        "required": [],
        "optional": ["test_taker_id", "user_id", "test_id", "question_id", "score_pct"],
    },
    "raw_attempts": {
        "filename": "v13_raw_attempts.csv",
        "description": "Attempt-level dashboard input after extraction and DQ annotation.",
        "required": [
            "test_taker_id", "user_id", "raw_institute", "institute_key", "institute_std",
            "institute_mapping_status", "missing_institute_flag", "generic_institute_flag",
            "reverse_mapping_candidate_flag", "test_id", "class_id", "class_name",
            "content_provider_name", "subscription_status", "subscription_start",
            "subscription_end", "subscription_start_month", "marks", "no_of_questions",
            "score_pct", "attempted_questions", "correct_answer_count",
            "incorrect_answer_count", "unanswered_questions", "question_bank_size",
            "total_questions_expected", "question_denominator_source",
            "accuracy_attempted_pct", "accuracy_expected_pct", "completion_pct",
            "pass_flag_safe", "time_taken", "time_taken_minutes", "speed_qpm",
            "efficiency_marks_per_min", "created_at", "finished_at", "attempt_status",
            "attempt_sequence_number", "user_test_attempt_count", "is_first_attempt",
            "is_latest_attempt", "is_best_attempt", "single_attempt_flag",
            "low_repeat_evidence_flag", "multi_class_mapping_flag",
            "duplicate_attempt_cluster_size", "dedupe_rank",
            "is_dashboard_default_attempt", "default_score_eligibility", "dq_status",
            "dq_notes",
        ],
        "optional": [],
    },
    "user_subscription_base": {
        "filename": "v13_user_subscription_base.csv",
        "description": "User subscription and cohort context.",
        "required": [],
        "optional": ["user_id", "institute_std", "class_id", "subscription_status"],
    },
    "content_question_map": {
        "filename": "v13_content_question_map.csv",
        "description": "Content/question mapping used for Content/Topic TAS proxy context.",
        "required": [],
        "optional": ["content_id", "content_title", "test_id", "question_id"],
    },
    "school_subject_cas_proxy": {
        "filename": "v13_school_subject_cas_proxy.csv",
        "description": "School-Subject CAS Proxy by school/class/provider.",
        "required": [
            "institute_std", "class_id", "class_name", "content_provider_name",
            "learner_count", "attempt_count", "test_count", "avg_score_pct",
            "avg_accuracy_expected_pct", "avg_completion_pct", "cas_proxy_score_pct",
            "evidence_score", "evidence_level", "multi_class_attempt_count",
            "excluded_multi_class_attempt_count", "first_attempt_at", "latest_attempt_at",
            "dq_warning_count",
        ],
        "optional": [],
    },
    "content_topic_tas_proxy": {
        "filename": "v13_content_topic_tas_proxy.csv",
        "description": "Content/Topic TAS Proxy by class/content.",
        "required": [
            "institute_std", "class_id", "class_name", "content_id", "content_title",
            "content_order", "learner_count", "attempt_count", "question_count",
            "avg_score_pct", "avg_accuracy_expected_pct", "avg_completion_pct",
            "tas_proxy_score_pct", "content_mapping_confidence", "evidence_level",
        ],
        "optional": [],
    },
    "learning_gain_signals": {
        "filename": "v13_learning_gain_signals.csv",
        "description": "BLS/ALS proxy movement and learning gain evidence.",
        "required": [],
        "optional": ["user_id", "test_id", "bls_proxy_score_pct", "als_proxy_score_pct", "learning_gain_pct"],
    },
    "readiness_signals": {
        "filename": "v13_readiness_signals.csv",
        "description": "Learner readiness evidence signals.",
        "required": [
            "user_id", "institute_std", "class_id", "readiness_score_pct",
            "readiness_band", "readiness_eligible_flag", "readiness_block_reason",
            "readiness_confidence_level", "evidence_score", "evidence_level",
            "avg_score_pct", "avg_accuracy_expected_pct", "avg_completion_pct",
            "attempt_count", "test_count", "question_evidence_count",
            "repeat_attempt_available_flag", "low_evidence_flag",
        ],
        "optional": [],
    },
    "work_habits_signals": {
        "filename": "v13_work_habits_signals.csv",
        "description": "Work habits evidence signals.",
        "required": [],
        "optional": ["user_id", "work_habits_score_pct", "work_habits_band"],
    },
    "learner_readiness_summary": {
        "filename": "v13_learner_readiness_summary.csv",
        "description": "Learner-level readiness summary.",
        "required": [],
        "optional": ["user_id", "institute_std", "class_id", "readiness_band"],
    },
    "school_readiness_summary": {
        "filename": "v13_school_readiness_summary.csv",
        "description": "School-level readiness summary.",
        "required": [],
        "optional": ["institute_std", "learner_count", "readiness_score_pct"],
    },
    "test_readiness_summary": {
        "filename": "v13_test_readiness_summary.csv",
        "description": "Test-level readiness summary.",
        "required": [],
        "optional": ["test_id", "content_title", "avg_score_pct", "pass_rate_safe"],
    },
    "cohort_context": {
        "filename": "v13_cohort_context.csv",
        "description": "Subscription cohort context by school/class/month.",
        "required": [],
        "optional": ["institute_std", "class_id", "subscription_start_month"],
    },
    "school_class_bundle_signals": {
        "filename": "v13_school_class_bundle_signals.csv",
        "description": "School/class bundle-level evidence signals.",
        "required": [],
        "optional": ["institute_std", "class_id", "cas_proxy_score_pct"],
    },
    "dq_summary": {
        "filename": "v13_dq_summary.csv",
        "description": "Flexible data-quality diagnostic report.",
        "required": [],
        "optional": [],
    },
    "metric_definitions": {
        "filename": "v13_metric_definitions.csv",
        "description": "Metric definitions, formulas, grains, and caveats.",
        "required": [
            "metric_name", "metric_type", "definition", "formula", "grain",
            "source_tables", "proxy_status", "default_filter", "known_limitations",
        ],
        "optional": [],
    },
    "data_dictionary": {
        "filename": "v13_data_dictionary.csv",
        "description": "Data dictionary for generated artifacts.",
        "required": [],
        "optional": ["artifact", "column_name", "definition"],
    },
    "build_summary": {
        "filename": "v13_build_summary.csv",
        "description": "Build summary and artifact-generation diagnostics.",
        "required": [],
        "optional": ["artifact", "rows", "status"],
    },
}


PAGE_ARTIFACTS = {
    "home": ["build_summary", "dq_summary", "school_readiness_summary", "raw_attempts"],
    "definitions": ["metric_definitions", "data_dictionary"],
    "data_quality": ["dq_summary", "build_summary"],
    "school_subject": ["school_subject_cas_proxy"],
    "test_topic": ["test_readiness_summary", "content_topic_tas_proxy", "content_question_map"],
    "learner": ["learner_readiness_summary", "readiness_signals", "learning_gain_signals", "work_habits_signals"],
    "cohort": ["cohort_context", "user_subscription_base"],
    "raw_explorer": list(ARTIFACTS.keys()),
}


def artifact_name_for_file(filename: str) -> str | None:
    for name, spec in ARTIFACTS.items():
        if spec["filename"] == filename:
            return name
    return None

