import pandas as pd
import streamlit as st

from utils.dq_policy import DQConfig, apply_dq_gate
from utils.dq_profiles import (
    dq_monitor_config,
    learner_diagnostic_config,
    published_performance_config,
)
from utils.dq_reporting import render_dq_summary
from utils.metrics import load_data_from_disk_or_session


st.title("DQ Monitors")


def _config_from_sidebar() -> DQConfig:
    st.sidebar.markdown("## DQ Scenario")

    scenario = st.sidebar.radio(
        "Starting point",
        [
            "Published: completed only",
            "Diagnostic: incomplete with evidence",
            "Exploration: all attempts",
            "Custom",
        ],
        index=0,
    )

    if scenario == "Published: completed only":
        base = published_performance_config()
    elif scenario == "Diagnostic: incomplete with evidence":
        base = learner_diagnostic_config()
    elif scenario == "Exploration: all attempts":
        base = DQConfig(
            completed_only=False,
            include_incomplete_if_has_evidence=True,
            dedupe_best_attempt=False,
            strict_pass_mark=True,
            show_incomplete=True,
            export_artifacts=False,
        )
    else:
        base = dq_monitor_config()

    st.sidebar.markdown("## Inclusion")
    completed_only = st.sidebar.toggle("Completed attempts only", value=base.completed_only)
    include_incomplete_if_has_evidence = st.sidebar.toggle(
        "Include incomplete with activity evidence",
        value=base.include_incomplete_if_has_evidence,
        disabled=not completed_only,
    )
    if not completed_only:
        include_incomplete_if_has_evidence = True

    dedupe_best_attempt = st.sidebar.toggle(
        "Dedupe best attempt per learner/test",
        value=base.dedupe_best_attempt,
    )
    show_incomplete = st.sidebar.toggle("Show incomplete attempts", value=base.show_incomplete)

    st.sidebar.markdown("## Required Checks")
    require_valid_marks = st.sidebar.toggle("Require valid marks", value=base.require_valid_marks)
    require_valid_no_of_questions = st.sidebar.toggle(
        "Require valid raw no_of_questions",
        value=base.require_valid_no_of_questions,
    )
    require_valid_time = st.sidebar.toggle("Require valid time taken", value=base.require_valid_time)
    require_question_level_support = st.sidebar.toggle(
        "Require question-level support",
        value=base.require_question_level_support,
    )
    require_usable_pass_mark = st.sidebar.toggle("Require usable pass mark", value=base.require_usable_pass_mark)
    exclude_time_outliers = st.sidebar.toggle("Exclude time outliers", value=base.exclude_time_outliers)

    st.sidebar.markdown("## Reporting")
    strict_pass_mark = st.sidebar.toggle(
        "Strict pass mark in pass KPIs",
        value=base.strict_pass_mark,
    )
    export_artifacts = st.sidebar.toggle("Export DQ artifacts", value=base.export_artifacts)

    return DQConfig(
        completed_only=completed_only,
        include_incomplete_if_has_evidence=include_incomplete_if_has_evidence,
        dedupe_best_attempt=dedupe_best_attempt,
        strict_pass_mark=strict_pass_mark,
        show_incomplete=show_incomplete,
        require_valid_marks=require_valid_marks,
        require_valid_no_of_questions=require_valid_no_of_questions,
        require_valid_time=require_valid_time,
        require_question_level_support=require_question_level_support,
        require_usable_pass_mark=require_usable_pass_mark,
        exclude_time_outliers=exclude_time_outliers,
        export_artifacts=export_artifacts,
    )


def _scenario_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    scenarios = {
        "Published completed only": published_performance_config(),
        "Diagnostic incomplete evidence": learner_diagnostic_config(),
        "Exploration all attempts": DQConfig(
            completed_only=False,
            include_incomplete_if_has_evidence=True,
            dedupe_best_attempt=False,
            strict_pass_mark=True,
            show_incomplete=True,
            export_artifacts=False,
        ),
    }

    rows = []
    for label, cfg in scenarios.items():
        cfg.export_artifacts = False
        included, report, excluded = apply_dq_gate(df_raw, config=cfg)
        salvage = report.get("salvage_stats", {})
        rows.append(
            {
                "Scenario": label,
                "Rows included": report.get("rows_included", 0),
                "Rows excluded": report.get("rows_excluded", 0),
                "Included rate": report.get("included_rate", 0),
                "Users": included["user_id"].nunique() if "user_id" in included.columns else 0,
                "Tests": included["test_id"].nunique() if "test_id" in included.columns else 0,
                "Incomplete usable raw": salvage.get("incomplete_usable_count_raw", 0),
                "Top exclusion": (
                    excluded["exclusion_reason"].value_counts().index[0]
                    if not excluded.empty and "exclusion_reason" in excluded.columns
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def _top_missing_values(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "rows"])
    values = df[column].astype("string").fillna("").str.strip()
    values = values.replace({"": "UNKNOWN", "-": "UNKNOWN"})
    return values.value_counts().head(25).rename_axis(column).reset_index(name="rows")


df_raw = load_data_from_disk_or_session()
if df_raw is None or df_raw.empty:
    st.warning("No dataset loaded. Upload in sidebar or add data/verify_df_fixed.csv.")
    st.stop()

config = _config_from_sidebar()

st.subheader("Scenario Comparison")
scenario_df = _scenario_rows(df_raw)
scenario_display = scenario_df.copy()
scenario_display["Included rate"] = (scenario_display["Included rate"] * 100).round(1).astype(str) + "%"
st.dataframe(scenario_display, use_container_width=True)

df_eligible, dq_report, df_exclusions = apply_dq_gate(df_raw, config=config)
render_dq_summary(dq_report)

if df_eligible.empty:
    st.warning("No rows are eligible under the active DQ policy. Toggle checks off to inspect which gate is binding.")

st.subheader("Active Policy")
policy = dq_report.get("policy", {})
policy_df = pd.DataFrame(
    [{"Check": key, "On": value} for key, value in policy.items() if isinstance(value, bool)]
)
st.dataframe(policy_df, use_container_width=True)

st.subheader("Coverage")
coverage = dq_report.get("coverage_rates_on_included", {})
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Institute coverage", f"{coverage.get('institute_coverage_rate', 0) * 100:.1f}%")
c2.metric("City coverage", f"{coverage.get('city_coverage_rate', 0) * 100:.1f}%")
c3.metric("Country coverage", f"{coverage.get('country_coverage_rate', 0) * 100:.1f}%")
c4.metric("Question support", f"{coverage.get('question_level_support_rate', 0) * 100:.1f}%")
c5.metric("Usable pass marks", f"{coverage.get('strict_pass_mark_coverage_rate', 0) * 100:.1f}%")

st.subheader("Flag Rates on Included Rows")
flags = dq_report.get("flag_rates_on_included", {})
if flags:
    flag_df = pd.DataFrame(
        [{"Flag": key, "Rate": value, "Rows": int(round(value * len(df_eligible)))} for key, value in flags.items()]
    )
    flag_df["Rate"] = (flag_df["Rate"] * 100).round(1).astype(str) + "%"
    st.dataframe(flag_df, use_container_width=True)

st.subheader("Exclusions")
if df_exclusions.empty:
    st.success("No excluded rows under the active DQ policy.")
else:
    st.dataframe(
        df_exclusions["exclusion_reason"]
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="rows"),
        use_container_width=True,
    )
    with st.expander("Excluded row sample"):
        show_cols = [
            c
            for c in [
                "exclusion_reason",
                "user_id",
                "username",
                "test_id",
                "marks",
                "no_of_questions",
                "time_taken",
                "finished_at",
                "is_incomplete",
                "incomplete_but_usable",
                "pass_mark_ambiguous",
                "missing_question_level_support",
                "time_taken_outlier",
            ]
            if c in df_exclusions.columns
        ]
        st.dataframe(df_exclusions[show_cols].head(500), use_container_width=True)

st.subheader("Included Row Sample")
if df_eligible.empty:
    st.info("No included rows to sample.")
else:
    show_cols = [
        c
        for c in [
            "user_id",
            "username",
            "test_id",
            "marks",
            "no_of_questions",
            "time_taken",
            "finished_at",
            "is_incomplete",
            "incomplete_but_usable",
            "pass_mark_ambiguous",
            "missing_question_level_support",
            "time_taken_outlier",
        ]
        if c in df_eligible.columns
    ]
    st.dataframe(df_eligible[show_cols].head(500), use_container_width=True)

st.subheader("Unmapped Segmentation Values")
for label, column in [
    ("Institute", "institute"),
    ("City", "city"),
    ("Country", "country"),
]:
    with st.expander(label):
        st.dataframe(_top_missing_values(df_raw, column), use_container_width=True)

st.subheader("Trend Checks")
if "created_at" in df_raw.columns:
    trend = df_raw.copy()
    trend["created_at"] = pd.to_datetime(trend["created_at"], errors="coerce")
    trend = trend[trend["created_at"].notna()].copy()
    if not trend.empty:
        trend["week"] = trend["created_at"].dt.to_period("W").dt.start_time
        trend["finished_at_missing"] = pd.to_datetime(
            trend.get("finished_at", pd.Series(pd.NaT, index=trend.index)),
            errors="coerce",
        ).isna()
        weekly = trend.groupby("week").agg(
            attempts=("week", "size"),
            finished_at_missing_rate=("finished_at_missing", "mean"),
        ).reset_index()
        st.line_chart(weekly, x="week", y="finished_at_missing_rate")
    else:
        st.info("created_at exists, but no parseable timestamps were found.")
else:
    st.info("No created_at column available for trend checks.")
