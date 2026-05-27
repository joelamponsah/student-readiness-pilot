"""Named DQ policy profiles for dashboard pages."""

from utils.dq_policy import DQConfig


def published_performance_config() -> DQConfig:
    """Conservative base for published KPIs, rankings, and rollups."""
    return DQConfig(
        completed_only=True,
        include_incomplete_if_has_evidence=False,
        dedupe_best_attempt=True,
        strict_pass_mark=True,
        show_incomplete=False,
        export_artifacts=True,
    )


def dq_monitor_config() -> DQConfig:
    """DQ monitor base; same eligible set as published views, with reporting."""
    return published_performance_config()


def learner_diagnostic_config() -> DQConfig:
    """Learner drilldown base where partial attempts can be diagnostic evidence."""
    return DQConfig(
        completed_only=True,
        include_incomplete_if_has_evidence=True,
        dedupe_best_attempt=False,
        strict_pass_mark=True,
        show_incomplete=False,
        export_artifacts=True,
    )
