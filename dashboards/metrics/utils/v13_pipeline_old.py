"""Shared v1.3 pipeline for raw attempts -> DQ -> readiness artifacts.

This module stays inside the v1.3 Test / Exercise Readiness boundary.
It preserves the existing readiness engine and Learn Smarter proxy helpers,
but centralizes the artifact-building sequence so pages can later consume the
same outputs without drifting formulas.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from utils.dq_policy import apply_dq_gate
from utils.dq_profiles import learner_diagnostic_config
from utils.insights import apply_insight_engine
from utils.learn_smarter_v13 import add_test_exercise_readiness_fields
from utils.metrics import (
    compute_basic_metrics2,
    compute_difficulty_df,
    compute_sab_behavioral,
    compute_user_coverage_features,
    compute_user_pass_features,
)


def _copy(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    return df.copy()


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _first_non_null(series: pd.Series):
    if series is None or len(series) == 0:
        return np.nan
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    value = non_null.iloc[0]
    return value


def _mode_or_first(series: pd.Series):
    if series is None or len(series) == 0:
        return np.nan
    s = series.dropna()
    if s.empty:
        return np.nan
    counts = s.astype(str).value_counts()
    if counts.empty:
        return s.iloc[0]
    return counts.index[0]


def _string_or_nan(value) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return text if text else np.nan


def _boolean_mask(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.Series(df[column], index=df.index).fillna(False).astype(bool)
    return pd.Series(False, index=df.index)


def _maybe_add_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _maybe_add_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def standardize_v13_fields(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Preserve raw fields and add stable display/standardization columns."""
    df = _copy(raw_df)
    if df.empty:
        return df

    df = _maybe_add_datetime(df, ["created_at", "finished_at", "updated_at"])
    df = _maybe_add_numeric(
        df,
        [
            "marks",
            "attempted_questions",
            "correct_answers",
            "wrong_answers",
            "time_taken",
            "duration",
            "no_of_questions",
            "total_questions",
            "question_limit",
            "question_bank_count",
            "max_marks_db",
            "pass_mark",
        ],
    )

    if "attempt_id" not in df.columns and "test_taker_id" in df.columns:
        df["attempt_id"] = df["test_taker_id"]
    df["attempt_id_std"] = df.get("attempt_id", df.get("test_taker_id"))

    # Institute standardization: prefer the source-standardized column if present.
    institute_source = None
    if "institute_standardized" in df.columns:
        institute_source = "institute_standardized"
        df["institute_std"] = df["institute_standardized"].fillna("Unknown").astype(str).str.strip()
        df["institute_std"] = df["institute_std"].replace("", "Unknown")
        df["institute_mapping_source"] = "source_column"
        df["institute_mapping_confidence"] = np.where(df["institute_std"].eq("Unknown"), "low", "high")
    elif "institute" in df.columns:
        institute_source = "institute"
        clean = df["institute"].fillna("").astype(str).str.strip()
        df["institute_raw"] = df["institute"]
        df["institute_clean_key"] = clean.str.lower()
        df["institute_std"] = clean.replace("", "Unknown").str.title()
        df["institute_mapping_source"] = "derived_title_case"
        df["institute_mapping_confidence"] = np.where(df["institute_std"].eq("Unknown"), "low", "medium")
    else:
        df["institute_std"] = "Unknown"
        df["institute_mapping_source"] = "missing"
        df["institute_mapping_confidence"] = "low"

    if "institute_raw" not in df.columns and institute_source == "institute_standardized":
        df["institute_raw"] = df.get("institute", df.get("institute_standardized"))
    if "institute_clean_key" not in df.columns:
        raw_institute = df["institute_std"].fillna("Unknown").astype(str).str.strip()
        df["institute_clean_key"] = raw_institute.str.lower().replace("", "unknown")
    df["institute_unmapped_flag"] = df["institute_std"].fillna("Unknown").astype(str).eq("Unknown")

    # Learner display ID: keep user_id canonical; add a display string for UI pages.
    learner_candidates = []
    for col in ["username", "learner_id", "student_id"]:
        if col in df.columns:
            learner_candidates.append(df[col].astype("string").fillna("").str.strip())
    if learner_candidates:
        display = learner_candidates[0].copy()
        for candidate in learner_candidates[1:]:
            display = display.where(display.ne(""), candidate)
        display = display.where(display.ne(""), df["user_id"].astype(str))
        df["learner_id_display"] = display
        df["learner_id_source"] = np.where(df.get("username", pd.Series("", index=df.index)).astype("string").fillna("").str.strip().ne(""), "username",
                                           np.where(df.get("learner_id", pd.Series("", index=df.index)).astype("string").fillna("").str.strip().ne(""), "learner_id",
                                                    np.where(df.get("student_id", pd.Series("", index=df.index)).astype("string").fillna("").str.strip().ne(""), "student_id", "user_id")))
        df["learner_id_confidence"] = np.where(df["learner_id_source"].eq("user_id"), "low", "high")
    else:
        df["learner_id_display"] = df["user_id"].astype(str)
        df["learner_id_source"] = "user_id"
        df["learner_id_confidence"] = "low"

    if "class_id" in df.columns:
        df["class_id_raw"] = df["class_id"]
        df["class_id_std"] = df["class_id"]
        df["class_label_display"] = df["class_id"].astype("string")
    else:
        df["class_id_raw"] = np.nan
        df["class_id_std"] = np.nan
        df["class_label_display"] = np.nan

    # Preserve geo/source context without inferring topic/subject/year.
    for col in ["city", "country", "subscriber_id"]:
        if col in df.columns and f"{col}_raw" not in df.columns:
            df[f"{col}_raw"] = df[col]

    if "attempt_id_std" not in df.columns:
        df["attempt_id_std"] = df.get("test_taker_id", np.nan)

    return df


def build_dq_attempts(standardized_attempts: pd.DataFrame) -> pd.DataFrame:
    """Annotate all rows with DQ metadata while keeping the full row set intact."""
    df = _copy(standardized_attempts)
    if df.empty:
        return df

    df["__v13_row_order"] = np.arange(len(df))
    config = learner_diagnostic_config()
    config.dedupe_best_attempt = False
    config.export_artifacts = False

    df_clean, _, df_exclusions = apply_dq_gate(df, config=config)
    df_clean = df_clean.copy()
    df_exclusions = df_exclusions.copy()

    df_clean["dq_included"] = True
    df_exclusions["dq_included"] = False
    df_clean["dq_eligible_proxy_sequence"] = True
    df_exclusions["dq_eligible_proxy_sequence"] = False
    df_clean["dq_eligible_published"] = df_clean["completion_status"].eq("verified_complete")
    df_exclusions["dq_eligible_published"] = False
    df_clean["dq_bucket"] = "included"
    df_exclusions["dq_bucket"] = "excluded"

    full = pd.concat([df_clean, df_exclusions], ignore_index=False, sort=False)
    if "__v13_row_order" in full.columns:
        full = full.sort_values("__v13_row_order", kind="mergesort")
    full = full.reset_index(drop=True)

    return full


def build_published_kpi_dataset(dq_attempts: pd.DataFrame) -> pd.DataFrame:
    """Strict published KPI slice: best completed attempt per learner/test."""
    if dq_attempts is None or dq_attempts.empty:
        return pd.DataFrame()

    df = dq_attempts.loc[_boolean_mask(dq_attempts, "dq_eligible_published")].copy()
    if df.empty:
        return df

    df = compute_basic_metrics2(df)
    sort_cols = [col for col in ["user_id", "test_id", "marks", "created_at", "attempt_id_std", "test_taker_id"] if col in df.columns]
    ascending = [True, True, False, False, False, False][: len(sort_cols)]
    df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort")

    dedupe_keys = [col for col in ["user_id", "test_id"] if col in df.columns]
    if dedupe_keys:
        df = df.drop_duplicates(dedupe_keys, keep="first")

    df = df.reset_index(drop=True)
    df["published_best_attempt"] = True
    return df


def build_proxy_sequence_dataset(dq_attempts: pd.DataFrame) -> pd.DataFrame:
    """Proxy sequence keeps repeated eligible attempts visible for BLS/ALS/CAS."""
    if dq_attempts is None or dq_attempts.empty:
        return pd.DataFrame()

    df = dq_attempts.loc[_boolean_mask(dq_attempts, "dq_eligible_proxy_sequence")].copy()
    if df.empty:
        return df

    df = compute_basic_metrics2(df)
    df = df.sort_values(
        [col for col in ["user_id", "test_id", "created_at", "attempt_id_std", "test_taker_id"] if col in df.columns],
        kind="mergesort",
    ).reset_index(drop=True)
    df = add_test_exercise_readiness_fields(df)
    return df


def build_existing_readiness_outputs(published_kpi: pd.DataFrame) -> pd.DataFrame:
    """Preserve the existing readiness engine while feeding it v1.3-cleaned data."""
    if published_kpi is None or published_kpi.empty:
        return pd.DataFrame()

    df = compute_basic_metrics2(published_kpi)

    sab_df = compute_sab_behavioral(df)
    pass_user = compute_user_pass_features(df)
    coverage_user = compute_user_coverage_features(df)

    sab_df = sab_df.merge(pass_user, on="user_id", how="left")
    sab_df = sab_df.merge(coverage_user, on="user_id", how="left")

    user_meta_cols = [
        "user_id",
        "learner_id_display",
        "learner_id_source",
        "learner_id_confidence",
        "institute_std",
        "institute_mapping_source",
        "institute_mapping_confidence",
        "city",
        "country",
        "class_id_std",
        "subscriber_id",
    ]
    available_meta = [c for c in user_meta_cols if c in df.columns]
    if available_meta:
        user_meta = df[available_meta].copy()
        rename_map = {}
        for col in available_meta:
            if col != "user_id":
                rename_map[col] = col
        user_meta = _user_first_non_null(user_meta)
        sab_df = sab_df.merge(user_meta, on="user_id", how="left", suffixes=("", "_meta"))

    sab_df = apply_insight_engine(sab_df)
    return sab_df


def build_difficulty_outputs(published_kpi: pd.DataFrame) -> pd.DataFrame:
    """Difficulty/DCI remains context only, derived from published KPI data."""
    if published_kpi is None or published_kpi.empty:
        return pd.DataFrame()
    return compute_difficulty_df(published_kpi)


def _user_first_non_null(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["user_id"]
    agg = {}
    for col in df.columns:
        if col in group_cols:
            continue
        agg[col] = _first_non_null
    return df.groupby(group_cols, dropna=False, as_index=False).agg(agg)


def _safe_merge(base: pd.DataFrame, other: pd.DataFrame, on: List[str], cols: Iterable[str]) -> pd.DataFrame:
    if other is None or other.empty:
        return base
    available = [c for c in cols if c in other.columns]
    if not available:
        return base
    merge_cols = list(dict.fromkeys(on + available))
    return base.merge(other[merge_cols].copy(), on=on, how="left")


def build_user_test_readiness_summary(
    proxy_sequence: pd.DataFrame,
    published_kpi: pd.DataFrame,
    readiness_user: pd.DataFrame,
    difficulty_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per learner/test, preserving BLS/ALS proxy traceability."""
    if proxy_sequence is None or proxy_sequence.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    sort_cols = [col for col in ["created_at", "attempt_id_std", "test_taker_id"] if col in proxy_sequence.columns]

    for (user_id, test_id), group in proxy_sequence.groupby(["user_id", "test_id"], dropna=False):
        attempts = group.sort_values(sort_cols, kind="mergesort") if sort_cols else group.copy()
        first = attempts.iloc[0]
        later = attempts.iloc[1:]

        if len(later):
            current = later.iloc[-1]
            potential = later.sort_values(
                [col for col in ["v13_score_pct", "created_at", "attempt_id_std", "test_taker_id"] if col in later.columns],
                ascending=[False, True, True, True][: len([c for c in ["v13_score_pct", "created_at", "attempt_id_std", "test_taker_id"] if c in later.columns])],
                kind="mergesort",
            ).iloc[0]
        else:
            current = pd.Series(dtype="object")
            potential = pd.Series(dtype="object")

        bls_score = first.get("v13_score_pct", np.nan)
        current_score = current.get("v13_score_pct", np.nan) if len(later) else np.nan
        potential_score = potential.get("v13_score_pct", np.nan) if len(later) else np.nan

        completion_mix = "|".join(sorted({str(x) for x in attempts.get("completion_status", pd.Series(dtype="object")).dropna().astype(str).unique()}))
        if not completion_mix:
            completion_mix = np.nan

        rows.append(
            {
                "user_id": user_id,
                "learner_id_display": _first_non_null(attempts["learner_id_display"]) if "learner_id_display" in attempts.columns else _string_or_nan(user_id),
                "test_id": test_id,
                "test_name": _first_non_null(attempts["test_name"]) if "test_name" in attempts.columns else np.nan,
                "assessment_theme_inferred": np.nan,
                "class_id_std": _first_non_null(attempts["class_id_std"]) if "class_id_std" in attempts.columns else np.nan,
                "institute_std": _first_non_null(attempts["institute_std"]) if "institute_std" in attempts.columns else np.nan,
                "subscriber_id": _first_non_null(attempts["subscriber_id"]) if "subscriber_id" in attempts.columns else np.nan,
                "attempt_count": int(len(attempts)),
                "bls_attempt_id": first.get("attempt_id_std", first.get("attempt_id", first.get("test_taker_id"))),
                "bls_created_at": first.get("created_at"),
                "bls_score_raw": first.get("marks"),
                "bls_score_denominator": first.get("accuracy_denominator"),
                "bls_score_denominator_source": first.get("accuracy_denominator_source"),
                "bls_score_pct": bls_score,
                "current_als_attempt_id": current.get("attempt_id_std", current.get("attempt_id", current.get("test_taker_id"))) if len(later) else np.nan,
                "current_als_created_at": current.get("created_at") if len(later) else pd.NaT,
                "current_als_score_raw": current.get("marks") if len(later) else np.nan,
                "current_als_score_denominator": current.get("accuracy_denominator") if len(later) else np.nan,
                "current_als_score_denominator_source": current.get("accuracy_denominator_source") if len(later) else np.nan,
                "current_als_score_pct": current_score,
                "potential_als_attempt_id": potential.get("attempt_id_std", potential.get("attempt_id", potential.get("test_taker_id"))) if len(later) else np.nan,
                "potential_als_created_at": potential.get("created_at") if len(later) else pd.NaT,
                "potential_als_score_raw": potential.get("marks") if len(later) else np.nan,
                "potential_als_score_denominator": potential.get("accuracy_denominator") if len(later) else np.nan,
                "potential_als_score_denominator_source": potential.get("accuracy_denominator_source") if len(later) else np.nan,
                "potential_als_score_pct": potential_score,
                "learning_gain_pct": (current_score - bls_score) if pd.notna(current_score) and pd.notna(bls_score) else np.nan,
                "potential_gain_pct": (potential_score - bls_score) if pd.notna(potential_score) and pd.notna(bls_score) else np.nan,
                "proxy_evidence_band": "high" if len(attempts) >= 3 else ("medium" if len(attempts) == 2 else "low"),
                "completion_status_mix": completion_mix,
                "question_pool_comparability": np.nan,
                "learn_smarter_mapping_status": "test_exercise_proxy_no_topic_id",
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    if readiness_user is not None and not readiness_user.empty:
        readiness_cols = [
            "user_id",
            "robust_SAB_scaled",
            "mean_accuracy",
            "mean_speed",
            "test_count",
            "pass_rate",
            "avg_pass_ratio",
            "coverage_factor",
            "coverage_risk",
            "insight_code",
            "exam_status",
            "readiness_probability_pct",
            "risk_band",
            "stakeholder_insight",
            "coach_feedback",
            "redemption_plan",
        ]
        summary = _safe_merge(summary, readiness_user, ["user_id"], readiness_cols)

    if published_kpi is not None and not published_kpi.empty:
        published_cols = [
            "user_id",
            "test_id",
            "marks",
            "accuracy_total",
            "accuracy_total_safe",
            "accuracy_denominator",
            "accuracy_denominator_source",
            "avg_accuracy_safe",
            "pass_mark",
        ]
        summary = _safe_merge(summary, published_kpi, ["user_id", "test_id"], published_cols)
        rename_map = {}
        for c in ["marks", "accuracy_total", "accuracy_total_safe", "accuracy_denominator", "accuracy_denominator_source", "avg_accuracy_safe", "pass_mark"]:
            if c in summary.columns and c not in ["bls_score_raw", "bls_score_pct"]:
                rename_map[c] = f"published_{c}"
        if rename_map:
            summary = summary.rename(columns=rename_map)

    if difficulty_df is not None and not difficulty_df.empty:
        diff_cols = [
            "test_id",
            "difficulty",
            "difficulty_label",
            "DCI",
            "test_stability",
            "pass_rate",
            "mean_accuracy",
            "takers",
        ]
        summary = _safe_merge(summary, difficulty_df, ["test_id"], diff_cols)
        if {"difficulty_label", "test_stability"}.issubset(summary.columns):
            summary["question_pool_comparability"] = summary.apply(
                lambda r: f"{r['difficulty_label']} | {r['test_stability']}"
                if pd.notna(r.get("difficulty_label")) or pd.notna(r.get("test_stability"))
                else np.nan,
                axis=1,
            )
            summary["difficulty_context_note"] = summary.apply(
                lambda r: f"Difficulty={r.get('difficulty_label', np.nan)}; DCI={r.get('DCI', np.nan)}; stability={r.get('test_stability', np.nan)}",
                axis=1,
            )

    if "current_als_score_pct" in summary.columns:
        cas = summary.groupby("test_id", dropna=False)["current_als_score_pct"].transform("mean")
        summary["cas_proxy_score_pct"] = cas
        summary["cas_proxy_coverage_pct"] = summary.groupby("test_id", dropna=False)["current_als_score_pct"].transform(lambda s: s.notna().mean())
    else:
        summary["cas_proxy_score_pct"] = np.nan
        summary["cas_proxy_coverage_pct"] = np.nan

    return summary


def build_group_readiness_summary(user_test_summary: pd.DataFrame) -> pd.DataFrame:
    """Group-level rollups for available source-backed dimensions only."""
    if user_test_summary is None or user_test_summary.empty:
        return pd.DataFrame()

    group_specs = [
        ("class", "class_id_std"),
        ("institute", "institute_std"),
        ("subscriber", "subscriber_id"),
        ("test", "test_id"),
    ]
    frames: List[pd.DataFrame] = []

    for group_level, group_col in group_specs:
        if group_col not in user_test_summary.columns:
            continue
        grouped = user_test_summary.groupby([group_col, "test_id"], dropna=False).agg(
            learner_count=("user_id", "nunique"),
            repeated_group_count=("attempt_count", lambda s: int((pd.to_numeric(s, errors="coerce") >= 2).sum())),
            mean_bls_score_pct=("bls_score_pct", "mean"),
            mean_current_als_score_pct=("current_als_score_pct", "mean"),
            mean_potential_als_score_pct=("potential_als_score_pct", "mean"),
            mean_learning_gain_pct=("learning_gain_pct", "mean"),
            cas_proxy_score_pct=("cas_proxy_score_pct", "mean"),
            formula_readiness_avg=("readiness_probability_pct", "mean"),
            robust_SAB_avg=("robust_SAB_scaled", "mean"),
            high_evidence_rate=("proxy_evidence_band", lambda s: float((s == "high").mean())),
            medium_evidence_rate=("proxy_evidence_band", lambda s: float((s == "medium").mean())),
            low_evidence_rate=("proxy_evidence_band", lambda s: float((s == "low").mean())),
            DCI=("DCI", "mean") if "DCI" in user_test_summary.columns else ("user_id", "size"),
        ).reset_index()

        if "difficulty_label" in user_test_summary.columns:
            difficulty_label = user_test_summary.groupby([group_col, "test_id"], dropna=False)["difficulty_label"].agg(_mode_or_first).reset_index(name="difficulty_label")
            grouped = grouped.merge(difficulty_label, on=[group_col, "test_id"], how="left")
        else:
            grouped["difficulty_label"] = np.nan

        if "test_stability" in user_test_summary.columns:
            test_stability = user_test_summary.groupby([group_col, "test_id"], dropna=False)["test_stability"].agg(_mode_or_first).reset_index(name="test_stability")
            grouped = grouped.merge(test_stability, on=[group_col, "test_id"], how="left")
        else:
            grouped["test_stability"] = np.nan

        if "difficulty_context_note" in user_test_summary.columns:
            grouped = grouped.merge(
                user_test_summary.groupby([group_col, "test_id"], dropna=False)["difficulty_context_note"].agg(_first_non_null).reset_index(name="difficulty_context_note"),
                on=[group_col, "test_id"],
                how="left",
            )
        else:
            grouped["difficulty_context_note"] = np.nan

        grouped.insert(0, "group_level", group_level)
        grouped.insert(1, "group_value", grouped[group_col].astype(str))
        grouped["group_id"] = grouped.apply(lambda r: f"{r['group_level']}:{r['group_value']}|test:{r['test_id']}", axis=1)
        frames.append(grouped)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


def build_smoke_report(raw_df: pd.DataFrame, artifacts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compact audit table for row counts, proxy coverage, and readiness outputs."""
    standardized_attempts = artifacts.get("standardized_attempts", pd.DataFrame())
    dq_attempts = artifacts.get("dq_attempts", pd.DataFrame())
    published_kpi = artifacts.get("published_kpi", pd.DataFrame())
    proxy_sequence = artifacts.get("proxy_sequence", pd.DataFrame())
    readiness_user = artifacts.get("readiness_user", pd.DataFrame())
    difficulty_df = artifacts.get("difficulty_df", pd.DataFrame())
    user_test_summary = artifacts.get("user_test_summary", pd.DataFrame())
    group_summary = artifacts.get("group_summary", pd.DataFrame())

    def _nunique(df: pd.DataFrame, col: str) -> int:
        if df is None or df.empty or col not in df.columns:
            return 0
        return int(df[col].nunique(dropna=True))

    report = {
        "raw_rows": int(len(raw_df)) if raw_df is not None else 0,
        "raw_users": _nunique(raw_df, "user_id"),
        "raw_tests": _nunique(raw_df, "test_id"),
        "standardized_rows": int(len(standardized_attempts)),
        "standardized_users": _nunique(standardized_attempts, "user_id"),
        "standardized_tests": _nunique(standardized_attempts, "test_id"),
        "dq_rows": int(len(dq_attempts)),
        "dq_users": _nunique(dq_attempts, "user_id"),
        "published_rows": int(len(published_kpi)),
        "published_users": _nunique(published_kpi, "user_id"),
        "proxy_sequence_rows": int(len(proxy_sequence)),
        "proxy_sequence_users": _nunique(proxy_sequence, "user_id"),
        "user_test_summary_rows": int(len(user_test_summary)),
        "user_test_summary_users": _nunique(user_test_summary, "user_id"),
        "group_summary_rows": int(len(group_summary)),
        "BLS_rows": int(user_test_summary["bls_score_pct"].notna().sum()) if "bls_score_pct" in user_test_summary.columns else 0,
        "Current_ALS_rows": int(user_test_summary["current_als_score_pct"].notna().sum()) if "current_als_score_pct" in user_test_summary.columns else 0,
        "Potential_ALS_rows": int(user_test_summary["potential_als_score_pct"].notna().sum()) if "potential_als_score_pct" in user_test_summary.columns else 0,
        "CAS_proxy_rows": int(user_test_summary["cas_proxy_score_pct"].notna().sum()) if "cas_proxy_score_pct" in user_test_summary.columns else 0,
        "readiness_probability_non_null_users": int(readiness_user["readiness_probability_pct"].notna().sum()) if "readiness_probability_pct" in readiness_user.columns else 0,
        "robust_SAB_scaled_non_null_users": int(readiness_user["robust_SAB_scaled"].notna().sum()) if "robust_SAB_scaled" in readiness_user.columns else 0,
        "difficulty_rows": int(len(difficulty_df)),
        "difficulty_DCI_non_null_rows": int(difficulty_df["DCI"].notna().sum()) if "DCI" in difficulty_df.columns else 0,
        "completion_status_counts": dq_attempts["completion_status"].value_counts(dropna=False).to_dict() if "completion_status" in dq_attempts.columns else {},
        "denominator_source_counts": dq_attempts["accuracy_denominator_source"].value_counts(dropna=False).to_dict() if "accuracy_denominator_source" in dq_attempts.columns else {},
    }
    return pd.DataFrame([report])


def build_v13_artifacts(raw_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build the shared v1.3 artifact bundle from raw attempts."""
    standardized_attempts = standardize_v13_fields(raw_df)
    dq_attempts = build_dq_attempts(standardized_attempts)
    published_kpi = build_published_kpi_dataset(dq_attempts)
    proxy_sequence = build_proxy_sequence_dataset(dq_attempts)
    readiness_user = build_existing_readiness_outputs(published_kpi)
    difficulty_df = build_difficulty_outputs(published_kpi)
    user_test_summary = build_user_test_readiness_summary(proxy_sequence, published_kpi, readiness_user, difficulty_df)
    group_summary = build_group_readiness_summary(user_test_summary)

    artifacts = {
        "standardized_attempts": standardized_attempts,
        "dq_attempts": dq_attempts,
        "published_kpi": published_kpi,
        "proxy_sequence": proxy_sequence,
        "readiness_user": readiness_user,
        "difficulty_df": difficulty_df,
        "user_test_summary": user_test_summary,
        "group_summary": group_summary,
    }
    artifacts["smoke_report"] = build_smoke_report(raw_df, artifacts)
    return artifacts
