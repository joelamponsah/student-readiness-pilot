from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Config (toggles from sidebar)
# -----------------------------
@dataclass
class DQConfig:
    completed_only: bool = True
    include_incomplete_if_has_evidence: bool = False  # NEW
    dedupe_best_attempt: bool = True
    strict_pass_mark: bool = True
    show_incomplete: bool = False  # for exploration only; NOT for published KPIs
    require_valid_marks: bool = True
    # no_of_questions is audited as unreliable; keep this off for published
    # metrics unless explicitly exploring raw source consistency.
    require_valid_no_of_questions: bool = False
    require_valid_time: bool = False
    require_question_level_support: bool = False
    require_usable_pass_mark: bool = False
    exclude_time_outliers: bool = False

    # Artifact-first mode
    artifact_dir: str = "data/dq_artifacts"
    export_artifacts: bool = True


# -----------------------------
# Helpers
# -----------------------------
def _to_datetime(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _safe_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()

def _robust_outlier_flag(x: pd.Series, k: float = 3.5) -> pd.Series:
    """
    Robust outlier flag via MAD z-score.
    Flags only (does not drop).
    """
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=x.index)
    z = 0.6745 * (x - med) / mad
    return (np.abs(z) > k).fillna(False)


def _compute_max_marks_candidate(df: pd.DataFrame) -> pd.Series:
    """
    Best-effort max marks candidate:
    - Prefer max_marks_db from COUNT(test_questions), per the full audit.
    - Else fall back to total_questions for legacy exports where it represents
      the same DB question count.
    - Else fall back to question_limit.
    no_of_questions is intentionally excluded because the audit found corrupted
    no_of_questions values.
    This is imperfect; we use it only to flag obvious pass_mark anomalies.
    """
    max_marks = pd.Series(np.nan, index=df.index)

    if "max_marks_db" in df.columns:
        max_marks = pd.to_numeric(df["max_marks_db"], errors="coerce")

    if "total_questions" in df.columns:
        tq = pd.to_numeric(df["total_questions"], errors="coerce")
        max_marks = max_marks.fillna(tq)

    if "question_limit" in df.columns:
        ql = pd.to_numeric(df["question_limit"], errors="coerce")
        max_marks = max_marks.fillna(ql)

    return max_marks


def _compute_pass_mark_effective(pass_mark: pd.Series, max_marks: pd.Series) -> pd.Series:
    """
    Convert pass_mark into an explicit attempt-level threshold.
    Values above max marks are treated as percent-like and flagged as ambiguous.
    """
    pm = pd.to_numeric(pass_mark, errors="coerce")
    mx = pd.to_numeric(max_marks, errors="coerce")
    effective = pd.Series(np.nan, index=pm.index)

    absolute = pm.notna() & mx.notna() & (pm > 0) & (pm <= mx)
    percent_like = pm.notna() & mx.notna() & (pm > mx)

    effective.loc[absolute] = pm.loc[absolute]
    effective.loc[percent_like] = np.ceil((pm.loc[percent_like] / 100.0) * mx.loc[percent_like])
    return effective


def export_dq_artifacts(
    df_raw: pd.DataFrame,
    df_eligible: pd.DataFrame,
    df_exclusions: pd.DataFrame,
    artifact_dir: str,
) -> None:
    os.makedirs(artifact_dir, exist_ok=True)

    # Parquet
    try:
        df_raw.to_parquet(os.path.join(artifact_dir, "fact_attempts_raw.parquet"), index=False)
        df_eligible.to_parquet(os.path.join(artifact_dir, "fact_attempts_eligible.parquet"), index=False)
        df_exclusions.to_parquet(os.path.join(artifact_dir, "dq_exclusions.parquet"), index=False)
    except Exception:
        # Parquet may fail if pyarrow missing; keep CSVs anyway
        pass

    # CSV
    df_raw.to_csv(os.path.join(artifact_dir, "fact_attempts_raw.csv"), index=False)
    df_eligible.to_csv(os.path.join(artifact_dir, "fact_attempts_eligible.csv"), index=False)
    df_exclusions.to_csv(os.path.join(artifact_dir, "dq_exclusions.csv"), index=False)


# -----------------------------
# Main gate
# -----------------------------
def apply_dq_gate(
    df_raw: pd.DataFrame,
    config: Optional[DQConfig] = None,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """
    Returns:
      df_clean      -> eligible dataset for KPIs (deduped + completed + validity)
      dq_report     -> dict with counts, rates, and policy toggles
      df_exclusions -> row-level exclusions with reasons (for auditability)
    """
    if config is None:
        config = DQConfig()

    if df_raw is None or len(df_raw) == 0:
        empty = pd.DataFrame()
        report = {
            "policy": config.__dict__,
            "rows_raw": 0,
            "rows_included": 0,
            "rows_excluded": 0,
            "exclusion_reasons": {},
        }
        return empty, report, empty

    df = df_raw.copy()

    missing_finished_at_column = "finished_at" not in df.columns
    if missing_finished_at_column:
        df["finished_at"] = pd.NaT

    # Coerce types
    df = _to_datetime(df, ["created_at", "updated_at", "finished_at"])
    df = _to_numeric(df, ["marks", "no_of_questions", "time_taken", "attempted_questions", "correct_answers", "pass_mark", "total_questions", "question_limit", "max_marks_db"])

    # Required columns safety
    for c in ["user_id", "test_id", "test_taker_id"]:
        if c not in df.columns:
            df[c] = np.nan

    # -----------------------------
    # Flags (do not drop here)
    # -----------------------------
    df["is_incomplete"] = df["finished_at"].isna()

    # Segmentation data flags
    if "city" in df.columns:
        city = _safe_str_series(df["city"])
        df["city_placeholder"] = (city == "-") | (city == "")
    else:
        df["city_placeholder"] = False

    if "institute_standardized" in df.columns:
        inst = _safe_str_series(df["institute_standardized"])
        df["institute_missing"] = inst.eq("")
    elif "institute" in df.columns:
        inst = _safe_str_series(df["institute"])
        df["institute_missing"] = inst.eq("")
    else:
        df["institute_missing"] = True

    #if "country" in df.columns:
        ctry = _safe_str_series(df["country"])
       # df["country_missing"] = ctry.eq("")
   # else:
      #  df["country_missing"] = True

    # Question-level support flag
    aq = pd.to_numeric(df.get("attempted_questions", np.nan), errors="coerce")
    ca = pd.to_numeric(df.get("correct_answers", np.nan), errors="coerce")
    df["missing_question_level_support"] = (
        aq.isna() | (aq <= 0) | ca.isna() | (ca < 0) | (ca > aq)
    ).fillna(True)

    # time_taken outlier flag (minutes)
    df["time_taken_outlier"] = _robust_outlier_flag(df.get("time_taken", np.nan))

    # no_of_questions suspect (do NOT drop by default; audit says this is a
    # diagnostic field, not a trusted denominator)
    noq = pd.to_numeric(df.get("no_of_questions", np.nan), errors="coerce")
    max_marks_candidate = _compute_max_marks_candidate(df)
    df["no_of_questions_suspect"] = (
        noq.isna()
        | (noq <= 0)
        | (aq.notna() & (aq > 0) & (noq < aq))
        | (max_marks_candidate.notna() & noq.notna() & (noq > max_marks_candidate))
        | (max_marks_candidate.notna() & aq.notna() & (aq > max_marks_candidate))
    ).fillna(True)

    # Pass mark ambiguity
    pm = pd.to_numeric(df.get("pass_mark", np.nan), errors="coerce")
    df["max_marks_effective"] = max_marks_candidate
    df["pass_mark_effective"] = _compute_pass_mark_effective(pm, max_marks_candidate)
    df["pass_mark_ambiguous"] = (
        pm.isna() | (pm <= 0) | (max_marks_candidate.notna() & (pm > max_marks_candidate))
    ).fillna(True)
    df["pass_mark_usable"] = df["pass_mark_effective"].notna() & ~df["pass_mark_ambiguous"]

    marks = pd.to_numeric(df.get("marks", np.nan), errors="coerce")
    tt = pd.to_numeric(df.get("time_taken", np.nan), errors="coerce")

    df["has_marks"] = marks.notna() & (marks >= 0)
    df["has_positive_marks"] = marks.notna() & (marks > 0)
    df["has_time_taken"] = tt.notna() & (tt > 0)
    df["has_attempted_questions"] = aq.notna() & (aq > 0)
    df["has_correct_answers"] = ca.notna() & (ca > 0)

    # Incomplete rows are weak diagnostic evidence only when they show positive
    # learner activity. A present zero mark alone is not enough evidence.
    df["has_partial_activity_evidence"] = (
        df["has_time_taken"]
        | df["has_attempted_questions"]
        | df["has_correct_answers"]
        | df["has_positive_marks"]
    )
    # Completion status is source-aware:
    # - with finished_at present, completion is verified directly
    # - without finished_at, we fall back to activity evidence and label it unknown-but-usable
    if missing_finished_at_column:
        df["completion_source"] = "fallback_activity_evidence"
        df["completion_status"] = np.where(
            df["has_partial_activity_evidence"],
            "unknown_but_usable",
            "incomplete",
        )
    else:
        df["completion_source"] = "finished_at"
        df["completion_status"] = np.where(
            df["finished_at"].notna(),
            "verified_complete",
            np.where(df["has_partial_activity_evidence"], "unknown_but_usable", "incomplete"),
        )

    df["is_incomplete"] = df["completion_status"].eq("incomplete")
    df["incomplete_but_usable"] = df["completion_status"].eq("unknown_but_usable")

    # -----------------------------
    # Eligibility filters (published KPI base)
    # -----------------------------
    if config.completed_only:
        if missing_finished_at_column:
            # Conservative fallback: include verified rows and unknown-but-usable rows so
            # published mode can still run on datasets that never recorded finished_at.
            df["_eligible_completed"] = df["completion_status"].isin(["verified_complete", "unknown_but_usable"])
        else:
            df["_eligible_completed"] = (
                df["completion_status"].eq("verified_complete")
                | (config.include_incomplete_if_has_evidence & df["completion_status"].eq("unknown_but_usable"))
            )
    else:
        df["_eligible_completed"] = True
   
    tt = pd.to_numeric(df.get("time_taken", np.nan), errors="coerce")
    mk = pd.to_numeric(df.get("marks", np.nan), errors="coerce")
    noq = pd.to_numeric(df.get("no_of_questions", np.nan), errors="coerce")

    valid_marks = mk.notna() & (mk >= 0)
    valid_noq = noq.notna() & (noq > 0)
    valid_time = tt.notna() & (tt > 0)

    validity_checks = []
    if config.require_valid_marks:
        validity_checks.append(valid_marks)
    if config.require_valid_no_of_questions:
        validity_checks.append(valid_noq)
    if config.require_valid_time:
        validity_checks.append(valid_time)
    if config.require_question_level_support:
        validity_checks.append(~df["missing_question_level_support"])
    if config.require_usable_pass_mark:
        validity_checks.append(df["pass_mark_usable"])
    if config.exclude_time_outliers:
        validity_checks.append(~df["time_taken_outlier"])

    if validity_checks:
        df["_eligible_validity"] = pd.concat(
            [pd.Series(check, index=df.index).fillna(False) for check in validity_checks],
            axis=1,
        ).all(axis=1)
    else:
        df["_eligible_validity"] = True

    # Separate "speed eligible" flag used by speed KPIs
    df["speed_eligible"] = valid_time.fillna(False)

    df["_eligible_base"] = df["_eligible_completed"] & df["_eligible_validity"]

    # -----------------------------
    # Dedupe: best completed attempt per (user_id, test_id)
    # -----------------------------
    df["is_duplicate_candidate"] = False
    if config.dedupe_best_attempt:
        # Only dedupe within base-eligible rows; everything else is excluded anyway.
        d = df[df["_eligible_base"]].copy()

        # Sort per tie-breakers:
        # 1) highest marks desc
        # 2) latest created_at desc
        # 3) highest test_taker_id desc
        d["_marks_sort"] = pd.to_numeric(d["marks"], errors="coerce").fillna(-1)
        d["_tta_sort"] = pd.to_numeric(d["test_taker_id"], errors="coerce").fillna(-1)
        d["_created_sort"] = pd.to_datetime(d["created_at"], errors="coerce")

        d = d.sort_values(
            by=["user_id", "test_id", "_marks_sort", "_created_sort", "_tta_sort"],
            ascending=[True, True, False, False, False],
            kind="mergesort",  # stable
        )

        # Keep first per group
        keep_idx = d.groupby(["user_id", "test_id"], dropna=False).head(1).index

        # Mark duplicates among base-eligible
        dup_idx = d.index.difference(keep_idx)
        df.loc[dup_idx, "is_duplicate_candidate"] = True

        # Eligible base AFTER dedupe
        df["_eligible_deduped"] = df.index.isin(keep_idx)
        df["_eligible"] = df["_eligible_base"] & df["_eligible_deduped"]
    else:
        df["_eligible"] = df["_eligible_base"]

    # Optional exploration: allow incomplete rows to remain visible (still flagged)
    if config.show_incomplete and not config.completed_only:
        # show_incomplete only affects visibility, not published KPI eligibility.
        pass
    
    # -----------------------------
    # Build exclusions with reasons
    # -----------------------------
    df["exclusion_reason"] = ""

    # priority-ordered reasons (first match wins)
    reasons = [
        ("incomplete_no_evidence", df["completion_status"].eq("incomplete")),
        ("incomplete_attempt", df["completion_status"].eq("unknown_but_usable") & ~pd.Series(df["_eligible_completed"], index=df.index)),
        ("invalid_time_taken", (~valid_time) if config.require_valid_time else False),
        ("invalid_marks", (~valid_marks) if config.require_valid_marks else False),
        ("invalid_no_of_questions", (~valid_noq) if config.require_valid_no_of_questions else False),
        ("missing_question_level_support", df["missing_question_level_support"] if config.require_question_level_support else False),
        ("unusable_pass_mark", (~df["pass_mark_usable"]) if config.require_usable_pass_mark else False),
        ("time_taken_outlier", df["time_taken_outlier"] if config.exclude_time_outliers else False),
        ("duplicate_best_attempt_rule", df["is_duplicate_candidate"] if config.dedupe_best_attempt else False),
    ]

    excluded = ~df["_eligible"]
    for name, mask in reasons:
        m = excluded & pd.Series(mask, index=df.index).fillna(False) & (df["exclusion_reason"] == "")
        df.loc[m, "exclusion_reason"] = name

    # fallback
    df.loc[excluded & (df["exclusion_reason"] == ""), "exclusion_reason"] = "other_or_policy"

    df_exclusions = df.loc[excluded].copy()

    # df_clean = eligible rows only
    df_clean = df.loc[df["_eligible"]].copy()

    # -----------------------------
    # Reporting
    # -----------------------------
    rows_raw = len(df)
    rows_included = len(df_clean)
    rows_excluded = len(df_exclusions)

    reason_counts = df_exclusions["exclusion_reason"].value_counts(dropna=False).to_dict()

    dq_report = {
        "policy": config.__dict__,
        "schema_warnings": {
            "missing_finished_at_column": bool(missing_finished_at_column),
        },
        "completion_source_counts": df["completion_source"].value_counts(dropna=False).to_dict(),
        "completion_status_counts": df["completion_status"].value_counts(dropna=False).to_dict(),
        "rows_raw": int(rows_raw),
        "rows_included": int(rows_included),
        "rows_excluded": int(rows_excluded),
        "included_rate": float(rows_included / rows_raw) if rows_raw else 0.0,
        "exclusion_reasons": {k: int(v) for k, v in reason_counts.items()},
        "flag_rates_on_included": {
            "pass_mark_ambiguous_rate": float(df_clean["pass_mark_ambiguous"].mean()) if rows_included else 0.0,
            "missing_question_level_support_rate": float(df_clean["missing_question_level_support"].mean()) if rows_included else 0.0,
            "time_taken_outlier_rate": float(df_clean["time_taken_outlier"].mean()) if rows_included else 0.0,
            "no_of_questions_suspect_rate": float(df_clean["no_of_questions_suspect"].mean()) if rows_included else 0.0,
            "institute_missing_rate": float(df_clean["institute_missing"].mean()) if rows_included else 0.0,
            "country_missing_rate": float(df_clean["country_missing"].mean()) if rows_included else 0.0,
            "city_placeholder_rate": float(df_clean["city_placeholder"].mean()) if rows_included else 0.0,
        },
        "coverage_rates_on_included": {
            "institute_coverage_rate": float((~df_clean["institute_missing"]).mean()) if rows_included else 0.0,
            "city_coverage_rate": float((~df_clean["city_placeholder"]).mean()) if rows_included else 0.0,
            "country_coverage_rate": float((~df_clean["country_missing"]).mean()) if rows_included else 0.0,
            "question_level_support_rate": float((~df_clean["missing_question_level_support"]).mean()) if rows_included else 0.0,
            "strict_pass_mark_coverage_rate": float(df_clean["pass_mark_usable"].mean()) if rows_included else 0.0,
        },
    }
    dq_report["salvage_stats"] = {
        "incomplete_rate_raw": float(df["is_incomplete"].mean()),
        "incomplete_usable_rate_raw": float(df["incomplete_but_usable"].mean()),
        "incomplete_usable_count_raw": int(df["incomplete_but_usable"].sum()),
    }
    # -----------------------------
    # Artifact-first exports
    # -----------------------------
    if config.export_artifacts:
        export_dq_artifacts(df_raw=df_raw, df_eligible=df_clean, df_exclusions=df_exclusions, artifact_dir=config.artifact_dir)

    # Cleanup internal columns (keep flags)
    for c in ["_eligible_completed", "_eligible_validity", "_eligible_base", "_eligible_deduped", "_eligible",
              "_marks_sort", "_created_sort", "_tta_sort"]:
        if c in df_clean.columns:
            df_clean.drop(columns=[c], inplace=True, errors="ignore")
        if c in df_exclusions.columns:
            df_exclusions.drop(columns=[c], inplace=True, errors="ignore")

    return df_clean, dq_report, df_exclusions
