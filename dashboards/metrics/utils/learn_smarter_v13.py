from __future__ import annotations

import numpy as np
import pandas as pd


def _score_denominator(df: pd.DataFrame) -> pd.Series:
    denominator = pd.Series(np.nan, index=df.index, dtype="float64")
    for column in ["max_marks_effective", "total_questions", "no_of_questions"]:
        if column in df.columns:
            denominator = denominator.fillna(pd.to_numeric(df[column], errors="coerce"))
    denominator = denominator.mask(denominator <= 0)
    return denominator


def add_test_exercise_readiness_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add v1.3 Learn Smarter-aligned test/exercise proxy fields.

    These fields are intentionally conservative. The current dataset does not
    contain verified lesson boundaries or a true class identifier, so BLS, ALS,
    and CAS are marked as partial proxies derived from eligible test attempts.
    """
    out = df.copy()
    if out.empty:
        return out

    required = {"user_id", "test_id", "marks"}
    missing = required.difference(out.columns)
    if missing:
        raise KeyError(f"Missing required v1.3 readiness columns: {sorted(missing)}")

    if "created_at" in out.columns:
        out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
    else:
        out["created_at"] = pd.NaT

    denominator = _score_denominator(out)
    marks = pd.to_numeric(out["marks"], errors="coerce")
    out["v13_score_pct"] = (marks / denominator).replace([np.inf, -np.inf], np.nan) * 100
    out["v13_score_pct"] = out["v13_score_pct"].clip(lower=0, upper=100)

    sort_cols = ["user_id", "test_id", "created_at"]
    if "test_taker_id" in out.columns:
        sort_cols.append("test_taker_id")
    ordered = out.sort_values(sort_cols, kind="mergesort")

    group_cols = ["user_id", "test_id"]
    out.loc[ordered.index, "v13_attempt_sequence_user_test"] = (
        ordered.groupby(group_cols).cumcount() + 1
    )
    out.loc[ordered.index, "v13_attempt_count_user_test"] = ordered.groupby(group_cols)[
        "test_id"
    ].transform("size")

    out["is_bls_proxy"] = out["v13_attempt_sequence_user_test"].eq(1)
    out["is_als_proxy"] = (
        out["v13_attempt_count_user_test"].gt(1)
        & out["v13_attempt_sequence_user_test"].eq(out["v13_attempt_count_user_test"])
    )
    out["bls_proxy_score_pct"] = out["v13_score_pct"].where(out["is_bls_proxy"])
    out["als_proxy_score_pct"] = out["v13_score_pct"].where(out["is_als_proxy"])

    out["cas_test_avg_score_pct"] = out.groupby("test_id")["v13_score_pct"].transform("mean")

    institute_col = next(
        (
            column
            for column in ["institute_std", "institute_standardized", "institute"]
            if column in out.columns
        ),
        None,
    )
    if institute_col:
        out["cas_institute_test_avg_score_pct"] = out.groupby(
            [institute_col, "test_id"], dropna=False
        )["v13_score_pct"].transform("mean")
    else:
        out["cas_institute_test_avg_score_pct"] = np.nan

    out["learn_smarter_mapping_status"] = "partial_test_exercise_proxy"
    out["learn_smarter_mapping_note"] = (
        "BLS/ALS use learner-test attempt order; CAS uses test/cohort averages. "
        "No verified lesson boundary or true class identifier is present."
    )

    return out
