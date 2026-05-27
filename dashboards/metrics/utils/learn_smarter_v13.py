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

    out["is_inferred_bls_proxy"] = out["v13_attempt_sequence_user_test"].eq(1)
    out["is_current_als_proxy"] = (
        out["v13_attempt_count_user_test"].gt(1)
        & out["v13_attempt_sequence_user_test"].eq(out["v13_attempt_count_user_test"])
    )
    later_attempt = out["v13_attempt_sequence_user_test"].gt(1)
    later_scores = out["v13_score_pct"].where(later_attempt)
    out["potential_als_proxy_score_pct"] = later_scores.groupby(
        [out["user_id"], out["test_id"]]
    ).transform("max")
    out["is_potential_als_proxy"] = later_attempt & out["v13_score_pct"].eq(
        out["potential_als_proxy_score_pct"]
    )

    out["inferred_bls_proxy_score_pct"] = out["v13_score_pct"].where(out["is_inferred_bls_proxy"])
    out["current_als_proxy_score_pct"] = out["v13_score_pct"].where(out["is_current_als_proxy"])
    out["learning_gain_proxy_pct"] = (
        out["current_als_proxy_score_pct"]
        - out.groupby(group_cols)["inferred_bls_proxy_score_pct"].transform("max")
    )
    out["potential_gain_proxy_pct"] = (
        out["potential_als_proxy_score_pct"]
        - out.groupby(group_cols)["inferred_bls_proxy_score_pct"].transform("max")
    )

    out["cas_proxy_basis"] = "current_als_proxy"
    current_als_scores = out["current_als_proxy_score_pct"].where(out["is_current_als_proxy"])
    out["cas_proxy_test_avg_score_pct"] = current_als_scores.groupby(out["test_id"]).transform("mean")

    institute_col = next(
        (
            column
            for column in ["institute_std", "institute_standardized", "institute"]
            if column in out.columns
        ),
        None,
    )
    if institute_col:
        out["cas_proxy_institute_test_avg_score_pct"] = current_als_scores.groupby(
            [out[institute_col], out["test_id"]], dropna=False
        ).transform("mean")
    else:
        out["cas_proxy_institute_test_avg_score_pct"] = np.nan

    question_count = denominator.fillna(0)
    has_question_support = ~out.get(
        "missing_question_level_support", pd.Series(True, index=out.index)
    ).fillna(True)
    has_pass_support = ~out.get("pass_mark_ambiguous", pd.Series(True, index=out.index)).fillna(True)
    repeated = out["v13_attempt_count_user_test"].ge(2)

    out["question_pool_comparability"] = "unknown_without_question_ids"
    out["proxy_evidence_band"] = np.select(
        [
            repeated & question_count.ge(30) & has_question_support & has_pass_support,
            repeated & question_count.ge(10),
        ],
        ["high", "medium"],
        default="low",
    )
    out["proxy_evidence_note"] = np.select(
        [
            ~repeated,
            question_count.lt(10),
            ~has_question_support,
            ~has_pass_support,
        ],
        [
            "Only one eligible attempt in grouping; ALS and gain proxies are unavailable.",
            "Question count is small; proxy interpretation is weak.",
            "Question-level support is missing or weak.",
            "Pass-mark support is ambiguous.",
        ],
        default=(
            "Proxy is based on repeated eligible attempts, but question-pool comparability "
            "remains unknown without question IDs or blueprint equivalence."
        ),
    )

    out["learn_smarter_mapping_status"] = "partial_test_exercise_proxy"
    out["learn_smarter_mapping_note"] = (
        "Inferred BLS Proxy, Current ALS Proxy, Potential ALS Proxy, and CAS Proxy use "
        "learner-test attempt order. They are not true BLS/ALS/CAS because historical "
        "attempts lack explicit assessment-phase labels, true class IDs, and question-pool "
        "comparability metadata."
    )

    # Backward-compatible aliases for earlier v1.3 exploratory code. Product
    # labels should use the explicit proxy names above.
    out["bls_proxy_score_pct"] = out["inferred_bls_proxy_score_pct"]
    out["als_proxy_score_pct"] = out["current_als_proxy_score_pct"]
    out["cas_test_avg_score_pct"] = out["cas_proxy_test_avg_score_pct"]
    out["cas_institute_test_avg_score_pct"] = out["cas_proxy_institute_test_avg_score_pct"]

    return out
