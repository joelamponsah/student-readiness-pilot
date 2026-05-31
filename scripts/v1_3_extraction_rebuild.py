#!/usr/bin/env python3
"""
Rebuild the v1.3 extraction from the broad source scope.

This script is the repo-side replacement for the older Drive notebook flow that
was still producing the narrow 2,473-user extract. It starts from the widest
available attempt table, preserves every attempt row, and only then applies the
v1.3 DQ gate and proxy sequence logic.

Run from repo root after providing database connection details via environment
variables:

  DATABASE_URL
    or
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

Optional env vars:
  DB_SCHEMA              default: public
  DB_USER_TABLE          default: users
  OUTPUT_DIR             default: data/v1_3_dataset_build
  BASELINE_VERIFY_PATH   default: data/verify_df_fixed.csv
  LEGACY_VERIFY_PATH     default: data/old_verify_df_fixed.csv

Outputs:
  - raw_v13_extraction.csv
  - dq_attempts.csv
  - proxy_sequence_attempts.csv
  - v13_user_test_readiness_summary.csv
  - published_kpi_user_test.csv
  - extraction_reconciliation_report.csv
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "dashboards" / "metrics"
if str(METRICS_PATH) not in sys.path:
    sys.path.insert(0, str(METRICS_PATH))

from utils.dq_policy import DQConfig, apply_dq_gate  # noqa: E402
from utils.dq_profiles import learner_diagnostic_config, published_performance_config  # noqa: E402
from utils.learn_smarter_v13 import add_test_exercise_readiness_fields  # noqa: E402


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name, default)
    return value if value not in ("", None) else None


def build_engine():
    database_url = _env("DATABASE_URL")
    if database_url:
        return create_engine(database_url)

    host = _env("DB_HOST")
    port = _env("DB_PORT", "5432")
    name = _env("DB_NAME")
    user = _env("DB_USER")
    password = _env("DB_PASSWORD")
    if not all([host, port, name, user, password]):
        raise RuntimeError(
            "Database connection details are missing. Set DATABASE_URL or DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD."
        )

    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{name}")


def read_sql(engine, table: str, schema: Optional[str] = None) -> pd.DataFrame:
    qualified = f'"{schema}".{table}' if schema else table
    return pd.read_sql_query(text(f"SELECT * FROM {qualified}"), engine)


def read_optional_table(engine, table: str, schema: Optional[str] = None) -> Optional[pd.DataFrame]:
    try:
        return read_sql(engine, table, schema=schema)
    except Exception:
        return None


def _ensure_datetime(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def _ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_raw_attempts(engine, schema: Optional[str] = None, user_table: str = "users") -> pd.DataFrame:
    # Base table: keep every test_takers row.
    test_takers = read_sql(engine, "test_takers", schema=schema)
    test_takers = _ensure_datetime(test_takers, ["created_at", "updated_at", "finished_at", "membership_created_at", "membership_updated_at"])
    test_takers = _ensure_numeric(
        test_takers,
        [
            "marks",
            "no_of_questions",
            "question_limit",
            "attempted_questions",
            "correct_answers",
            "wrong_answers",
            "time_taken",
            "pass_mark",
            "total_questions",
            "total_marks",
            "class_id",
            "subscriber_id",
            "group_id",
        ],
    )

    # Optional user enrichment.
    users = read_optional_table(engine, user_table, schema=schema)
    if users is not None and not users.empty and "user_id" in users.columns:
        user_cols = [c for c in ["user_id", "f_name", "l_name", "username", "country", "institute", "city"] if c in users.columns]
        users = users[user_cols].drop_duplicates(subset=["user_id"])
        rename_map = {
            "f_name": "first_name",
            "l_name": "last_name",
            "username": "learner_id",
        }
        users = users.rename(columns={k: v for k, v in rename_map.items() if k in users.columns})
        test_takers = test_takers.merge(users, on="user_id", how="left", suffixes=("", "_user"))

    # Test metadata.
    tests = read_optional_table(engine, "tests", schema=schema)
    if tests is not None and not tests.empty and "test_id" in tests.columns:
        test_cols = [
            c
            for c in [
                "test_id",
                "name",
                "description",
                "instructions",
                "duration",
                "occurrence",
                "question_limit",
                "pass_mark",
                "created_at",
            ]
            if c in tests.columns
        ]
        tests = tests[test_cols].drop_duplicates(subset=["test_id"])
        test_takers = test_takers.merge(tests, on="test_id", how="left", suffixes=("", "_test"))

    # Response-level evidence.
    test_results = read_optional_table(engine, "test_results", schema=schema)
    test_answers = read_optional_table(engine, "test_answers", schema=schema)
    if test_results is not None and not test_results.empty and "test_taker_id" in test_results.columns:
        results = test_results.copy()
        results = _ensure_datetime(results, ["created_at", "updated_at"])
        results = _ensure_numeric(results, ["test_taker_id", "test_question_id", "chosen_answer_id", "test_answer_id"])

        if test_answers is not None and not test_answers.empty and "answer_id" in test_answers.columns:
            answers = test_answers.copy()
            answers = _ensure_numeric(answers, ["answer_id", "correct"])
            if "correct" in answers.columns:
                answers = answers.rename(columns={"correct": "is_correct"})
            else:
                answers["is_correct"] = np.nan

            join_key = "chosen_answer_id" if "chosen_answer_id" in results.columns else "test_answer_id" if "test_answer_id" in results.columns else None
            if join_key is not None:
                results = results.merge(answers[["answer_id", "is_correct"]], left_on=join_key, right_on="answer_id", how="left")
            else:
                results["is_correct"] = np.nan
        else:
            results["is_correct"] = np.nan

        agg = results.groupby("test_taker_id", dropna=False).agg(
            answer_rows=("test_question_id", "count"),
            attempted_questions=("test_question_id", "count"),
            correct_answers=("is_correct", "sum"),
            wrong_answers=("is_correct", lambda x: int(x.notna().sum() - x.fillna(0).sum())),
            duplicate_attempt_question_rows=("test_question_id", lambda x: int(x.duplicated().sum())),
            answer_grade_sum_diagnostic=("is_correct", "sum"),
        ).reset_index()
        agg["wrong_answers"] = agg["wrong_answers"].fillna(0)
        test_takers = test_takers.merge(agg, on="test_taker_id", how="left", suffixes=("", "_results"))

    # Supporting tables for class-level context and question pool diagnostics.
    test_questions = read_optional_table(engine, "test_questions", schema=schema)
    if test_questions is not None and not test_questions.empty and "test_id" in test_questions.columns:
        tq = test_questions.copy()
        tq = _ensure_numeric(tq, ["test_id", "question_id", "grade"])
        question_counts = tq.groupby("test_id", dropna=False).agg(
            test_question_count=("question_id", "count"),
            max_marks_db=("grade", "sum"),
        ).reset_index()
        test_takers = test_takers.merge(question_counts, on="test_id", how="left", suffixes=("", "_questions"))

    test_classes = read_optional_table(engine, "test_classes", schema=schema)
    if test_classes is not None and not test_classes.empty and "test_id" in test_classes.columns:
        tc = test_classes.copy()
        for col in ["test_id", "class_id", "group_id", "subscriber_id"]:
            if col in tc.columns:
                tc[col] = pd.to_numeric(tc[col], errors="coerce")
        class_ctx = tc.drop_duplicates(subset=[c for c in ["test_id", "class_id", "group_id", "subscriber_id"] if c in tc.columns])
        test_takers = test_takers.merge(class_ctx, on="test_id", how="left", suffixes=("", "_class"))

    class_answers = read_optional_table(engine, "class_answers", schema=schema)
    class_questions = read_optional_table(engine, "class_questions", schema=schema)
    if class_answers is not None and not class_answers.empty:
        test_takers["has_class_answers"] = True
    else:
        test_takers["has_class_answers"] = False
    if class_questions is not None and not class_questions.empty:
        test_takers["has_class_questions"] = True
    else:
        test_takers["has_class_questions"] = False

    # Preserve the old notebook column names where possible.
    if "f_name" not in test_takers.columns and "first_name" in test_takers.columns:
        test_takers["f_name"] = test_takers["first_name"]
    if "l_name" not in test_takers.columns and "last_name" in test_takers.columns:
        test_takers["l_name"] = test_takers["last_name"]
    if "username" not in test_takers.columns and "learner_id" in test_takers.columns:
        test_takers["username"] = test_takers["learner_id"]

    return test_takers


def comparison_report(raw_df: pd.DataFrame, baseline_path: Path, legacy_path: Path) -> dict:
    out = {
        "raw_rows": int(len(raw_df)),
        "raw_users": int(raw_df["user_id"].nunique()) if "user_id" in raw_df.columns else 0,
        "raw_groups": int(raw_df[["user_id", "test_id"]].drop_duplicates().shape[0]) if {"user_id", "test_id"}.issubset(raw_df.columns) else 0,
    }

    for label, path in [("baseline", baseline_path), ("legacy", legacy_path)]:
        if path.exists():
            ref = pd.read_csv(path)
            out[f"{label}_rows"] = int(len(ref))
            out[f"{label}_users"] = int(ref["user_id"].nunique()) if "user_id" in ref.columns else 0
            out[f"{label}_groups"] = int(ref[["user_id", "test_id"]].drop_duplicates().shape[0]) if {"user_id", "test_id"}.issubset(ref.columns) else 0
            if "user_id" in ref.columns:
                ref_users = set(ref["user_id"].dropna().astype(str))
                raw_users = set(raw_df["user_id"].dropna().astype(str)) if "user_id" in raw_df.columns else set()
                out[f"{label}_users_missing_from_raw"] = int(len(ref_users - raw_users))
                out[f"{label}_users_overlap_with_raw"] = int(len(ref_users & raw_users))
        else:
            out[f"{label}_path_missing"] = str(path)

    return out


def main() -> None:
    schema = _env("DB_SCHEMA", "public")
    user_table = _env("DB_USER_TABLE", "users")
    output_dir = Path(_env("OUTPUT_DIR", "data/v1_3_dataset_build"))
    baseline_path = Path(_env("BASELINE_VERIFY_PATH", "data/verify_df_fixed.csv"))
    legacy_path = Path(_env("LEGACY_VERIFY_PATH", "data/old_verify_df_fixed.csv"))

    output_dir.mkdir(parents=True, exist_ok=True)

    engine = build_engine()
    raw_df = build_raw_attempts(engine, schema=schema, user_table=user_table)
    raw_df.to_csv(output_dir / "raw_v13_extraction.csv", index=False)

    # DQ gate: published KPIs stay strict; proxy sequence keeps repeated eligible attempts.
    published_cfg = published_performance_config()
    published_cfg.export_artifacts = False
    published_df, published_report, published_exclusions = apply_dq_gate(raw_df, config=published_cfg)

    proxy_cfg = learner_diagnostic_config()
    proxy_cfg.dedupe_best_attempt = False
    proxy_cfg.export_artifacts = False
    proxy_df, proxy_report, proxy_exclusions = apply_dq_gate(raw_df, config=proxy_cfg)

    published_df.to_csv(output_dir / "published_kpi_user_test.csv", index=False)
    published_exclusions.to_csv(output_dir / "dq_exclusions_published.csv", index=False)
    proxy_df.to_csv(output_dir / "dq_attempts.csv", index=False)
    proxy_exclusions.to_csv(output_dir / "dq_exclusions_proxy.csv", index=False)

    proxy_sequence_df = proxy_df.copy()
    proxy_sequence_df = add_test_exercise_readiness_fields(proxy_sequence_df)
    proxy_sequence_df.to_csv(output_dir / "proxy_sequence_attempts.csv", index=False)

    summary_df = proxy_sequence_df[
        [
            c
            for c in [
                "user_id",
                "test_id",
                "test_name",
                "subscriber_id",
                "first_name",
                "last_name",
                "learner_id",
                "country",
                "institute",
                "city",
                "class_id",
                "group_id",
                "v13_attempt_count_user_test",
                "v13_attempt_rank_user_test",
                "v13_attempt_rank_from_end",
                "inferred_bls_proxy_score_pct",
                "current_als_proxy_score_pct",
                "potential_als_proxy_score_pct",
                "learning_gain_proxy_pct",
                "potential_gain_proxy_pct",
                "proxy_evidence_band",
                "learn_smarter_mapping_status",
                "v13_score_pct",
            ]
            if c in proxy_sequence_df.columns
        ]
    ].drop_duplicates(subset=["user_id", "test_id"])
    summary_df.to_csv(output_dir / "v13_user_test_readiness_summary.csv", index=False)

    report = {
        "raw": {
            "rows": int(len(raw_df)),
            "users": int(raw_df["user_id"].nunique()) if "user_id" in raw_df.columns else 0,
            "groups": int(raw_df[["user_id", "test_id"]].drop_duplicates().shape[0]) if {"user_id", "test_id"}.issubset(raw_df.columns) else 0,
        },
        "published": {
            "rows": int(len(published_df)),
            "users": int(published_df["user_id"].nunique()) if "user_id" in published_df.columns else 0,
            "groups": int(published_df[["user_id", "test_id"]].drop_duplicates().shape[0]) if {"user_id", "test_id"}.issubset(published_df.columns) else 0,
            "completion_status_counts": published_report.get("completion_status_counts", {}),
            "exclusion_reasons": published_report.get("exclusion_reasons", {}),
        },
        "proxy": {
            "rows": int(len(proxy_df)),
            "users": int(proxy_df["user_id"].nunique()) if "user_id" in proxy_df.columns else 0,
            "groups": int(proxy_df[["user_id", "test_id"]].drop_duplicates().shape[0]) if {"user_id", "test_id"}.issubset(proxy_df.columns) else 0,
            "completion_status_counts": proxy_report.get("completion_status_counts", {}),
            "exclusion_reasons": proxy_report.get("exclusion_reasons", {}),
        },
        "summary": {
            "rows": int(len(summary_df)),
            "users": int(summary_df["user_id"].nunique()) if "user_id" in summary_df.columns else 0,
            "groups": int(summary_df[["user_id", "test_id"]].drop_duplicates().shape[0]) if {"user_id", "test_id"}.issubset(summary_df.columns) else 0,
        },
        "reconciliation": comparison_report(raw_df, baseline_path=baseline_path, legacy_path=legacy_path),
    }

    report_path = output_dir / "extraction_reconciliation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    # Human-readable CSV for quick inspection.
    flat = []
    for section, values in report.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat.append({"section": section, "metric": key, "value": value})
        else:
            flat.append({"section": section, "metric": "value", "value": values})
    pd.DataFrame(flat).to_csv(output_dir / "extraction_reconciliation_report.csv", index=False)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
