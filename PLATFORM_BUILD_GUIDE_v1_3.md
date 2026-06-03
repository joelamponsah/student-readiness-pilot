# Platform Build Guide v1.3

## Scope

v1.3 is the Test / Exercise Readiness release. It keeps the v1.2 DQ baseline and adds conservative proxy fields for BLS, ALS, and CAS.

The dashboard now expects `data/raw_attempts.csv` as the primary input. This is the raw attempt-level contract. `data/verify_df_fixed.csv` remains available only as a legacy/reference baseline.

## Operating Rules

- Published KPIs use the published DQ slice.
- Proxy metrics use the proxy-sequence slice with repeated eligible attempts preserved.
- The loader must not dedupe attempts.
- The Streamlit home entry loads and saves `data/raw_attempts.csv` by default.
- `proxy_sequence_attempts.csv` is derived from the raw attempt input and must not be treated as a primary upload target.
- If `finished_at` is missing, DQ completion must fall back to activity evidence and remain explicitly source-aware.
- Zero-attempt rows must stay visible long enough to be flagged, then neutralized.
- Full-test accuracy must use the delivered attempt denominator, not the full randomized question-bank count.
- Denominator priority is delivered result evidence, consistent `no_of_questions`, consistent `question_limit`, then low-confidence legacy fallback.
- Attempted-question accuracy must use `correct_answers / attempted_questions` from the test-results rollup.
- `max_marks_db` / `question_bank_count` from `COUNT(test_questions)` is context for random-pool size, not the normal score denominator.
- `no_of_questions` and `question_limit` are usable only when they reconcile with marks, correct answers, attempted answers, and result evidence.
- Legacy `total_questions` is ambiguous and must not override delivered denominator evidence.
- `dashboards/metrics/pages/1_Metrics.py` is the explanatory metrics page; DQ gating remains on `0_DQ_Monitors.py`.
- The Home page should describe `raw_attempts.csv` as the required dashboard input and `verify_df_fixed.csv` as legacy/reference only.
- User Summary should treat inactive zero-attempt rows as non-attempts when showing average accuracy.
- The current source does not provide `topic_id`, `subject_id`, or `year_group`.
- Use `class_id` with `subscriber_id` and `created_at` for cohort logic.
- `test_name` may be used only as a derived assessment theme or subject-like label when the naming is consistent enough.

## Verification

- Raw loader row count matches the source CSV.
- Published mode is not empty on the repo dataset.
- Diagnostic mode preserves repeated attempts.
- Current ALS Proxy and Potential ALS Proxy appear only where repeated attempts exist.
- `robust_SAB_scaled` stays within 0-100.
- Proxy gain stays within -100 to 100.
- `accuracy_denominator_source` should normally be `delivered_result_questions`, `answer_grade_sum_diagnostic`, `no_of_questions`, or `question_limit`.
- `max_marks_db_is_bank_count` should be true when DB question rows exceed the delivered attempt size.
- `no_of_questions_suspect` flags counts that fail reconciliation; it does not exclude rows by itself unless a policy toggle requires it.
