# v1.3 Change Log

## 2026-05-29 - Correct delivered-attempt denominator handling

- Corrected the denominator policy after reconciling the audit with randomized question-pool behavior.
- Separated `question_bank_count` / `max_marks_db` from the delivered scoring denominator.
- Standardized full-test accuracy to use `delivered_denominator`, selected from delivered result evidence, consistent `no_of_questions`, or consistent `question_limit`.
- Added separate attempted-question normalization using `correct_answers / attempted_questions` from the test-results rollup.
- Kept `max_marks_db = COUNT(test_questions WHERE test_id = X)` as question-bank context and a low-confidence last resort only.
- Added denominator flags: `delivered_denominator_source`, `denominator_confidence`, `max_marks_db_is_bank_count`, `no_of_questions_consistent`, `question_limit_consistent`, and `denominator_conflict_flag`.
- Updated dashboard readiness helpers and User Summary totals to avoid diluting marks with full question-bank counts.

## 2026-05-27 - Rename Metrics page and tighten summary interpretation

- Renamed the main explanatory metrics page to `dashboards/metrics/pages/1_Metrics.py`.
- Kept DQ gating on `0_DQ_Monitors.py` and removed DQ summary from the metrics page.
- Clarified the metric math and the published KPI vs proxy-sequence split on the Metrics page.
- Noted that User Summary should exclude inactive zero-attempt rows from displayed average accuracy.
- Added the v1.3 schema boundary across the pages: no source `topic_id`, `subject_id`, or `year_group`; use `class_id` and `created_at` for cohort logic; treat `test_name` subject labels as inferred.

## 2026-05-27 - Fix loader, DQ fallback, and proxy-sequence separation

- Removed attempt-level dedupe from `load_data_from_disk_or_session()`.
- Made DQ completion logic source-aware when `finished_at` is missing.
- Separated published KPI data from proxy-sequence data.
- Fixed zero-attempt handling and unified the accuracy denominator.
- Moved the Institute Summary proxy block to the correct position after institute selection.

## 2026-05-27 - Convert Streamlit entry point to Home page

- Added `dashboards/metrics/Home.py` as the main entry page.
- Kept `dashboards/metrics/app.py` as a compatibility wrapper.
