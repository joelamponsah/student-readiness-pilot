# v1.3 Change Log

## 2026-05-29 - Align accuracy denominator with full audit

- Added `test_takers.no_of_questions` to the v1.3 dataset builder as a raw DQ field, not as a trusted denominator.
- Standardized full-test accuracy to use `max_marks_db = COUNT(test_questions WHERE test_id = X)`.
- Added separate attempted-question normalization using `correct_answers / attempted_questions` from the test-results rollup.
- Kept `question_limit` only as a fallback when DB question counts are unavailable.
- Flagged `no_of_questions` as suspect when it is missing, non-positive, less than attempted questions, or greater than `max_marks_db`.
- Updated dashboard readiness helpers and User Summary totals to avoid using raw `no_of_questions` as the default denominator.

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
