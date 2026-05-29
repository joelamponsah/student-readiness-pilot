# Platform Build Guide v1.3

## Scope

v1.3 is the Test / Exercise Readiness release. It keeps the v1.2 DQ baseline and adds conservative proxy fields for BLS, ALS, and CAS.

## Operating Rules

- Published KPIs use the published DQ slice.
- Proxy metrics use the proxy-sequence slice with repeated eligible attempts preserved.
- The loader must not dedupe attempts.
- If `finished_at` is missing, DQ completion must fall back to activity evidence and remain explicitly source-aware.
- Zero-attempt rows must stay visible long enough to be flagged, then neutralized.
- Full-test accuracy must use `max_marks_db = COUNT(test_questions WHERE test_id = X)` when available.
- Attempted-question accuracy must use `correct_answers / attempted_questions` from the test-results rollup.
- `no_of_questions` is not a trusted denominator; keep it as a raw DQ/anomaly field.
- Legacy `total_questions` may be used only where it represents the DB question count from `test_questions`.
- `dashboards/metrics/pages/1_Metrics.py` is the explanatory metrics page; DQ gating remains on `0_DQ_Monitors.py`.
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
- `accuracy_denominator_source` is `max_marks_db` for normal v1.3 exports.
- `no_of_questions_suspect` flags impossible raw question counts instead of excluding full-test accuracy by itself.
