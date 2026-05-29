# v1.3 Scope Audit

v1.3 is a Test / Exercise Readiness bridge release. It does not claim full Learn Smarter behavior.

## Required Guardrails

- Preserve the v1.2 DQ gate.
- Keep proxy BLS/ALS/CAS language explicit and partial.
- Use published KPI data only for published rollups.
- Use proxy-sequence data for repeated-attempt proxy metrics.
- Treat missing `finished_at` as a source-aware fallback, not as proof of incompletion.

## Audit Notes

- Loader dedupe removed from session loading.
- Zero-attempt rows are preserved long enough to flag inactive attempts.
- Full-test accuracy uses `max_marks_db = COUNT(test_questions WHERE test_id = X)`.
- Attempted-question accuracy uses `correct_answers / attempted_questions` from the test-results rollup.
- `no_of_questions` remains a raw DQ/anomaly field and is not a trusted denominator.
- Duplicate `(test_taker_id, test_question_id)` rows must be deduped before answer rollups.
- Institute Summary proxy metrics are rendered after institute selection.
