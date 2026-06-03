# Developer Diagnostics and Troubleshooting v1.3

## Smoke Checks

1. Confirm the loader preserves the raw row count from `data/raw_attempts.csv`.
2. Confirm published DQ mode still returns eligible rows on the repo dataset.
3. Confirm diagnostic mode preserves repeated eligible attempts.
4. Confirm proxy metrics are based on the proxy-sequence slice, not the deduped published slice.
5. Confirm `inactive` zero-attempt rows are flagged.
6. Confirm accuracy values stay in the expected range.
   - `full_test_accuracy = marks / delivered_denominator` should stay between 0 and 1.
   - `attempted_accuracy = correct_answers / attempted_questions` should stay between 0 and 1 where question-result rows exist.
   - `accuracy_denominator_source` should normally be delivered result evidence, `no_of_questions`, or `question_limit`.
   - `max_marks_db_is_bank_count` should be true when the DB count is the full randomized pool.
7. Confirm `robust_SAB_scaled` stays within 0-100.
8. Confirm proxy gain stays within -100 to 100.
9. Confirm Institute Summary loads without a `NameError`.
10. Confirm the `Metrics` page explains the calculation rules without showing the DQ summary block.
11. Confirm User Summary average accuracy excludes inactive zero-attempt rows.
12. Confirm the pages state that `topic_id`, `subject_id`, and `year_group` are not available in the current source.
13. Confirm `class_id` is used for cohort logic and `test_name`-derived subject labels are clearly marked as inferred.
14. Confirm `no_of_questions` is used only when it reconciles with delivered evidence.
15. Confirm duplicate `(test_taker_id, test_question_id)` rows are deduped in the notebook answer rollup.
16. Confirm attempts without test-results rows are visible in the smoke report as missing test-results coverage.
17. Confirm the dashboard home text treats `raw_attempts.csv` as the required input and `verify_df_fixed.csv` as legacy/reference only.
18. Confirm `proxy_sequence_attempts.csv` is treated as a derived artifact, not a primary upload target.

## Manual Proxy Verification

- Inferred BLS Proxy should appear on first eligible attempts.
- Current ALS Proxy should appear on later repeated eligible attempts.
- Potential ALS Proxy should reflect the best later repeated eligible attempt.
- CAS Proxy should be treated as a conservative test/institute average, not a true class average.
- Proxy percentages should reconcile to `marks / delivered_denominator * 100`.
- Attempted accuracy should be used as behavior/context, not as the BLS/ALS/CAS proxy score.
