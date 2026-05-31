# v1.3 Extraction Scope Audit

Status: provisional

## What Was Compared

1. GitHub `data/verify_df_fixed.csv`
2. GitHub `data/old_verify_df_fixed.csv`
3. Generated `dq_attempts.csv`

## Current Counts

| File | Rows | Unique users | User-test groups |
| --- | ---: | ---: | ---: |
| `data/verify_df_fixed.csv` | 30,795 | 9,385 | 15,357 |
| `data/old_verify_df_fixed.csv` | 12,354 | 2,473 | 5,581 |
| `dq_attempts.csv` | 12,533 | 2,473 | 5,582 |

## Interpretation

- The v1.3 raw extract is not reconciling to the full GitHub baseline.
- It matches the smaller legacy scope almost exactly on unique users.
- `dq_attempts.csv` has 2,473 users, the same user count as `old_verify_df_fixed.csv`, and only 5 extra users beyond that legacy file.
- The full baseline has 9,385 users, so the raw v1.3 extract is missing 6,917 users relative to the main GitHub file.

## Notebook Source Scope

The Drive notebook builds its dataset from exported tables centered on `test_takers`, `test_results`, `test_answers`, `tests`, `test_questions`, `test_classes`, `class_answers`, and `class_questions`.
The current evidence shows that the export feeding v1.3 is still scoped like the old dataset family, not the full baseline family represented by `data/verify_df_fixed.csv`.

## Practical Conclusion

- The 2,230-user summary cannot be treated as final coverage until the extraction scope is corrected or explicitly justified.
- Dashboard fixes alone do not solve this. The raw source scope has to be widened first.

## Required Next Check

- Rebuild the extraction from the broadest attempt-level source available.
- Verify that the new raw extract starts much closer to 9,385 users before DQ filtering.
- Only then re-evaluate proxy coverage and summary coverage.

## Replacement Extractor

- Use [`scripts/v1_3_extraction_rebuild.py`](/home/jorleansco/student-readiness-pilot/scripts/v1_3_extraction_rebuild.py) as the repo-side source of truth for the broader extraction flow.
- The older Drive notebook remains useful for reference, but it is not the authoritative extraction path anymore.
