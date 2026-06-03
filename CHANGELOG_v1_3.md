# v1.3 Change Log

## 2026-06-03 - Switch dashboard contract to raw attempt input

- Introduced `data/raw_attempts.csv` as the required dashboard input contract for v1.3.
- Updated the Streamlit home entry to load and save `data/raw_attempts.csv` by default.
- Kept `data/verify_df_fixed.csv` as a legacy/reference artifact only.
- Documented `proxy_sequence_attempts.csv` as a derived artifact, not a primary app input.
- Added the raw-attempts build notebook and validation reports for the new contract.

## 2026-05-29 - Correct randomized-pool accuracy denominator

- Added `test_takers.no_of_questions` to the v1.3 dataset builder.
- Standardized attempt accuracy to prefer delivered question count from `no_of_questions`, then `question_limit`, with `tests.total_questions` only as a legacy fallback.
- Flagged `total_questions` as suspect when it appears to represent the full randomized question bank instead of the delivered attempt size.
- Updated dashboard readiness helpers and User Summary totals to avoid using full bank size as the default denominator.

## 2026-05-27 - Rename Metrics page and tighten user accuracy display

- Renamed the main metrics page to `dashboards/metrics/pages/1_Metrics.py`.
- Removed the DQ summary block from the metrics page so DQ gating stays on DQ Monitors.
- Added the calculation notes for accuracy, speed, efficiency, and proxy metrics.
- Updated User Summary to exclude inactive zero-attempt rows from the average accuracy display.
- Added the v1.3 schema boundary across the pages: no source `topic_id`, `subject_id`, or `year_group`; use `class_id` and `created_at` for cohort logic; treat `test_name` subject labels as inferred.

## 2026-05-27 - Fix loader, DQ fallback, and proxy-sequence separation

- Removed attempt-level dedupe from `load_data_from_disk_or_session()`.
- Made DQ completion logic source-aware when `finished_at` is missing.
- Kept `verified_complete` distinct from `unknown_but_usable`; published KPI mode stays strict while diagnostic/proxy modes may include unknown-but-usable rows.
- Separated published KPI data from proxy-sequence data.
- Fixed zero-attempt handling and unified the accuracy denominator.
- Moved the Institute Summary proxy block to the correct position after institute selection.

## 2026-05-27 - Replace narrow Drive extraction with broad repo-side extractor

- Added `scripts/v1_3_extraction_rebuild.py` as the repo-side replacement for the older Drive notebook extraction flow.
- The new extractor starts from `test_takers` as the base table, left-joins outward, and writes a reconciliation report against `data/verify_df_fixed.csv` and `data/old_verify_df_fixed.csv`.
- Marked the v1.3 summary artifact provisional until the broad-source extract is regenerated and reconciled.

## 2026-05-27 - Convert Streamlit entry point to Home page

- Added `dashboards/metrics/Home.py` as the main entry page.
- Kept `dashboards/metrics/app.py` as a compatibility wrapper.
