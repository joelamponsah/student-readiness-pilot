# v1.3 Cleanup Audit

Date: 2026-05-27
Branch: v1.3-test-exercise-readiness

## Cleanup Rule

v1.3 keeps pages that are compatible with the Test / Exercise Readiness scope and the frozen v1.2 DQ foundation. Pages are removed when they are legacy duplicates, non-Streamlit page remnants, under-construction experiments, or active views that calculate metrics from raw data without the v1.2 DQ gate.

## Pages Kept

| File | Reason |
| --- | --- |
| `0_DQ_Monitors.py` | Required v1.3 DQ scenario, exclusion, and coverage audit page. |
| `1_Basic_Metrics.py` | Uses published DQ policy and exposes DQ summary before metrics. |
| `4_Ranking_and_Leaderboard.py` | Uses published DQ policy and states leaderboard eligibility. |
| `7_User_Summary.py` | Maintained user drilldown with diagnostic DQ controls. |
| `Institute_Summary.py` | Maintained institute rollup with published DQ policy and mapping support. |

## Pages Removed

| File | Reason |
| --- | --- |
| `10_Cumulative Results.py` | Under-construction raw-data page; not DQ-gated. |
| `2_Accuracy _Speed_by_User_ and_Tests.py` | Raw-data exploratory view; duplicates safer maintained metrics. |
| `2_Test_and_Topic_Trends` | Non-`.py` remnant; reads raw CSV and references metrics not computed there. |
| `3_Difficulty_DCI_Stability.py` | Raw-data exploratory view; DCI remains available through maintained utilities/pages. |
| `5_Advanced_Metrics.py` | Raw-data exploratory SAB page; functionality is covered in maintained summaries. |
| `6_Exam_Readiness_Model` | Non-`.py` experimental model page with unsupported feature columns. |
| `8_Tests_Overview.py` | Raw-data test overview; not DQ-gated for v1.3 published use. |
| `V2_User_Summary` | Legacy duplicate outside v1.3 scope. |
| `coverage_method` | Legacy duplicate user summary experiment. |
| `old_User_Summary` | Legacy duplicate user summary experiment. |
| `old_User_Summary.py` | Compatibility wrapper no longer needed after cleanup. |
| `old_institute_summary` | Legacy duplicate institute summary. |
| `user_test_v1` | Legacy duplicate user summary experiment. |

## Non-Page Files Not Removed

- `dashboards/streamlit_app_repo.zip` is a packaged artifact, not an active Streamlit page. It should be reviewed separately before deletion.
- `data/old_mapping.csv`, `data/old_verify_df_fixed.csv`, and `data/1mapping.csv` are legacy data artifacts. They are not deleted in this cleanup because data-removal policy needs a separate review.

## Result

The v1.3 dashboard page set is now smaller and aligned to the defensible release surface:

1. DQ Monitors
2. Basic Metrics
3. Ranking and Leaderboard
4. User Summary
5. Institute Summary
