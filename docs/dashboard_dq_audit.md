# Dashboard DQ Audit

Status: working audit after DQ eligibility v1  
Scope: `dashboards/metrics/pages`

## Summary

The DQ-critical pages now mostly follow the updated bootstrap direction: published rollups use completed-only, deduped, strict-pass-mark policy; learner drilldown is the only page allowed to include incomplete-with-evidence as diagnostic evidence.

The remaining risk is consistency. Several older/non-DQ pages still exist and may compute directly from raw data or legacy metric helpers. They should either be aligned, hidden, or explicitly labeled as exploratory.

## Page Matrix

| Page | Current DQ status | Incomplete-with-evidence | Dedupe | Pass-mark handling | DQ disclosure | Risk |
| --- | --- | --- | --- | --- | --- | --- |
| `0_DQ_Monitors.py` | Uses `apply_dq_gate` | Excluded | Yes | Strict | Yes | Low |
| `1_Basic_Metrics.py` | Uses `apply_dq_gate` | Excluded | Yes | Strict | Yes | Low |
| `4_Ranking_and_Leaderboard.py` | Uses `apply_dq_gate` | Excluded | Yes | Strict | Yes | Low |
| `7_User_Summary.py` | Uses `apply_dq_gate` | Allowed by sidebar/fallback for diagnostic view | No by fallback | Strict for pass KPIs | Partial | Medium |
| `Institute_Summary.py` | Uses `apply_dq_gate` | Excluded | Yes | Strict | Yes | Low |
| `2_Accuracy _Speed_by_User_ and_Tests.py` | Not yet audited/aligned | Unknown | Unknown | Unknown | Unknown | High |
| `3_Difficulty_DCI_Stability.py` | Not yet audited/aligned | Unknown | Unknown | Unknown | Unknown | High |
| `5_Advanced_Metrics.py` | Not yet audited/aligned | Unknown | Unknown | Unknown | Unknown | High |
| `8_Tests_Overview.py` | Not yet audited/aligned | Unknown | Unknown | Unknown | Unknown | High |
| `10_Cumulative Results.py` | Not yet audited/aligned | Unknown | Unknown | Unknown | Unknown | High |

## Findings

1. The DQ monitor, basic metrics, ranking, and institute summary pages use the conservative published-performance policy.
2. The user summary page intentionally supports a diagnostic mode where incomplete-with-evidence can be included, but this needs stronger visible labeling around partial evidence.
3. The legacy old user summary page has been removed.
4. The remaining old/non-DQ pages should not be treated as publication-ready until audited.

## Recommended Next Changes

1. Audit the remaining high-risk pages and either align them to `apply_dq_gate` or mark them exploratory.
2. Update dashboard language to say "test-based readiness foundation" rather than implying the full Learn Smarter model is already implemented.
3. Replace deprecated Streamlit `use_container_width` calls with `width="stretch"` before that API is removed.

## Completed In This Pass

1. Added named DQ policy profiles in `utils/dq_profiles.py`.
2. Updated DQ-critical pages to use named profiles instead of hand-written `DQConfig(...)` blocks.
3. Added a visible diagnostic caption when learner drilldown includes incomplete-with-evidence.
4. Tightened incomplete-with-evidence so a zero mark alone is not enough evidence.
