# Platform Build Guide v1.3

## Scope

v1.3 is the Test / Exercise Readiness release. It keeps the v1.2 DQ baseline and adds conservative proxy fields for BLS, ALS, and CAS.

## Operating Rules

- Published KPIs use the published DQ slice.
- Proxy metrics use the proxy-sequence slice with repeated eligible attempts preserved.
- The loader must not dedupe attempts.
- If `finished_at` is missing, DQ completion must fall back to activity evidence and remain explicitly source-aware.
- Zero-attempt rows must stay visible long enough to be flagged, then neutralized.
- Accuracy must use the canonical denominator order: `max_marks_effective -> total_questions -> no_of_questions`.
- `dashboards/metrics/pages/1_Metrics.py` is the explanatory metrics page; DQ gating remains on `0_DQ_Monitors.py`.
- User Summary should treat inactive zero-attempt rows as non-attempts when showing average accuracy.

## Verification

- Raw loader row count matches the source CSV.
- Published mode is not empty on the repo dataset.
- Diagnostic mode preserves repeated attempts.
- Current ALS Proxy and Potential ALS Proxy appear only where repeated attempts exist.
- `robust_SAB_scaled` stays within 0-100.
- Proxy gain stays within -100 to 100.
