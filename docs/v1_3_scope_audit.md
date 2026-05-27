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
- Accuracy uses the canonical denominator order.
- Institute Summary proxy metrics are rendered after institute selection.
