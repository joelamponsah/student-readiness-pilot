# Developer Diagnostics and Troubleshooting v1.3

## Purpose

This guide gives developers a practical checklist for validating v1.3 Test / Exercise Readiness behavior.

## Validate DQ Gating

Run a basic DQ smoke test:

```bash
PYTHONPATH=dashboards/metrics python3 - <<'PY'
from utils.metrics import load_data_from_disk_or_session
from utils.dq_policy import apply_dq_gate
from utils.dq_profiles import published_performance_config

df = load_data_from_disk_or_session("data/verify_df_fixed.csv")
eligible, report, exclusions = apply_dq_gate(df, published_performance_config())
print("raw_rows", report["rows_raw"])
print("included_rows", report["rows_included"])
print("excluded_rows", report["rows_excluded"])
print("top_exclusions", report.get("exclusion_reasons", {}))
PY
```

Expected behavior:

- Raw rows are greater than included rows when DQ exclusions apply.
- Published policy does not include incomplete-with-evidence by default.
- Exclusions include clear row-level reasons.

## Spot Duplicate Attempt Problems

Check whether dedupe is active in the selected policy:

- `dedupe_best_attempt=True` for published performance.
- Deduping should happen before rankings and published learner summaries.
- User diagnostic views may show broader evidence only when labeled.

If rankings or institute rollups look inflated, inspect:

- repeated `user_id` + `test_id`
- repeated `test_taker_id`
- repeated `created_at`

## Inspect Pass-Mark Ambiguity

Pass-mark ambiguity is expected in this dataset. Developers should check:

- `pass_mark_ambiguous`
- `pass_mark_effective`
- `pass_mark_usable`

Published pass KPIs should use strict pass-mark handling or disclose ambiguity.

## Test Whether Readiness Uses Trusted Inputs

For published outputs, verify that:

- the page calls `apply_dq_gate`
- metric functions receive the eligible dataframe, not raw data
- incomplete attempts are not silently mixed into performance KPIs
- pass-mark ambiguity is excluded or flagged where pass metrics are shown

For `Basic Metrics`, confirm the sidebar view mode is correct:

- `Published` should be the default for the release baseline.
- `Diagnostic preview` should only be used when you want to inspect repeated-attempt proxy behavior.

## Verify BLS / ALS / CAS Proxy Fields

Use the v1.3 helper:

```bash
PYTHONPATH=dashboards/metrics python3 - <<'PY'
from utils.metrics import load_data_from_disk_or_session
from utils.dq_policy import apply_dq_gate
from utils.dq_profiles import published_performance_config
from utils.learn_smarter_v13 import add_test_exercise_readiness_fields

df = load_data_from_disk_or_session("data/verify_df_fixed.csv")
eligible, report, exclusions = apply_dq_gate(df, published_performance_config())
mapped = add_test_exercise_readiness_fields(eligible)
print(mapped[[
    "user_id",
    "test_id",
    "v13_score_pct",
    "inferred_bls_proxy_score_pct",
    "current_als_proxy_score_pct",
    "potential_als_proxy_score_pct",
    "cas_proxy_test_avg_score_pct",
    "proxy_evidence_band",
    "learn_smarter_mapping_status",
]].head())
PY
```

Expected behavior:

- Inferred BLS Proxy appears only on the first eligible learner/test attempt.
- Current ALS Proxy appears only on the latest later eligible learner/test attempt.
- Potential ALS Proxy reflects the best later eligible learner/test attempt.
- CAS Proxy is based on the selected ALS proxy and is not a true class average.
- Evidence band is low when repeated attempts or question coverage are weak.
- Mapping status remains partial.

## Common Failure Modes

| Symptom | Likely cause | Check |
| --- | --- | --- |
| Published counts look too high | Raw data used instead of DQ eligible data | Trace page input dataframe |
| Incomplete attempts appear in rankings | Wrong DQ profile | Check `completed_only` and scenario |
| Pass rate changes unexpectedly | Pass-mark ambiguity handling changed | Inspect `strict_pass_mark` |
| BLS/ALS appears complete | Proxy labels are being overclaimed | Check mapping status and docs |
| CAS is interpreted as class average | No class identifier exists | Use cohort/test average wording |
| Learning gain looks too strong | Random question-pool variation may be driving score change | Check `question_pool_comparability` and `proxy_evidence_band` |
| Institution search misses known school/bank | Override mapping not loaded | Check `data/mapping_overrides.csv` |

## Dashboard Disagreements

When two pages disagree:

1. Confirm both pages load the same dataset.
2. Confirm both pages apply the same DQ profile.
3. Confirm whether one page is diagnostic and the other is published.
4. Confirm whether institute standardization is applied before filtering.
5. Compare row counts before and after DQ gating.

## Trace Metrics From Source to Output

The normal published path should be:

`raw CSV -> apply_dq_gate -> eligible attempts -> metric helper -> dashboard output`

Any page that skips the DQ gate should be treated as legacy or exploratory until reviewed.
