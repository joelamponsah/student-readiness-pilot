# v1.3 Scope Audit

Date: 2026-05-27
Branch: v1.3-test-exercise-readiness
Source prompt: v1.3 Codex Execution Prompt - Test / Exercise Readiness - 2026-05-26

## Technical Restatement

v1.3 is a bridge release from the frozen v1.2 DQ baseline into Learn Smarter-aligned test and exercise readiness. It must preserve the v1.2 DQ gate, standardization, exclusion disclosure, and diagnostic/published policy separation while introducing BLS, ALS, and CAS only where existing trusted data can support them.

v1.3 is not the full Learn Smarter model build.

## Grounded Facts

- The working dataset available in the repo is `data/verify_df_fixed.csv`.
- Available attempt fields include `test_taker_id`, `test_id`, `user_id`, `marks`, `no_of_questions`, `created_at`, `updated_at`, `time_taken`, `finished_at`, `institute`, `institute_standardized`, `name`, `description`, `instructions`, `time_limit`, `occurrence`, `total_questions`, `pass_mark`, `attempted_questions`, and `correct_answers`.
- The current DQ policy lives in `dashboards/metrics/utils/dq_policy.py`.
- Published-style views already use DQ-gated attempt data and expose completion, dedupe, validity, pass-mark, and exclusion controls.
- Existing dashboard language still uses older "Learner Readiness" and "Exam Readiness" framing in several places.

## Assumptions

- In v1.3, "test / exercise readiness" means readiness derived from trusted test or exercise attempts after DQ gating.
- `test_id` and `name` are the strongest available test/exercise identifiers.
- `institute_standardized` or `institute_std` may support organization-level rollups, but there is no true class section identifier in the current dataset.
- BLS and ALS are not explicitly recorded as before/after lesson events in the current dataset.

## Open Questions

- Is there a separate lesson/session/content table that can connect attempts to before-lesson and after-lesson timing?
- Is `occurrence` a reliable attempt-order or scheduling indicator across all tests?
- Is there a true class, stream, course, or group identifier outside the current CSV?
- Should v1.3 expose proxy fields in the UI immediately, or first ship them as documented derived artifacts?

## Current Implementation vs v1.3 Scope

| Area | Current state | v1.3 action |
| --- | --- | --- |
| DQ gate | Present and configurable | Preserve as authoritative |
| Published vs diagnostic policy | Present via DQ profiles and toggles | Preserve and document |
| Institute mapping | Uses mapping overrides | Preserve as standardization layer |
| Old user summary | Present as legacy page | Do not update unless explicitly requested |
| Dashboard framing | Still says Learner/Exam Readiness in places | Rename landing/framing to Test / Exercise Readiness |
| BLS | Not explicitly modeled | Add Inferred BLS Proxy only, clearly flagged |
| ALS | Not explicitly modeled | Add Current ALS Proxy and Potential ALS Proxy only, clearly flagged |
| CAS | No true class identifier | Add CAS Proxy based on selected ALS proxy only, clearly flagged |

## Defensible v1.3 BLS / ALS / CAS Mapping

### Inferred BLS Proxy

Current dataset does not contain a verified lesson boundary. The only defensible v1.3 proxy is:

- first DQ-eligible attempt by `user_id` and `test_id`
- marked as `inferred_bls_proxy_score_pct`
- flagged as partial, because it is attempt-order based, not lesson-event based

### Current ALS Proxy

Current dataset does not contain a verified post-lesson marker. The only defensible v1.3 proxy is:

- latest repeated DQ-eligible attempt by `user_id` and `test_id`
- marked as `current_als_proxy_score_pct` when there is more than one attempt
- flagged as partial, because it is repeat-attempt based, not lesson-event based

### Potential ALS Proxy

The learner's best DQ-eligible later attempt within the same grouping may be used as demonstrated potential:

- marked as `potential_als_proxy_score_pct`
- not used as current readiness, because it can overstate stable performance

### CAS Proxy

Current dataset does not contain a true class identifier. The defensible v1.3 proxy is:

- test-level average of the selected ALS proxy
- institute/test-level average of the selected ALS proxy when standardized institute is available
- flagged as cohort/test average, not true class average

### Question-Pool Confidence

Current data does not include question IDs or blueprint equivalence. Because tests may draw random questions, direct score changes can reflect question-set variation as well as learning. v1.3 proxy outputs must include evidence/confidence indicators and must not overclaim learning gain.

## First Implementation Plan

1. Add v1.3 docs: changelog, build guide, diagnostics guide, and this scope audit.
2. Update dashboard landing copy to Test / Exercise Readiness.
3. Add a v1.3 utility for auditable BLS/ALS/CAS proxy fields.
4. Smoke test imports and DQ gating on the real repo dataset.
5. In a later pass, decide whether to surface proxy fields in User Summary, Institute Summary, or a new v1.3 readiness page.
