# DQ Eligibility V1

Status: working v1 policy  
Scope: test-based Learner Readiness foundation inside Learn Smarter

## Purpose

This policy defines which attempt rows may be used for published dashboard metrics in the current test-based readiness layer. It exists because the source data has known integrity risks: missing completion timestamps, duplicate attempts, unreliable question counts, pass-mark ambiguity, question-result coverage gaps, and messy segmentation values.

The default rule is conservative: published performance, rankings, institute rollups, and pass metrics must use trusted eligible attempts. Partial or ambiguous evidence may be useful for diagnosis, but it must not silently enter published KPI bases.

## Attempt Classes

### Completed Eligible Attempt

An attempt is completed eligible when:

- `finished_at` is present.
- `marks` is present and non-negative.
- `no_of_questions` is present and greater than zero.
- Required identifiers such as `user_id` and `test_id` are present enough for the view being computed.
- The row survives the active dedupe policy when dedupe is required.

Completed eligible attempts are the default base for published performance and ranking metrics.

### Incomplete With Evidence

An incomplete attempt has `finished_at` missing.

It may be treated as incomplete-with-evidence only when it has positive learner activity evidence, such as:

- `time_taken > 0`
- `attempted_questions > 0`
- `correct_answers > 0`
- `marks > 0`
- future verified `test_results` activity for the attempt

A present zero mark alone is not enough evidence. Incomplete-with-evidence rows are diagnostic evidence, not default published performance evidence.

### Incomplete Without Evidence

An incomplete attempt without positive learner activity evidence is excluded from eligible KPI bases. It should appear only in DQ monitoring and exclusion reporting.

## Metric Policy

| Use case | Include incomplete-with-evidence? | Dedupe? | Pass-mark policy | Disclosure |
| --- | --- | --- | --- | --- |
| Published performance KPIs | No | Yes | Exclude or flag ambiguous pass marks | Required |
| Rankings / leaderboards | No | Yes | Strict pass marks | Required |
| Institute / geography rollups | No | Yes | Strict pass marks | Required |
| Learner diagnostic drilldown | Allowed, clearly labeled | Optional | Strict for pass KPIs | Required |
| DQ monitor | Show separately | Yes for eligible base | Report ambiguity | Required |
| Future broader engagement model | Possible weak signal | TBD | Not pass-based | TBD |

## Current Named Policy Modes

### Published Performance

Use for performance dashboards, rankings, and institute-level views.

- `completed_only=True`
- `include_incomplete_if_has_evidence=False`
- `dedupe_best_attempt=True`
- `strict_pass_mark=True`

### Learner Diagnostic

Use only for learner drilldown where partial evidence may help explain behavior.

- `completed_only=True`
- `include_incomplete_if_has_evidence=True`
- `dedupe_best_attempt=False` unless the page explicitly separates raw from eligible attempts
- `strict_pass_mark=True`

This mode must label eligible attempts, raw attempts, and partial evidence separately.

### DQ Monitor

Use for audit and monitoring.

- `completed_only=True`
- `include_incomplete_if_has_evidence=False`
- `dedupe_best_attempt=True`
- `strict_pass_mark=True`
- expose exclusions, salvage counts, and coverage rates

## Required Disclosures

Any published view that aggregates beyond a single learner must disclose:

- raw rows versus included rows
- exclusion reason counts
- duplicate handling
- pass-mark usable coverage
- question-level support coverage
- institute / city / country coverage where segmentation is used

## Open Items

- Freeze whether learner drilldown should dedupe by default or show all eligible attempts.
- Add verified `test_results` support once those tables are available in the repo pipeline.
- Decide whether incomplete-with-evidence can become a weak engagement signal in a later Learn Smarter model.
- Replace the boolean `include_incomplete_if_has_evidence` with a clearer completion policy enum in a future refactor.
