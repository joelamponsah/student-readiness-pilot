# DQ Eligibility V1

This document defines the conservative eligibility policy used by the v1 dashboard layer.

## Eligibility Principles

- Keep raw rows until they have been annotated.
- Explain exclusions with explicit flags.
- Do not remove repeated attempts in the loader.
- Published summaries may dedupe to the best eligible attempt.
- Diagnostic and proxy-sequence views must preserve repeated eligible attempts.

## Completed Attempt Rule

An attempt is completed when:

- `finished_at` exists in the source and is populated; or
- `finished_at` is missing from the source, but activity evidence is present and the row is marked as fallback/unknown usable.

The fallback must not be described as verified completion.

## Required Metric Support

A completed eligible attempt should have:

- valid marks;
- a full-test denominator, preferably `max_marks_db` from `COUNT(test_questions WHERE test_id = X)`;
- nonzero attempted-question evidence;
- parseable attempt timing where timing metrics are used.

## Completion Flags

Use these fields when available:

- `missing_finished_at_column`
- `completion_source`
- `completion_status`

Recommended values:

- `completion_source = finished_at`
- `completion_source = fallback_activity_evidence`
- `completion_source = missing_finished_at_column`
- `completion_status = verified_complete`
- `completion_status = incomplete`
- `completion_status = unknown_but_usable`

## Published vs Proxy Sequence

Published KPI dataset:

- completed/usable rows only;
- may dedupe best attempt by learner/test;
- used for rankings, institution rollups, and public summaries.

Proxy sequence dataset:

- completed/usable rows only;
- must preserve repeated eligible attempts;
- used for inferred BLS proxy, current ALS proxy, potential ALS proxy, learning gain proxy, and CAS proxy.

## Accuracy Denominator Policy

The full audit found `no_of_questions` can contain impossible values and should not be a trusted score denominator.

Use:

```text
full_test_accuracy = marks / max_marks_db
max_marks_db = COUNT(test_questions WHERE test_id = X)
```

For attempted-question behavior/context, use:

```text
attempted_accuracy = correct_answers / attempted_questions
```

`no_of_questions` is retained for DQ checks only. It is suspect when missing,
non-positive, less than attempted questions, or greater than `max_marks_db`.
