# v1.3 Test / Exercise Readiness Review Plan

## Purpose
This document captures the recommended v1.3 direction before implementation. v1.3 remains a bridge release, not the full Learn Smarter model.
It keeps the current readiness formulas visible and compares them to the new BLS / ALS / CAS proxy layer.

## Technical Opinion
The two-track model is sound: preserve the current formula readiness path and add a separate Learn Smarter proxy path.
Difficulty and DCI should remain context, not a hard correction.
Metrics should become the analytical reference page, but DQ Monitors must remain the only place that decides inclusion.
Published KPI mode should stay strict on `verified_complete`; `unknown_but_usable` is a real label, but only diagnostic/proxy views may include it.

## What To Build Now
Publish one stable v1.3 summary artifact at user plus test level.
Publish one grouped summary artifact at topic or test level, then add class, year group, and institute rollups only when source keys exist.
Keep row-level proxy outputs for traceability, and keep summary-level proxy scores for dashboard readability.
Keep CAS as a proxy aggregation only.

## Defer To V2
True Learn Smarter modeling.
Hard difficulty adjustment or calibration of proxy scores.
A unified readiness score that merges formula readiness and proxy readiness.
Read, listen, watch, discuss, or points weighting.

## Minimum Data Contract
Required now: user_id, test_id, created_at, marks, attempted_questions, correct_answers, time_taken, duration, pass_mark, test_name, institute, class_id, and subscriber_id.
Strongly useful: finished_at, total_questions, no_of_questions, attempt_id, and test_taker_id.
Not currently available in the source: topic_id, subject_id, topic_name, subject_name, and year_group.
V1.3 may use test_name as a derived assessment theme or subject-like label when naming is consistent enough, but it must stay explicit that this is inferred rather than source-backed.
V2 only: question_id, question-topic mapping, question difficulty metadata, and blueprint or pool metadata.

## Recommended Sequence
1. Lock the canonical source fields for attempts and learner identity.
2. Confirm the grouping keys that really exist in the dataset.
3. Produce the v1.3 summary artifact, then the grouped summary artifact.
4. Add difficulty and DCI as context columns only.
5. Wire the Metrics page to explain formulas, proxy outputs, and signal comparison.

## Drive and Local Storage
This plan has been saved to Google Drive and mirrored locally as a markdown file for repo-side review.
