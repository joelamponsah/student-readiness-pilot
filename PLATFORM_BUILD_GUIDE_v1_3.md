# Platform Build Guide v1.3

## Purpose

This guide explains how to run the v1.3 Test / Exercise Readiness dashboard safely without breaking the trusted v1.2 baseline.

v1.3 is Learn Smarter-aligned, but it is not the full Learn Smarter model build.

## Project Structure

- `data/verify_df_fixed.csv`: default local dataset used by the dashboard.
- `data/mapping.csv`: base institute mapping.
- `data/mapping_overrides.csv`: confirmed mapping overrides, including school and B2B bank standardization.
- `dashboards/metrics/Home.py`: Streamlit home page.
- `dashboards/metrics/pages/`: dashboard pages.
- `dashboards/metrics/utils/dq_policy.py`: authoritative DQ gate.
- `dashboards/metrics/utils/dq_profiles.py`: named DQ policy profiles.
- `dashboards/metrics/utils/learn_smarter_v13.py`: v1.3 BLS/ALS/CAS proxy helper.
- `docs/`: audit and policy documentation.

## Active Dashboard Pages

The v1.3 page surface is intentionally limited to maintained views:

- `0_DQ_Monitors.py`
- `1_Basic_Metrics.py`
- `4_Ranking_and_Leaderboard.py`
- `7_User_Summary.py`
- `Institute_Summary.py`

## Launch

From the repo root:

```bash
PYTHONPATH=dashboards/metrics .venv/bin/streamlit run dashboards/metrics/Home.py
```

The app loads `data/verify_df_fixed.csv` by default. You may also upload a processed CSV from the Streamlit sidebar.

## Data Expectations

The current v1.3 code expects attempt-level test/exercise data with fields such as:

- `user_id`
- `test_id`
- `test_taker_id`
- `marks`
- `no_of_questions`
- `total_questions`
- `created_at`
- `finished_at`
- `time_taken`
- `attempted_questions`
- `correct_answers`
- `pass_mark`
- `name`
- `institute` or standardized institute fields

## Safe v1.3 Operating Rules

- Use the DQ gate before calculating published readiness outputs.
- Keep published KPIs completed-only unless a page is explicitly diagnostic.
- Keep incomplete-with-evidence separate from published performance.
- Do not use BLS/ALS/CAS proxy fields as final framework metrics.
- Do not overwrite or reinterpret v1.2 historical outputs without documenting the policy version.

## Core Outputs

Implemented in v1.3:

- DQ-gated test/exercise readiness framing.
- Inferred BLS Proxy from first eligible attempt by learner/test.
- Current ALS Proxy from latest repeated eligible attempt by learner/test.
- Potential ALS Proxy from best repeated eligible attempt by learner/test.
- CAS Proxy from selected ALS proxy averages.
- Evidence/confidence indicators for random question-pool comparability limits.

Not implemented in v1.3:

- Full PPS model.
- Full read/listen/watch/discuss integration.
- Full meaningful-usage weighting.
- Full Learn Smarter model build.
- Tutor assistant or recommendation engine logic.
