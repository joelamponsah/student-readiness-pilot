# v1.3 Change Log

## 2026-05-27 - Fix loader, DQ fallback, and proxy-sequence separation

- Removed attempt-level dedupe from `load_data_from_disk_or_session()`.
- Made DQ completion logic source-aware when `finished_at` is missing.
- Separated published KPI data from proxy-sequence data.
- Fixed zero-attempt handling and unified the accuracy denominator.
- Moved the Institute Summary proxy block to the correct position after institute selection.

## 2026-05-27 - Convert Streamlit entry point to Home page

- Added `dashboards/metrics/Home.py` as the main entry page.
- Kept `dashboards/metrics/app.py` as a compatibility wrapper.
