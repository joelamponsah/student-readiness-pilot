# v1.3 Change Log

## 2026-05-27 - Start v1.3 Test / Exercise Readiness branch

- Affected files / modules:
  - `dashboards/metrics/app.py`
  - `dashboards/metrics/utils/learn_smarter_v13.py`
  - `docs/v1_3_scope_audit.md`
  - `CHANGELOG_v1_3.md`
  - `PLATFORM_BUILD_GUIDE_v1_3.md`
  - `DEV_DIAGNOSTICS_AND_TROUBLESHOOTING_v1_3.md`
- Reason:
  - Begin v1.3 from the frozen v1.2 DQ baseline and align project framing to Test / Exercise Readiness.
- Category:
  - DQ-related: yes, preserved existing DQ gate assumptions.
  - Metric-related: yes, introduced documented BLS/ALS/CAS proxy fields.
  - Framework-related: yes, aligned v1.3 to Learn Smarter without claiming full Learn Smarter coverage.
  - UI-related: yes, updated landing page framing.
  - Documentation-related: yes, created operational docs.
- Backward compatibility / migration note:
  - No existing DQ policy behavior is changed.
  - BLS/ALS/CAS fields are additive and explicitly partial proxies.
  - Existing pages remain available; legacy pages are not updated in this change.
