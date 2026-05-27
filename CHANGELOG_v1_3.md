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

## 2026-05-27 - Apply proxy policy update and clean legacy pages

- Affected files / modules:
  - `dashboards/metrics/utils/learn_smarter_v13.py`
  - `docs/v1_3_scope_audit.md`
  - `docs/v1_3_cleanup_audit.md`
  - `PLATFORM_BUILD_GUIDE_v1_3.md`
  - `DEV_DIAGNOSTICS_AND_TROUBLESHOOTING_v1_3.md`
  - removed legacy/experimental dashboard page files listed in `docs/v1_3_cleanup_audit.md`
- Reason:
  - Align implementation with the updated execution prompt and `v1.3 BLS ALS CAS Proxy Policy - 2026-05-26`.
  - Reduce v1.3 dashboard surface to maintained, DQ-compatible pages.
- Category:
  - DQ-related: yes, removed active raw-data exploratory pages from the v1.3 page surface.
  - Metric-related: yes, added Current ALS Proxy, Potential ALS Proxy, CAS Proxy, gain proxies, and evidence bands.
  - Framework-related: yes, strengthened proxy-only wording and random question-pool caveats.
  - UI-related: yes, removed obsolete dashboard pages.
  - Documentation-related: yes, added cleanup audit and updated v1.3 docs.
- Backward compatibility / migration note:
  - Core maintained pages remain available.
  - Removed pages were legacy, experimental, non-`.py` remnants, or raw-data views that conflicted with v1.3 DQ discipline.
  - Earlier exploratory alias columns remain in `learn_smarter_v13.py`, but product language should use explicit proxy names.
