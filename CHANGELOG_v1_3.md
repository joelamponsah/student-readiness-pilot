# v1.3 Change Log

## 2026-05-27 - Convert Streamlit entry point to Home page

- Affected files / modules:
  - `dashboards/metrics/Home.py`
  - `dashboards/metrics/app.py`
  - `.devcontainer/devcontainer.json`
  - `PLATFORM_BUILD_GUIDE_v1_3.md`
- Reason:
  - Replace the generic landing stub with a real Home page that explains the project scope and loads the dataset used by the dashboard.
  - Keep `app.py` as a compatibility wrapper so existing launch commands still work.
- Category:
  - DQ-related: yes, the loaded dataset remains shared across the DQ-gated page set.
  - Metric-related: no new metrics, just the home page wiring.
  - Framework-related: yes, the app now uses a proper Home entry point.
  - UI-related: yes, the first screen now reads as a home page instead of a generic hub.
  - Documentation-related: yes, launch references are updated to point to `Home.py`.
- Backward compatibility / migration note:
  - Existing `streamlit run dashboards/metrics/app.py` launch commands still work.
  - The new `Home.py` is the preferred entry point for the v1.3 dashboard.

## 2026-05-27 - Align maintained summary pages with v1.3 proxy metrics

- Affected files / modules:
  - `dashboards/metrics/pages/7_User_Summary.py`
  - `dashboards/metrics/pages/Institute_Summary.py`
- Reason:
  - Bring the maintained learner and institute summary pages into line with the v1.3 Learn Smarter proxy model.
  - Expose the proxy summary, evidence band, and CAS Proxy caveat directly on the two stakeholder-facing summary pages.
- Category:
  - DQ-related: yes, summaries now show proxy outputs only after the active DQ gate.
  - Metric-related: yes, adds proxy row counts, mean proxy scores, gain proxies, and evidence-band breakdowns.
  - Framework-related: yes, keeps product language aligned to the partial test/exercise proxy framing.
  - UI-related: yes, inserts a compact summary block near the top of both maintained summary pages.
  - Documentation-related: yes, this entry records the maintained-page alignment change.
- Backward compatibility / migration note:
  - Existing learner and institute metrics remain in place.
  - The new proxy summary is additive and does not remove any existing charts or tables.

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

## 2026-05-27 - Add landing hub and v1.3 basic metrics proxy summary

- Affected files / modules:
  - `dashboards/metrics/app.py`
  - `dashboards/metrics/pages/1_Basic_Metrics.py`
- Reason:
  - Turn the Streamlit entry point into a proper navigation hub.
  - Update `Basic Metrics` to show DQ-gated core metrics plus the proxy metrics used by v1.3.
- Category:
  - DQ-related: yes, `Basic Metrics` continues to run on the selected DQ policy before metrics are computed.
  - Metric-related: yes, `Basic Metrics` now surfaces Inferred BLS Proxy, Current ALS Proxy, Potential ALS Proxy, CAS Proxy coverage, and gain proxies.
  - Framework-related: yes, the landing hub now routes users through the maintained v1.3 page set.
  - UI-related: yes, replaced the old upload-first landing page with page navigation and dataset stats.
  - Documentation-related: yes, this change is reflected in the build and diagnostics guides.
- Backward compatibility / migration note:
  - Page paths are unchanged.
  - The landing page now connects the maintained pages explicitly.
  - Diagnostic preview in `Basic Metrics` is additive and does not alter the published default.
