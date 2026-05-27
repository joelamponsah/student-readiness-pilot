# Developer Diagnostics and Troubleshooting v1.3

## Smoke Checks

1. Confirm the loader preserves the raw row count.
2. Confirm published DQ mode still returns eligible rows on the repo dataset.
3. Confirm diagnostic mode preserves repeated eligible attempts.
4. Confirm proxy metrics are based on the proxy-sequence slice, not the deduped published slice.
5. Confirm `inactive` zero-attempt rows are flagged.
6. Confirm accuracy values stay in the expected range.
7. Confirm `robust_SAB_scaled` stays within 0-100.
8. Confirm proxy gain stays within -100 to 100.
9. Confirm Institute Summary loads without a `NameError`.

## Manual Proxy Verification

- Inferred BLS Proxy should appear on first eligible attempts.
- Current ALS Proxy should appear on later repeated eligible attempts.
- Potential ALS Proxy should reflect the best later repeated eligible attempt.
- CAS Proxy should be treated as a conservative test/institute average, not a true class average.
