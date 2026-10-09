# Checkpoint parity policy

Parity is checked per cell against the accepted frozen JSON artifact for the same seed, history fraction, target, and method. The reconstructed checkpoint is evaluated only on the frozen AUDIT split and canonical VALIDATION split; TEST is inaccessible.

The primary node-macro MAE uses an absolute tolerance of `1e-10` and a relative tolerance of `1e-6`; supporting finite metrics use the same declared tolerances. A cell passes only when every reported audit/validation metric in the accepted contract is finite and within both criteria (`abs(actual-reference) <= 1e-10 + 1e-6*abs(reference)`). Missing, extra, or unexplained metrics fail closed. The complete parity report must contain all 180 cells, `test_accessed=false`, and `parity_passed=true` for every cell.

Any mismatch halts TEST readiness. It must be recorded with source key, expected value, observed value, absolute discrepancy, and relative discrepancy. The original accepted result JSON is never overwritten. There is no tuning, seed dropping, post-hoc selection, retraining, or algorithm change in response to a mismatch.
