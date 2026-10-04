# Canonical TEST Opening Protocol

Frozen commit: `1681741779df301dcb64587090ce262d2ad1d443`.

Opening TEST is an evaluation event, not a development iteration. Before opening TEST, freeze code, seeds 42/123/2026, all hyperparameters, and the 25%/50% regimes. Evaluate the frozen Local, FedAvg, FedProx, FedPer, FedFomo-style, and BTD-FL methods on reference-grid TEST and industrial TEST according to the pre-registered data splits. Do not tune after TEST is opened.

Write one-shot artifacts with explicit `test_evaluated=true`, split provenance, seed, method, history fraction, and commit metadata. Reference-grid and industrial TEST outputs must be separate from development-validation artifacts. Negative TEST results are retained and reported without method changes or post-hoc seed selection.
