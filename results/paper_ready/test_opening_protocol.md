# Canonical TEST Opening Protocol

Opening TEST is an evaluation event, not a development iteration.

## Frozen identities

- Scientific method/code frozen commit: `1681741779df301dcb64587090ce262d2ad1d443`
- Accepted development and external-result commit: `e8dd4c3fbc4936175d3e9f0fee5c7c06d6303d3b`
- Paper-ready reporting/manifest commit: `REPORTING_COMMIT_TO_BE_RECORDED_AFTER_THIS_PACKAGE_IS_COMMITTED`

The reporting placeholder must be replaced with the SHA of the committed paper-ready package before execution, without changing methods, data selection, or reported development/external results.

## Preregistered evaluation matrix

- Seeds: 42, 123, 2026. No seed may be dropped.
- Target available-history fractions: 0.25 and 0.50.
- Methods: Local, FedAvg, FedProx, FedPer, FedFomo-style, and BTD-FL.
- Reference targets: each of `39_bus_semi_urban_reference_grid`, `50_bus_rural_reference_grid`, `56_bus_semi_urban_reference_grid`, and `80_bus_rural_reference_grid` is evaluated on its frozen canonical TEST split.
- Industrial target: `norway_industrial_mvlv` is evaluated on its Stage-1 frozen canonical TEST indices `[11341, 13344)` only.

This is 3 seeds x 2 history fractions x 6 methods for each domain. Reference reporting includes all four target grids; industrial reporting includes the single industrial target.

## Selection prohibition

No reference or industrial TEST observation, target, loss, or metric may be used for donor selection, FedFomo-style peer weighting, communication-round selection, epoch selection, early stopping, graph construction, scaling, hyperparameter selection, or model revision. Frozen protocol-selected states are evaluated once. There is no retraining, tuning, donor reselection, seed replacement, or reporting-rule change in response to TEST.

## One-shot execution order

1. Verify the three frozen commit identities and a clean accepted-artifact checksum manifest.
2. Verify all development/external artifacts still state `test_evaluated=false`; industrial artifacts must also state `industrial_test_evaluated=false` and `reference_grid_tests_evaluated=false`.
3. Materialize the complete evaluation matrix before reading any TEST target.
4. Evaluate reference-grid TEST for all matrix cells in deterministic seed, fraction, method, target order.
5. Evaluate industrial TEST for all matrix cells in deterministic seed, fraction, method order.
6. Write raw per-cell artifacts before aggregation; never overwrite development/external artifacts.
7. Validate completeness, finite metrics, commit identity, split provenance, and exact matrix membership.
8. Produce the frozen aggregate report once, including every seed and every method.

## Output contract

Write only under `results/final_test_opening/`. Per-cell filenames are:

- `reference_test_<fraction>pct_seed<seed>_<method>_<target>.json`
- `industrial_test_<fraction>pct_seed<seed>_<method>.json`

Aggregate filenames are `reference_test_summary.json`, `industrial_test_summary.json`, and `final_test_opening_manifest.json`. Every artifact must include: scientific commit, accepted-results commit, reporting-manifest commit, seed, history fraction, method, domain, target, exact TEST index bounds, checkpoint/protocol identity, metric definitions and units, `test_evaluated=true`, TEST access timestamp, and flags proving TEST was not used for selection. Industrial artifacts additionally record `industrial_test_evaluated=true` and `reference_grid_tests_evaluated` with the actual domain-specific state.

## Failure and reporting rules

Negative TEST results remain in the complete report and cause no method changes. Partial failures are recorded with error provenance; no partial aggregate is presented as complete, and successful cells are not rerun selectively. Missing cells trigger a halted release, not seed dropping. All six methods, both fractions, all seeds, and all target grids are reported without selective omission. No further development begins from TEST outcomes within this preregistered evaluation event.

This document defines future execution only. It contains no TEST result value and authorizes no TEST access in the reporting commit.
