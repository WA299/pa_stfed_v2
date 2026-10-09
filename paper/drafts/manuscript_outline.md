# Manuscript outline

## Introduction

- Motivate forecasting when a target grid has limited local history and the available clients have different topologies and load behavior.
- State the scientific question: whether target-conditioned directed temporal transfer improves scarce-history forecasting while preserving the target's spatial/topology state.
- Define the evidence scope: four Norway reference grids for development validation and a separate Norwegian industrial MV/LV domain for external validation.
- State the freeze: all methods, protocols, metrics, and accepted result artifacts are fixed; canonical TEST remains locked.

## Related Work

- Temporal sequence models for load forecasting.
- Graph and physical-topology models for grid forecasting.
- Federated averaging and heterogeneous-client personalization.
- Peer-weighted federated methods, including the explicitly named FedFomo-style comparator.
- Transfer learning and selective parameter transfer under domain shift.
- Position the contribution as an evaluation of target-conditioned directed benefit selection and raw temporal-state transfer, rather than a new model architecture.

## Methodology

### Forecasting model and frozen feature contract

- Use the accepted PUCRSTAttnV2ConditionalUtility architecture and the frozen `P + calendar` input (`0,2,3,4,5,6`), history length 168, and one-step horizon.
- Keep temporal and spatial parameter groups distinct. BTD-FL transfers the selected donor's raw temporal group only; target-local graph buffers and spatial initialization remain local.

### Federated clients and history regimes

- Reference clients: 39-bus semi-urban, 50-bus rural, 56-bus semi-urban, and 80-bus rural reference grids.
- External target: `norway_industrial_mvlv`, whose native values are hourly `Load_kWh` measurements.
- Evaluate target history fractions 0.25 and 0.50 with seeds 42, 123, and 2026.
- Reference canonical validation is development validation because it participated in method development. Industrial canonical validation is external industrial validation in a different operational domain.

### Methods

- Scarce Local, FedAvg, FedProx, FedPer, FedFomo-style, and final BTD-FL direct temporal transfer.
- BTD-FL trains a local target proxy, probes each donor on target FIT/CALIBRATION, selects the current-run maximum positive calibration benefit, discards the adapted probe state, and trains a fresh target full model with only the selected raw temporal state injected.
- The zero-transfer fallback is retained by the frozen protocol; it was not triggered in the accepted results.

### Metrics and evidence controls

- Primary endpoint: validation node-macro MAE. Supporting endpoints are RMSE, WAPE, sMAPE, and grid-aggregate metrics where available.
- Reference and industrial MAE are kept in their own domains and units; industrial `Load_kWh` values are not relabeled as instantaneous MW.
- Every manuscript number is traced through `results/paper_ready/manifests/paper_results_manifest.json` to an accepted JSON artifact and key path.

## Experimental Setup

- Chronological scarce-history splits, FIT-only scaling and graph construction, and calibration-only selection/early stopping follow the accepted frozen protocols.
- Reference-grid summary artifacts: `results/federated/formal_25pct/multiseed/multiseed_25pct_summary.json` and `results/federated/formal_50pct/multiseed/multiseed_50pct_summary.json`.
- Industrial summary artifacts: `results/federated/industrial_external/industrial_external_25pct_summary.json` and `results/federated/industrial_external/industrial_external_50pct_summary.json`.
- Centralized evidence uses `results/ablations/puc_rstattn_v2_ablation_summary.json` and `results/centralized/centralized_validation_summary.json`.
- Three seeds describe robustness and consistency; they do not establish statistical significance.
- Canonical TEST is locked and is not used in this manuscript package.

## Results

Follow this order in the manuscript.

### 1. Temporal dominance

- Report the GRU anchor, `temporal_residual_only`, Graph WaveNet, and full PUC-RSTAttn V2 development-validation node-macro MAE from the centralized accepted artifacts.
- Use the exact temporal-only value `0.00032925806226992394` and the explicit ablation key.

### 2. Four-reference-grid 25% and 50% validation

- Report mean +/- standard deviation across seeds for Scarce Local, FedAvg, FedProx, FedPer, FedFomo-style, and BTD-FL.
- Report BTD-FL relative validation-MAE improvement and seed wins against each comparator for both fractions.
- Label the canonical validation split as development validation.

### 3. External industrial 25% and 50% validation

- Report industrial node-macro MAE, RMSE, WAPE, sMAPE, and grid-aggregate metrics within the industrial domain.
- Report BTD-FL relative node-MAE improvement and seed wins against each comparator.
- State that WAPE and sMAPE are not uniformly improved and that industrial values are hourly load/energy measurements.

### 4. Directed-benefit mechanism analysis

- Show the selected donor and four directed calibration benefits for every industrial seed/fraction row.
- Summarize reference and industrial positive/non-positive benefit counts and fallback counts.
- Distinguish final raw temporal transfer, full-model transfer, historical adapted-state reuse, and no-benefit-selection ablations.
- State the control limitation: no-benefit-selection versus final BTD is not a perfectly controlled WHO-only ablation because the accepted no-benefit artifact also reuses adapted probe state.

## Discussion

- Interpret temporal dominance and the weak/inconsistent physical/spatial contribution without significance claims.
- Discuss consistent primary node-MAE gains across three seeds while separating the reference development domain from industrial external validation.
- Discuss industrial negative transfer for FedAvg/FedProx and the non-uniform percentage metrics.
- Discuss donor identity as run-specific target-conditioned selection, not an intrinsically best donor.
- State that the absence of fallback events does not validate fallback performance.
- Describe limitations: three seeds, development use of reference canonical validation, one external industrial dataset, and no TEST evidence in this package.

## Conclusion

- Restate the bounded finding: frozen BTD-FL direct temporal transfer improves the primary scarce-history node-MAE endpoint in the accepted reference and industrial validation summaries.
- Emphasize that the result is a target-conditioned robustness finding with explicit domain and metric boundaries.
- Defer all canonical TEST claims until the separately locked one-shot TEST protocol is executed.
