# Formal BTD-FL Benchmark: 25% History

BTD-FL: Benefit-Triggered Topology-Decoupled Federated Learning.
Canonical validation is evaluation-only; canonical test is never evaluated.
Donor selection uses train-internal calibration only. No causal claim is made.

## Scarce-Target Validation Node-MAE

| Scarce target | scarce_local | FedAvg | FedProx | FedPer |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000396354 | 0.000393631 | 0.000422026 | 0.000403363 |
| 50_bus_rural_reference_grid | 0.000357883 | 0.000346234 | 0.000377785 | 0.000375166 |
| 56_bus_semi_urban_reference_grid | 0.000446129 | 0.000439404 | 0.000467889 | 0.000461117 |
| 80_bus_rural_reference_grid | 0.000205147 | 0.000247644 | 0.00027522 | 0.000232657 |

## Four-Scenario Macro Node-MAE

| Split | scarce_local | FedAvg | FedProx | FedPer |
|---|---:|---:|---:|---:|
| audit | 0.000320515 | 0.000331821 | 0.000362489 | 0.000342585 |
| validation | 0.000351378 | 0.000356728 | 0.00038573 | 0.000368076 |

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
