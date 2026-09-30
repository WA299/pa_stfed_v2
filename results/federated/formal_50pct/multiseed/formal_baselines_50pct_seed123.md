# Formal BTD-FL Benchmark: 25% History

BTD-FL: Benefit-Triggered Topology-Decoupled Federated Learning.
Canonical validation is evaluation-only; canonical test is never evaluated.
Donor selection uses train-internal calibration only. No causal claim is made.

## Scarce-Target Validation Node-MAE

| Scarce target | scarce_local | FedAvg | FedProx | FedPer |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000394764 | 0.000396508 | 0.000421364 | 0.000402837 |
| 50_bus_rural_reference_grid | 0.000354336 | 0.000351299 | 0.000380743 | 0.000395032 |
| 56_bus_semi_urban_reference_grid | 0.000449389 | 0.000443003 | 0.000467422 | 0.000449656 |
| 80_bus_rural_reference_grid | 0.000205886 | 0.000259276 | 0.000279479 | 0.000227099 |

## Four-Scenario Macro Node-MAE

| Split | scarce_local | FedAvg | FedProx | FedPer |
|---|---:|---:|---:|---:|
| audit | 0.000321427 | 0.00033832 | 0.000363881 | 0.000344762 |
| validation | 0.000351094 | 0.000362521 | 0.000387252 | 0.000368656 |

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
