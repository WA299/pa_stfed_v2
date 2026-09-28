# Formal BTD-FL Benchmark: 25% History

BTD-FL: Benefit-Triggered Topology-Decoupled Federated Learning.
Canonical validation is evaluation-only; canonical test is never evaluated.
Donor selection uses train-internal calibration only. No causal claim is made.

## Scarce-Target Validation Node-MAE

| Scarce target | scarce_local | FedAvg | FedProx | FedPer |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000416861 | 0.000399741 | 0.000423543 | 0.000414531 |
| 50_bus_rural_reference_grid | 0.000382804 | 0.000351589 | 0.000381519 | 0.000426072 |
| 56_bus_semi_urban_reference_grid | 0.000476449 | 0.000446986 | 0.000471707 | 0.000456232 |
| 80_bus_rural_reference_grid | 0.000220244 | 0.000268909 | 0.000282003 | 0.000255356 |

## Four-Scenario Macro Node-MAE

| Split | scarce_local | FedAvg | FedProx | FedPer |
|---|---:|---:|---:|---:|
| audit | 0.000330304 | 0.00033838 | 0.000364086 | 0.000349184 |
| validation | 0.000374089 | 0.000366806 | 0.000389693 | 0.000388048 |

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
