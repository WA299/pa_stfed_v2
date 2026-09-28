# Formal BTD-FL Benchmark: 25% History

BTD-FL: Benefit-Triggered Topology-Decoupled Federated Learning.
Canonical validation is evaluation-only; canonical test is never evaluated.
Donor selection uses train-internal calibration only. No causal claim is made.

## Scarce-Target Validation Node-MAE

| Scarce target | scarce_local | FedAvg | FedProx | FedPer |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000415254 | 0.000397073 | 0.000424122 | 0.000400957 |
| 50_bus_rural_reference_grid | 0.000396997 | 0.000342839 | 0.00037836 | 0.000388265 |
| 56_bus_semi_urban_reference_grid | 0.000470739 | 0.000447099 | 0.000472603 | 0.000450634 |
| 80_bus_rural_reference_grid | 0.000223634 | 0.000242928 | 0.000275264 | 0.000254161 |

## Four-Scenario Macro Node-MAE

| Split | scarce_local | FedAvg | FedProx | FedPer |
|---|---:|---:|---:|---:|
| audit | 0.00033081 | 0.000330271 | 0.000362717 | 0.000340652 |
| validation | 0.000376656 | 0.000357485 | 0.000387587 | 0.000373504 |

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
