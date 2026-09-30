# Formal BTD-FL Benchmark: 25% History

BTD-FL: Benefit-Triggered Topology-Decoupled Federated Learning.
Canonical validation is evaluation-only; canonical test is never evaluated.
Donor selection uses train-internal calibration only. No causal claim is made.

## Scarce-Target Validation Node-MAE

| Scarce target | scarce_local | FedAvg | FedProx | FedPer |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000398449 | 0.000392452 | 0.000421989 | 0.00039531 |
| 50_bus_rural_reference_grid | 0.000350382 | 0.000340252 | 0.000379876 | 0.000374727 |
| 56_bus_semi_urban_reference_grid | 0.000444619 | 0.000442259 | 0.000467302 | 0.000448747 |
| 80_bus_rural_reference_grid | 0.000204128 | 0.000235648 | 0.000278819 | 0.000220956 |

## Four-Scenario Macro Node-MAE

| Split | scarce_local | FedAvg | FedProx | FedPer |
|---|---:|---:|---:|---:|
| audit | 0.000319601 | 0.000328859 | 0.000364223 | 0.00033608 |
| validation | 0.000349394 | 0.000352653 | 0.000386997 | 0.000359935 |

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
