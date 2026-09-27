# Formal BTD-FL Benchmark: 25% History

BTD-FL: Benefit-Triggered Topology-Decoupled Federated Learning.
Canonical validation is evaluation-only; canonical test is never evaluated.
Donor selection uses train-internal calibration only. No causal claim is made.

## Scarce-Target Validation Node-MAE

| Scarce target | scarce_local | FedAvg | FedProx | FedPer | BTD-FL |
|---|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000416143 | 0.000394535 | 0.000423913 | 0.000410201 | 0.000390059 |
| 50_bus_rural_reference_grid | 0.000385882 | 0.000353098 | 0.000381095 | 0.000407751 | 0.000336294 |
| 56_bus_semi_urban_reference_grid | 0.000470452 | 0.000446297 | 0.000472186 | 0.000458034 | 0.000446751 |
| 80_bus_rural_reference_grid | 0.000211706 | 0.00024526 | 0.000271423 | 0.000242482 | 0.000206175 |

## Four-Scenario Macro Node-MAE

| Split | scarce_local | FedAvg | FedProx | FedPer | BTD-FL |
|---|---:|---:|---:|---:|---:|
| audit | 0.000327046 | 0.000331328 | 0.000361729 | 0.000340569 | 0.000313392 |
| validation | 0.000371046 | 0.000359797 | 0.000387154 | 0.000379617 | 0.000344819 |

## BTD-FL Relative Improvements vs Comparators

| Comparator | Audit macro improvement | Validation macro improvement | Validation wins |
|---|---:|---:|---:|
| scarce_local | 0.0417488 | 0.0706828 | 4 |
| fedavg | 0.0541343 | 0.0416291 | 3 |
| fedprox | 0.133626 | 0.109349 | 4 |
| fedper | 0.0797995 | 0.0916647 | 4 |

## Selected Donors and Graph Metadata

| Target | Candidate donors | Selected donor | Graph fit | Graph selection | Selected edge count |
|---|---|---|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 50_bus_rural_reference_grid, 56_bus_semi_urban_reference_grid, 80_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 546 | 273 | 84 |
| 50_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid, 56_bus_semi_urban_reference_grid, 80_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | 546 | 273 | 60 |
| 56_bus_semi_urban_reference_grid | 39_bus_semi_urban_reference_grid, 50_bus_rural_reference_grid, 80_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | 546 | 273 | 132 |
| 80_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid, 50_bus_rural_reference_grid, 56_bus_semi_urban_reference_grid | 50_bus_rural_reference_grid | 546 | 273 | 95 |

## Current-Run Calibration Benefits

| Target \ Donor | 39_bus_semi_urban_reference_grid | 50_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | - | 0.0296257 | 0.066049 | 0.00678424 |
| 50_bus_rural_reference_grid | 0.0547871 | - | 0.0479127 | 0.00312723 |
| 56_bus_semi_urban_reference_grid | 0.0503394 | 0.0258238 | - | 0.0106164 |
| 80_bus_rural_reference_grid | 0.0536431 | 0.0637568 | 0.0511381 | - |

## Scarce-Target Audit Node-MAE

| Target | scarce_local | BTD-FL | Relative improvement |
|---|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000372181 | 0.000350049 | 0.0594646 |
| 50_bus_rural_reference_grid | 0.000306918 | 0.000292519 | 0.0469148 |
| 56_bus_semi_urban_reference_grid | 0.000417712 | 0.000404865 | 0.0307549 |
| 80_bus_rural_reference_grid | 0.000211373 | 0.000206135 | 0.02478 |

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
