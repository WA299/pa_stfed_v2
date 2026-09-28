# FedFomo-style Heterogeneous-Grid Comparator: 25% History

peer trainable parameters are evaluated using each target client's local topology buffers and scaler; topology buffers are never exchanged
No claim of exact paper FedFomo is made. Validation is evaluation-only; test is never evaluated.

## Four-scenario macro node-MAE

| Method | Audit node MAE | Validation node MAE |
|---|---:|---:|
| FedFomo-style | 0.000326891 | 0.000371751 |

## Frozen comparator improvements

| Comparator | Audit relative improvement | Validation relative improvement | Validation wins |
|---|---:|---:|---:|
| scarce_local | 0.0103326 | 0.00625215 | 2 |
| fedavg | 0.0339513 | -0.0134788 | 2 |
| fedprox | 0.102159 | 0.0460429 | 3 |
| fedper | 0.0638415 | 0.0419978 | 4 |
| BTD-FL Direct Temporal Transfer (final) | -0.0564871 | -0.0799244 | 0 |

## Scenario metrics

| Scarce target | Audit node MAE | Audit grid MAE | Validation node MAE | Validation grid MAE |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000359632 | 0.00423718 | 0.00039953 | 0.00496091 |
| 50_bus_rural_reference_grid | 0.000313437 | 0.00254285 | 0.000414359 | 0.00414278 |
| 56_bus_semi_urban_reference_grid | 0.000411899 | 0.00711762 | 0.000451897 | 0.00787276 |
| 80_bus_rural_reference_grid | 0.000222596 | 0.00305558 | 0.000221216 | 0.00279672 |

## Scarce-target round peer weights

| Round | Baseline self calibration loss | Positive peers | Fallback | Peer normalized weights |
|---:|---:|---:|---|---|
| 1 | 0.000583736 | 4 | False | {"39_bus_semi_urban_reference_grid": 0.3988475026923311, "50_bus_rural_reference_grid": 0.1825088509378656, "56_bus_semi_urban_reference_grid": 0.2024070685106128, "80_bus_rural_reference_grid": 0.21623657785919048} |
| 2 | 0.00040774 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.7374459157940135, "56_bus_semi_urban_reference_grid": 0.2625540842059865} |
| 3 | 0.000388302 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.8722523253763349, "56_bus_semi_urban_reference_grid": 0.12774767462366515} |
| 4 | 0.000385045 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.8942603993469439, "56_bus_semi_urban_reference_grid": 0.10573960065305618} |
| 5 | 0.00038328 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.7697407763579471, "56_bus_semi_urban_reference_grid": 0.2302592236420528} |
| 6 | 0.000381564 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.7142732244085632, "56_bus_semi_urban_reference_grid": 0.2857267755914367} |
| 7 | 0.000377867 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.7486438454635093, "56_bus_semi_urban_reference_grid": 0.2513561545364908} |
| 8 | 0.000373256 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.877648459777529, "56_bus_semi_urban_reference_grid": 0.12235154022247094} |
| 9 | 0.000369819 | 1 | False | {"39_bus_semi_urban_reference_grid": 1.0} |
| 10 | 0.000368759 | 1 | False | {"39_bus_semi_urban_reference_grid": 1.0} |

validation_used_for_peer_weighting: false
audit_used_for_peer_weighting: false
test_evaluated: false
communication_round_selection: "fixed_final_round"
