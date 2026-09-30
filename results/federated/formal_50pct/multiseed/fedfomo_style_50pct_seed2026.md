# FedFomo-style Heterogeneous-Grid Comparator: 25% History

peer trainable parameters are evaluated using each target client's local topology buffers and scaler; topology buffers are never exchanged
No claim of exact paper FedFomo is made. Validation is evaluation-only; test is never evaluated.

## Four-scenario macro node-MAE

| Method | Audit node MAE | Validation node MAE |
|---|---:|---:|
| FedFomo-style | 0.000323027 | 0.000348478 |

## Frozen comparator improvements

| Comparator | Audit relative improvement | Validation relative improvement | Validation wins |
|---|---:|---:|---:|
| scarce_local | -0.0107215 | 0.00262258 | 2 |
| fedavg | 0.0177342 | 0.0118381 | 2 |
| fedprox | 0.113105 | 0.0995316 | 4 |
| fedper | 0.0388395 | 0.0318303 | 4 |
| BTD-FL Direct Temporal Transfer (final) | -0.0369667 | -0.0389132 | 0 |

## Scenario metrics

| Scarce target | Audit node MAE | Audit grid MAE | Validation node MAE | Validation grid MAE |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000359874 | 0.00388942 | 0.00039412 | 0.00432119 |
| 50_bus_rural_reference_grid | 0.000298617 | 0.00246209 | 0.000353282 | 0.00345932 |
| 56_bus_semi_urban_reference_grid | 0.000411298 | 0.00664578 | 0.00044087 | 0.00669932 |
| 80_bus_rural_reference_grid | 0.000222319 | 0.00303226 | 0.000205641 | 0.00279902 |

## Scarce-target round peer weights

| Round | Baseline self calibration loss | Positive peers | Fallback | Peer normalized weights |
|---:|---:|---:|---|---|
| 1 | 0.000630727 | 4 | False | {"39_bus_semi_urban_reference_grid": 0.27458302278663793, "50_bus_rural_reference_grid": 0.21936425419387964, "56_bus_semi_urban_reference_grid": 0.2353728524651514, "80_bus_rural_reference_grid": 0.270679870554331} |
| 2 | 0.000395486 | 3 | False | {"39_bus_semi_urban_reference_grid": 0.5646115784905156, "56_bus_semi_urban_reference_grid": 0.3372763408673845, "80_bus_rural_reference_grid": 0.09811208064209984} |
| 3 | 0.000375578 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.7703999820409838, "56_bus_semi_urban_reference_grid": 0.2296000179590161} |
| 4 | 0.00037155 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.7866701114289167, "56_bus_semi_urban_reference_grid": 0.21332988857108326} |
| 5 | 0.000367519 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.5539408046806478, "56_bus_semi_urban_reference_grid": 0.4460591953193523} |
| 6 | 0.000363851 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.8282551317703983, "56_bus_semi_urban_reference_grid": 0.17174486822960167} |
| 7 | 0.000359874 | 1 | False | {"39_bus_semi_urban_reference_grid": 1.0} |
| 8 | 0.000357722 | 1 | False | {"39_bus_semi_urban_reference_grid": 1.0} |
| 9 | 0.000355786 | 1 | False | {"39_bus_semi_urban_reference_grid": 1.0} |
| 10 | 0.000354014 | 1 | False | {"39_bus_semi_urban_reference_grid": 1.0} |

validation_used_for_peer_weighting: false
audit_used_for_peer_weighting: false
test_evaluated: false
communication_round_selection: "fixed_final_round"
