# FedFomo-style Heterogeneous-Grid Comparator: 25% History

peer trainable parameters are evaluated using each target client's local topology buffers and scaler; topology buffers are never exchanged
No claim of exact paper FedFomo is made. Validation is evaluation-only; test is never evaluated.

## Four-scenario macro node-MAE

| Method | Audit node MAE | Validation node MAE |
|---|---:|---:|
| FedFomo-style | 0.000322667 | 0.000348751 |

## Frozen comparator improvements

| Comparator | Audit relative improvement | Validation relative improvement | Validation wins |
|---|---:|---:|---:|
| scarce_local | -0.0038555 | 0.00667251 | 2 |
| fedavg | 0.0462681 | 0.0379847 | 3 |
| fedprox | 0.113264 | 0.0994207 | 4 |
| fedper | 0.0640897 | 0.053993 | 4 |
| BTD-FL Direct Temporal Transfer (final) | -0.049837 | -0.0463149 | 0 |

## Scenario metrics

| Scarce target | Audit node MAE | Audit grid MAE | Validation node MAE | Validation grid MAE |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000362305 | 0.00395314 | 0.000398267 | 0.00456023 |
| 50_bus_rural_reference_grid | 0.000295056 | 0.00266202 | 0.000349677 | 0.00380339 |
| 56_bus_semi_urban_reference_grid | 0.000412372 | 0.00674714 | 0.000440757 | 0.00718891 |
| 80_bus_rural_reference_grid | 0.000220933 | 0.00302584 | 0.000206303 | 0.00281291 |

## Scarce-target round peer weights

| Round | Baseline self calibration loss | Positive peers | Fallback | Peer normalized weights |
|---:|---:|---:|---|---|
| 1 | 0.000654823 | 4 | False | {"39_bus_semi_urban_reference_grid": 0.3274109261385378, "50_bus_rural_reference_grid": 0.21052085609866897, "56_bus_semi_urban_reference_grid": 0.21687968646410802, "80_bus_rural_reference_grid": 0.24518853129868529} |
| 2 | 0.000391953 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.5954206634025372, "56_bus_semi_urban_reference_grid": 0.40457933659746276} |
| 3 | 0.000375542 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.77497066137172, "56_bus_semi_urban_reference_grid": 0.22502933862828006} |
| 4 | 0.000371593 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.9309810568326743, "56_bus_semi_urban_reference_grid": 0.06901894316732562} |
| 5 | 0.000368822 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.6525678818849322, "56_bus_semi_urban_reference_grid": 0.3474321181150679} |
| 6 | 0.000365559 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.9190100828032, "56_bus_semi_urban_reference_grid": 0.08098991719680006} |
| 7 | 0.00036252 | 1 | False | {"39_bus_semi_urban_reference_grid": 1.0} |
| 8 | 0.00036078 | 1 | False | {"39_bus_semi_urban_reference_grid": 1.0} |
| 9 | 0.000359544 | 1 | False | {"39_bus_semi_urban_reference_grid": 1.0} |
| 10 | 0.000357981 | 1 | False | {"39_bus_semi_urban_reference_grid": 1.0} |

validation_used_for_peer_weighting: false
audit_used_for_peer_weighting: false
test_evaluated: false
communication_round_selection: "fixed_final_round"
