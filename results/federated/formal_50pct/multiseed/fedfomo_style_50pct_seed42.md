# FedFomo-style Heterogeneous-Grid Comparator: 25% History

peer trainable parameters are evaluated using each target client's local topology buffers and scaler; topology buffers are never exchanged
No claim of exact paper FedFomo is made. Validation is evaluation-only; test is never evaluated.

## Four-scenario macro node-MAE

| Method | Audit node MAE | Validation node MAE |
|---|---:|---:|
| FedFomo-style | 0.00032535 | 0.000349955 |

## Frozen comparator improvements

| Comparator | Audit relative improvement | Validation relative improvement | Validation wins |
|---|---:|---:|---:|
| scarce_local | -0.0150868 | 0.00405088 | 3 |
| fedavg | 0.0195011 | 0.0189878 | 1 |
| fedprox | 0.102455 | 0.0927465 | 4 |
| fedper | 0.0503063 | 0.0492319 | 4 |
| BTD-FL Direct Temporal Transfer (final) | -0.0406054 | -0.0365803 | 0 |

## Scenario metrics

| Scarce target | Audit node MAE | Audit grid MAE | Validation node MAE | Validation grid MAE |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000359597 | 0.00373424 | 0.000394116 | 0.00428297 |
| 50_bus_rural_reference_grid | 0.000298936 | 0.00272204 | 0.000350555 | 0.0036323 |
| 56_bus_semi_urban_reference_grid | 0.000413354 | 0.00691388 | 0.000442301 | 0.00719219 |
| 80_bus_rural_reference_grid | 0.000229514 | 0.00299831 | 0.000212847 | 0.0027431 |

## Scarce-target round peer weights

| Round | Baseline self calibration loss | Positive peers | Fallback | Peer normalized weights |
|---:|---:|---:|---|---|
| 1 | 0.000587253 | 4 | False | {"39_bus_semi_urban_reference_grid": 0.2806584488580686, "50_bus_rural_reference_grid": 0.22144814145063724, "56_bus_semi_urban_reference_grid": 0.2139855456565679, "80_bus_rural_reference_grid": 0.2839078640347263} |
| 2 | 0.000395621 | 3 | False | {"39_bus_semi_urban_reference_grid": 0.5915531704596252, "56_bus_semi_urban_reference_grid": 0.37834227613231286, "80_bus_rural_reference_grid": 0.030104553408061955} |
| 3 | 0.0003753 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.7790810344503046, "56_bus_semi_urban_reference_grid": 0.22091896554969534} |
| 4 | 0.000371207 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.8766560452786332, "56_bus_semi_urban_reference_grid": 0.12334395472136679} |
| 5 | 0.000367448 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.7933224933439648, "56_bus_semi_urban_reference_grid": 0.20667750665603513} |
| 6 | 0.000364117 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.8180768044475449, "56_bus_semi_urban_reference_grid": 0.18192319555245506} |
| 7 | 0.000360804 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.9229488349049648, "56_bus_semi_urban_reference_grid": 0.07705116509503517} |
| 8 | 0.000358773 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.9894299068769937, "56_bus_semi_urban_reference_grid": 0.010570093123006254} |
| 9 | 0.000356941 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.9674251863687856, "56_bus_semi_urban_reference_grid": 0.03257481363121445} |
| 10 | 0.000354223 | 1 | False | {"39_bus_semi_urban_reference_grid": 1.0} |

validation_used_for_peer_weighting: false
audit_used_for_peer_weighting: false
test_evaluated: false
communication_round_selection: "fixed_final_round"
