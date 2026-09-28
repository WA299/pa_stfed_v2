# FedFomo-style Heterogeneous-Grid Comparator: 25% History

peer trainable parameters are evaluated using each target client's local topology buffers and scaler; topology buffers are never exchanged
No claim of exact paper FedFomo is made. Validation is evaluation-only; test is never evaluated.

## Four-scenario macro node-MAE

| Method | Audit node MAE | Validation node MAE |
|---|---:|---:|
| FedFomo-style | 0.000329001 | 0.000374571 |

## Frozen comparator improvements

| Comparator | Audit relative improvement | Validation relative improvement | Validation wins |
|---|---:|---:|---:|
| scarce_local | 0.00546865 | 0.00553499 | 2 |
| fedavg | 0.00384554 | -0.0477961 | 1 |
| fedprox | 0.092956 | 0.0335821 | 3 |
| fedper | 0.0342041 | -0.00285604 | 1 |
| BTD-FL Direct Temporal Transfer (final) | -0.0476116 | -0.0722728 | 0 |

## Scenario metrics

| Scarce target | Audit node MAE | Audit grid MAE | Validation node MAE | Validation grid MAE |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000359989 | 0.00441796 | 0.000402355 | 0.00484697 |
| 50_bus_rural_reference_grid | 0.000315582 | 0.00274884 | 0.000418209 | 0.00444586 |
| 56_bus_semi_urban_reference_grid | 0.000413382 | 0.00683295 | 0.000453384 | 0.00749277 |
| 80_bus_rural_reference_grid | 0.000227049 | 0.00306127 | 0.000224337 | 0.00267652 |

## Scarce-target round peer weights

| Round | Baseline self calibration loss | Positive peers | Fallback | Peer normalized weights |
|---:|---:|---:|---|---|
| 1 | 0.000565117 | 4 | False | {"39_bus_semi_urban_reference_grid": 0.40571209216058635, "50_bus_rural_reference_grid": 0.17472904277119436, "56_bus_semi_urban_reference_grid": 0.20139806250560302, "80_bus_rural_reference_grid": 0.21816080256261625} |
| 2 | 0.000408811 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.7289949192607561, "56_bus_semi_urban_reference_grid": 0.2710050807392439} |
| 3 | 0.000388483 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.7860626404042483, "56_bus_semi_urban_reference_grid": 0.21393735959575175} |
| 4 | 0.000384966 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.7043674109297917, "56_bus_semi_urban_reference_grid": 0.2956325890702084} |
| 5 | 0.000381817 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.6789392049516093, "56_bus_semi_urban_reference_grid": 0.3210607950483908} |
| 6 | 0.000377902 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.6311854268201134, "56_bus_semi_urban_reference_grid": 0.36881457317988675} |
| 7 | 0.00037318 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.7040674728435172, "56_bus_semi_urban_reference_grid": 0.29593252715648277} |
| 8 | 0.000369157 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.9207335142155211, "56_bus_semi_urban_reference_grid": 0.07926648578447892} |
| 9 | 0.000367208 | 1 | False | {"39_bus_semi_urban_reference_grid": 1.0} |
| 10 | 0.000366443 | 1 | False | {"39_bus_semi_urban_reference_grid": 1.0} |

validation_used_for_peer_weighting: false
audit_used_for_peer_weighting: false
test_evaluated: false
communication_round_selection: "fixed_final_round"
