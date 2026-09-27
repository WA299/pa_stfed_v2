# FedFomo-style Heterogeneous-Grid Comparator: 25% History

peer trainable parameters are evaluated using each target client's local topology buffers and scaler; topology buffers are never exchanged
No claim of exact paper FedFomo is made. Validation is evaluation-only; test is never evaluated.

## Four-scenario macro node-MAE

| Method | Audit node MAE | Validation node MAE |
|---|---:|---:|
| FedFomo-style | 0.000323949 | 0.000360143 |

## Frozen comparator improvements

| Comparator | Audit relative improvement | Validation relative improvement | Validation wins |
|---|---:|---:|---:|
| scarce_local | 0.00946881 | 0.0293843 | 3 |
| fedavg | 0.0222716 | -0.000960559 | 1 |
| fedprox | 0.104442 | 0.0697686 | 4 |
| fedper | 0.0488013 | 0.0512986 | 3 |
| BTD-FL Direct Temporal Transfer (final) | -0.0331379 | -0.0476337 | 0 |

## Scenario metrics

| Scarce target | Audit node MAE | Audit grid MAE | Validation node MAE | Validation grid MAE |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000358254 | 0.0041393 | 0.000399253 | 0.00463429 |
| 50_bus_rural_reference_grid | 0.000298029 | 0.00276433 | 0.00036131 | 0.00349131 |
| 56_bus_semi_urban_reference_grid | 0.000415889 | 0.0072423 | 0.000460095 | 0.00889665 |
| 80_bus_rural_reference_grid | 0.000223625 | 0.00315234 | 0.000219914 | 0.00294673 |

## Scarce-target round peer weights

| Round | Baseline self calibration loss | Positive peers | Fallback | Peer normalized weights |
|---:|---:|---:|---|---|
| 1 | 0.000541045 | 4 | False | {"39_bus_semi_urban_reference_grid": 0.42528878746097604, "50_bus_rural_reference_grid": 0.17141746834401908, "56_bus_semi_urban_reference_grid": 0.179519206000083, "80_bus_rural_reference_grid": 0.22377453819492182} |
| 2 | 0.000409133 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.6923098030756759, "56_bus_semi_urban_reference_grid": 0.3076901969243241} |
| 3 | 0.000388625 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.7639385482891675, "56_bus_semi_urban_reference_grid": 0.23606145171083256} |
| 4 | 0.000384271 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.7715191568661942, "56_bus_semi_urban_reference_grid": 0.22848084313380576} |
| 5 | 0.000380539 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.6571040379808541, "56_bus_semi_urban_reference_grid": 0.3428959620191459} |
| 6 | 0.000377115 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.6873134746504359, "56_bus_semi_urban_reference_grid": 0.3126865253495641} |
| 7 | 0.000372657 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.8162252241123686, "56_bus_semi_urban_reference_grid": 0.18377477588763141} |
| 8 | 0.000368358 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.9642850635519356, "56_bus_semi_urban_reference_grid": 0.03571493644806448} |
| 9 | 0.000366588 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.9588231938580821, "56_bus_semi_urban_reference_grid": 0.04117680614191788} |
| 10 | 0.000364873 | 2 | False | {"39_bus_semi_urban_reference_grid": 0.9774513698096325, "56_bus_semi_urban_reference_grid": 0.022548630190367465} |

validation_used_for_peer_weighting: false
audit_used_for_peer_weighting: false
test_evaluated: false
communication_round_selection: "fixed_final_round"
