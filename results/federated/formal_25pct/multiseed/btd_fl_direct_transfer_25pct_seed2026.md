# BTD-FL Direct Temporal Transfer: 25% History

Target-conditioned donor adaptations are calibration-only benefit probes. The final transferred initialization is the raw selected donor temporal state.
Canonical validation is evaluation-only. Canonical test is never evaluated.

## Selected donors and benefits

| Target | Selected donor | Benefit | Fallback | Matches frozen formal run |
|---|---|---:|---|---|
| 39_bus_semi_urban_reference_grid | 56_bus_semi_urban_reference_grid | 0.0783091 | False | True |
| 50_bus_rural_reference_grid | 80_bus_rural_reference_grid | 0.0642458 | False | False |
| 56_bus_semi_urban_reference_grid | 39_bus_semi_urban_reference_grid | 0.0462019 | False | True |
| 80_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | 0.0714517 | False | False |

## Current-run calibration benefit matrix

| Target \ Donor | 39_bus_semi_urban_reference_grid | 50_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | - | 0.0561503 | 0.0783091 | 0.0527838 |
| 50_bus_rural_reference_grid | 0.0484119 | - | 0.0620637 | 0.0642458 |
| 56_bus_semi_urban_reference_grid | 0.0462019 | 0.0396112 | - | 0.034574 |
| 80_bus_rural_reference_grid | 0.0714517 | 0.0683493 | 0.0667889 | - |

## Train audit metrics

| Target | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Grid MAE | Grid RMSE | Grid WAPE % | Grid sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.0003481 | 0.000515298 | 32.6037 | 32.3791 | 0.00350777 | 0.00431072 | 11.0864 | 11.2435 |
| 50_bus_rural_reference_grid | 0.00029125 | 0.000436898 | 38.4201 | 39.6956 | 0.00253459 | 0.00339022 | 8.40802 | 8.52035 |
| 56_bus_semi_urban_reference_grid | 0.000405533 | 0.000585091 | 33.1519 | 33.5177 | 0.00627569 | 0.00762379 | 10.657 | 10.8946 |
| 80_bus_rural_reference_grid | 0.000211311 | 0.000367124 | 45.4453 | 47.3386 | 0.00313626 | 0.00396899 | 16.3705 | 17.7595 |
| Four-target macro | 0.000314048 | 0.000476103 | 37.4052 | 38.2328 | 0.00386358 | 0.00482343 | 11.6305 | 12.1045 |

## Canonical validation metrics

| Target | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Grid MAE | Grid RMSE | Grid WAPE % | Grid sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000386982 | 0.000565458 | 27.959 | 28.0073 | 0.00426467 | 0.00538739 | 10.5258 | 10.7082 |
| 50_bus_rural_reference_grid | 0.000363674 | 0.000542151 | 31.2517 | 34.2805 | 0.00241938 | 0.00312247 | 6.27463 | 6.27655 |
| 56_bus_semi_urban_reference_grid | 0.00044218 | 0.000627356 | 25.5899 | 26.753 | 0.00717015 | 0.00865478 | 9.00013 | 9.29916 |
| 80_bus_rural_reference_grid | 0.000204462 | 0.000361511 | 59.2524 | 55.8519 | 0.00289423 | 0.00370095 | 11.9774 | 12.5832 |
| Four-target macro | 0.000349324 | 0.000524119 | 36.0132 | 36.2232 | 0.00418711 | 0.00521639 | 9.4445 | 9.71677 |

## Relative improvement vs frozen comparators

| Comparator | Audit | Validation | Validation wins |
|---|---:|---:|---:|

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
adapted_probe_states_used_for_final_initialization: false
