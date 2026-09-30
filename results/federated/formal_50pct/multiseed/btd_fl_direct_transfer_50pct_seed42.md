# BTD-FL Direct Temporal Transfer: 25% History

Target-conditioned donor adaptations are calibration-only benefit probes. The final transferred initialization is the raw selected donor temporal state.
Canonical validation is evaluation-only. Canonical test is never evaluated.

## Selected donors and benefits

| Target | Selected donor | Benefit | Fallback | Matches frozen formal run |
|---|---|---:|---|---|
| 39_bus_semi_urban_reference_grid | 56_bus_semi_urban_reference_grid | 0.0481642 | False | True |
| 50_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 0.0638401 | False | False |
| 56_bus_semi_urban_reference_grid | 39_bus_semi_urban_reference_grid | 0.0490272 | False | True |
| 80_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | 0.0294263 | False | False |

## Current-run calibration benefit matrix

| Target \ Donor | 39_bus_semi_urban_reference_grid | 50_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | - | 0.0264597 | 0.0481642 | 0.0100783 |
| 50_bus_rural_reference_grid | 0.0515354 | - | 0.0638401 | 0.0150172 |
| 56_bus_semi_urban_reference_grid | 0.0490272 | 0.0235815 | - | 0.0135338 |
| 80_bus_rural_reference_grid | 0.0294263 | 0.020012 | 0.0281354 | - |

## Train audit metrics

| Target | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Grid MAE | Grid RMSE | Grid WAPE % | Grid sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000350943 | 0.000524089 | 32.9023 | 32.3117 | 0.00350683 | 0.00435684 | 11.1587 | 11.2662 |
| 50_bus_rural_reference_grid | 0.000272788 | 0.000408209 | 37.3344 | 36.8437 | 0.00229058 | 0.00295865 | 7.75969 | 7.80937 |
| 56_bus_semi_urban_reference_grid | 0.000401995 | 0.000580122 | 33.325 | 33.6064 | 0.00588447 | 0.00713153 | 10.1733 | 10.4764 |
| 80_bus_rural_reference_grid | 0.000224893 | 0.000396108 | 41.2375 | 38.6618 | 0.00288073 | 0.00381362 | 14.2626 | 15.3125 |
| Four-target macro | 0.000312655 | 0.000477132 | 36.1998 | 35.3559 | 0.00364065 | 0.00456516 | 10.8386 | 11.2161 |

## Canonical validation metrics

| Target | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Grid MAE | Grid RMSE | Grid WAPE % | Grid sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000386737 | 0.00056391 | 27.8213 | 27.8534 | 0.00403347 | 0.00508034 | 9.95515 | 10.1814 |
| 50_bus_rural_reference_grid | 0.000325276 | 0.000506528 | 30.359 | 31.4879 | 0.00291143 | 0.00373472 | 7.55075 | 7.57906 |
| 56_bus_semi_urban_reference_grid | 0.000432867 | 0.000614232 | 25.2942 | 26.5067 | 0.00669162 | 0.00814112 | 8.39947 | 8.72047 |
| 80_bus_rural_reference_grid | 0.00020554 | 0.00036628 | 40.8452 | 46.4276 | 0.0028025 | 0.00375103 | 11.5978 | 12.2184 |
| Four-target macro | 0.000337605 | 0.000512738 | 31.0799 | 33.0689 | 0.00410975 | 0.00517681 | 9.3758 | 9.67483 |

## Relative improvement vs frozen comparators

| Comparator | Audit | Validation | Validation wins |
|---|---:|---:|---:|

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
adapted_probe_states_used_for_final_initialization: false
