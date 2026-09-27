# BTD-FL Direct Temporal Transfer: 25% History

Target-conditioned donor adaptations are calibration-only benefit probes. The final transferred initialization is the raw selected donor temporal state.
Canonical validation is evaluation-only. Canonical test is never evaluated.

## Selected donors and benefits

| Target | Selected donor | Benefit | Fallback | Matches frozen formal run |
|---|---|---:|---|---|
| 39_bus_semi_urban_reference_grid | 56_bus_semi_urban_reference_grid | 0.066049 | False | True |
| 50_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | 0.0547871 | False | True |
| 56_bus_semi_urban_reference_grid | 39_bus_semi_urban_reference_grid | 0.0503394 | False | True |
| 80_bus_rural_reference_grid | 50_bus_rural_reference_grid | 0.0637568 | False | True |

## Current-run calibration benefit matrix

| Target \ Donor | 39_bus_semi_urban_reference_grid | 50_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | - | 0.0296257 | 0.066049 | 0.00678424 |
| 50_bus_rural_reference_grid | 0.0547871 | - | 0.0479127 | 0.00312723 |
| 56_bus_semi_urban_reference_grid | 0.0503394 | 0.0258238 | - | 0.0106164 |
| 80_bus_rural_reference_grid | 0.0536431 | 0.0637568 | 0.0511381 | - |

## Train audit metrics

| Target | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Grid MAE | Grid RMSE | Grid WAPE % | Grid sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000350549 | 0.000520114 | 32.7854 | 32.637 | 0.00384389 | 0.00466771 | 12.1487 | 12.3963 |
| 50_bus_rural_reference_grid | 0.000291846 | 0.000433972 | 38.9301 | 39.9453 | 0.00245226 | 0.00325859 | 8.13491 | 8.19513 |
| 56_bus_semi_urban_reference_grid | 0.000402978 | 0.000581941 | 32.8965 | 33.2487 | 0.00614545 | 0.00740576 | 10.4359 | 10.7064 |
| 80_bus_rural_reference_grid | 0.000208861 | 0.000364057 | 42.4689 | 41.9137 | 0.0030316 | 0.00380829 | 15.8242 | 17.0626 |
| Four-target macro | 0.000313559 | 0.000475021 | 36.7702 | 36.9362 | 0.0038683 | 0.00478509 | 11.6359 | 12.0901 |

## Canonical validation metrics

| Target | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Grid MAE | Grid RMSE | Grid WAPE % | Grid sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000389131 | 0.000569427 | 28.0468 | 28.1095 | 0.00438617 | 0.00544506 | 10.8257 | 11.1216 |
| 50_bus_rural_reference_grid | 0.00033289 | 0.000513668 | 30.3936 | 32.598 | 0.00302237 | 0.00386626 | 7.83845 | 7.93417 |
| 56_bus_semi_urban_reference_grid | 0.000443694 | 0.000629384 | 25.7489 | 26.8658 | 0.00714799 | 0.00854567 | 8.97231 | 9.32289 |
| 80_bus_rural_reference_grid | 0.000209357 | 0.000369642 | 58.0977 | 55.1145 | 0.00304136 | 0.00375015 | 12.5863 | 13.3183 |
| Four-target macro | 0.000343768 | 0.00052053 | 35.5718 | 35.672 | 0.00439947 | 0.00540179 | 10.0557 | 10.4242 |

## Relative improvement vs frozen comparators

| Comparator | Audit | Validation | Validation wins |
|---|---:|---:|---:|
| scarce_local | 0.0412401 | 0.0735161 | 4 |
| fedavg | 0.0536322 | 0.044551 | 4 |
| fedprox | 0.133167 | 0.112064 | 4 |
| fedper | 0.0793111 | 0.094434 | 4 |
| btd_fl | -0.000530839 | 0.00304886 | 3 |

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
adapted_probe_states_used_for_final_initialization: false
