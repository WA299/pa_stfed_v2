# BTD-FL Direct Temporal Transfer: 25% History

Target-conditioned donor adaptations are calibration-only benefit probes. The final transferred initialization is the raw selected donor temporal state.
Canonical validation is evaluation-only. Canonical test is never evaluated.

## Selected donors and benefits

| Target | Selected donor | Benefit | Fallback | Matches frozen formal run |
|---|---|---:|---|---|
| 39_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid | 0.0664168 | False | False |
| 50_bus_rural_reference_grid | 80_bus_rural_reference_grid | 0.103279 | False | False |
| 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid | 0.0586294 | False | False |
| 80_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | 0.0857323 | False | False |

## Current-run calibration benefit matrix

| Target \ Donor | 39_bus_semi_urban_reference_grid | 50_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | - | 0.025511 | 0.0651944 | 0.0664168 |
| 50_bus_rural_reference_grid | 0.0497895 | - | 0.0599463 | 0.103279 |
| 56_bus_semi_urban_reference_grid | 0.0471644 | 0.0231153 | - | 0.0586294 |
| 80_bus_rural_reference_grid | 0.0857323 | 0.0700565 | 0.0559643 | - |

## Train audit metrics

| Target | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Grid MAE | Grid RMSE | Grid WAPE % | Grid sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000347503 | 0.000519429 | 32.2486 | 31.9931 | 0.00361153 | 0.00444009 | 11.4143 | 11.5384 |
| 50_bus_rural_reference_grid | 0.000279966 | 0.000430204 | 34.5608 | 35.4696 | 0.00257086 | 0.00340327 | 8.52833 | 8.60397 |
| 56_bus_semi_urban_reference_grid | 0.000405449 | 0.000583895 | 32.8129 | 33.1303 | 0.00574747 | 0.00711959 | 9.76005 | 9.92261 |
| 80_bus_rural_reference_grid | 0.000204736 | 0.000358443 | 41.6761 | 39.8982 | 0.00290318 | 0.00371635 | 15.1539 | 16.3609 |
| Four-target macro | 0.000309413 | 0.000472993 | 35.3246 | 35.1228 | 0.00370826 | 0.00466983 | 11.2141 | 11.6065 |

## Canonical validation metrics

| Target | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Grid MAE | Grid RMSE | Grid WAPE % | Grid sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000394724 | 0.000574938 | 28.3312 | 28.2629 | 0.0041371 | 0.00523446 | 10.2109 | 10.4508 |
| 50_bus_rural_reference_grid | 0.000329028 | 0.000514096 | 28.6233 | 32.0022 | 0.00320321 | 0.00403433 | 8.30747 | 8.4834 |
| 56_bus_semi_urban_reference_grid | 0.000450429 | 0.000633664 | 25.9071 | 26.985 | 0.00659419 | 0.0080891 | 8.27718 | 8.64794 |
| 80_bus_rural_reference_grid | 0.00020277 | 0.000359456 | 46.7582 | 48.7024 | 0.00285388 | 0.00364173 | 11.8105 | 12.5219 |
| Four-target macro | 0.000344238 | 0.000520538 | 32.405 | 33.9881 | 0.0041971 | 0.00524991 | 9.65151 | 10.026 |

## Relative improvement vs frozen comparators

| Comparator | Audit | Validation | Validation wins |
|---|---:|---:|---:|

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
adapted_probe_states_used_for_final_initialization: false
