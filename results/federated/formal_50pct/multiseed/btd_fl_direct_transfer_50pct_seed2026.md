# BTD-FL Direct Temporal Transfer: 25% History

Target-conditioned donor adaptations are calibration-only benefit probes. The final transferred initialization is the raw selected donor temporal state.
Canonical validation is evaluation-only. Canonical test is never evaluated.

## Selected donors and benefits

| Target | Selected donor | Benefit | Fallback | Matches frozen formal run |
|---|---|---:|---|---|
| 39_bus_semi_urban_reference_grid | 56_bus_semi_urban_reference_grid | 0.057521 | False | True |
| 50_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 0.0753731 | False | False |
| 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid | 0.0433475 | False | False |
| 80_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 0.0331383 | False | False |

## Current-run calibration benefit matrix

| Target \ Donor | 39_bus_semi_urban_reference_grid | 50_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | - | 0.0542022 | 0.057521 | 0.0500816 |
| 50_bus_rural_reference_grid | 0.0732112 | - | 0.0753731 | 0.0556262 |
| 56_bus_semi_urban_reference_grid | 0.0420599 | 0.043266 | - | 0.0433475 |
| 80_bus_rural_reference_grid | 0.0219675 | 0.0199875 | 0.0331383 | - |

## Train audit metrics

| Target | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Grid MAE | Grid RMSE | Grid WAPE % | Grid sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000347307 | 0.000522513 | 32.4664 | 31.8427 | 0.00365389 | 0.00446819 | 11.6266 | 11.8633 |
| 50_bus_rural_reference_grid | 0.000274816 | 0.000411469 | 37.5417 | 36.855 | 0.00238678 | 0.00308755 | 8.08556 | 8.14085 |
| 56_bus_semi_urban_reference_grid | 0.000402997 | 0.00058093 | 33.3159 | 33.5808 | 0.0058223 | 0.0070436 | 10.0658 | 10.351 |
| 80_bus_rural_reference_grid | 0.000220926 | 0.000390655 | 41.8476 | 40.9639 | 0.00288175 | 0.00376436 | 14.2677 | 15.315 |
| Four-target macro | 0.000311512 | 0.000476392 | 36.2929 | 35.8106 | 0.00368618 | 0.00459092 | 11.0114 | 11.4176 |

## Canonical validation metrics

| Target | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Grid MAE | Grid RMSE | Grid WAPE % | Grid sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000383605 | 0.000560525 | 27.6877 | 27.7668 | 0.00413245 | 0.00512926 | 10.1995 | 10.5493 |
| 50_bus_rural_reference_grid | 0.000321318 | 0.000506475 | 29.5974 | 31.4968 | 0.00315397 | 0.00400979 | 8.17976 | 8.25159 |
| 56_bus_semi_urban_reference_grid | 0.000434715 | 0.000616605 | 25.3901 | 26.57 | 0.00660559 | 0.00800646 | 8.29149 | 8.63755 |
| 80_bus_rural_reference_grid | 0.000202065 | 0.000359799 | 46.7667 | 53.1716 | 0.00270794 | 0.00356608 | 11.2065 | 11.8472 |
| Four-target macro | 0.000335426 | 0.000510851 | 32.3605 | 34.7513 | 0.00414999 | 0.0051779 | 9.4693 | 9.8214 |

## Relative improvement vs frozen comparators

| Comparator | Audit | Validation | Validation wins |
|---|---:|---:|---:|

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
adapted_probe_states_used_for_final_initialization: false
