# BTD-FL Direct Temporal Transfer: 25% History

Target-conditioned donor adaptations are calibration-only benefit probes. The final transferred initialization is the raw selected donor temporal state.
Canonical validation is evaluation-only. Canonical test is never evaluated.

## Selected donors and benefits

| Target | Selected donor | Benefit | Fallback | Matches frozen formal run |
|---|---|---:|---|---|
| 39_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid | 0.0565343 | False | False |
| 50_bus_rural_reference_grid | 80_bus_rural_reference_grid | 0.110877 | False | False |
| 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid | 0.0636581 | False | False |
| 80_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | 0.0240401 | False | False |

## Current-run calibration benefit matrix

| Target \ Donor | 39_bus_semi_urban_reference_grid | 50_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | - | 0.0294749 | 0.0465817 | 0.0565343 |
| 50_bus_rural_reference_grid | 0.0491918 | - | 0.0672667 | 0.110877 |
| 56_bus_semi_urban_reference_grid | 0.0513807 | 0.0261959 | - | 0.0636581 |
| 80_bus_rural_reference_grid | 0.0240401 | -0.00428125 | 0.0227542 | - |

## Train audit metrics

| Target | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Grid MAE | Grid RMSE | Grid WAPE % | Grid sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000346062 | 0.000524252 | 32.2881 | 31.7065 | 0.0037392 | 0.00453843 | 11.898 | 12.1459 |
| 50_bus_rural_reference_grid | 0.00026596 | 0.000402424 | 34.6096 | 35.0828 | 0.00221804 | 0.00292218 | 7.51395 | 7.51519 |
| 56_bus_semi_urban_reference_grid | 0.000397366 | 0.000578944 | 32.6916 | 32.8703 | 0.00588326 | 0.00713787 | 10.1712 | 10.3794 |
| 80_bus_rural_reference_grid | 0.00022001 | 0.000393147 | 42.0948 | 41.5945 | 0.00308124 | 0.00389135 | 15.2554 | 16.3841 |
| Four-target macro | 0.000307349 | 0.000474692 | 35.421 | 35.3135 | 0.00373044 | 0.00462246 | 11.2096 | 11.6062 |

## Canonical validation metrics

| Target | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Grid MAE | Grid RMSE | Grid WAPE % | Grid sMAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000383606 | 0.000563392 | 27.6595 | 27.7584 | 0.0041729 | 0.00517101 | 10.2993 | 10.6006 |
| 50_bus_rural_reference_grid | 0.00031369 | 0.000492431 | 28.3031 | 30.1863 | 0.00268302 | 0.00352191 | 6.95837 | 6.96304 |
| 56_bus_semi_urban_reference_grid | 0.000431182 | 0.000613911 | 25.0706 | 26.2344 | 0.0066594 | 0.00809014 | 8.35903 | 8.64413 |
| 80_bus_rural_reference_grid | 0.000204777 | 0.000365852 | 46.2777 | 52.8082 | 0.00286306 | 0.00367517 | 11.8485 | 12.5132 |
| Four-target macro | 0.000333314 | 0.000508897 | 31.8277 | 34.2468 | 0.0040946 | 0.00511456 | 9.36629 | 9.68024 |

## Relative improvement vs frozen comparators

| Comparator | Audit | Validation | Validation wins |
|---|---:|---:|---:|

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
adapted_probe_states_used_for_final_initialization: false
