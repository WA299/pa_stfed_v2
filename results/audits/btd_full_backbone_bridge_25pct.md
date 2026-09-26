# BTD Full-Backbone Bridge: 25% History

Validation-only diagnostic. Canonical test data is never evaluated.
Donors are read from the completed calibration-selected transfer audit.

## Selected Donors

| Target | Selected donor |
|---|---|
| 39_bus_semi_urban_reference_grid | 56_bus_semi_urban_reference_grid |
| 50_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid |
| 56_bus_semi_urban_reference_grid | 39_bus_semi_urban_reference_grid |
| 80_bus_rural_reference_grid | 50_bus_rural_reference_grid |

## A/B/C Train Audit

| Target | From scratch MAE | Local warmstart MAE | Transfer warmstart MAE | B vs C improvement |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000374221 | 0.000372181 | 0.000350049 | 0.0594646 |
| 50_bus_rural_reference_grid | 0.000317518 | 0.000306918 | 0.000292519 | 0.0469148 |
| 56_bus_semi_urban_reference_grid | 0.000422523 | 0.000417712 | 0.000404865 | 0.0307549 |
| 80_bus_rural_reference_grid | 0.000224702 | 0.000211373 | 0.000206135 | 0.02478 |

## A/B/C Canonical Validation

| Target | From scratch MAE | Local warmstart MAE | Transfer warmstart MAE | B vs C improvement |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000420218 | 0.000416143 | 0.000390059 | 0.0626826 |
| 50_bus_rural_reference_grid | 0.000422514 | 0.000385882 | 0.000336294 | 0.128505 |
| 56_bus_semi_urban_reference_grid | 0.000468926 | 0.000470452 | 0.000446751 | 0.0503807 |
| 80_bus_rural_reference_grid | 0.000219162 | 0.000211706 | 0.000206175 | 0.0261291 |

## Four-grid Macro: Train Audit

| Variant | Node MAE | Node RMSE | Node WAPE (%) | Node sMAPE (%) | Grid MAE | Grid RMSE | Grid WAPE (%) | Grid sMAPE (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_from_scratch | 0.000334741 | 0.000499667 | 40.097 | 40.6223 | 0.00433366 | 0.0052587 | 12.7378 | 13.4355 |
| local_proxy_warmstart_full | 0.000327046 | 0.000493974 | 38.2588 | 37.4984 | 0.00420196 | 0.00511131 | 12.4804 | 13.2218 |
| selected_transfer_proxy_warmstart_full | 0.000313392 | 0.00047517 | 36.056 | 35.529 | 0.00386344 | 0.00476528 | 11.4888 | 11.9759 |
B vs C relative improvement: 0.0417488

## Four-grid Macro: Canonical Validation

| Variant | Node MAE | Node RMSE | Node WAPE (%) | Node sMAPE (%) | Grid MAE | Grid RMSE | Grid WAPE (%) | Grid sMAPE (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_from_scratch | 0.000382705 | 0.000559463 | 39.3068 | 38.2545 | 0.00515095 | 0.00619268 | 11.5127 | 12.052 |
| local_proxy_warmstart_full | 0.000371046 | 0.000549295 | 38.0639 | 36.4952 | 0.00482392 | 0.00583402 | 11.054 | 11.6755 |
| selected_transfer_proxy_warmstart_full | 0.000344819 | 0.000521678 | 31.2236 | 33.2707 | 0.00443817 | 0.00542549 | 9.97996 | 10.3877 |
B vs C relative improvement: 0.0706828

Targets improved on audit: 4
Targets improved on validation: 4
validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
