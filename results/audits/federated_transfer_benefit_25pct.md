# Directed Cross-Grid Transfer Benefit Audit (25% History)

Validation-only external diagnostic. Canonical test split is never evaluated.
Donor selection uses train-internal calibration only.

## Scarcity Metadata

| Target | Raw train | Available | Eligible | Fit | Calibration | Audit | Scaler start | Scaler end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 6132 | 1533 | 1365 | 819 | 273 | 273 | 4599 | 5586 |
| 50_bus_rural_reference_grid | 6132 | 1533 | 1365 | 819 | 273 | 273 | 4599 | 5586 |
| 56_bus_semi_urban_reference_grid | 6132 | 1533 | 1365 | 819 | 273 | 273 | 4599 | 5586 |
| 80_bus_rural_reference_grid | 6132 | 1533 | 1365 | 819 | 273 | 273 | 4599 | 5586 |

## Directed Calibration Benefit Matrix

| Target \ Donor | 39_bus_semi_urban_reference_grid | 50_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | - | 0.0296257 | 0.066049 | 0.00678424 |
| 50_bus_rural_reference_grid | 0.0547871 | - | 0.0479127 | 0.00312723 |
| 56_bus_semi_urban_reference_grid | 0.0503394 | 0.0258238 | - | 0.0106164 |
| 80_bus_rural_reference_grid | 0.0536431 | 0.0637568 | 0.0511381 | - |

## Directed Audit Benefit Matrix

| Target \ Donor | 39_bus_semi_urban_reference_grid | 50_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | - | 0.0316922 | 0.0619941 | 0.0104699 |
| 50_bus_rural_reference_grid | 0.0509021 | - | 0.0534041 | -0.00244006 |
| 56_bus_semi_urban_reference_grid | 0.0462936 | 0.0285517 | - | 0.00778477 |
| 80_bus_rural_reference_grid | 0.0703058 | 0.0741983 | 0.063304 | - |

## Directed Validation Benefit Matrix

| Target \ Donor | 39_bus_semi_urban_reference_grid | 50_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | - | 0.0317092 | 0.0614168 | 0.00889128 |
| 50_bus_rural_reference_grid | 0.123573 | - | 0.0330468 | 0.0887792 |
| 56_bus_semi_urban_reference_grid | 0.0500787 | 0.027172 | - | 0.0116212 |
| 80_bus_rural_reference_grid | 0.0656008 | 0.0647381 | 0.0675849 | - |

## Selected Donors

| Target | Donor | Calibration benefit | Audit benefit | Validation benefit | Fallback |
|---|---|---:|---:|---:|---|
| 39_bus_semi_urban_reference_grid | 56_bus_semi_urban_reference_grid | 0.066049 | 0.0619941 | 0.0614168 | False |
| 50_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | 0.0547871 | 0.0509021 | 0.123573 | False |
| 56_bus_semi_urban_reference_grid | 39_bus_semi_urban_reference_grid | 0.0503394 | 0.0462936 | 0.0500787 | False |
| 80_bus_rural_reference_grid | 50_bus_rural_reference_grid | 0.0637568 | 0.0741983 | 0.0647381 | False |

## Per-Target Train Audit Comparison

| Target | Local node MAE | Selected-policy node MAE | Relative improvement |
|---|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000373144 | 0.000350011 | 0.0619941 |
| 50_bus_rural_reference_grid | 0.000308481 | 0.000292779 | 0.0509021 |
| 56_bus_semi_urban_reference_grid | 0.000424568 | 0.000404914 | 0.0462936 |
| 80_bus_rural_reference_grid | 0.00022341 | 0.000206834 | 0.0741983 |

## Per-Target Canonical Validation Comparison

| Target | Local node MAE | Selected-policy node MAE | Relative improvement |
|---|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000415884 | 0.000390342 | 0.0614168 |
| 50_bus_rural_reference_grid | 0.000382084 | 0.000334869 | 0.123573 |
| 56_bus_semi_urban_reference_grid | 0.000470391 | 0.000446834 | 0.0500787 |
| 80_bus_rural_reference_grid | 0.000220344 | 0.00020608 | 0.0647381 |

## Train Audit Macro Summary

| Policy | Node MAE | Node RMSE | Node WAPE (%) | Node sMAPE (%) | Grid MAE | Grid RMSE | Grid WAPE (%) | Grid sMAPE (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| target_local_proxy | 0.000332401 | 0.000499436 | 39.4528 | 39.7685 | 0.00464155 | 0.00554505 | 13.4967 | 14.288 |
| selected_transfer_policy | 0.000313634 | 0.000475131 | 36.5327 | 35.8647 | 0.00387435 | 0.00479652 | 11.4917 | 11.9689 |
Relative node-MAE improvement: 0.0564578

## Canonical Validation Macro Summary

| Policy | Node MAE | Node RMSE | Node WAPE (%) | Node sMAPE (%) | Grid MAE | Grid RMSE | Grid WAPE (%) | Grid sMAPE (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| target_local_proxy | 0.000372176 | 0.000552437 | 37.2675 | 37.9036 | 0.00547209 | 0.00648909 | 12.1941 | 12.9042 |
| selected_transfer_policy | 0.000344531 | 0.00052113 | 33.8317 | 33.9807 | 0.00441824 | 0.00542338 | 9.84381 | 10.2196 |
Relative node-MAE improvement: 0.0742787
