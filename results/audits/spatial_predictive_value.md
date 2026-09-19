# V2 Spatial Predictive Value Audit

time_scope: `formal_train_only`; audit_split: `chronological_80_20_within_train`.

| Grid | Loads | Fit | Holdout | Model | MAE | RMSE | WAPE (%) | ΔMAE (%) | ΔRMSE (%) | ΔWAPE (%) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 28 | 4771 | 1193 | self_only | 0.000405051 | 0.000549197 | 39.0284 | 0 | 0 | 0 |
| 39_bus_semi_urban_reference_grid | 28 | 4771 | 1193 | grid_context | 0.000386854 | 0.000536199 | 37.1054 | -4.492 | -2.367 | -4.927 |
| 39_bus_semi_urban_reference_grid | 28 | 4771 | 1193 | electrical_neighbors | 0.000391617 | 0.000540275 | 37.6522 | -3.317 | -1.625 | -3.526 |
| 39_bus_semi_urban_reference_grid | 28 | 4771 | 1193 | statistical_neighbors | 0.000384255 | 0.000536853 | 36.861 | -5.134 | -2.248 | -5.553 |
| 50_bus_rural_reference_grid | 21 | 4771 | 1193 | self_only | 0.000319449 | 0.000441705 | 40.9609 | 0 | 0 | 0 |
| 50_bus_rural_reference_grid | 21 | 4771 | 1193 | grid_context | 0.000310797 | 0.000434555 | 39.0215 | -2.709 | -1.619 | -4.735 |
| 50_bus_rural_reference_grid | 21 | 4771 | 1193 | electrical_neighbors | 0.000312803 | 0.000437277 | 39.5736 | -2.081 | -1.002 | -3.387 |
| 50_bus_rural_reference_grid | 21 | 4771 | 1193 | statistical_neighbors | 0.000318314 | 0.000445444 | 39.6834 | -0.3554 | 0.8465 | -3.119 |
| 56_bus_semi_urban_reference_grid | 44 | 4771 | 1193 | self_only | 0.000443754 | 0.000590122 | 39.3231 | 0 | 0 | 0 |
| 56_bus_semi_urban_reference_grid | 44 | 4771 | 1193 | grid_context | 0.000422491 | 0.000572935 | 37.447 | -4.792 | -2.912 | -4.771 |
| 56_bus_semi_urban_reference_grid | 44 | 4771 | 1193 | electrical_neighbors | 0.000429823 | 0.000580584 | 38.0672 | -3.139 | -1.616 | -3.194 |
| 56_bus_semi_urban_reference_grid | 44 | 4771 | 1193 | statistical_neighbors | 0.000425219 | 0.00058102 | 37.6679 | -4.177 | -1.542 | -4.209 |
| 80_bus_rural_reference_grid | 32 | 4771 | 1193 | self_only | 0.000300598 | 0.000441841 | 52.6137 | 0 | 0 | 0 |
| 80_bus_rural_reference_grid | 32 | 4771 | 1193 | grid_context | 0.000297626 | 0.000439434 | 52.0188 | -0.9885 | -0.5448 | -1.131 |
| 80_bus_rural_reference_grid | 32 | 4771 | 1193 | electrical_neighbors | 0.000295616 | 0.000438478 | 51.1844 | -1.657 | -0.7611 | -2.717 |
| 80_bus_rural_reference_grid | 32 | 4771 | 1193 | statistical_neighbors | 0.000293461 | 0.00043524 | 50.16 | -2.374 | -1.494 | -4.664 |

Neighbor selection uses audit_fit only; formal validation/test are not evaluated.
