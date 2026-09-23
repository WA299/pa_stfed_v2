# Strong-Temporal Conditional Predictive-Utility Audit

Train-only deterministic Ridge audit; validation and test labels are not accessed.

| Grid | Model | Node MAE | Node RMSE | Node WAPE (%) | Node sMAPE (%) | Aggregate MAE | Aggregate RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | strong_self | 0.000383538 | 0.000528589 | 36.7608 | 36.566 | 0.00234146 | 0.00301594 |
| 39_bus_semi_urban_reference_grid | conditional_utility_top3 | 0.000381835 | 0.000529118 | 36.5987 | 36.4903 | 0.00234704 | 0.00304911 |
| 39_bus_semi_urban_reference_grid | conditional_utility_uniform_mean_context | 0.000381508 | 0.000528659 | 36.5648 | 36.4438 | 0.00234643 | 0.00304751 |
| 50_bus_rural_reference_grid | strong_self | 0.000300306 | 0.000420164 | 36.6763 | 39.3364 | 0.00189605 | 0.00241939 |
| 50_bus_rural_reference_grid | conditional_utility_top3 | 0.000302034 | 0.000422151 | 37.0107 | 39.7284 | 0.0019148 | 0.00245331 |
| 50_bus_rural_reference_grid | conditional_utility_uniform_mean_context | 0.000299989 | 0.000420515 | 36.594 | 39.3703 | 0.00190805 | 0.00243504 |
| 56_bus_semi_urban_reference_grid | strong_self | 0.000418379 | 0.000564701 | 36.8736 | 38.1188 | 0.00336681 | 0.00425861 |
| 56_bus_semi_urban_reference_grid | conditional_utility_top3 | 0.000413718 | 0.000566757 | 36.438 | 38.1037 | 0.00364821 | 0.0046888 |
| 56_bus_semi_urban_reference_grid | conditional_utility_uniform_mean_context | 0.000414179 | 0.000565631 | 36.5118 | 38.0382 | 0.00354391 | 0.0045314 |
| 80_bus_rural_reference_grid | strong_self | 0.000308933 | 0.000449924 | 52.5615 | 54.8612 | 0.00358411 | 0.00453144 |
| 80_bus_rural_reference_grid | conditional_utility_top3 | 0.000307954 | 0.000449651 | 52.5335 | 55.1054 | 0.00358566 | 0.00450994 |
| 80_bus_rural_reference_grid | conditional_utility_uniform_mean_context | 0.000307736 | 0.00044964 | 52.323 | 55.0986 | 0.00358946 | 0.00451572 |

## Four-grid unweighted macro

| Model | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
|---|---|---:|---:|---:|---:|
| strong_self | node_macro | 0.000352789 | 0.000490844 | 40.718 | 42.2206 |
| strong_self | grid_aggregate | 0.00279711 | 0.00355634 | 9.1353 | 9.49196 |
| conditional_utility_top3 | node_macro | 0.000351385 | 0.000491919 | 40.6452 | 42.357 |
| conditional_utility_top3 | grid_aggregate | 0.00287393 | 0.00367529 | 9.2863 | 9.66479 |
| conditional_utility_uniform_mean_context | node_macro | 0.000350853 | 0.000491111 | 40.4984 | 42.2377 |
| conditional_utility_uniform_mean_context | grid_aggregate | 0.00284696 | 0.00363242 | 9.23661 | 9.61514 |
