# Fixed-topology STGCN baseline: 80_bus_rural_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| STGCN | node_macro | 0.00020087 | 0.000358977 | 46.3623 | 51.3937 |
| STGCN | grid_aggregate | 0.00316398 | 0.00394009 | 13.0938 | 14.2072 |
| Persistence 1h | node_macro | 0.000233666 | 0.000394782 | 40.8617 | 38.1808 |
| Persistence 1h | grid_aggregate | 0.00283403 | 0.00390906 | 11.7283 | 12.6145 |

## Best epoch summary

- best_epoch: 30
- best_validation_node_macro_mae: 0.00020087
- best_train_scaled_mae: 0.183873
- epochs_run: 38

## STGCN vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | stgcn_lower | stgcn_lower | stgcn_higher | stgcn_higher |
| grid_aggregate | stgcn_higher | stgcn_higher | stgcn_higher | stgcn_higher |

adjacency_mode: binary_physical
topology_used: True
adjacency_type: binary_physical_normalized
electrical_edge_features_used: False
test_evaluated: False
