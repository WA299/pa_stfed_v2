# Fixed-topology STGCN baseline: 39_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| STGCN | node_macro | 0.000431284 | 0.000624622 | 31.4062 | 31.7401 |
| STGCN | grid_aggregate | 0.00546212 | 0.00649715 | 13.4813 | 14.0009 |
| Persistence 1h | node_macro | 0.000490575 | 0.000693005 | 36.1604 | 34.5479 |
| Persistence 1h | grid_aggregate | 0.00345857 | 0.00439136 | 8.53624 | 8.57834 |

## Best epoch summary

- best_epoch: 43
- best_validation_node_macro_mae: 0.000431284
- best_train_scaled_mae: 0.335178
- epochs_run: 50

## STGCN vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | stgcn_lower | stgcn_lower | stgcn_lower | stgcn_lower |
| grid_aggregate | stgcn_higher | stgcn_higher | stgcn_higher | stgcn_higher |

topology_used: True
adjacency_type: binary_physical_normalized
electrical_edge_features_used: False
test_evaluated: False
