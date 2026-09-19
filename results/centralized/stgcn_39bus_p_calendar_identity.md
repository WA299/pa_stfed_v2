# Fixed-topology STGCN baseline: 39_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| STGCN | node_macro | 0.0004246 | 0.000618634 | 30.8145 | 30.9298 |
| STGCN | grid_aggregate | 0.00521083 | 0.00621725 | 12.861 | 13.4229 |
| Persistence 1h | node_macro | 0.000490575 | 0.000693005 | 36.1604 | 34.5479 |
| Persistence 1h | grid_aggregate | 0.00345857 | 0.00439136 | 8.53624 | 8.57834 |

## Best epoch summary

- best_epoch: 46
- best_validation_node_macro_mae: 0.0004246
- best_train_scaled_mae: 0.329604
- epochs_run: 50

## STGCN vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | stgcn_lower | stgcn_lower | stgcn_lower | stgcn_lower |
| grid_aggregate | stgcn_higher | stgcn_higher | stgcn_higher | stgcn_higher |

adjacency_mode: identity
topology_used: False
adjacency_type: identity
electrical_edge_features_used: False
test_evaluated: False
