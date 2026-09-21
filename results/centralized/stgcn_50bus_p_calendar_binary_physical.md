# Fixed-topology STGCN baseline: 50_bus_rural_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| STGCN | node_macro | 0.00033736 | 0.00053105 | 31.4531 | 32.5715 |
| STGCN | grid_aggregate | 0.00290566 | 0.00375211 | 7.53578 | 7.66453 |
| Persistence 1h | node_macro | 0.000374557 | 0.000564059 | 34.5653 | 33.336 |
| Persistence 1h | grid_aggregate | 0.00253653 | 0.00333136 | 6.57844 | 6.57141 |

## Best epoch summary

- best_epoch: 48
- best_validation_node_macro_mae: 0.00033736
- best_train_scaled_mae: 0.108993
- epochs_run: 50

## STGCN vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | stgcn_lower | stgcn_lower | stgcn_lower | stgcn_lower |
| grid_aggregate | stgcn_higher | stgcn_higher | stgcn_higher | stgcn_higher |

adjacency_mode: binary_physical
topology_used: True
adjacency_type: binary_physical_normalized
electrical_edge_features_used: False
test_evaluated: False
