# Graph WaveNet baseline: 80_bus_rural_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Graph WaveNet | node_macro | 0.000180246 | 0.000332146 | 44.0113 | 44.4304 |
| Graph WaveNet | grid_aggregate | 0.00267402 | 0.00349574 | 11.0661 | 11.7267 |
| Persistence 1h | node_macro | 0.000233666 | 0.000394782 | 40.8617 | 38.1808 |
| Persistence 1h | grid_aggregate | 0.00283403 | 0.00390906 | 11.7283 | 12.6145 |

## Best epoch summary

- best_epoch: 50
- best_validation_node_macro_mae: 0.000180246
- best_train_scaled_mae: 0.156468
- epochs_run: 50

## Graph WaveNet vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | graph_wavenet_lower | graph_wavenet_lower | graph_wavenet_higher | graph_wavenet_higher |
| grid_aggregate | graph_wavenet_lower | graph_wavenet_lower | graph_wavenet_lower | graph_wavenet_lower |

topology_used: True
graph_type: physical_plus_adaptive
receptive_field: 13
electrical_edge_features_used: False
test_evaluated: False
