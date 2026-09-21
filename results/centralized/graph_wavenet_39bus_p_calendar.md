# Graph WaveNet baseline: 39_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Graph WaveNet | node_macro | 0.000387925 | 0.000574635 | 28.1088 | 28.4199 |
| Graph WaveNet | grid_aggregate | 0.00503781 | 0.00619965 | 12.434 | 12.5338 |
| Persistence 1h | node_macro | 0.000490575 | 0.000693005 | 36.1604 | 34.5479 |
| Persistence 1h | grid_aggregate | 0.00345857 | 0.00439136 | 8.53624 | 8.57834 |

## Best epoch summary

- best_epoch: 49
- best_validation_node_macro_mae: 0.000387925
- best_train_scaled_mae: 0.289892
- epochs_run: 50

## Graph WaveNet vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | graph_wavenet_lower | graph_wavenet_lower | graph_wavenet_lower | graph_wavenet_lower |
| grid_aggregate | graph_wavenet_higher | graph_wavenet_higher | graph_wavenet_higher | graph_wavenet_higher |

topology_used: True
graph_type: physical_plus_adaptive
receptive_field: 13
electrical_edge_features_used: False
test_evaluated: False
