# Graph WaveNet baseline: 50_bus_rural_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Graph WaveNet | node_macro | 0.000311158 | 0.000501112 | 29.9662 | 30.2866 |
| Graph WaveNet | grid_aggregate | 0.00296255 | 0.00383741 | 7.68333 | 7.69251 |
| Persistence 1h | node_macro | 0.000374557 | 0.000564059 | 34.5653 | 33.336 |
| Persistence 1h | grid_aggregate | 0.00253653 | 0.00333136 | 6.57844 | 6.57141 |

## Best epoch summary

- best_epoch: 47
- best_validation_node_macro_mae: 0.000311158
- best_train_scaled_mae: 0.103885
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
