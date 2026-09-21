# Graph WaveNet baseline: 56_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Graph WaveNet | node_macro | 0.000452294 | 0.000641542 | 26.4818 | 27.5905 |
| Graph WaveNet | grid_aggregate | 0.00692373 | 0.00843414 | 8.69082 | 8.86571 |
| Persistence 1h | node_macro | 0.000537618 | 0.00073951 | 31.5906 | 31.9867 |
| Persistence 1h | grid_aggregate | 0.00523067 | 0.00660912 | 6.56565 | 6.68602 |

## Best epoch summary

- best_epoch: 23
- best_validation_node_macro_mae: 0.000452294
- best_train_scaled_mae: 0.297468
- epochs_run: 31

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
