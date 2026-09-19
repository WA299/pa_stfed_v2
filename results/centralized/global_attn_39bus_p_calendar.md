# Shared GRU + global spatial attention: 39_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Global attention | node_macro | 0.000396407 | 0.000579967 | 28.9954 | 29.0549 |
| Global attention | grid_aggregate | 0.00475003 | 0.00569698 | 11.7237 | 12.2228 |
| Persistence 1h | node_macro | 0.000490575 | 0.000693005 | 36.1604 | 34.5479 |
| Persistence 1h | grid_aggregate | 0.00345857 | 0.00439136 | 8.53624 | 8.57834 |

## Best epoch summary

- best_epoch: 50
- best_validation_node_macro_mae: 0.000396407
- best_train_scaled_mae: 0.306884
- epochs_run: 50

## Global attention vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | global_attention_lower | global_attention_lower | global_attention_lower | global_attention_lower |
| grid_aggregate | global_attention_higher | global_attention_higher | global_attention_higher | global_attention_higher |

temporal_encoder: shared_gru
spatial_attention: global_scaled_dot_product
topology_used: False
attention_scope: all_nodes
electrical_distance_used: False
electrical_edge_features_used: False
test_evaluated: False
