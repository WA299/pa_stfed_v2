# Shared GRU + global spatial attention: 80_bus_rural_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Global attention | node_macro | 0.000201323 | 0.000359613 | 46.5532 | 52.0815 |
| Global attention | grid_aggregate | 0.00293017 | 0.00366873 | 12.1262 | 12.7944 |
| Persistence 1h | node_macro | 0.000233666 | 0.000394782 | 40.8617 | 38.1808 |
| Persistence 1h | grid_aggregate | 0.00283403 | 0.00390906 | 11.7283 | 12.6145 |

## Best epoch summary

- best_epoch: 49
- best_validation_node_macro_mae: 0.000201323
- best_train_scaled_mae: 0.181622
- epochs_run: 50

## Global attention vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | global_attention_lower | global_attention_lower | global_attention_higher | global_attention_higher |
| grid_aggregate | global_attention_higher | global_attention_lower | global_attention_higher | global_attention_higher |

temporal_encoder: shared_gru
spatial_attention: global_scaled_dot_product
topology_used: False
attention_scope: all_nodes
attention_source: load_buses_only
electrical_distance_used: False
electrical_edge_features_used: False
test_evaluated: False
