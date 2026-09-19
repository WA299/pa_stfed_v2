# Shared GRU + global spatial attention: 80_bus_rural_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Global attention | node_macro | 0.000199543 | 0.000352883 | 63.8517 | 60.1551 |
| Global attention | grid_aggregate | 0.00310136 | 0.00378507 | 12.8346 | 13.6997 |
| Persistence 1h | node_macro | 0.000233666 | 0.000394782 | 40.8617 | 38.1808 |
| Persistence 1h | grid_aggregate | 0.00283403 | 0.00390906 | 11.7283 | 12.6145 |

## Best epoch summary

- best_epoch: 50
- best_validation_node_macro_mae: 0.000199543
- best_train_scaled_mae: 0.177242
- epochs_run: 50

## Global attention vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | global_attention_lower | global_attention_lower | global_attention_higher | global_attention_higher |
| grid_aggregate | global_attention_higher | global_attention_lower | global_attention_higher | global_attention_higher |

temporal_encoder: shared_gru
spatial_attention: global_scaled_dot_product
topology_used: True
attention_scope: physical_hop2
attention_source: all_nodes
electrical_distance_used: False
electrical_edge_features_used: False
test_evaluated: False
