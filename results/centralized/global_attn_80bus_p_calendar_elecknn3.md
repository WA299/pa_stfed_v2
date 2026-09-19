# Shared GRU + global spatial attention: 80_bus_rural_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Global attention | node_macro | 0.000201143 | 0.000360002 | 57.2816 | 60.633 |
| Global attention | grid_aggregate | 0.00300438 | 0.00370542 | 12.4333 | 13.2005 |
| Persistence 1h | node_macro | 0.000233666 | 0.000394782 | 40.8617 | 38.1808 |
| Persistence 1h | grid_aggregate | 0.00283403 | 0.00390906 | 11.7283 | 12.6145 |

## Best epoch summary

- best_epoch: 50
- best_validation_node_macro_mae: 0.000201143
- best_train_scaled_mae: 0.181264
- epochs_run: 50

## Global attention vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | global_attention_lower | global_attention_lower | global_attention_higher | global_attention_higher |
| grid_aggregate | global_attention_higher | global_attention_lower | global_attention_higher | global_attention_higher |

temporal_encoder: shared_gru
spatial_attention: global_scaled_dot_product
topology_used: True
attention_scope: electrical_load_knn3
attention_source: candidate_mask_defined
electrical_distance_used: True
electrical_distance_role: candidate_selection_only
electrical_bias_used: False
knn_k: 3
electrical_edge_features_used: False
test_evaluated: False
