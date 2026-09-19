# Electrical-distance-biased attention: 80_bus_rural_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Electrical attention | node_macro | 0.000205745 | 0.000364038 | 76.9964 | 68.5808 |
| Electrical attention | grid_aggregate | 0.0031296 | 0.00381167 | 12.9515 | 13.741 |
| Persistence 1h | node_macro | 0.000233666 | 0.000394782 | 40.8617 | 38.1808 |
| Persistence 1h | grid_aggregate | 0.00283403 | 0.00390906 | 11.7283 | 12.6145 |

## Best epoch summary

- best_epoch: 48
- best_validation_node_macro_mae: 0.000205745
- best_train_scaled_mae: 0.181935
- epochs_run: 50

## Electrical attention vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | electrical_attention_lower | electrical_attention_lower | electrical_attention_higher | electrical_attention_higher |
| grid_aggregate | electrical_attention_higher | electrical_attention_lower | electrical_attention_higher | electrical_attention_higher |

temporal_encoder: shared_gru
spatial_attention: global_scaled_dot_product_with_electrical_bias
attention_scope: all_nodes
topology_used: True
electrical_distance_used: True
electrical_distance_type: impedance_abs_distance
distance_normalization: median_positive_offdiagonal_then_log1p
lambda_learnable: True
lambda_initial: 1
lambda_final: 0.846284
electrical_edge_features_used: False
test_evaluated: False
