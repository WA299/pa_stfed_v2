# Electrical-distance-biased attention: 39_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Electrical attention | node_macro | 0.000395663 | 0.000578387 | 28.935 | 28.988 |
| Electrical attention | grid_aggregate | 0.00460192 | 0.00555563 | 11.3582 | 11.7929 |
| Persistence 1h | node_macro | 0.000490575 | 0.000693005 | 36.1604 | 34.5479 |
| Persistence 1h | grid_aggregate | 0.00345857 | 0.00439136 | 8.53624 | 8.57834 |

## Best epoch summary

- best_epoch: 50
- best_validation_node_macro_mae: 0.000395663
- best_train_scaled_mae: 0.306718
- epochs_run: 50

## Electrical attention vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | electrical_attention_lower | electrical_attention_lower | electrical_attention_lower | electrical_attention_lower |
| grid_aggregate | electrical_attention_higher | electrical_attention_higher | electrical_attention_higher | electrical_attention_higher |

temporal_encoder: shared_gru
spatial_attention: global_scaled_dot_product_with_electrical_bias
attention_scope: all_nodes
topology_used: True
electrical_distance_used: True
electrical_distance_type: impedance_abs_distance
distance_normalization: median_positive_offdiagonal_then_log1p
lambda_learnable: True
lambda_initial: 1
lambda_final: 0.312519
electrical_edge_features_used: False
test_evaluated: False
