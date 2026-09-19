# GRU + GAT spatiotemporal baseline: 39_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| GRU + GAT | node_macro | 0.000400356 | 0.000579311 | 29.3394 | 29.4149 |
| GRU + GAT | grid_aggregate | 0.00414788 | 0.00514938 | 10.2375 | 10.5281 |
| Persistence 1h | node_macro | 0.000490575 | 0.000693005 | 36.1604 | 34.5479 |
| Persistence 1h | grid_aggregate | 0.00345857 | 0.00439136 | 8.53624 | 8.57834 |

## Best epoch summary

- best_epoch: 50
- best_validation_node_macro_mae: 0.000400356
- best_train_scaled_mae: 0.311193
- epochs_run: 50

## GRU + GAT vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | gat_st_lower | gat_st_lower | gat_st_lower | gat_st_lower |
| grid_aggregate | gat_st_higher | gat_st_higher | gat_st_higher | gat_st_higher |

temporal_encoder: shared_gru
spatial_attention: gat
topology_used: True
adjacency_role: attention_mask
electrical_edge_features_used: False
electrical_distance_used: False
test_evaluated: False
