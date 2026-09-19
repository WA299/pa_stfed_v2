# GRU + GAT spatiotemporal baseline: 80_bus_rural_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| GRU + GAT | node_macro | 0.000198698 | 0.000353191 | 45.3866 | 52.8399 |
| GRU + GAT | grid_aggregate | 0.00291627 | 0.0036652 | 12.0687 | 12.8114 |
| Persistence 1h | node_macro | 0.000233666 | 0.000394782 | 40.8617 | 38.1808 |
| Persistence 1h | grid_aggregate | 0.00283403 | 0.00390906 | 11.7283 | 12.6145 |

## Best epoch summary

- best_epoch: 47
- best_validation_node_macro_mae: 0.000198698
- best_train_scaled_mae: 0.180619
- epochs_run: 50

## GRU + GAT vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | gat_st_lower | gat_st_lower | gat_st_higher | gat_st_higher |
| grid_aggregate | gat_st_higher | gat_st_lower | gat_st_higher | gat_st_higher |

temporal_encoder: shared_gru
spatial_attention: gat
topology_used: True
adjacency_role: attention_mask
electrical_edge_features_used: False
electrical_distance_used: False
test_evaluated: False
