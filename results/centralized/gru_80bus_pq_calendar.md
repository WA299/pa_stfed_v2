# Centralized GRU: 80_bus_rural_reference_grid / pq_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| SharedNodeGRU | node_macro | 0.000198405 | 0.000356595 | 37.9789 | 44.8509 |
| SharedNodeGRU | grid_aggregate | 0.00286168 | 0.00360597 | 11.8428 | 12.5228 |
| Persistence 1h | node_macro | 0.000233666 | 0.000394782 | 40.8617 | 38.1808 |
| Persistence 1h | grid_aggregate | 0.00283403 | 0.00390906 | 11.7283 | 12.6145 |

## Best epoch summary

- best_epoch: 46
- best_validation_node_macro_mae: 0.000198405
- best_train_scaled_mae: 0.177836
- epochs_run: 50

## GRU vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | gru_lower | gru_lower | gru_lower | gru_higher |
| grid_aggregate | gru_higher | gru_lower | gru_higher | gru_lower |

test_evaluated: False
