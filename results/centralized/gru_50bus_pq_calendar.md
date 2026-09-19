# Centralized GRU: 50_bus_rural_reference_grid / pq_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| SharedNodeGRU | node_macro | 0.000333585 | 0.000514073 | 32.0858 | 34.756 |
| SharedNodeGRU | grid_aggregate | 0.00275966 | 0.00357729 | 7.15712 | 7.19237 |
| Persistence 1h | node_macro | 0.000374557 | 0.000564059 | 34.5653 | 33.336 |
| Persistence 1h | grid_aggregate | 0.00253653 | 0.00333136 | 6.57844 | 6.57141 |

## Best epoch summary

- best_epoch: 50
- best_validation_node_macro_mae: 0.000333585
- best_train_scaled_mae: 0.105083
- epochs_run: 50

## GRU vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | gru_lower | gru_lower | gru_lower | gru_higher |
| grid_aggregate | gru_higher | gru_higher | gru_higher | gru_higher |

test_evaluated: False
