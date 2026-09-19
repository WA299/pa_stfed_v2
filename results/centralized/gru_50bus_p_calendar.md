# Centralized GRU: 50_bus_rural_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| SharedNodeGRU | node_macro | 0.000335095 | 0.000509787 | 31.9404 | 33.9837 |
| SharedNodeGRU | grid_aggregate | 0.00265037 | 0.00345746 | 6.87369 | 6.89188 |
| Persistence 1h | node_macro | 0.000374557 | 0.000564059 | 34.5653 | 33.336 |
| Persistence 1h | grid_aggregate | 0.00253653 | 0.00333136 | 6.57844 | 6.57141 |

## Best epoch summary

- best_epoch: 50
- best_validation_node_macro_mae: 0.000335095
- best_train_scaled_mae: 0.10676
- epochs_run: 50

## GRU vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | gru_lower | gru_lower | gru_lower | gru_higher |
| grid_aggregate | gru_higher | gru_higher | gru_higher | gru_higher |

test_evaluated: False
