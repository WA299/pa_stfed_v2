# Centralized GRU: 80_bus_rural_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| SharedNodeGRU | node_macro | 0.000199762 | 0.000358149 | 42.4904 | 49.4964 |
| SharedNodeGRU | grid_aggregate | 0.00287526 | 0.00363064 | 11.8989 | 12.5906 |
| Persistence 1h | node_macro | 0.000233666 | 0.000394782 | 40.8617 | 38.1808 |
| Persistence 1h | grid_aggregate | 0.00283403 | 0.00390906 | 11.7283 | 12.6145 |

## Best epoch summary

- best_epoch: 49
- best_validation_node_macro_mae: 0.000199762
- best_train_scaled_mae: 0.180192
- epochs_run: 50

## GRU vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | gru_lower | gru_lower | gru_higher | gru_higher |
| grid_aggregate | gru_higher | gru_lower | gru_higher | gru_lower |

test_evaluated: False
