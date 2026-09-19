# Centralized GRU Smoke Baseline: 39-bus

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| SharedNodeGRU | node_macro | 0.000403975 | 0.000583605 | 29.5967 | 29.6896 |
| SharedNodeGRU | grid_aggregate | 0.00419447 | 0.00516585 | 10.3525 | 10.7451 |
| Persistence 1h | node_macro | 0.000490575 | 0.000693005 | 36.1604 | 34.5479 |
| Persistence 1h | grid_aggregate | 0.00345857 | 0.00439136 | 8.53624 | 8.57834 |

## Best epoch summary

- best_epoch: 30
- best_validation_node_macro_mae: 0.000403975
- best_train_scaled_mae: 0.316003
- epochs_run: 30

## GRU vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | gru_lower | gru_lower | gru_lower | gru_lower |
| grid_aggregate | gru_higher | gru_higher | gru_higher | gru_higher |

test_evaluated: False
