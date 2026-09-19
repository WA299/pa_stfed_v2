# Centralized GRU: 39_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| SharedNodeGRU | node_macro | 0.000396929 | 0.000575579 | 29.0493 | 29.1097 |
| SharedNodeGRU | grid_aggregate | 0.00424555 | 0.00519702 | 10.4786 | 10.8925 |
| Persistence 1h | node_macro | 0.000490575 | 0.000693005 | 36.1604 | 34.5479 |
| Persistence 1h | grid_aggregate | 0.00345857 | 0.00439136 | 8.53624 | 8.57834 |

## Best epoch summary

- best_epoch: 50
- best_validation_node_macro_mae: 0.000396929
- best_train_scaled_mae: 0.307829
- epochs_run: 50

## GRU vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | gru_lower | gru_lower | gru_lower | gru_lower |
| grid_aggregate | gru_higher | gru_higher | gru_higher | gru_higher |

test_evaluated: False
