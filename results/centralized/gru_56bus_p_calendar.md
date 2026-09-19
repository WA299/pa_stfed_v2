# Centralized GRU: 56_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| SharedNodeGRU | node_macro | 0.000447376 | 0.000627382 | 26.3344 | 27.6745 |
| SharedNodeGRU | grid_aggregate | 0.00659355 | 0.00796741 | 8.27637 | 8.57188 |
| Persistence 1h | node_macro | 0.000537618 | 0.00073951 | 31.5906 | 31.9867 |
| Persistence 1h | grid_aggregate | 0.00523067 | 0.00660912 | 6.56565 | 6.68602 |

## Best epoch summary

- best_epoch: 50
- best_validation_node_macro_mae: 0.000447376
- best_train_scaled_mae: 0.28816
- epochs_run: 50

## GRU vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | gru_lower | gru_lower | gru_lower | gru_lower |
| grid_aggregate | gru_higher | gru_higher | gru_higher | gru_higher |

test_evaluated: False
