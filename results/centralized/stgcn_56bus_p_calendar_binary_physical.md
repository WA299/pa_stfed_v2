# Fixed-topology STGCN baseline: 56_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics. Test labels and test metrics are intentionally not used.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| STGCN | node_macro | 0.000474903 | 0.000673455 | 27.6809 | 29.1969 |
| STGCN | grid_aggregate | 0.00943017 | 0.0107178 | 11.837 | 12.5108 |
| Persistence 1h | node_macro | 0.000537618 | 0.00073951 | 31.5906 | 31.9867 |
| Persistence 1h | grid_aggregate | 0.00523067 | 0.00660912 | 6.56565 | 6.68602 |

## Best epoch summary

- best_epoch: 35
- best_validation_node_macro_mae: 0.000474903
- best_train_scaled_mae: 0.306904
- epochs_run: 43

## STGCN vs persistence_1h factual comparison

| Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | --- | --- | --- |
| node_macro | stgcn_lower | stgcn_lower | stgcn_lower | stgcn_lower |
| grid_aggregate | stgcn_higher | stgcn_higher | stgcn_higher | stgcn_higher |

adjacency_mode: binary_physical
topology_used: True
adjacency_type: binary_physical_normalized
electrical_edge_features_used: False
test_evaluated: False
