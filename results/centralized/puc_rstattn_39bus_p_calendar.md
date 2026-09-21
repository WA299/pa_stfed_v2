# PUC-RSTAttn: 39_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics; no test labels are accessed.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| PUC-RSTAttn final | node_macro | 0.000398136 | 0.000575477 | 29.1731 | 29.2742 |
| PUC-RSTAttn final | grid_aggregate | 0.00406159 | 0.00509807 | 10.0246 | 10.298 |
| Temporal anchor | node_macro | 0.000398893 | 0.000571201 | 29.2738 | 29.3967 |
| Temporal anchor | grid_aggregate | 0.00366899 | 0.00467917 | 9.05558 | 9.15752 |
| Persistence 1h | node_macro | 0.000490575 | 0.000693005 | 36.1604 | 34.5479 |
| Persistence 1h | grid_aggregate | 0.00345857 | 0.00439136 | 8.53624 | 8.57834 |

## Training summary

- best_epoch: 50
- epochs_run: 50
- best_validation_node_macro_mae: 0.000398136

## Graph diagnostics

{
  "positive_edge_count": 84,
  "targets_with_selected_neighbors": {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 28
  },
  "mean_positive_utility": 0.027294315976342635,
  "mean_normalized_impedance": 0.9679095149040222,
  "mean_normalized_hop": 0.9380952715873718,
  "utility_fit_samples": 4771,
  "utility_selection_samples": 1193
}

test_evaluated: False
