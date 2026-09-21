# PUC-RSTAttn: 80_bus_rural_reference_grid / p_calendar

Validation-only metrics; no test labels are accessed.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| PUC-RSTAttn final | node_macro | 0.000201558 | 0.000363408 | 40.8816 | 46.1969 |
| PUC-RSTAttn final | grid_aggregate | 0.00285738 | 0.00364125 | 11.8249 | 12.5634 |
| Temporal anchor | node_macro | 0.000201838 | 0.000361574 | 42.0405 | 46.5022 |
| Temporal anchor | grid_aggregate | 0.00282377 | 0.00361976 | 11.6859 | 12.4201 |
| Persistence 1h | node_macro | 0.000233666 | 0.000394782 | 40.8617 | 38.1808 |
| Persistence 1h | grid_aggregate | 0.00283403 | 0.00390906 | 11.7283 | 12.6145 |

## Training summary

- best_epoch: 50
- epochs_run: 50
- best_validation_node_macro_mae: 0.000201558

## Graph diagnostics

{
  "positive_edge_count": 95,
  "targets_with_selected_neighbors": {
    "0": 0,
    "1": 0,
    "2": 1,
    "3": 31
  },
  "mean_positive_utility": 0.023237799655847494,
  "mean_normalized_impedance": 0.9808717966079712,
  "mean_normalized_hop": 0.9455377459526062,
  "utility_fit_samples": 4771,
  "utility_selection_samples": 1193
}

test_evaluated: False
