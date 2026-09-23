# PUC-RSTAttn_V2_conditional_utility: 80_bus_rural_reference_grid / p_calendar

Validation-only metrics; no test labels are accessed.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| final | node_macro | 0.000191545 | 0.000354421 | 47.3367 | 49.3567 |
| final | grid_aggregate | 0.00289652 | 0.00364962 | 11.9869 | 12.7191 |
| temporal_residual | node_macro | 0.000192875 | 0.000356522 | 47.3329 | 48.8459 |
| temporal_residual | grid_aggregate | 0.00289652 | 0.00364962 | 11.9869 | 12.7191 |
| gru_anchor | node_macro | 0.000221812 | 0.00035755 | 50.0894 | 54.2639 |
| gru_anchor | grid_aggregate | 0.0033666 | 0.00412112 | 13.9323 | 14.9576 |
| persistence_1h | node_macro | 0.000233666 | 0.000394782 | 40.8617 | 38.1808 |
| persistence_1h | grid_aggregate | 0.00283403 | 0.00390906 | 11.7283 | 12.6145 |

## Validation diagnostics

{
  "mean_temporal_gate": 0.3597200200671241,
  "mean_spatial_gate_active": 0.23232532186167582,
  "mean_abs_temporal_correction": 0.10051609646706354,
  "mean_abs_centered_spatial_residual": 0.005421567608469299,
  "mean_spatial_attention_entropy": 23.38641357421875,
  "aggregate_preservation_max_abs_error": 1.4702479044596355e-06
}

test_evaluated: False
