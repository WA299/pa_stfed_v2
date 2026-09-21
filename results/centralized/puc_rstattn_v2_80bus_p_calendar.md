# PUC-RSTAttn V2: 80_bus_rural_reference_grid / p_calendar

Validation-only metrics; no test labels are accessed.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| final | node_macro | 0.000194356 | 0.000356072 | 50.3242 | 51.0451 |
| final | grid_aggregate | 0.00289949 | 0.00367837 | 11.9992 | 12.8265 |
| temporal_residual | node_macro | 0.000195762 | 0.000355328 | 48.3771 | 52.4333 |
| temporal_residual | grid_aggregate | 0.00289949 | 0.00367837 | 11.9992 | 12.8265 |
| gru_anchor | node_macro | 0.000217978 | 0.000356757 | 48.6349 | 53.228 |
| gru_anchor | grid_aggregate | 0.00315706 | 0.00393331 | 13.0651 | 13.9931 |
| persistence_1h | node_macro | 0.000233666 | 0.000394782 | 40.8617 | 38.1808 |
| persistence_1h | grid_aggregate | 0.00283403 | 0.00390906 | 11.7283 | 12.6145 |

## Validation diagnostics

{
  "mean_temporal_gate": 0.3155184700375512,
  "mean_spatial_gate_active": 0.3753089024907067,
  "mean_abs_temporal_correction": 0.0767686798991192,
  "mean_abs_centered_spatial_residual": 0.007328477089426347,
  "mean_spatial_attention_entropy": 29.40045892624628,
  "aggregate_preservation_max_abs_error": 1.5383674984886533e-06
}

test_evaluated: False
