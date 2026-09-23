# PUC-RSTAttn_V2_conditional_utility: 39_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics; no test labels are accessed.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| final | node_macro | 0.000382019 | 0.000557324 | 27.8361 | 27.9554 |
| final | grid_aggregate | 0.00431854 | 0.00532857 | 10.6588 | 10.9461 |
| temporal_residual | node_macro | 0.000385677 | 0.000559549 | 28.063 | 28.2351 |
| temporal_residual | grid_aggregate | 0.00431854 | 0.00532857 | 10.6588 | 10.9461 |
| gru_anchor | node_macro | 0.00040531 | 0.000586304 | 29.6784 | 29.7154 |
| gru_anchor | grid_aggregate | 0.00413804 | 0.00509914 | 10.2132 | 10.603 |
| persistence_1h | node_macro | 0.000490575 | 0.000693005 | 36.1604 | 34.5479 |
| persistence_1h | grid_aggregate | 0.00345857 | 0.00439136 | 8.53624 | 8.57834 |

## Validation diagnostics

{
  "mean_temporal_gate": 0.33568998816467466,
  "mean_spatial_gate_active": 0.5692729162318366,
  "mean_abs_temporal_correction": 0.20174558283317656,
  "mean_abs_centered_spatial_residual": 0.0365105453612549,
  "mean_spatial_attention_entropy": 26.956967580886115,
  "aggregate_preservation_max_abs_error": 2.0464261372884116e-06
}

test_evaluated: False
