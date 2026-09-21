# PUC-RSTAttn V2: 39_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics; no test labels are accessed.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| final | node_macro | 0.000376738 | 0.000550121 | 27.4241 | 27.5046 |
| final | grid_aggregate | 0.00407731 | 0.00506964 | 10.0634 | 10.2778 |
| temporal_residual | node_macro | 0.000381778 | 0.0005542 | 27.7098 | 27.8356 |
| temporal_residual | grid_aggregate | 0.00407731 | 0.00506964 | 10.0634 | 10.2778 |
| gru_anchor | node_macro | 0.00040581 | 0.000587735 | 29.6701 | 29.7303 |
| gru_anchor | grid_aggregate | 0.00429027 | 0.0052629 | 10.589 | 11.0429 |
| persistence_1h | node_macro | 0.000490575 | 0.000693005 | 36.1604 | 34.5479 |
| persistence_1h | grid_aggregate | 0.00345857 | 0.00439136 | 8.53624 | 8.57834 |

## Validation diagnostics

{
  "mean_temporal_gate": 0.28672640344926287,
  "mean_spatial_gate_active": 0.4644632942619778,
  "mean_abs_temporal_correction": 0.22365424320811317,
  "mean_abs_centered_spatial_residual": 0.03276770499845346,
  "mean_spatial_attention_entropy": 26.23624084109352,
  "aggregate_preservation_max_abs_error": 1.9811448596772695e-06
}

test_evaluated: False
