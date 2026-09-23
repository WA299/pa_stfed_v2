# PUC-RSTAttn_V2_conditional_utility: 50_bus_rural_reference_grid / p_calendar

Validation-only metrics; no test labels are accessed.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| final | node_macro | 0.000319956 | 0.000505991 | 32.2856 | 36.6114 |
| final | grid_aggregate | 0.00307006 | 0.00392821 | 7.96215 | 8.06339 |
| temporal_residual | node_macro | 0.000327886 | 0.000506609 | 32.0461 | 33.3472 |
| temporal_residual | grid_aggregate | 0.00307006 | 0.00392821 | 7.96216 | 8.06339 |
| gru_anchor | node_macro | 0.000343004 | 0.000521147 | 33.7899 | 39.0331 |
| gru_anchor | grid_aggregate | 0.00266108 | 0.00344248 | 6.90145 | 6.94932 |
| persistence_1h | node_macro | 0.000374557 | 0.000564059 | 34.5653 | 33.336 |
| persistence_1h | grid_aggregate | 0.00253653 | 0.00333136 | 6.57844 | 6.57141 |

## Validation diagnostics

{
  "mean_temporal_gate": 0.5707385625158038,
  "mean_spatial_gate_active": 0.6963934657119569,
  "mean_abs_temporal_correction": 0.04128100527893929,
  "mean_abs_centered_spatial_residual": 0.009403432830281201,
  "mean_spatial_attention_entropy": 18.826882952735538,
  "aggregate_preservation_max_abs_error": 1.0047640119280134e-06
}

test_evaluated: False
