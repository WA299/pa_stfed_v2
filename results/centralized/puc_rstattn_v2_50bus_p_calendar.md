# PUC-RSTAttn V2: 50_bus_rural_reference_grid / p_calendar

Validation-only metrics; no test labels are accessed.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| final | node_macro | 0.000319189 | 0.000497551 | 30.9487 | 32.7366 |
| final | grid_aggregate | 0.00262815 | 0.00343107 | 6.81606 | 6.83364 |
| temporal_residual | node_macro | 0.000337537 | 0.000505596 | 35.1303 | 36.1473 |
| temporal_residual | grid_aggregate | 0.00262815 | 0.00343107 | 6.81606 | 6.83365 |
| gru_anchor | node_macro | 0.000340415 | 0.000514457 | 32.4829 | 34.5592 |
| gru_anchor | grid_aggregate | 0.00251442 | 0.00327994 | 6.5211 | 6.53837 |
| persistence_1h | node_macro | 0.000374557 | 0.000564059 | 34.5653 | 33.336 |
| persistence_1h | grid_aggregate | 0.00253653 | 0.00333136 | 6.57844 | 6.57141 |

## Validation diagnostics

{
  "mean_temporal_gate": 0.516476489958309,
  "mean_spatial_gate_active": 0.4999593205395199,
  "mean_abs_temporal_correction": 0.033328261403810414,
  "mean_abs_centered_spatial_residual": 0.008913490610818068,
  "mean_spatial_attention_entropy": 14.661847432454428,
  "aggregate_preservation_max_abs_error": 1.044500441778274e-06
}

test_evaluated: False
