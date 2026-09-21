# PUC-RSTAttn V2: 56_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics; no test labels are accessed.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| final | node_macro | 0.000436791 | 0.000616188 | 25.6937 | 27.0097 |
| final | grid_aggregate | 0.00703414 | 0.00837216 | 8.8294 | 9.11656 |
| temporal_residual | node_macro | 0.000441952 | 0.000620121 | 25.94 | 27.2532 |
| temporal_residual | grid_aggregate | 0.00703414 | 0.00837216 | 8.82941 | 9.11656 |
| gru_anchor | node_macro | 0.000454166 | 0.000638879 | 26.6508 | 27.9501 |
| gru_anchor | grid_aggregate | 0.00688464 | 0.00830536 | 8.64175 | 8.97788 |
| persistence_1h | node_macro | 0.000537618 | 0.00073951 | 31.5906 | 31.9867 |
| persistence_1h | grid_aggregate | 0.00523067 | 0.00660912 | 6.56565 | 6.68602 |

## Validation diagnostics

{
  "mean_temporal_gate": 0.6053766721770877,
  "mean_spatial_gate_active": 0.40252632328442167,
  "mean_abs_temporal_correction": 0.14356286575396857,
  "mean_abs_centered_spatial_residual": 0.04236163349733466,
  "mean_spatial_attention_entropy": 38.130061285836355,
  "aggregate_preservation_max_abs_error": 3.2555489313034785e-06
}

test_evaluated: False
