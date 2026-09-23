# PUC-RSTAttn_V2_conditional_utility: 56_bus_semi_urban_reference_grid / p_calendar

Validation-only metrics; no test labels are accessed.

| Method | Scope | MAE | RMSE | WAPE (%) | sMAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| final | node_macro | 0.000436766 | 0.000617251 | 25.673 | 26.9376 |
| final | grid_aggregate | 0.00706242 | 0.00843115 | 8.86491 | 9.12196 |
| temporal_residual | node_macro | 0.000438567 | 0.000617358 | 25.711 | 26.9764 |
| temporal_residual | grid_aggregate | 0.00706242 | 0.00843115 | 8.86491 | 9.12196 |
| gru_anchor | node_macro | 0.000453928 | 0.000636484 | 26.6398 | 27.9374 |
| gru_anchor | grid_aggregate | 0.006555 | 0.00795531 | 8.22797 | 8.52134 |
| persistence_1h | node_macro | 0.000537618 | 0.00073951 | 31.5906 | 31.9867 |
| persistence_1h | grid_aggregate | 0.00523067 | 0.00660912 | 6.56565 | 6.68602 |

## Validation diagnostics

{
  "mean_temporal_gate": 0.510993521128382,
  "mean_spatial_gate_active": 0.5675447001343682,
  "mean_abs_temporal_correction": 0.15716628144894326,
  "mean_abs_centered_spatial_residual": 0.0326433221233033,
  "mean_spatial_attention_entropy": 41.56672695704869,
  "aggregate_preservation_max_abs_error": 3.6443982805524555e-06
}

test_evaluated: False
