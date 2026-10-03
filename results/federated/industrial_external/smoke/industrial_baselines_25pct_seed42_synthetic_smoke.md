# Industrial External Benchmark: 25%

Industrial Load_kWh remains in native hourly-interval units. No source or industrial TEST was evaluated.

## Industrial target metrics

| Method | Audit node MAE | Validation node MAE |
|---|---:|---:|
| industrial_scarce_local | 0.785273 | 0.776237 |
| fedavg | 0.784117 | 0.775034 |
| fedprox | 0.784233 | 0.775259 |
| fedper | 0.784529 | 0.775136 |

Guardrails:

- industrial_test_locked = true
- industrial_test_evaluated = false
- reference_grid_tests_evaluated = false
