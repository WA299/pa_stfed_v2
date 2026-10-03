# Industrial External Benchmark: 25%

Industrial Load_kWh remains in native hourly-interval units. No source or industrial TEST was evaluated.

## Industrial target metrics

| Method | Audit node MAE | Validation node MAE |
|---|---:|---:|
| industrial_scarce_local | 4.55396 | 4.55358 |
| fedavg | 10.7982 | 10.878 |
| fedprox | 12.636 | 12.6137 |
| fedper | 6.57968 | 6.50138 |

Guardrails:

- industrial_test_locked = true
- industrial_test_evaluated = false
- reference_grid_tests_evaluated = false
