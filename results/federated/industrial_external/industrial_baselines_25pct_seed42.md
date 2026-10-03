# Industrial External Benchmark: 25%

Industrial Load_kWh remains in native hourly-interval units. No source or industrial TEST was evaluated.

## Industrial target metrics

| Method | Audit node MAE | Validation node MAE |
|---|---:|---:|
| industrial_scarce_local | 4.5429 | 4.53516 |
| fedavg | 9.67838 | 9.51235 |
| fedprox | 12.7114 | 12.6185 |
| fedper | 5.78134 | 5.66083 |

Guardrails:

- industrial_test_locked = true
- industrial_test_evaluated = false
- reference_grid_tests_evaluated = false
