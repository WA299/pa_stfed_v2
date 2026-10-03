# Industrial External Benchmark: 50%

Industrial Load_kWh remains in native hourly-interval units. No source or industrial TEST was evaluated.

## Industrial target metrics

| Method | Audit node MAE | Validation node MAE |
|---|---:|---:|
| industrial_scarce_local | 3.97342 | 4.15231 |
| fedavg | 8.18458 | 8.40449 |
| fedprox | 11.1688 | 11.3613 |
| fedper | 6.38848 | 6.45563 |

Guardrails:

- industrial_test_locked = true
- industrial_test_evaluated = false
- reference_grid_tests_evaluated = false
