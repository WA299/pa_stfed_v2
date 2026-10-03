# Industrial External Benchmark: 50%

Industrial Load_kWh remains in native hourly-interval units. No source or industrial TEST was evaluated.

## Industrial target metrics

| Method | Audit node MAE | Validation node MAE |
|---|---:|---:|
| industrial_scarce_local | 3.72259 | 3.87317 |
| fedavg | 7.13487 | 7.35882 |
| fedprox | 10.6112 | 10.7881 |
| fedper | 6.14181 | 6.1351 |

Guardrails:

- industrial_test_locked = true
- industrial_test_evaluated = false
- reference_grid_tests_evaluated = false
