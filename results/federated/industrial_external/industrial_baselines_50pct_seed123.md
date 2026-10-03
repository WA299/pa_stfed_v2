# Industrial External Benchmark: 50%

Industrial Load_kWh remains in native hourly-interval units. No source or industrial TEST was evaluated.

## Industrial target metrics

| Method | Audit node MAE | Validation node MAE |
|---|---:|---:|
| industrial_scarce_local | 3.67676 | 3.86222 |
| fedavg | 9.43641 | 9.6113 |
| fedprox | 11.4559 | 11.6693 |
| fedper | 6.2141 | 6.20129 |

Guardrails:

- industrial_test_locked = true
- industrial_test_evaluated = false
- reference_grid_tests_evaluated = false
