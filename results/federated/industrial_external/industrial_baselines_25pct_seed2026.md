# Industrial External Benchmark: 25%

Industrial Load_kWh remains in native hourly-interval units. No source or industrial TEST was evaluated.

## Industrial target metrics

| Method | Audit node MAE | Validation node MAE |
|---|---:|---:|
| industrial_scarce_local | 4.62928 | 4.55239 |
| fedavg | 8.18256 | 8.11447 |
| fedprox | 11.6736 | 11.6452 |
| fedper | 6.08961 | 5.96443 |

Guardrails:

- industrial_test_locked = true
- industrial_test_evaluated = false
- reference_grid_tests_evaluated = false
