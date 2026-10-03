# Industrial External Benchmark: 50%

Industrial Load_kWh remains in native hourly-interval units. No source or industrial TEST was evaluated.

## Industrial target metrics

| Method | Audit node MAE | Validation node MAE |
|---|---:|---:|
| industrial_scarce_local | 0.787069 | 0.774704 |
| fedavg | 0.786691 | 0.774655 |
| fedprox | 0.786812 | 0.774874 |
| fedper | 0.786755 | 0.774692 |

Guardrails:

- industrial_test_locked = true
- industrial_test_evaluated = false
- reference_grid_tests_evaluated = false
