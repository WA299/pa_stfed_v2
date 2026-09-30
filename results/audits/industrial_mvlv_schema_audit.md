# Industrial MV/LV schema audit

Stage 1 schema/quality audit only. No neural training or forecasting metric was evaluated.

## Topology

- Nodes/branches: 76 / 75
- Connected/tree: True / True
- Source/slack buses: r1v47.0b1, r2v47.0b1
- Load buses: 45
- BASE_KV counts: `{"0.23": 1, "0.230": 2, "0.415": 52, "11": 19, "47": 2}`

## Time series

- Common interval: `2020-09-07 01:00:00` through `2022-03-17 00:00:00`
- Common rows: 13344; missing hours: 0
- Duplicate groups identical and deduplicated: `True`

## Splits

- train: [0, 9340) = 9340 rows
- validation: [9340, 11341) = 2001 rows
- test: [11341, 13344) = 2003 rows
- 25% scarce: available=2335, eligible=2167, fit/calibration/audit=1300/433/434
- 50% scarce: available=4670, eligible=4502, fit/calibration/audit=2701/900/901

## Guardrails

- `test_evaluated = false`
- `test_locked = true`
- `training_executed = false`
