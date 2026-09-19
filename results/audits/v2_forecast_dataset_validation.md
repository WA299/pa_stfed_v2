# V2 Forecasting Window Dataset Validation

Generated at (UTC): `2026-09-19T10:50:40.458487+00:00`

Task: history 168 hours, horizon 1 hour, target next-hour active P. No model training or test metric.

| Grid | Mode | Samples | History shape | Target shape | Train/Val/Test | Split bounds | No future leakage | Train load-only scaler | Calendar unchanged | Target roundtrip | All checks |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 39_bus_semi_urban_reference_grid | p_calendar | 8592 | `[168, 39, 6]` | `[39]` | 5964/1314/1314 | True | True | True | True | True | True |
| 39_bus_semi_urban_reference_grid | pq_calendar | 8592 | `[168, 39, 7]` | `[39]` | 5964/1314/1314 | True | True | True | True | True | True |
| 50_bus_rural_reference_grid | p_calendar | 8592 | `[168, 50, 6]` | `[50]` | 5964/1314/1314 | True | True | True | True | True | True |
| 50_bus_rural_reference_grid | pq_calendar | 8592 | `[168, 50, 7]` | `[50]` | 5964/1314/1314 | True | True | True | True | True | True |
| 56_bus_semi_urban_reference_grid | p_calendar | 8592 | `[168, 56, 6]` | `[56]` | 5964/1314/1314 | True | True | True | True | True | True |
| 56_bus_semi_urban_reference_grid | pq_calendar | 8592 | `[168, 56, 7]` | `[56]` | 5964/1314/1314 | True | True | True | True | True | True |
| 80_bus_rural_reference_grid | p_calendar | 8592 | `[168, 80, 6]` | `[80]` | 5964/1314/1314 | True | True | True | True | True | True |
| 80_bus_rural_reference_grid | pq_calendar | 8592 | `[168, 80, 7]` | `[80]` | 5964/1314/1314 | True | True | True | True | True | True |

## Target timestamp ranges

### 39_bus_semi_urban_reference_grid

#### p_calendar

| Split | Samples | First target | Last target | Target bounds |
| --- | ---: | --- | --- | --- |
| train | 5964 | 2021-01-08 00:00:00 | 2021-09-13 11:00:00 | True |
| validation | 1314 | 2021-09-13 12:00:00 | 2021-11-07 05:00:00 | True |
| test | 1314 | 2021-11-07 06:00:00 | 2021-12-31 23:00:00 | True |

Scaler: split=train, source indices [0, 6132) ending 2021-09-13 11:00:00; load-bus-only=True, calendar-unchanged=True, target-roundtrip=True; test labels evaluated: False.

#### pq_calendar

| Split | Samples | First target | Last target | Target bounds |
| --- | ---: | --- | --- | --- |
| train | 5964 | 2021-01-08 00:00:00 | 2021-09-13 11:00:00 | True |
| validation | 1314 | 2021-09-13 12:00:00 | 2021-11-07 05:00:00 | True |
| test | 1314 | 2021-11-07 06:00:00 | 2021-12-31 23:00:00 | True |

Scaler: split=train, source indices [0, 6132) ending 2021-09-13 11:00:00; load-bus-only=True, calendar-unchanged=True, target-roundtrip=True; test labels evaluated: False.

### 50_bus_rural_reference_grid

#### p_calendar

| Split | Samples | First target | Last target | Target bounds |
| --- | ---: | --- | --- | --- |
| train | 5964 | 2021-01-08 00:00:00 | 2021-09-13 11:00:00 | True |
| validation | 1314 | 2021-09-13 12:00:00 | 2021-11-07 05:00:00 | True |
| test | 1314 | 2021-11-07 06:00:00 | 2021-12-31 23:00:00 | True |

Scaler: split=train, source indices [0, 6132) ending 2021-09-13 11:00:00; load-bus-only=True, calendar-unchanged=True, target-roundtrip=True; test labels evaluated: False.

#### pq_calendar

| Split | Samples | First target | Last target | Target bounds |
| --- | ---: | --- | --- | --- |
| train | 5964 | 2021-01-08 00:00:00 | 2021-09-13 11:00:00 | True |
| validation | 1314 | 2021-09-13 12:00:00 | 2021-11-07 05:00:00 | True |
| test | 1314 | 2021-11-07 06:00:00 | 2021-12-31 23:00:00 | True |

Scaler: split=train, source indices [0, 6132) ending 2021-09-13 11:00:00; load-bus-only=True, calendar-unchanged=True, target-roundtrip=True; test labels evaluated: False.

### 56_bus_semi_urban_reference_grid

#### p_calendar

| Split | Samples | First target | Last target | Target bounds |
| --- | ---: | --- | --- | --- |
| train | 5964 | 2021-01-08 00:00:00 | 2021-09-13 11:00:00 | True |
| validation | 1314 | 2021-09-13 12:00:00 | 2021-11-07 05:00:00 | True |
| test | 1314 | 2021-11-07 06:00:00 | 2021-12-31 23:00:00 | True |

Scaler: split=train, source indices [0, 6132) ending 2021-09-13 11:00:00; load-bus-only=True, calendar-unchanged=True, target-roundtrip=True; test labels evaluated: False.

#### pq_calendar

| Split | Samples | First target | Last target | Target bounds |
| --- | ---: | --- | --- | --- |
| train | 5964 | 2021-01-08 00:00:00 | 2021-09-13 11:00:00 | True |
| validation | 1314 | 2021-09-13 12:00:00 | 2021-11-07 05:00:00 | True |
| test | 1314 | 2021-11-07 06:00:00 | 2021-12-31 23:00:00 | True |

Scaler: split=train, source indices [0, 6132) ending 2021-09-13 11:00:00; load-bus-only=True, calendar-unchanged=True, target-roundtrip=True; test labels evaluated: False.

### 80_bus_rural_reference_grid

#### p_calendar

| Split | Samples | First target | Last target | Target bounds |
| --- | ---: | --- | --- | --- |
| train | 5964 | 2021-01-08 00:00:00 | 2021-09-13 11:00:00 | True |
| validation | 1314 | 2021-09-13 12:00:00 | 2021-11-07 05:00:00 | True |
| test | 1314 | 2021-11-07 06:00:00 | 2021-12-31 23:00:00 | True |

Scaler: split=train, source indices [0, 6132) ending 2021-09-13 11:00:00; load-bus-only=True, calendar-unchanged=True, target-roundtrip=True; test labels evaluated: False.

#### pq_calendar

| Split | Samples | First target | Last target | Target bounds |
| --- | ---: | --- | --- | --- |
| train | 5964 | 2021-01-08 00:00:00 | 2021-09-13 11:00:00 | True |
| validation | 1314 | 2021-09-13 12:00:00 | 2021-11-07 05:00:00 | True |
| test | 1314 | 2021-11-07 06:00:00 | 2021-12-31 23:00:00 | True |

Scaler: split=train, source indices [0, 6132) ending 2021-09-13 11:00:00; load-bus-only=True, calendar-unchanged=True, target-roundtrip=True; test labels evaluated: False.

All checks pass: **True**.