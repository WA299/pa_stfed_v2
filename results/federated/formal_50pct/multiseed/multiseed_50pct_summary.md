# Frozen 50%-History Multi-Seed Robustness Summary

Seeds 42, 123, and 2026 were fixed in advance. No hyperparameters or scientific definitions vary by seed.

Canonical validation is robustness evaluation only; canonical test remains locked. Three seeds do not support significance claims.

## Primary Metrics

| Method | Validation node-MAE mean | std | Audit node-MAE mean | std |
|---|---:|---:|---:|---:|
| scarce_local | 0.00035062209 | 8.7581048e-07 | 0.00032051424 | 7.4579467e-07 |
| fedavg | 0.0003573008 | 4.0490592e-06 | 0.00033300017 | 3.9512962e-06 |
| fedprox | 0.00038665942 | 6.6552976e-07 | 0.000363531 | 7.4978101e-07 |
| fedper | 0.00036555554 | 3.9813908e-06 | 0.0003411424 | 3.6881826e-06 |
| fedfomo_style | 0.0003490613 | 6.4150387e-07 | 0.00032368137 | 1.1893213e-06 |
| btd_fl_direct_transfer | 0.00033544811 | 1.7520255e-06 | 0.00031050524 | 2.2799333e-06 |

## Per-Seed Primary Node-MAE

| Method | Seed 42 validation | Seed 123 validation | Seed 2026 validation | Seed 42 audit | Seed 123 audit | Seed 2026 audit |
|---|---:|---:|---:|---:|---:|---:|
| scarce_local | 0.00035137812 | 0.00035109371 | 0.00034939444 | 0.00032051487 | 0.00032142733 | 0.00031960052 |
| fedavg | 0.00035672823 | 0.0003625213 | 0.00035265287 | 0.00033182128 | 0.00033832002 | 0.00032885919 |
| fedprox | 0.00038572984 | 0.00038725191 | 0.00038699651 | 0.0003624891 | 0.00036388137 | 0.00036422252 |
| fedper | 0.0003680758 | 0.00036865585 | 0.00035993497 | 0.00034258456 | 0.0003447623 | 0.00033608033 |
| fedfomo_style | 0.00034995473 | 0.00034875103 | 0.00034847813 | 0.0003253504 | 0.0003226666 | 0.00032302713 |
| btd_fl_direct_transfer | 0.00033760504 | 0.00033331365 | 0.00033542565 | 0.00031265491 | 0.00030734922 | 0.00031151158 |

## BTD-FL Direct Transfer Robustness

| Comparator | Mean relative improvement | std | BTD wins by seed | Scarce-target validation wins / 12 |
|---|---:|---:|---:|---:|
| scarce_local | 0.043273086 | 0.0052203466 | 3 | 11 |
| fedavg | 0.061008541 | 0.013966361 | 3 | 12 |
| fedprox | 0.13243563 | 0.0059569998 | 3 | 12 |
| fedper | 0.082248476 | 0.011345007 | 3 | 12 |
| fedfomo_style | 0.039003276 | 0.0038241053 | 3 | 12 |

## Donor Selection Robustness

| Target | Seed 42 donor | Seed 123 donor | Seed 2026 donor |
|---|---|---|---|
| 39_bus_semi_urban_reference_grid | 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid |
| 50_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid |
| 56_bus_semi_urban_reference_grid | 39_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid | 80_bus_rural_reference_grid |
| 80_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | 39_bus_semi_urban_reference_grid | 56_bus_semi_urban_reference_grid |

Directed benefit signs: 35 positive, 1 non-positive across 36 observations; fallbacks: 0.

## Guardrails
validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
seed42_retrained: false
