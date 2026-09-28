# Frozen 25%-History Multi-Seed Robustness Summary

Seeds 42, 123, and 2026 were fixed in advance. Seed 42 is historical and is read, not retrained. No hyperparameters or scientific definitions vary by seed.

Canonical validation is robustness evaluation only; canonical test remains locked. Three seeds do not support significance claims.

## Primary Metrics

| Method | Validation node-MAE mean | std | Audit node-MAE mean | std |
|---|---:|---:|---:|---:|
| scarce_local | 0.00037393046 | 2.2930193e-06 | 0.00032938658 | 1.6679189e-06 |
| fedavg | 0.0003613629 | 3.9633006e-06 | 0.00033332624 | 3.5992055e-06 |
| fedprox | 0.00038814487 | 1.109028e-06 | 0.00036284402 | 9.6646772e-07 |
| fedper | 0.00038038967 | 5.962401e-06 | 0.00034346845 | 4.0412918e-06 |
| fedfomo_style | 0.0003688216 | 6.2437679e-06 | 0.00032661367 | 2.0715843e-06 |
| btd_fl_direct_transfer | 0.00034577673 | 2.5159437e-06 | 0.00031234004 | 2.0791965e-06 |

## Per-Seed Primary Node-MAE

| Method | Seed 42 validation | Seed 123 validation | Seed 2026 validation | Seed 42 audit | Seed 123 audit | Seed 2026 audit |
|---|---:|---:|---:|---:|---:|---:|
| scarce_local | 0.00037104596 | 0.00037408948 | 0.00037665593 | 0.00032704593 | 0.00033030401 | 0.0003308098 |
| fedavg | 0.00035979743 | 0.00036680649 | 0.00035748477 | 0.0003313284 | 0.00033837952 | 0.00033027079 |
| fedprox | 0.00038715426 | 0.00038969324 | 0.00038758712 | 0.00036172869 | 0.00036408588 | 0.0003627175 |
| fedper | 0.00037961685 | 0.00038804777 | 0.0003735044 | 0.00034056944 | 0.0003491835 | 0.00034065241 |
| fedfomo_style | 0.00036014303 | 0.00037175061 | 0.00037457115 | 0.0003239492 | 0.00032689111 | 0.00032900072 |
| btd_fl_direct_transfer | 0.00034376809 | 0.00034423763 | 0.00034932447 | 0.00031355852 | 0.00030941325 | 0.00031404836 |

## BTD-FL Direct Transfer Robustness

| Comparator | Mean relative improvement | std | BTD wins by seed | Scarce-target validation wins / 12 |
|---|---:|---:|---:|---:|
| scarce_local | 0.075292762 | 0.0032098173 | 3 | 12 |
| fedavg | 0.042968649 | 0.01583918 | 3 | 10 |
| fedprox | 0.109143 | 0.0076035938 | 3 | 12 |
| fedper | 0.090690296 | 0.019838985 | 3 | 12 |
| fedfomo_style | 0.06229289 | 0.012199076 | 3 | 12 |

## Donor Selection Robustness

| Target | Seed 42 donor | Seed 123 donor | Seed 2026 donor |
|---|---|---|---|
| 39_bus_semi_urban_reference_grid | 56_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid | 56_bus_semi_urban_reference_grid |
| 50_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid | 80_bus_rural_reference_grid |
| 56_bus_semi_urban_reference_grid | 39_bus_semi_urban_reference_grid | 80_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid |
| 80_bus_rural_reference_grid | 50_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | 39_bus_semi_urban_reference_grid |

Directed benefit signs: 36 positive, 0 non-positive across 36 observations; fallbacks: 0.

## Guardrails
validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
seed42_retrained: false
