# 25% vs 50% History Robustness

This comparison reuses frozen summaries only. No monotonicity or significance claim is implied.

## Primary Validation/Audit Node-MAE

| Method | 25% validation mean +/- std | 50% validation mean +/- std | 25% audit mean +/- std | 50% audit mean +/- std |
|---|---:|---:|---:|---:|
| scarce_local | 0.00037393046 +/- 2.2930193e-06 | 0.00035062209 +/- 8.7581048e-07 | 0.00032938658 +/- 1.6679189e-06 | 0.00032051424 +/- 7.4579467e-07 |
| fedavg | 0.0003613629 +/- 3.9633006e-06 | 0.0003573008 +/- 4.0490592e-06 | 0.00033332624 +/- 3.5992055e-06 | 0.00033300017 +/- 3.9512962e-06 |
| fedprox | 0.00038814487 +/- 1.109028e-06 | 0.00038665942 +/- 6.6552976e-07 | 0.00036284402 +/- 9.6646772e-07 | 0.000363531 +/- 7.4978101e-07 |
| fedper | 0.00038038967 +/- 5.962401e-06 | 0.00036555554 +/- 3.9813908e-06 | 0.00034346845 +/- 4.0412918e-06 | 0.0003411424 +/- 3.6881826e-06 |
| fedfomo_style | 0.0003688216 +/- 6.2437679e-06 | 0.0003490613 +/- 6.4150387e-07 | 0.00032661367 +/- 2.0715843e-06 | 0.00032368137 +/- 1.1893213e-06 |
| btd_fl_direct_transfer | 0.00034577673 +/- 2.5159437e-06 | 0.00033544811 +/- 1.7520255e-06 | 0.00031234004 +/- 2.0791965e-06 | 0.00031050524 +/- 2.2799333e-06 |

## BTD-FL Direct Transfer Comparisons

| Comparator | 25% mean improvement | 50% mean improvement | 25% seed wins | 50% seed wins | 25% target-seed wins / 12 | 50% target-seed wins / 12 |
|---|---:|---:|---:|---:|---:|---:|
| scarce_local | 0.075292762 | 0.043273086 | 3 | 3 | 12 | 11 |
| fedavg | 0.042968649 | 0.061008541 | 3 | 3 | 10 | 12 |
| fedprox | 0.109143 | 0.13243563 | 3 | 3 | 12 | 12 |
| fedper | 0.090690296 | 0.082248476 | 3 | 3 | 12 | 12 |
| fedfomo_style | 0.06229289 | 0.039003276 | 3 | 3 | 12 | 12 |

## Guardrails

test_evaluated: false
canonical test remains locked
donor direction and fallback counts are reported from each input summary
