# Formal BTD-FL Ablation: btd_full_model_transfer

Mechanism changed: all compatible trainable neural parameters transferred.
The accepted main BTD-FL JSON is a frozen reference only; no baseline is retrained.
Canonical validation is evaluation-only. Canonical test is never evaluated.

## Per-target audit node/grid metrics

| Target | Ablation node MAE | Main BTD-FL node MAE | Scarce-local node MAE | Ablation grid MAE |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.00035338 | 0.000350049 | 0.000372181 | 0.00371768 |
| 50_bus_rural_reference_grid | 0.000294664 | 0.000292519 | 0.000306918 | 0.00239907 |
| 56_bus_semi_urban_reference_grid | 0.000406324 | 0.000404865 | 0.000417712 | 0.00667921 |
| 80_bus_rural_reference_grid | 0.000219945 | 0.000206135 | 0.000211373 | 0.00312221 |

## Per-target canonical-validation node/grid metrics

| Target | Ablation node MAE | Main BTD-FL node MAE | Scarce-local node MAE | Ablation grid MAE |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000392749 | 0.000390059 | 0.000416143 | 0.00415544 |
| 50_bus_rural_reference_grid | 0.00033231 | 0.000336294 | 0.000385882 | 0.00261432 |
| 56_bus_semi_urban_reference_grid | 0.000442249 | 0.000446751 | 0.000470452 | 0.00751946 |
| 80_bus_rural_reference_grid | 0.000219756 | 0.000206175 | 0.000211706 | 0.003043 |

## Donors and relative change

| Target | Donor | Rule | Audit vs BTD-FL | Validation vs BTD-FL | Audit vs local | Validation vs local |
|---|---|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 56_bus_semi_urban_reference_grid | frozen_main_btd_fl_selected_donor | -0.00951709 | -0.00689724 | 0.0505135 | 0.0562177 |
| 50_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | frozen_main_btd_fl_selected_donor | -0.00733268 | 0.0118464 | 0.0399262 | 0.138829 |
| 56_bus_semi_urban_reference_grid | 39_bus_semi_urban_reference_grid | frozen_main_btd_fl_selected_donor | -0.00360315 | 0.0100761 | 0.0272626 | 0.0599491 |
| 80_bus_rural_reference_grid | 50_bus_rural_reference_grid | frozen_main_btd_fl_selected_donor | -0.066995 | -0.0658744 | -0.0405549 | -0.0380241 |

## Four-target macro

| Split | Ablation node MAE | Main BTD-FL node MAE | Scarce-local node MAE | Ablation grid MAE |
|---|---:|---:|---:|---:|
| audit | 0.000318578 | 0.000313392 | 0.000327046 | 0.00397954 |
| validation | 0.000346766 | 0.000344819 | 0.000371046 | 0.00433306 |

Win counts: versus main BTD-FL = 2; versus scarce-local = 3.

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
