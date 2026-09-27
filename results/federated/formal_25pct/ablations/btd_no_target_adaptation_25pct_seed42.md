# Formal BTD-FL Ablation: btd_no_target_adaptation

Mechanism changed: donor temporal state injected without target proxy adaptation.
The accepted main BTD-FL JSON is a frozen reference only; no baseline is retrained.
Canonical validation is evaluation-only. Canonical test is never evaluated.

## Per-target audit node/grid metrics

| Target | Ablation node MAE | Main BTD-FL node MAE | Scarce-local node MAE | Ablation grid MAE |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000350549 | 0.000350049 | 0.000372181 | 0.00384389 |
| 50_bus_rural_reference_grid | 0.000291846 | 0.000292519 | 0.000306918 | 0.00245226 |
| 56_bus_semi_urban_reference_grid | 0.000402978 | 0.000404865 | 0.000417712 | 0.00614545 |
| 80_bus_rural_reference_grid | 0.000208861 | 0.000206135 | 0.000211373 | 0.0030316 |

## Per-target canonical-validation node/grid metrics

| Target | Ablation node MAE | Main BTD-FL node MAE | Scarce-local node MAE | Ablation grid MAE |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000389131 | 0.000390059 | 0.000416143 | 0.00438617 |
| 50_bus_rural_reference_grid | 0.00033289 | 0.000336294 | 0.000385882 | 0.00302237 |
| 56_bus_semi_urban_reference_grid | 0.000443694 | 0.000446751 | 0.000470452 | 0.00714799 |
| 80_bus_rural_reference_grid | 0.000209357 | 0.000206175 | 0.000211706 | 0.00304136 |

## Donors and relative change

| Target | Donor | Rule | Audit vs BTD-FL | Validation vs BTD-FL | Audit vs local | Validation vs local |
|---|---|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 56_bus_semi_urban_reference_grid | frozen_main_btd_fl_selected_donor | -0.00142775 | 0.00237782 | 0.0581218 | 0.0649114 |
| 50_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | frozen_main_btd_fl_selected_donor | 0.00230169 | 0.0101208 | 0.0491085 | 0.137326 |
| 56_bus_semi_urban_reference_grid | 39_bus_semi_urban_reference_grid | frozen_main_btd_fl_selected_donor | 0.0046605 | 0.00684147 | 0.0352721 | 0.0568775 |
| 80_bus_rural_reference_grid | 50_bus_rural_reference_grid | frozen_main_btd_fl_selected_donor | -0.0132235 | -0.0154347 | 0.0118842 | 0.0110976 |

## Four-target macro

| Split | Ablation node MAE | Main BTD-FL node MAE | Scarce-local node MAE | Ablation grid MAE |
|---|---:|---:|---:|---:|
| audit | 0.000313559 | 0.000313392 | 0.000327046 | 0.0038683 |
| validation | 0.000343768 | 0.000344819 | 0.000371046 | 0.00439947 |

Win counts: versus main BTD-FL = 3; versus scarce-local = 4.

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
