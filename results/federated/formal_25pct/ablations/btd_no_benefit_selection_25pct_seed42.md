# Formal BTD-FL Ablation: btd_no_benefit_selection

Mechanism changed: donor identity uses fixed lexicographic rule.
The accepted main BTD-FL JSON is a frozen reference only; no baseline is retrained.
Canonical validation is evaluation-only. Canonical test is never evaluated.

## Per-target audit node/grid metrics

| Target | Ablation node MAE | Main BTD-FL node MAE | Scarce-local node MAE | Ablation grid MAE |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000361034 | 0.000350049 | 0.000372181 | 0.00452146 |
| 50_bus_rural_reference_grid | 0.000292519 | 0.000292519 | 0.000306918 | 0.00244588 |
| 56_bus_semi_urban_reference_grid | 0.000404865 | 0.000404865 | 0.000417712 | 0.00642109 |
| 80_bus_rural_reference_grid | 0.000208387 | 0.000206135 | 0.000211373 | 0.00284235 |

## Per-target canonical-validation node/grid metrics

| Target | Ablation node MAE | Main BTD-FL node MAE | Scarce-local node MAE | Ablation grid MAE |
|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 0.000403456 | 0.000390059 | 0.000416143 | 0.00518262 |
| 50_bus_rural_reference_grid | 0.000336294 | 0.000336294 | 0.000385882 | 0.00306392 |
| 56_bus_semi_urban_reference_grid | 0.000446751 | 0.000446751 | 0.000470452 | 0.00766561 |
| 80_bus_rural_reference_grid | 0.000206764 | 0.000206175 | 0.000211706 | 0.00286344 |

## Donors and relative change

| Target | Donor | Rule | Audit vs BTD-FL | Validation vs BTD-FL | Audit vs local | Validation vs local |
|---|---|---|---:|---:|---:|---:|
| 39_bus_semi_urban_reference_grid | 50_bus_rural_reference_grid | fixed_deterministic_no_benefit | -0.0313815 | -0.0343479 | 0.0299493 | 0.0304877 |
| 50_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | fixed_deterministic_no_benefit | 0 | 0 | 0.0469148 | 0.128505 |
| 56_bus_semi_urban_reference_grid | 39_bus_semi_urban_reference_grid | fixed_deterministic_no_benefit | 0 | 0 | 0.0307549 | 0.0503807 |
| 80_bus_rural_reference_grid | 39_bus_semi_urban_reference_grid | fixed_deterministic_no_benefit | -0.0109231 | -0.00286118 | 0.0141276 | 0.0233427 |

## Four-target macro

| Split | Ablation node MAE | Main BTD-FL node MAE | Scarce-local node MAE | Ablation grid MAE |
|---|---:|---:|---:|---:|
| audit | 0.000316701 | 0.000313392 | 0.000327046 | 0.00405769 |
| validation | 0.000348316 | 0.000344819 | 0.000371046 | 0.0046939 |

Win counts: versus main BTD-FL = 0; versus scarce-local = 4.

## Guardrails

validation_used_for_selection: false
audit_used_for_selection: false
test_evaluated: false
