# Formal BTD-FL Mechanism Ablations: 25% History

The accepted main BTD-FL JSON is the frozen reference. Invariant FedAvg, FedProx, FedPer, and scarce-local results are not retrained.

| Ablation | Audit node MAE | Validation node MAE | Wins vs main BTD-FL | Wins vs scarce-local |
|---|---:|---:|---:|---:|
| btd_full_model_transfer | 0.000318578 | 0.000346766 | 2 | 3 |
| btd_no_benefit_selection | 0.000316701 | 0.000348316 | 0 | 4 |
| btd_no_target_adaptation | 0.000313559 | 0.000343768 | 3 | 4 |

test_evaluated: false
validation_used_for_selection: false
audit_used_for_selection: false
