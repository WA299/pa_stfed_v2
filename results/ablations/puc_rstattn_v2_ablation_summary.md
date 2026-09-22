# PUC-RSTAttn V2 Formal Ablation Summary

Validation-only metrics; frozen full-model results are loaded from centralized outputs.

## Per-grid node macro

| Grid | Model | MAE | RMSE | WAPE (%) | sMAPE (%) |
|---|---|---:|---:|---:|---:|
| 39bus | gru_anchor_only | 0.000396929 | 0.000575579 | 29.0493 | 29.1097 |
| 39bus | temporal_residual_only | 0.000375164 | 0.000542233 | 27.2702 | 27.3695 |
| 39bus | utility_uniform_spatial | 0.000376994 | 0.000550085 | 27.4973 | 27.6215 |
| 39bus | utility_prior_no_physics | 0.000376442 | 0.000548342 | 27.3972 | 27.5061 |
| 39bus | full_puc_rstattn_v2 | 0.000376738 | 0.000550121 | 27.4241 | 27.5046 |
| 50bus | gru_anchor_only | 0.000335095 | 0.000509787 | 31.9404 | 33.9837 |
| 50bus | temporal_residual_only | 0.000319913 | 0.000503242 | 30.6807 | 33.4531 |
| 50bus | utility_uniform_spatial | 0.000321364 | 0.000506719 | 31.5694 | 34.7484 |
| 50bus | utility_prior_no_physics | 0.000320547 | 0.000498527 | 30.8356 | 32.7007 |
| 50bus | full_puc_rstattn_v2 | 0.000319189 | 0.000497551 | 30.9487 | 32.7366 |
| 56bus | gru_anchor_only | 0.000447376 | 0.000627382 | 26.3344 | 27.6745 |
| 56bus | temporal_residual_only | 0.000431032 | 0.00060633 | 25.2908 | 26.5394 |
| 56bus | utility_uniform_spatial | 0.000437897 | 0.000617708 | 25.7516 | 27.065 |
| 56bus | utility_prior_no_physics | 0.000437014 | 0.000617524 | 25.7004 | 27.0221 |
| 56bus | full_puc_rstattn_v2 | 0.000436791 | 0.000616188 | 25.6937 | 27.0097 |
| 80bus | gru_anchor_only | 0.000199762 | 0.000358149 | 42.4904 | 49.4964 |
| 80bus | temporal_residual_only | 0.000190923 | 0.000350724 | 44.8543 | 50.6516 |
| 80bus | utility_uniform_spatial | 0.000190411 | 0.000354663 | 42.1724 | 47.7669 |
| 80bus | utility_prior_no_physics | 0.000195097 | 0.00035823 | 50.6282 | 50.9767 |
| 80bus | full_puc_rstattn_v2 | 0.000194356 | 0.000356072 | 50.3242 | 51.0451 |

## Per-grid grid aggregate

| Grid | Model | MAE | RMSE | WAPE (%) | sMAPE (%) |
|---|---|---:|---:|---:|---:|
| 39bus | gru_anchor_only | 0.00424555 | 0.00519702 | 10.4786 | 10.8925 |
| 39bus | temporal_residual_only | 0.00355891 | 0.00451377 | 8.78389 | 8.95634 |
| 39bus | utility_uniform_spatial | 0.00415786 | 0.0051481 | 10.2622 | 10.5369 |
| 39bus | utility_prior_no_physics | 0.00394982 | 0.00494621 | 9.7487 | 9.93887 |
| 39bus | full_puc_rstattn_v2 | 0.00407731 | 0.00506964 | 10.0634 | 10.2778 |
| 50bus | gru_anchor_only | 0.00265037 | 0.00345746 | 6.87369 | 6.89188 |
| 50bus | temporal_residual_only | 0.00282609 | 0.00366028 | 7.32942 | 7.39868 |
| 50bus | utility_uniform_spatial | 0.00294465 | 0.00379431 | 7.63689 | 7.69526 |
| 50bus | utility_prior_no_physics | 0.00262429 | 0.00342198 | 6.80606 | 6.82825 |
| 50bus | full_puc_rstattn_v2 | 0.00262815 | 0.00343107 | 6.81606 | 6.83364 |
| 56bus | gru_anchor_only | 0.00659355 | 0.00796741 | 8.27637 | 8.57188 |
| 56bus | temporal_residual_only | 0.00608521 | 0.00748333 | 7.63829 | 7.84779 |
| 56bus | utility_uniform_spatial | 0.00708774 | 0.00842911 | 8.89669 | 9.21562 |
| 56bus | utility_prior_no_physics | 0.00717888 | 0.00852648 | 9.01108 | 9.29498 |
| 56bus | full_puc_rstattn_v2 | 0.00703414 | 0.00837216 | 8.8294 | 9.11656 |
| 80bus | gru_anchor_only | 0.00287526 | 0.00363064 | 11.8989 | 12.5906 |
| 80bus | temporal_residual_only | 0.00297919 | 0.00366869 | 12.329 | 13.1389 |
| 80bus | utility_uniform_spatial | 0.00295473 | 0.00368398 | 12.2278 | 13.0497 |
| 80bus | utility_prior_no_physics | 0.00292167 | 0.00370513 | 12.091 | 12.9088 |
| 80bus | full_puc_rstattn_v2 | 0.00289949 | 0.00367837 | 11.9992 | 12.8265 |

## Cross-grid node_macro

| Model | MAE | RMSE | WAPE (%) | sMAPE (%) |
|---|---:|---:|---:|---:|
| gru_anchor_only | 0.00034479 | 0.000517724 | 32.4536 | 35.0661 |
| temporal_residual_only | 0.000329258 | 0.000500632 | 32.024 | 34.5034 |
| utility_uniform_spatial | 0.000331667 | 0.000507294 | 31.7477 | 34.3005 |
| utility_prior_no_physics | 0.000332275 | 0.000505656 | 33.6403 | 34.5514 |
| full_puc_rstattn_v2 | 0.000331769 | 0.000504983 | 33.5977 | 34.574 |

## Cross-grid grid_aggregate

| Model | MAE | RMSE | WAPE (%) | sMAPE (%) |
|---|---:|---:|---:|---:|
| gru_anchor_only | 0.00409118 | 0.00506313 | 9.3819 | 9.7367 |
| temporal_residual_only | 0.00386235 | 0.00483152 | 9.02016 | 9.33542 |
| utility_uniform_spatial | 0.00428624 | 0.00526388 | 9.75589 | 10.1244 |
| utility_prior_no_physics | 0.00416866 | 0.00514995 | 9.41421 | 9.74273 |
| full_puc_rstattn_v2 | 0.00415977 | 0.00513781 | 9.42701 | 9.76361 |

## Numeric node-macro MAE deltas

| Comparison | Absolute delta | Relative (%) |
|---|---:|---:|
| temporal_residual_only_vs_gru_anchor_only | -1.55322e-05 | -4.50483 |
| utility_uniform_spatial_vs_temporal_residual_only | 2.40877e-06 | 0.731576 |
| utility_prior_no_physics_vs_utility_uniform_spatial | 6.08069e-07 | 0.183337 |
| full_puc_rstattn_v2_vs_utility_prior_no_physics | -5.06163e-07 | -0.152332 |
