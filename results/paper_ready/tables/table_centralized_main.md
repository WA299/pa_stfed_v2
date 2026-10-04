# Centralized Architecture Evidence

| Method | Split | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Source | JSON key path |
|---|---|---|---|---|---|---|---|
| GRU anchor | development validation | 0.00034479027031545902 | 0.0005177241315166731 | 32.453634014593263 | 35.066072840433826 | results/ablations/puc_rstattn_v2_ablation_summary.json | macro_across_grids.node_macro.gru_anchor_only.metrics |
| temporal_residual_only | development validation | 0.00032925806226992394 | 0.00050063229858023474 | 32.024002258025504 | 34.503392949516893 | results/ablations/puc_rstattn_v2_ablation_summary.json | macro_across_grids.node_macro.temporal_residual_only.metrics |
| Graph WaveNet | development validation | 0.00033290577237537751 | 0.00051235858793632558 | 32.142007260454911 | 32.681817441314649 | results/centralized/centralized_validation_summary.json | macro_across_grids.node_macro.graph_wavenet.metrics |
| PUC-RSTAttn V2 | development validation | 0.0003317687407702807 | 0.00050498295223042533 | 33.597672406344493 | 34.574001633452262 | results/ablations/puc_rstattn_v2_ablation_summary.json | macro_across_grids.node_macro.full_puc_rstattn_v2.metrics |
