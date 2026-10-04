# Centralized architecture evidence

| Method | Split | Node MAE | Node RMSE | Node WAPE % | Node sMAPE % | Source | JSON key path |
|---|---|---|---|---|---|---|---|
| GRU anchor | development validation | 0.00034479027 | 0.00051772413 | 32.453634 | 35.066073 | results\centralized\centralized_validation_summary.json | macro_across_grids.node_macro.shared_gru.metrics |
| temporal_residual_only | development validation | 0.00033176874 | 0.00050498295 | 33.597672 | 34.574002 | results\centralized\centralized_validation_summary.json | macro_across_grids.node_macro.temporal_residual_only.metrics |
| Graph WaveNet | development validation | 0.00033290577 | 0.00051235859 | 32.142007 | 32.681817 | results\centralized\centralized_validation_summary.json | macro_across_grids.node_macro.graph_wavenet.metrics |
| conditional-utility V2 | development validation | 0.00033176874 | 0.00050498295 | 33.597672 | 34.574002 | results\centralized\centralized_validation_summary.json | macro_across_grids.node_macro.puc_rstattn_v2.metrics |
