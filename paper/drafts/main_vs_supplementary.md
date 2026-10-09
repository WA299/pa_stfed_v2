# Main text versus supplementary material

## Main text

1. **Dataset and protocol table**
   - Four reference grids and `norway_industrial_mvlv`.
   - 25%/50% target history, three seeds, history 168, horizon 1, frozen methods, FIT/CALIBRATION/AUDIT roles, and TEST lock.
2. **Reference multiseed main table**
   - 25% and 50% validation/audit node-macro MAE mean and standard deviation for Local, FedAvg, FedProx, FedPer, FedFomo-style, and BTD-FL.
   - Separate BTD comparison columns/table for relative improvement, standard deviation, and seed wins.
3. **Industrial multiseed main table**
   - 25% and 50% industrial validation node-macro MAE, RMSE, WAPE, sMAPE, and grid-aggregate MAE mean and standard deviation.
   - Separate industrial BTD comparison table with mean relative MAE improvement, standard deviation, and seed wins.
4. **BTD conceptual schematic**
   - Target local proxy and FIT-only graph/scaler.
   - Four donor probes, calibration-only directed benefit, positive argmax selection, discarded adapted probe, raw temporal-only transfer, target-local spatial state, and audit/validation evaluation.
5. **Relative-gain figure**
   - BTD validation node-MAE relative improvement versus Local, FedAvg, and FedFomo-style at both history fractions, with reference and industrial domains distinguished.
6. **Industrial benefit matrix**
   - Four donor columns (39/50/56/80), six seed/history rows, target-conditioned directed calibration benefits, and explicit selected-donor markers.

## Supplementary material

1. **Centralized full metrics**
   - GRU anchor, temporal-only, Graph WaveNet, utility variants, and full PUC-RSTAttn V2 with MAE, RMSE, WAPE, and sMAPE.
2. **Mechanism ablations**
   - Final direct raw temporal transfer, historical adapted-state BTD, no-benefit-selection, and full-model transfer.
   - Explicit WHO/WHAT/STATE columns and the no-benefit-selection control limitation.
3. **Complete donor table**
   - Per-target/per-seed reference donor behavior and per-seed industrial donor benefits, selected donors, and fallback flags.
4. **Additional metrics**
   - Audit metrics, industrial percentage metrics, grid-aggregate metrics, and per-seed values.
5. **History-fraction absolute MAE figure**
   - Separate reference and industrial axes to avoid cross-domain scale conflation.
6. **Provenance and preflight**
   - Full paper-results manifest, checkpoint inventory, 180-cell evaluation matrix, and TEST-opening protocol.
