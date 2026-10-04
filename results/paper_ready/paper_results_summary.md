# Frozen Results Digest

All statements below use frozen JSON artifacts and three-seed mean ± standard deviation; no statistical significance claim is made. Reference-grid canonical validation is development validation, while industrial canonical validation is external industrial validation.

## Temporal dominance

The centralized macro evidence includes the GRU anchor, temporal-residual-only model, Graph WaveNet, and conditional-utility V2 evidence. The temporal-residual-only variant improves the GRU anchor in node-macro MAE, while physical-topology variants are not uniformly better.

## Reference-grid robustness

At both 25% and 50% history, BTD-FL is compared using exact same-regime three-seed summaries. Relative improvements are reported in Table B and Figure 2; these are robustness summaries, not significance tests.

## Industrial external validation

BTD-FL's primary industrial node-macro MAE is compared within the industrial domain only. RMSE and grid-aggregate metrics provide supporting evidence, while node WAPE and sMAPE are not uniformly improved. Industrial values are hourly industrial load/energy measurements and are not instantaneous MW.

## Mechanism and benefit behavior

The ablations distinguish donor selection, transfer scope, and target adaptation. Directed benefits are target-conditioned calibration quantities, not causal benefits. Industrial positive/non-positive directed benefit counts are reported directly; zero-transfer fallback was not empirically triggered in the accepted summaries and no fallback-performance claim is made.

Canonical TEST remains locked and no TEST claim is made.
