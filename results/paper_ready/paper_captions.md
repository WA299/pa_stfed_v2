# Paper-ready Captions

**Table A.** Centralized architecture evidence on development validation across four reference grids. Metrics are unweighted grid-level node macros from explicit accepted method entries.

**Table B.** Reference-grid three-seed consistency and robustness at 25% and 50% target history. Canonical validation is development validation; the separate comparison table reports BTD-FL relative node-MAE improvement and seed wins.

**Table C.** External industrial validation on hourly industrial load/energy measurements. Absolute MAE is compared only within the industrial target domain. Node WAPE and sMAPE are not uniformly improved.

**Table D.** Seed-42 BTD-FL mechanism ablations separating WHO is selected, WHAT state is transferred, and whether target-adapted probe state is reused.

**Table E.** Target-conditioned directed transfer benefit and selected donor behavior across history fractions and seeds. These calibration quantities are not causal effects.

**Figure 1.** Mean +/- population standard deviation of validation node-macro MAE across seeds 42, 123, and 2026. Reference-grid development validation and external industrial validation use separate axes because their units and scales are not comparable.

**Figure 2.** BTD-FL relative validation node-MAE improvement (%) versus Local, FedAvg, and FedFomo-style. Values summarize three-seed consistency, not an inferential test.

**Figure 3.** Industrial target-conditioned directed calibration benefit by seed and available-history fraction. Columns follow donors 39, 50, 56, and 80; black outlines mark the frozen selected donor.
