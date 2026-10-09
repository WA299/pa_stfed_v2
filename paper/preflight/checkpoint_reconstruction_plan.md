# Frozen checkpoint reconstruction plan

The accepted matrix contains exactly 180 target-model cells: 144 reference-grid cells (3 seeds × 2 history fractions × 6 methods × 4 targets) and 36 industrial cells (3 × 2 × 6 × 1). Public `local` maps to `scarce_local` for reference grids and `industrial_scarce_local` for the industrial target.

Reconstruction is offline and uses only the frozen scientific commit (`1681741779df301dcb64587090ce262d2ad1d443`) and TRAIN/FIT/CALIBRATION data. It must export tensor-only state dictionaries at the existing selection points: calibration-best Local/BTD, final round-10 FedAvg/FedProx/FedPer/FedFomo-style. No TEST targets may be read during reconstruction.

Every checkpoint requires a sidecar with model/configuration, graph/topology, FIT-only scaler provenance, feature contract (`P + calendar`, indices `0,2,3,4,5,6`), target node order/mask, source and frozen commit identities, selection point, hash, environment, and all TEST flags false. Writes are atomic and the checkpoint is accepted only with a matching SHA256 and complete tensor state.

Before TEST unlock, all 180 identities must validate and every AUDIT/canonical-validation metric must pass the fixed parity policy. Missing, corrupt, incomplete, or unexplained mismatches halt TEST readiness. No post-hoc seed selection, tuning, or method change is permitted.

The implementation is currently plan-only; real reconstruction has not been executed.
