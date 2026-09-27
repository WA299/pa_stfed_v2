# Frozen 25%-History Multi-Seed Benchmark

Seeds `42`, `123`, and `2026` were fixed in advance to test robustness without post-hoc seed selection. Seed `42` is the accepted historical result and is read from the parent formal-result directory; it is never retrained by the multiseed launcher.

Seeds `123` and `2026` use exactly the accepted architecture, scarcity definition, 60/20/20 target split, fit-only scaler and graph, optimizer settings, federated budget, donor-selection rules, and evaluation isolation. The only experimental variable is the random seed.

Canonical validation is robustness evaluation only. Canonical test remains locked and is never loaded or evaluated. Interpret results using mean +/- standard deviation and the explicit per-seed values. Three seeds do not support statistical-significance claims.

Run the new seeds with `python scripts/run_multiseed_25pct.py --seeds 123 2026 --device cuda --summarize`, or aggregate completed artifacts with `python scripts/summarize_multiseed_25pct.py`.
