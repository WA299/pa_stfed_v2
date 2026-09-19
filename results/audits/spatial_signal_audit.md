# V2 Train-only Spatial Signal Audit

This is a structural diagnostic, not a significance test or a model result.

time_scope: `train_only`; node_scope: `load_buses_only`; distance_sources: `canonical_loader`.

| Grid | Load buses | Load-load pairs | Expected pairs | Train samples |
| --- | ---: | ---: | ---: | ---: |
| 39_bus_semi_urban_reference_grid | 28 | 378 | 378 | 6132 |

## 39_bus_semi_urban_reference_grid

| Sequence | Hop group | Pairs | Finite | Excluded | Mean | Median | Q25 | Q75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_standardized | hop=1 | 1 | 1 | 0 | 0.220192 | 0.220192 | 0.220192 | 0.220192 |
| raw_standardized | hop=2 | 26 | 26 | 0 | 0.495967 | 0.485941 | 0.405293 | 0.596318 |
| raw_standardized | hop=3-4 | 140 | 140 | 0 | 0.462053 | 0.456591 | 0.377686 | 0.551467 |
| raw_standardized | hop>=5 | 211 | 211 | 0 | 0.478886 | 0.487733 | 0.355795 | 0.611805 |
| hour_of_week_residual | hop=1 | 1 | 1 | 0 | 0.1926 | 0.1926 | 0.1926 | 0.1926 |
| hour_of_week_residual | hop=2 | 26 | 26 | 0 | 0.516301 | 0.537392 | 0.407699 | 0.617083 |
| hour_of_week_residual | hop=3-4 | 140 | 140 | 0 | 0.47589 | 0.475736 | 0.387124 | 0.570854 |
| hour_of_week_residual | hop>=5 | 211 | 211 | 0 | 0.479284 | 0.50408 | 0.339196 | 0.627293 |
| one_hour_change | hop=1 | 1 | 1 | 0 | 0.0113011 | 0.0113011 | 0.0113011 | 0.0113011 |
| one_hour_change | hop=2 | 26 | 26 | 0 | 0.0123174 | 0.0104341 | -0.00119998 | 0.0251812 |
| one_hour_change | hop=3-4 | 140 | 140 | 0 | 0.0122622 | 0.0149311 | -0.00359797 | 0.0281909 |
| one_hour_change | hop>=5 | 211 | 211 | 0 | 0.0164831 | 0.0152855 | 0.00371703 | 0.0307228 |

| Sequence | Spearman(corr, -hop) | Spearman(corr, -impedance) | Adjacent mean/median | Non-adjacent mean/median |
| --- | ---: | ---: | --- | --- |
| raw_standardized | -0.026677 | -0.0500772 | 0.220192 / 0.220192 | 0.473813 / 0.470751 |
| hour_of_week_residual | 0.0057643 | 0.0249079 | 0.1926 / 0.1926 | 0.480576 / 0.490853 |
| one_hour_change | -0.0963368 | -0.108544 | 0.0113011 / 0.0113011 | 0.0146284 / 0.0147567 |

Zero-variance/invalid policy: correlation=None; excluded from finite summaries and Spearman, with exclusion counts/reasons recorded

| 50_bus_rural_reference_grid | 21 | 210 | 210 | 6132 |

## 50_bus_rural_reference_grid

| Sequence | Hop group | Pairs | Finite | Excluded | Mean | Median | Q25 | Q75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_standardized | hop=1 | 0 | 0 | 0 | - | - | - | - |
| raw_standardized | hop=2 | 5 | 5 | 0 | 0.526457 | 0.462316 | 0.451369 | 0.656071 |
| raw_standardized | hop=3-4 | 19 | 19 | 0 | 0.484897 | 0.579268 | 0.48938 | 0.642806 |
| raw_standardized | hop>=5 | 186 | 186 | 0 | 0.438279 | 0.519956 | 0.375776 | 0.665495 |
| hour_of_week_residual | hop=1 | 0 | 0 | 0 | - | - | - | - |
| hour_of_week_residual | hop=2 | 5 | 5 | 0 | 0.548568 | 0.512048 | 0.478367 | 0.675105 |
| hour_of_week_residual | hop=3-4 | 19 | 19 | 0 | 0.499465 | 0.599302 | 0.501761 | 0.653518 |
| hour_of_week_residual | hop>=5 | 186 | 186 | 0 | 0.428455 | 0.549408 | 0.409739 | 0.686782 |
| one_hour_change | hop=1 | 0 | 0 | 0 | - | - | - | - |
| one_hour_change | hop=2 | 5 | 5 | 0 | 0.0122177 | 0.00363275 | -0.000870992 | 0.0318768 |
| one_hour_change | hop=3-4 | 19 | 19 | 0 | -0.00199368 | 0.0046518 | -0.0182467 | 0.0238352 |
| one_hour_change | hop>=5 | 186 | 186 | 0 | 0.0114796 | 0.00819945 | -0.00517958 | 0.0211813 |

| Sequence | Spearman(corr, -hop) | Spearman(corr, -impedance) | Adjacent mean/median | Non-adjacent mean/median |
| --- | ---: | ---: | --- | --- |
| raw_standardized | 0.0256889 | 0.0651191 | - / - | 0.444596 / 0.526492 |
| hour_of_week_residual | 0.022779 | 0.0664654 | - / - | 0.437739 / 0.555617 |
| one_hour_change | 0.0381031 | 0.146808 | - / - | 0.0102781 / 0.00797353 |

Zero-variance/invalid policy: correlation=None; excluded from finite summaries and Spearman, with exclusion counts/reasons recorded

| 56_bus_semi_urban_reference_grid | 44 | 946 | 946 | 6132 |

## 56_bus_semi_urban_reference_grid

| Sequence | Hop group | Pairs | Finite | Excluded | Mean | Median | Q25 | Q75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_standardized | hop=1 | 0 | 0 | 0 | - | - | - | - |
| raw_standardized | hop=2 | 81 | 81 | 0 | 0.589503 | 0.598098 | 0.515387 | 0.690164 |
| raw_standardized | hop=3-4 | 236 | 236 | 0 | 0.596746 | 0.62797 | 0.513637 | 0.700059 |
| raw_standardized | hop>=5 | 629 | 629 | 0 | 0.595218 | 0.611216 | 0.528143 | 0.683751 |
| hour_of_week_residual | hop=1 | 0 | 0 | 0 | - | - | - | - |
| hour_of_week_residual | hop=2 | 81 | 81 | 0 | 0.594367 | 0.609399 | 0.518387 | 0.711463 |
| hour_of_week_residual | hop=3-4 | 236 | 236 | 0 | 0.610756 | 0.642185 | 0.508491 | 0.71998 |
| hour_of_week_residual | hop>=5 | 629 | 629 | 0 | 0.605403 | 0.628331 | 0.533187 | 0.698718 |
| one_hour_change | hop=1 | 0 | 0 | 0 | - | - | - | - |
| one_hour_change | hop=2 | 81 | 81 | 0 | 0.0206735 | 0.0195805 | 0.00853871 | 0.0320111 |
| one_hour_change | hop=3-4 | 236 | 236 | 0 | 0.0237344 | 0.0182047 | 0.0048445 | 0.0315756 |
| one_hour_change | hop>=5 | 629 | 629 | 0 | 0.0198145 | 0.0173812 | 0.00305112 | 0.0332663 |

| Sequence | Spearman(corr, -hop) | Spearman(corr, -impedance) | Adjacent mean/median | Non-adjacent mean/median |
| --- | ---: | ---: | --- | --- |
| raw_standardized | -0.0355253 | 0.0777678 | - / - | 0.59511 / 0.613172 |
| hour_of_week_residual | -0.0379182 | 0.0737712 | - / - | 0.605793 / 0.628299 |
| one_hour_change | 0.0129891 | -0.041048 | - / - | 0.020866 / 0.0175428 |

Zero-variance/invalid policy: correlation=None; excluded from finite summaries and Spearman, with exclusion counts/reasons recorded

| 80_bus_rural_reference_grid | 32 | 496 | 496 | 6132 |

## 80_bus_rural_reference_grid

| Sequence | Hop group | Pairs | Finite | Excluded | Mean | Median | Q25 | Q75 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_standardized | hop=1 | 0 | 0 | 0 | - | - | - | - |
| raw_standardized | hop=2 | 4 | 4 | 0 | 0.31619 | 0.349315 | 0.0861613 | 0.579344 |
| raw_standardized | hop=3-4 | 42 | 42 | 0 | 0.255901 | 0.24691 | -0.0344951 | 0.54574 |
| raw_standardized | hop>=5 | 450 | 450 | 0 | 0.111284 | 0.063164 | -0.117525 | 0.351726 |
| hour_of_week_residual | hop=1 | 0 | 0 | 0 | - | - | - | - |
| hour_of_week_residual | hop=2 | 4 | 4 | 0 | 0.30922 | 0.347768 | 0.0628246 | 0.594163 |
| hour_of_week_residual | hop=3-4 | 42 | 42 | 0 | 0.253779 | 0.237813 | -0.0442932 | 0.556142 |
| hour_of_week_residual | hop>=5 | 450 | 450 | 0 | 0.107002 | 0.0473846 | -0.129437 | 0.35458 |
| one_hour_change | hop=1 | 0 | 0 | 0 | - | - | - | - |
| one_hour_change | hop=2 | 4 | 4 | 0 | -0.00102416 | -0.0112501 | -0.017876 | 0.00560168 |
| one_hour_change | hop=3-4 | 42 | 42 | 0 | 0.00237494 | 0.00498935 | -0.0107129 | 0.0154832 |
| one_hour_change | hop>=5 | 450 | 450 | 0 | 0.00790613 | 0.00658983 | -0.00510782 | 0.0197722 |

| Sequence | Spearman(corr, -hop) | Spearman(corr, -impedance) | Adjacent mean/median | Non-adjacent mean/median |
| --- | ---: | ---: | --- | --- |
| raw_standardized | 0.0408905 | 0.0890127 | - / - | 0.125182 / 0.0719514 |
| hour_of_week_residual | 0.0395712 | 0.0855827 | - / - | 0.121061 / 0.0640396 |
| one_hour_change | -0.101764 | -0.0739396 | - / - | 0.00736575 / 0.0061747 |

Zero-variance/invalid policy: correlation=None; excluded from finite summaries and Spearman, with exclusion counts/reasons recorded

