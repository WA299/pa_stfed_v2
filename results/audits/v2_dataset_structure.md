# V2 Dataset Structure

Generated at (UTC): `2026-09-18T20:55:27.974619+00:00`

This report lists relative paths, extensions, file sizes, and safely readable table headers/shapes.

## norway_4_lv_grids

Directories: 4  Files: 28  Tables read successfully: 28/28

| Relative path | Extension | Size (bytes) | Table status | Fields | Shape |
| --- | --- | ---: | --- | --- | --- |
| `39_bus_semi_urban_reference_grid/branch_data_extra.csv` | `.csv` | 1184 | success | Unnamed: 0, fbus, tbus, Branch type, Length [km] | (38, 5) |
| `39_bus_semi_urban_reference_grid/load_bus_extra.csv` | `.csv` | 834 | success | bus_i, Consumer type, ID | (29, 3) |
| `39_bus_semi_urban_reference_grid/mpc_base_mva.csv` | `.csv` | 19 | success | Unnamed: 0, Base MVA | (1, 2) |
| `39_bus_semi_urban_reference_grid/mpc_branch.csv` | `.csv` | 4140 | success | Unnamed: 0, fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax | (38, 14) |
| `39_bus_semi_urban_reference_grid/mpc_bus.csv` | `.csv` | 2819 | success | bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, basekV, zone, maxVm, minVm, Vmax, Vmin | (39, 15) |
| `39_bus_semi_urban_reference_grid/p_load.csv` | `.csv` | 2774133 | success | Date, 16, 14, 19, 35, 6, 13, 26, 17, 15, 31, 39, 37, 9, 10, 22, 33, 29, 11, 36, 23, 2, 24, 27, 20, 4, 7, 30, 30.1, 5 | (8760, 30) |
| `39_bus_semi_urban_reference_grid/q_load.csv` | `.csv` | 5852398 | success | Date, 16, 14, 19, 35, 6, 13, 26, 17, 15, 31, 39, 37, 9, 10, 22, 33, 29, 11, 36, 23, 2, 24, 27, 20, 4, 7, 30, 30.1, 5 | (8760, 30) |
| `50_bus_rural_reference_grid/branch_extra.csv` | `.csv` | 1957 | success | Unnamed: 0, From Bus, To Bus, Length [km], Branch type | (49, 5) |
| `50_bus_rural_reference_grid/load_bus_extra.csv` | `.csv` | 515 | success | Unnamed: 0, bus_i, Consumer type | (21, 3) |
| `50_bus_rural_reference_grid/mpc_base_mva.csv` | `.csv` | 18 | success | Base MVA | (1, 1) |
| `50_bus_rural_reference_grid/mpc_branch.csv` | `.csv` | 3997 | success | Unnamed: 0, fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax | (49, 14) |
| `50_bus_rural_reference_grid/mpc_bus.csv` | `.csv` | 2751 | success | bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, basekV, zone, Vmax, Vmin | (50, 13) |
| `50_bus_rural_reference_grid/p_load.csv` | `.csv` | 2007314 | success | Date, 6, 10, 11, 12, 15, 16, 18, 20, 25, 29, 30, 31, 33, 36, 39, 40, 42, 43, 48, 49, 50 | (8760, 22) |
| `50_bus_rural_reference_grid/q_load.csv` | `.csv` | 1674108 | success | Date, 6, 10, 11, 12, 15, 16, 18, 20, 25, 29, 30, 31, 33, 36, 39, 40, 42, 43, 48, 49, 50 | (8760, 22) |
| `56_bus_semi_urban_reference_grid/branch_data_extra.csv` | `.csv` | 1769 | success | Unnamed: 0, fbus, tbus, Branch type, Length [km] | (55, 5) |
| `56_bus_semi_urban_reference_grid/load_bus_extra.csv` | `.csv` | 1375 | success | bus_i, Consumer type, ID | (53, 3) |
| `56_bus_semi_urban_reference_grid/mpc_base_mva.csv` | `.csv` | 19 | success | Unnamed: 0, Base MVA | (1, 2) |
| `56_bus_semi_urban_reference_grid/mpc_branch.csv` | `.csv` | 5988 | success | Unnamed: 0, fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax | (55, 14) |
| `56_bus_semi_urban_reference_grid/mpc_bus.csv` | `.csv` | 4044 | success | bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, basekV, zone, maxVm, minVm, Vmax, Vmin | (56, 15) |
| `56_bus_semi_urban_reference_grid/p_load.csv` | `.csv` | 4914816 | success | Date, 54, 54.1, 54.2, 54.3, 54.4, 54.5, 35, 35.1, 48, 23, 55, 24, 49, 18, 45, 28, 34, 34.1, 47, 4, 5, 17, 3, 3.1, 40, 56, 27, 29, 16, 20, 6, 9, 39, 52, 52.1, 33, 30, 12, 44, 11, 38, 46, 22, 36, 51, 50, 7, 21, 26, 32, 43, 13, 15 | (8760, 54) |
| `56_bus_semi_urban_reference_grid/q_load.csv` | `.csv` | 10530088 | success | Date, 54, 54.1, 54.2, 54.3, 54.4, 54.5, 35, 35.1, 48, 23, 55, 24, 49, 18, 45, 28, 34, 34.1, 47, 4, 5, 17, 3, 3.1, 40, 56, 27, 29, 16, 20, 6, 9, 39, 52, 52.1, 33, 30, 12, 44, 11, 38, 46, 22, 36, 51, 50, 7, 21, 26, 32, 43, 13, 15 | (8760, 54) |
| `80_bus_rural_reference_grid/branch_extra.csv` | `.csv` | 3150 | success | Unnamed: 0, From Bus, To Bus, Length [km], Branch type | (79, 5) |
| `80_bus_rural_reference_grid/load_bus_extra.csv` | `.csv` | 779 | success | Unnamed: 0, bus_i, Consumer type | (32, 3) |
| `80_bus_rural_reference_grid/mpc_base_mva.csv` | `.csv` | 20 | success | Base MVA | (1, 1) |
| `80_bus_rural_reference_grid/mpc_branch.csv` | `.csv` | 6463 | success | Unnamed: 0, fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax | (79, 14) |
| `80_bus_rural_reference_grid/mpc_bus.csv` | `.csv` | 4520 | success | bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, basekV, zone, Vmax, Vmin | (80, 13) |
| `80_bus_rural_reference_grid/p_load.csv` | `.csv` | 2924750 | success | Date, 8, 11, 12, 13, 14, 15, 18, 20, 22, 25, 27, 30, 36, 39, 41, 45, 46, 48, 50, 53, 56, 57, 58, 59, 60, 64, 71, 73, 74, 75, 77, 80 | (8760, 33) |
| `80_bus_rural_reference_grid/q_load.csv` | `.csv` | 2174132 | success | Date, 8, 11, 12, 13, 14, 15, 18, 20, 22, 25, 27, 30, 36, 39, 41, 45, 46, 48, 50, 53, 56, 57, 58, 59, 60, 64, 71, 73, 74, 75, 77, 80 | (8760, 33) |

## norway_industrial_mvlv

Directories: 0  Files: 48  Tables read successfully: 48/48

| Relative path | Extension | Size (bytes) | Table status | Fields | Shape |
| --- | --- | ---: | --- | --- | --- |
| `branch.csv` | `.csv` | 5768 | success | F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX, PF, QF, PT, QT, MU_SF, MU_ST, MU_ANGMIN, MU_ANGMAX | (75, 21) |
| `bus.csv` | `.csv` | 3513 | success | BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN | (76, 13) |
| `gen.csv` | `.csv` | 154 | success | GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PC1, PC2, QC1MIN, QC1MAX, QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN | (0, 24) |
| `r1v0.23b2.txt` | `.txt` | 991004 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.23b3.txt` | `.txt` | 981776 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b10.txt` | `.txt` | 1039730 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b11.txt` | `.txt` | 1021074 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b12.txt` | `.txt` | 1011090 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b13.txt` | `.txt` | 1039484 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b14.txt` | `.txt` | 1033476 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b15.txt` | `.txt` | 1038646 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b16.txt` | `.txt` | 1039648 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b17.txt` | `.txt` | 1040395 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b18.txt` | `.txt` | 1011143 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b19.txt` | `.txt` | 1012246 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b20.txt` | `.txt` | 1034252 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b21.txt` | `.txt` | 1036515 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b22.txt` | `.txt` | 1041256 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b23.txt` | `.txt` | 1033283 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b24.txt` | `.txt` | 1040616 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b25.txt` | `.txt` | 1046948 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b26.txt` | `.txt` | 1017657 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b27.txt` | `.txt` | 1034067 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b28.txt` | `.txt` | 1009733 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b29.txt` | `.txt` | 1010729 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b30.txt` | `.txt` | 1012141 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b31.txt` | `.txt` | 1011940 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b32.txt` | `.txt` | 1010850 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b33.txt` | `.txt` | 1008657 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b34.txt` | `.txt` | 1010493 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b35.txt` | `.txt` | 1036201 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b36.txt` | `.txt` | 1010229 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b37.txt` | `.txt` | 1044038 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b38.txt` | `.txt` | 1037359 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b39.txt` | `.txt` | 978986 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b40.txt` | `.txt` | 1012861 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b41.txt` | `.txt` | 614224 | success | Bus_ID, Timestamp, Load_kWh | (16273, 3) |
| `r1v0.415b42.txt` | `.txt` | 1051987 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b43.txt` | `.txt` | 1051160 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b44.txt` | `.txt` | 1039838 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b7.txt` | `.txt` | 1060840 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b8.txt` | `.txt` | 1027439 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r1v0.415b9.txt` | `.txt` | 799620 | success | Bus_ID, Timestamp, Load_kWh | (21553, 3) |
| `r2v0.415b3.txt` | `.txt` | 509821 | success | Bus_ID, Timestamp, Load_kWh | (13345, 3) |
| `r2v0.415b4.txt` | `.txt` | 520825 | success | Bus_ID, Timestamp, Load_kWh | (13345, 3) |
| `r2v0.415b5.txt` | `.txt` | 1050454 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r2v0.415b6.txt` | `.txt` | 1025979 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
| `r2v0.415b8.txt` | `.txt` | 1024739 | success | Bus_ID, Timestamp, Load_kWh | (26690, 3) |
