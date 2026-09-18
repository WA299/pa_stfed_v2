# V2 Schema and ID Mapping Audit

Generated at (UTC): `2026-09-18T21:35:02.067115+00:00`

Scope is limited to schema, timestamps, and identifier mappings; no model or forecasting analysis is performed.

## LV grid: 39_bus_semi_urban_reference_grid

bus_count=39; branch_count=38; consumer_count=29; unique_load_bus_count=28
Multiple consumers on one bus observed: **yes**.
Multiple consumers per bus allowed by row mapping: **yes**.
P/Q profile one-to-one: **yes**.
Unmapped profile count: **0**.
p_header_bus_mismatch_count: **0**; q_header_bus_mismatch_count: **0**.
Positional mapping has header semantics support: **yes**.
profile_to_bus_mapping_verified: **yes**.

| File | Date parseable | First | Last | Rows | Interval (min) | Strict 8760 hourly |
| --- | --- | --- | --- | ---: | --- | --- |
| `p_load.csv` | yes | 2021-01-01 00:00:00 | 2021-12-31 23:00:00 | 8760 | 60.0 | yes |
| `q_load.csv` | yes | 2021-01-01 00:00:00 | 2021-12-31 23:00:00 | 8760 | 60.0 | yes |

Profile mapping uses raw CSV column position -> `load_bus_extra.csv` row order:

| Position | P raw header | P header bus | P match | Q raw header | Q header bus | Q match | bus_i | Consumer | ID | Bus exists | Status |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `16` | `16` | yes | `16` | `16` | yes | `16` | Single-family house | `5774` | yes | mapped |
| 2 | `14` | `14` | yes | `14` | `14` | yes | `14` | Single-family house | `1341` | yes | mapped |
| 3 | `19` | `19` | yes | `19` | `19` | yes | `19` | Single-family house | `7532` | yes | mapped |
| 4 | `35` | `35` | yes | `35` | `35` | yes | `35` | Single-family house | `947` | yes | mapped |
| 5 | `6` | `6` | yes | `6` | `6` | yes | `6` | Single-family house | `1860` | yes | mapped |
| 6 | `13` | `13` | yes | `13` | `13` | yes | `13` | Single-family house | `4621` | yes | mapped |
| 7 | `26` | `26` | yes | `26` | `26` | yes | `26` | Single-family house | `3905` | yes | mapped |
| 8 | `17` | `17` | yes | `17` | `17` | yes | `17` | Single-family house | `1803` | yes | mapped |
| 9 | `15` | `15` | yes | `15` | `15` | yes | `15` | Single-family house | `3125` | yes | mapped |
| 10 | `31` | `31` | yes | `31` | `31` | yes | `31` | Single-family house | `2489` | yes | mapped |
| 11 | `39` | `39` | yes | `39` | `39` | yes | `39` | Single-family house | `1071` | yes | mapped |
| 12 | `37` | `37` | yes | `37` | `37` | yes | `37` | Single-family house | `2713` | yes | mapped |
| 13 | `9` | `9` | yes | `9` | `9` | yes | `9` | Single-family house | `1334` | yes | mapped |
| 14 | `10` | `10` | yes | `10` | `10` | yes | `10` | Single-family house | `4247` | yes | mapped |
| 15 | `22` | `22` | yes | `22` | `22` | yes | `22` | Single-family house | `3160` | yes | mapped |
| 16 | `33` | `33` | yes | `33` | `33` | yes | `33` | Single-family house | `902` | yes | mapped |
| 17 | `29` | `29` | yes | `29` | `29` | yes | `29` | Single-family house | `3520` | yes | mapped |
| 18 | `11` | `11` | yes | `11` | `11` | yes | `11` | Single-family house | `2132` | yes | mapped |
| 19 | `36` | `36` | yes | `36` | `36` | yes | `36` | Single-family house | `3054` | yes | mapped |
| 20 | `23` | `23` | yes | `23` | `23` | yes | `23` | Single-family house | `1255` | yes | mapped |
| 21 | `2` | `2` | yes | `2` | `2` | yes | `2` | Single-family house | `850` | yes | mapped |
| 22 | `24` | `24` | yes | `24` | `24` | yes | `24` | Single-family house | `1789` | yes | mapped |
| 23 | `27` | `27` | yes | `27` | `27` | yes | `27` | Single-family house | `2970` | yes | mapped |
| 24 | `20` | `20` | yes | `20` | `20` | yes | `20` | Single-family house | `465` | yes | mapped |
| 25 | `4` | `4` | yes | `4` | `4` | yes | `4` | Single-family house | `1976` | yes | mapped |
| 26 | `7` | `7` | yes | `7` | `7` | yes | `7` | Single-family house | `5226` | yes | mapped |
| 27 | `30` | `30` | yes | `30` | `30` | yes | `30` | Townhouse | `6455` | yes | mapped |
| 28 | `30` | `30` | yes | `30.1` | `30` | yes | `30` | Townhouse | `306` | yes | mapped |
| 29 | `5` | `5` | yes | `5` | `5` | yes | `5` | Single-family house | `2325` | yes | mapped |

Header mismatch profiles: none
Mapping anomalies: none

## LV grid: 50_bus_rural_reference_grid

bus_count=50; branch_count=49; consumer_count=21; unique_load_bus_count=21
Multiple consumers on one bus observed: **no**.
Multiple consumers per bus allowed by row mapping: **yes**.
P/Q profile one-to-one: **yes**.
Unmapped profile count: **0**.
p_header_bus_mismatch_count: **0**; q_header_bus_mismatch_count: **0**.
Positional mapping has header semantics support: **yes**.
profile_to_bus_mapping_verified: **yes**.

| File | Date parseable | First | Last | Rows | Interval (min) | Strict 8760 hourly |
| --- | --- | --- | --- | ---: | --- | --- |
| `p_load.csv` | yes | 2021-01-01 00:00:00 | 2021-12-31 23:00:00 | 8760 | 60.0 | yes |
| `q_load.csv` | yes | 2021-01-01 00:00:00 | 2021-12-31 23:00:00 | 8760 | 60.0 | yes |

Profile mapping uses raw CSV column position -> `load_bus_extra.csv` row order:

| Position | P raw header | P header bus | P match | Q raw header | Q header bus | Q match | bus_i | Consumer | ID | Bus exists | Status |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `6` | `6` | yes | `6` | `6` | yes | `6` | Private building | `-` | yes | mapped |
| 2 | `10` | `10` | yes | `10` | `10` | yes | `10` | Private building | `-` | yes | mapped |
| 3 | `11` | `11` | yes | `11` | `11` | yes | `11` | Private building | `-` | yes | mapped |
| 4 | `12` | `12` | yes | `12` | `12` | yes | `12` | Private building | `-` | yes | mapped |
| 5 | `15` | `15` | yes | `15` | `15` | yes | `15` | Private building | `-` | yes | mapped |
| 6 | `16` | `16` | yes | `16` | `16` | yes | `16` | Private building | `-` | yes | mapped |
| 7 | `18` | `18` | yes | `18` | `18` | yes | `18` | Private building | `-` | yes | mapped |
| 8 | `20` | `20` | yes | `20` | `20` | yes | `20` | Private building | `-` | yes | mapped |
| 9 | `25` | `25` | yes | `25` | `25` | yes | `25` | Private building | `-` | yes | mapped |
| 10 | `29` | `29` | yes | `29` | `29` | yes | `29` | Private building | `-` | yes | mapped |
| 11 | `30` | `30` | yes | `30` | `30` | yes | `30` | Private building | `-` | yes | mapped |
| 12 | `31` | `31` | yes | `31` | `31` | yes | `31` | Private building | `-` | yes | mapped |
| 13 | `33` | `33` | yes | `33` | `33` | yes | `33` | Private building | `-` | yes | mapped |
| 14 | `36` | `36` | yes | `36` | `36` | yes | `36` | Private building | `-` | yes | mapped |
| 15 | `39` | `39` | yes | `39` | `39` | yes | `39` | Private building | `-` | yes | mapped |
| 16 | `40` | `40` | yes | `40` | `40` | yes | `40` | Private building | `-` | yes | mapped |
| 17 | `42` | `42` | yes | `42` | `42` | yes | `42` | Private building | `-` | yes | mapped |
| 18 | `43` | `43` | yes | `43` | `43` | yes | `43` | Private building | `-` | yes | mapped |
| 19 | `48` | `48` | yes | `48` | `48` | yes | `48` | Private building | `-` | yes | mapped |
| 20 | `49` | `49` | yes | `49` | `49` | yes | `49` | Private building | `-` | yes | mapped |
| 21 | `50` | `50` | yes | `50` | `50` | yes | `50` | Private building | `-` | yes | mapped |

Header mismatch profiles: none
Mapping anomalies: none

## LV grid: 56_bus_semi_urban_reference_grid

bus_count=56; branch_count=55; consumer_count=53; unique_load_bus_count=44
Multiple consumers on one bus observed: **yes**.
Multiple consumers per bus allowed by row mapping: **yes**.
P/Q profile one-to-one: **yes**.
Unmapped profile count: **0**.
p_header_bus_mismatch_count: **0**; q_header_bus_mismatch_count: **0**.
Positional mapping has header semantics support: **yes**.
profile_to_bus_mapping_verified: **yes**.

| File | Date parseable | First | Last | Rows | Interval (min) | Strict 8760 hourly |
| --- | --- | --- | --- | ---: | --- | --- |
| `p_load.csv` | yes | 2021-01-01 00:00:00 | 2021-12-31 23:00:00 | 8760 | 60.0 | yes |
| `q_load.csv` | yes | 2021-01-01 00:00:00 | 2021-12-31 23:00:00 | 8760 | 60.0 | yes |

Profile mapping uses raw CSV column position -> `load_bus_extra.csv` row order:

| Position | P raw header | P header bus | P match | Q raw header | Q header bus | Q match | bus_i | Consumer | ID | Bus exists | Status |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `54` | `54` | yes | `54` | `54` | yes | `54` | Flat | `1437` | yes | mapped |
| 2 | `54` | `54` | yes | `54.1` | `54` | yes | `54` | Flat | `1004` | yes | mapped |
| 3 | `54` | `54` | yes | `54.2` | `54` | yes | `54` | Flat | `5833` | yes | mapped |
| 4 | `54` | `54` | yes | `54.3` | `54` | yes | `54` | Flat | `671` | yes | mapped |
| 5 | `54` | `54` | yes | `54.4` | `54` | yes | `54` | Flat | `1933` | yes | mapped |
| 6 | `54` | `54` | yes | `54.5` | `54` | yes | `54` | Flat | `2718` | yes | mapped |
| 7 | `35` | `35` | yes | `35` | `35` | yes | `35` | Townhouse | `5825` | yes | mapped |
| 8 | `35` | `35` | yes | `35.1` | `35` | yes | `35` | Townhouse | `1538` | yes | mapped |
| 9 | `48` | `48` | yes | `48` | `48` | yes | `48` | Single-family house | `1860` | yes | mapped |
| 10 | `23` | `23` | yes | `23` | `23` | yes | `23` | Single-family house | `3125` | yes | mapped |
| 11 | `55` | `55` | yes | `55` | `55` | yes | `55` | Single-family house | `2489` | yes | mapped |
| 12 | `24` | `24` | yes | `24` | `24` | yes | `24` | Single-family house | `1803` | yes | mapped |
| 13 | `49` | `49` | yes | `49` | `49` | yes | `49` | Single-family house | `1071` | yes | mapped |
| 14 | `18` | `18` | yes | `18` | `18` | yes | `18` | Single-family house | `1206` | yes | mapped |
| 15 | `45` | `45` | yes | `45` | `45` | yes | `45` | Single-family house | `687` | yes | mapped |
| 16 | `28` | `28` | yes | `28` | `28` | yes | `28` | Single-family house | `1334` | yes | mapped |
| 17 | `34` | `34` | yes | `34` | `34` | yes | `34` | Townhouse | `3108` | yes | mapped |
| 18 | `34` | `34` | yes | `34.1` | `34` | yes | `34` | Townhouse | `6739` | yes | mapped |
| 19 | `47` | `47` | yes | `47` | `47` | yes | `47` | Single-family house | `2713` | yes | mapped |
| 20 | `4` | `4` | yes | `4` | `4` | yes | `4` | Single-family house | `902` | yes | mapped |
| 21 | `5` | `5` | yes | `5` | `5` | yes | `5` | Single-family house | `3520` | yes | mapped |
| 22 | `17` | `17` | yes | `17` | `17` | yes | `17` | Single-family house | `2132` | yes | mapped |
| 23 | `3` | `3` | yes | `3` | `3` | yes | `3` | Townhouse | `2601` | yes | mapped |
| 24 | `3` | `3` | yes | `3.1` | `3` | yes | `3` | Townhouse | `3614` | yes | mapped |
| 25 | `40` | `40` | yes | `40` | `40` | yes | `40` | Single-family house | `6435` | yes | mapped |
| 26 | `56` | `56` | yes | `56` | `56` | yes | `56` | Single-family house | `3054` | yes | mapped |
| 27 | `27` | `27` | yes | `27` | `27` | yes | `27` | Single-family house | `1255` | yes | mapped |
| 28 | `29` | `29` | yes | `29` | `29` | yes | `29` | Single-family house | `5998` | yes | mapped |
| 29 | `16` | `16` | yes | `16` | `16` | yes | `16` | Single-family house | `4029` | yes | mapped |
| 30 | `20` | `20` | yes | `20` | `20` | yes | `20` | Single-family house | `3509` | yes | mapped |
| 31 | `6` | `6` | yes | `6` | `6` | yes | `6` | Single-family house | `850` | yes | mapped |
| 32 | `9` | `9` | yes | `9` | `9` | yes | `9` | Single-family house | `952` | yes | mapped |
| 33 | `39` | `39` | yes | `39` | `39` | yes | `39` | Single-family house | `2535` | yes | mapped |
| 34 | `52` | `52` | yes | `52` | `52` | yes | `52` | Townhouse | `5555` | yes | mapped |
| 35 | `52` | `52` | yes | `52.1` | `52` | yes | `52` | Townhouse | `1207` | yes | mapped |
| 36 | `33` | `33` | yes | `33` | `33` | yes | `33` | Single-family house | `1587` | yes | mapped |
| 37 | `30` | `30` | yes | `30` | `30` | yes | `30` | Single-family house | `4080` | yes | mapped |
| 38 | `12` | `12` | yes | `12` | `12` | yes | `12` | Single-family house | `1224` | yes | mapped |
| 39 | `44` | `44` | yes | `44` | `44` | yes | `44` | Single-family house | `3361` | yes | mapped |
| 40 | `11` | `11` | yes | `11` | `11` | yes | `11` | Single-family house | `2970` | yes | mapped |
| 41 | `38` | `38` | yes | `38` | `38` | yes | `38` | Single-family house | `2763` | yes | mapped |
| 42 | `46` | `46` | yes | `46` | `46` | yes | `46` | Single-family house | `802` | yes | mapped |
| 43 | `22` | `22` | yes | `22` | `22` | yes | `22` | Single-family house | `3598` | yes | mapped |
| 44 | `36` | `36` | yes | `36` | `36` | yes | `36` | Single-family house | `465` | yes | mapped |
| 45 | `51` | `51` | yes | `51` | `51` | yes | `51` | Single-family house | `3167` | yes | mapped |
| 46 | `50` | `50` | yes | `50` | `50` | yes | `50` | Single-family house | `7085` | yes | mapped |
| 47 | `7` | `7` | yes | `7` | `7` | yes | `7` | Single-family house | `6518` | yes | mapped |
| 48 | `21` | `21` | yes | `21` | `21` | yes | `21` | Single-family house | `5226` | yes | mapped |
| 49 | `26` | `26` | yes | `26` | `26` | yes | `26` | Single-family house | `6590` | yes | mapped |
| 50 | `32` | `32` | yes | `32` | `32` | yes | `32` | Single-family house | `930` | yes | mapped |
| 51 | `43` | `43` | yes | `43` | `43` | yes | `43` | Single-family house | `7561` | yes | mapped |
| 52 | `13` | `13` | yes | `13` | `13` | yes | `13` | Single-family house | `2106` | yes | mapped |
| 53 | `15` | `15` | yes | `15` | `15` | yes | `15` | Single-family house | `441` | yes | mapped |

Header mismatch profiles: none
Mapping anomalies: none

## LV grid: 80_bus_rural_reference_grid

bus_count=80; branch_count=79; consumer_count=32; unique_load_bus_count=32
Multiple consumers on one bus observed: **no**.
Multiple consumers per bus allowed by row mapping: **yes**.
P/Q profile one-to-one: **yes**.
Unmapped profile count: **0**.
p_header_bus_mismatch_count: **0**; q_header_bus_mismatch_count: **0**.
Positional mapping has header semantics support: **yes**.
profile_to_bus_mapping_verified: **yes**.

| File | Date parseable | First | Last | Rows | Interval (min) | Strict 8760 hourly |
| --- | --- | --- | --- | ---: | --- | --- |
| `p_load.csv` | yes | 2021-01-01 00:00:00 | 2021-12-31 23:00:00 | 8760 | 60.0 | yes |
| `q_load.csv` | yes | 2021-01-01 00:00:00 | 2021-12-31 23:00:00 | 8760 | 60.0 | yes |

Profile mapping uses raw CSV column position -> `load_bus_extra.csv` row order:

| Position | P raw header | P header bus | P match | Q raw header | Q header bus | Q match | bus_i | Consumer | ID | Bus exists | Status |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `8` | `8` | yes | `8` | `8` | yes | `8` | Private building | `-` | yes | mapped |
| 2 | `11` | `11` | yes | `11` | `11` | yes | `11` | Private building | `-` | yes | mapped |
| 3 | `12` | `12` | yes | `12` | `12` | yes | `12` | Private building | `-` | yes | mapped |
| 4 | `13` | `13` | yes | `13` | `13` | yes | `13` | Private building | `-` | yes | mapped |
| 5 | `14` | `14` | yes | `14` | `14` | yes | `14` | Private building | `-` | yes | mapped |
| 6 | `15` | `15` | yes | `15` | `15` | yes | `15` | Private building | `-` | yes | mapped |
| 7 | `18` | `18` | yes | `18` | `18` | yes | `18` | Private building | `-` | yes | mapped |
| 8 | `20` | `20` | yes | `20` | `20` | yes | `20` | Private building | `-` | yes | mapped |
| 9 | `22` | `22` | yes | `22` | `22` | yes | `22` | Private building | `-` | yes | mapped |
| 10 | `25` | `25` | yes | `25` | `25` | yes | `25` | Private building | `-` | yes | mapped |
| 11 | `27` | `27` | yes | `27` | `27` | yes | `27` | Private building | `-` | yes | mapped |
| 12 | `30` | `30` | yes | `30` | `30` | yes | `30` | Private building | `-` | yes | mapped |
| 13 | `36` | `36` | yes | `36` | `36` | yes | `36` | Private building | `-` | yes | mapped |
| 14 | `39` | `39` | yes | `39` | `39` | yes | `39` | Private building | `-` | yes | mapped |
| 15 | `41` | `41` | yes | `41` | `41` | yes | `41` | Private building | `-` | yes | mapped |
| 16 | `45` | `45` | yes | `45` | `45` | yes | `45` | Private building | `-` | yes | mapped |
| 17 | `46` | `46` | yes | `46` | `46` | yes | `46` | Private building | `-` | yes | mapped |
| 18 | `48` | `48` | yes | `48` | `48` | yes | `48` | Private building | `-` | yes | mapped |
| 19 | `50` | `50` | yes | `50` | `50` | yes | `50` | Private building | `-` | yes | mapped |
| 20 | `53` | `53` | yes | `53` | `53` | yes | `53` | Private building | `-` | yes | mapped |
| 21 | `56` | `56` | yes | `56` | `56` | yes | `56` | Private building | `-` | yes | mapped |
| 22 | `57` | `57` | yes | `57` | `57` | yes | `57` | Private building | `-` | yes | mapped |
| 23 | `58` | `58` | yes | `58` | `58` | yes | `58` | Private building | `-` | yes | mapped |
| 24 | `59` | `59` | yes | `59` | `59` | yes | `59` | Private building | `-` | yes | mapped |
| 25 | `60` | `60` | yes | `60` | `60` | yes | `60` | Private building | `-` | yes | mapped |
| 26 | `64` | `64` | yes | `64` | `64` | yes | `64` | Private building | `-` | yes | mapped |
| 27 | `71` | `71` | yes | `71` | `71` | yes | `71` | Private building | `-` | yes | mapped |
| 28 | `73` | `73` | yes | `73` | `73` | yes | `73` | Private building | `-` | yes | mapped |
| 29 | `74` | `74` | yes | `74` | `74` | yes | `74` | Private building | `-` | yes | mapped |
| 30 | `75` | `75` | yes | `75` | `75` | yes | `75` | Private building | `-` | yes | mapped |
| 31 | `77` | `77` | yes | `77` | `77` | yes | `77` | Private building | `-` | yes | mapped |
| 32 | `80` | `80` | yes | `80` | `80` | yes | `80` | Private building | `-` | yes | mapped |

Header mismatch profiles: none
Mapping anomalies: none

## Industrial MV/LV

bus_count=76; load TXT file count=45.
Temperature files: -; weather files: -.

| Load TXT | Unique Bus_ID | Bus IDs exist in bus.csv | First | Last | Interval (min) |
| --- | --- | --- | --- | --- | --- |
| `r1v0.23b2.txt` | r1v0.23b2 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.23b3.txt` | r1v0.23b3 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b10.txt` | r1v0.415b10 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b11.txt` | r1v0.415b11 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b12.txt` | r1v0.415b12 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b13.txt` | r1v0.415b13 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b14.txt` | r1v0.415b14 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b15.txt` | r1v0.415b15 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b16.txt` | r1v0.415b16 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b17.txt` | r1v0.415b17 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b18.txt` | r1v0.415b18 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b19.txt` | r1v0.415b19 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b20.txt` | r1v0.415b20 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b21.txt` | r1v0.415b21 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b22.txt` | r1v0.415b22 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b23.txt` | r1v0.415b23 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b24.txt` | r1v0.415b24 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b25.txt` | r1v0.415b25 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b26.txt` | r1v0.415b26 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b27.txt` | r1v0.415b27 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b28.txt` | r1v0.415b28 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b29.txt` | r1v0.415b29 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b30.txt` | r1v0.415b30 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b31.txt` | r1v0.415b31 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b32.txt` | r1v0.415b32 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b33.txt` | r1v0.415b33 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b34.txt` | r1v0.415b34 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b35.txt` | r1v0.415b35 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b36.txt` | r1v0.415b36 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b37.txt` | r1v0.415b37 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b38.txt` | r1v0.415b38 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b39.txt` | r1v0.415b39 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b40.txt` | r1v0.415b40 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b41.txt` | r1v0.415b41 | yes | 2020-05-08 01:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b42.txt` | r1v0.415b42 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b43.txt` | r1v0.415b43 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b44.txt` | r1v0.415b44 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b7.txt` | r1v0.415b7 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b8.txt` | r1v0.415b8 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r1v0.415b9.txt` | r1v0.415b9 | yes | 2019-10-01 01:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r2v0.415b3.txt` | r2v0.415b3 | yes | 2020-09-07 01:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r2v0.415b4.txt` | r2v0.415b4 | yes | 2020-09-07 01:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r2v0.415b5.txt` | r2v0.415b5 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r2v0.415b6.txt` | r2v0.415b6 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |
| `r2v0.415b8.txt` | r2v0.415b8 | yes | 2019-03-01 00:00:00 | 2022-03-17 00:00:00 | 0.0, 60.0 |

Industrial mapping anomalies: none
