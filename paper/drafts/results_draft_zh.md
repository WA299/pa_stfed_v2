# 结果草稿（中文）

本稿的数值来自接受的冻结结果 JSON，并通过 `results/paper_ready/manifests/paper_results_manifest.json` 逐项追溯。参考电网 canonical validation 曾用于方法开发，故称为 **development validation**；工业 canonical validation 属于不同运行域的 **external industrial validation**。三个种子（42、123、2026）只描述一致性/稳健性，不构成统计显著性结论。工业目标值是每小时区间的 `Load_kWh` 工业负荷/能量测量，不是瞬时 MW。表格保留源 JSON 的精度；相对改善以 0–1 比例表示。

## 1. 时间建模主导性

四个参考电网的集中式 development-validation node-macro MAE 如表 1。`temporal_residual_only` 为 `0.00032925806226992394`，低于 GRU anchor 的 `0.000344790270315459`；Graph WaveNet 为 `0.0003329057723753775`，完整 PUC-RSTAttn V2 为 `0.0003317687407702807`。

| 方法 | Node-macro MAE | 接受结果来源：artifact#JSON key |
|---|---:|---|
| GRU anchor | 0.000344790270315459 | `results/ablations/puc_rstattn_v2_ablation_summary.json#macro_across_grids.node_macro.gru_anchor_only.metrics.mae` |
| temporal_residual_only | 0.00032925806226992394 | `results/ablations/puc_rstattn_v2_ablation_summary.json#macro_across_grids.node_macro.temporal_residual_only.metrics.mae` |
| Graph WaveNet | 0.0003329057723753775 | `results/centralized/centralized_validation_summary.json#macro_across_grids.node_macro.graph_wavenet.metrics.mae` |
| PUC-RSTAttn V2 | 0.0003317687407702807 | `results/ablations/puc_rstattn_v2_ablation_summary.json#macro_across_grids.node_macro.full_puc_rstattn_v2.metrics.mae` |

## 2. 四个参考电网：25% 与 50% history

表 2 报告四个参考电网在 development validation 上的 node-macro MAE mean +/- population std，均跨种子 42、123、2026。完整 JSON 路径给出 `.mean` 与 `.std` 两个值。

| 方法 | 25% validation MAE mean +/- std | 25% 精确 JSON key | 50% validation MAE mean +/- std | 50% 精确 JSON key |
|---|---:|---|---:|---|
| Scarce Local | 0.0003739304568965831 +/- 2.293019321928134e-06 | `metrics.scarce_local.validation.node_macro.mae.mean` / `.std` | 0.00035062209210361155 +/- 8.758104785082134e-07 | `metrics.scarce_local.validation.node_macro.mae.mean` / `.std` |
| FedAvg | 0.0003613628958204597 +/- 3.963300641403902e-06 | `metrics.fedavg.validation.node_macro.mae.mean` / `.std` | 0.00035730080002285055 +/- 4.049059150743375e-06 | `metrics.fedavg.validation.node_macro.mae.mean` / `.std` |
| FedProx | 0.00038814487293691203 +/- 1.1090279617434959e-06 | `metrics.fedprox.validation.node_macro.mae.mean` / `.std` | 0.00038665942105532166 +/- 6.655297576748568e-07 | `metrics.fedprox.validation.node_macro.mae.mean` / `.std` |
| FedPer | 0.0003803896747997208 +/- 5.962400972059978e-06 | `metrics.fedper.validation.node_macro.mae.mean` / `.std` | 0.000365555543365815 +/- 3.9813907598617465e-06 | `metrics.fedper.validation.node_macro.mae.mean` / `.std` |
| FedFomo-style | 0.00036882159763834764 +/- 6.243767923841199e-06 | `metrics.fedfomo_style.validation.node_macro.mae.mean` / `.std` | 0.000349061298024347 +/- 6.41503865379168e-07 | `metrics.fedfomo_style.validation.node_macro.mae.mean` / `.std` |
| BTD-FL | 0.00034577673225562013 +/- 2.515943738737929e-06 | `metrics.btd_fl_direct_transfer.validation.node_macro.mae.mean` / `.std` | 0.0003354481131077414 +/- 1.7520254865545847e-06 | `metrics.btd_fl_direct_transfer.validation.node_macro.mae.mean` / `.std` |

来源文件依次为 `results/federated/formal_25pct/multiseed/multiseed_25pct_summary.json` 与 `results/federated/formal_50pct/multiseed/multiseed_50pct_summary.json`。表 3 中相对改善保留 summary JSON 中的比例值，seed wins 为三种子中 BTD-FL validation MAE 较低的次数。

| Comparator | 25% mean relative improvement | 25% std | 25% wins / 3 | 50% mean relative improvement | 50% std | 50% wins / 3 |
|---|---:|---:|---:|---:|---:|---:|
| Scarce Local | 0.07529276239766151 | 0.00320981728123254 | 3 | 0.043273086366926665 | 0.005220346565042788 | 3 |
| FedAvg | 0.04296864908952084 | 0.015839179768413186 | 3 | 0.06100854131332002 | 0.013966360992410934 | 3 |
| FedProx | 0.10914299952300076 | 0.007603593799693188 | 3 | 0.13243563295641927 | 0.0059569998081508225 | 3 |
| FedPer | 0.09069029629333719 | 0.019838985444682745 | 3 | 0.08224847641260156 | 0.011345007244546663 | 3 |
| FedFomo-style | 0.062292889579418975 | 0.012199076130362412 | 3 | 0.039003276211258986 | 0.003824105313932098 | 3 |

比较值的 JSON 路径为 `btd_vs_comparators.<comparator>.mean_relative_improvement`、`.std_relative_improvement`、`.seeds_btd_wins`。所有结果均为三种子的一致性描述，不表示统计显著性。

## 3. 外部工业域：25% 与 50% history

表 4 的 primary endpoint 是工业 external-validation node-macro MAE。工业结果按本域报告，不与参考电网的绝对 MAE 混合。

| 方法 | 25% validation MAE mean +/- std | 25% 精确 JSON key | 50% validation MAE mean +/- std | 50% 精确 JSON key |
|---|---:|---|---:|---|
| Industrial Local | 4.547041646740443 +/- 0.008417325155579879 | `methods.industrial_scarce_local.validation.node_macro.mae.mean` / `.std` | 3.96256847851566 +/- 0.13424253160853794 | `methods.industrial_scarce_local.validation.node_macro.mae.mean` / `.std` |
| FedAvg | 9.501590643431959 +/- 1.1282122277147317 | `methods.fedavg.validation.node_macro.mae.mean` / `.std` | 8.458205679906355 +/- 0.9203560932356697 | `methods.fedavg.validation.node_macro.mae.mean` / `.std` |
| FedProx | 12.29245878993882 +/- 0.4576659408880685 | `methods.fedprox.validation.node_macro.mae.mean` / `.std` | 11.272902497597004 +/- 0.3651598729485767 | `methods.fedprox.validation.node_macro.mae.mean` / `.std` |
| FedPer | 6.042214572121369 +/- 0.3475319990531886 | `methods.fedper.validation.node_macro.mae.mean` / `.std` | 6.264005475385464 +/- 0.13817024860479318 | `methods.fedper.validation.node_macro.mae.mean` / `.std` |
| FedFomo-style | 4.9459699745053625 +/- 0.06632035849505322 | `methods.fedfomo_style.validation.node_macro.mae.mean` / `.std` | 4.546066445059885 +/- 0.05053210494713433 | `methods.fedfomo_style.validation.node_macro.mae.mean` / `.std` |
| BTD-FL | 3.7056023919283034 +/- 0.09877240033274068 | `methods.btd_fl_direct_transfer.validation.node_macro.mae.mean` / `.std` | 3.412532453334036 +/- 0.05701939410339808 | `methods.btd_fl_direct_transfer.validation.node_macro.mae.mean` / `.std` |

对应来源为 `results/federated/industrial_external/industrial_external_25pct_summary.json` 与 `industrial_external_50pct_summary.json`。表 5 的相对改善是 `comparisons_btd_vs_comparators.<comparator>.mean_relative_improvement`，标准差为 `.std_relative_improvement`，胜场数为 `.seed_win_count`。

| Comparator | 25% mean relative improvement | 25% std | 25% wins / 3 | 50% mean relative improvement | 50% std | 50% wins / 3 |
|---|---:|---:|---:|---:|---:|---:|
| Industrial Local | 0.18507651835077202 | 0.020719714267681853 | 3 | 0.13822819658933796 | 0.01916672868341732 | 3 |
| FedAvg | 0.6052779592111199 | 0.041015052188468105 | 3 | 0.592158886135364 | 0.04037040993809066 | 3 |
| FedProx | 0.6982240853764261 | 0.01135379842115401 | 3 | 0.6970751129673917 | 0.007229234689639923 | 3 |
| FedPer | 0.3856154850341617 | 0.019527757071938507 | 3 | 0.45513774088865117 | 0.004964830612997241 | 3 |
| FedFomo-style | 0.25091363559561924 | 0.00998084557546212 | 3 | 0.24936022167924454 | 0.007974580920631504 | 3 |

表 6 展示 BTD-FL 与 Industrial Local 的 additional validation metrics。值均来自工业 summary 的 `methods.<method>.validation.<scope>.<metric>.{mean,std}`。RMSE 和 grid-aggregate MAE 支持 primary node-MAE 结果；WAPE 和 sMAPE 不呈现统一改善：25% 时 BTD-FL 的 node WAPE/sMAPE 高于 Local，50% 时 node WAPE 略高于 Local 而 sMAPE 低于 Local。

| History | 方法 | Node RMSE mean +/- std | Node WAPE % mean +/- std | Node sMAPE % mean +/- std | Grid-aggregate MAE mean +/- std |
|---|---|---:|---:|---:|---:|
| 25% | Industrial Local | 9.070312595636809 +/- 0.0520931316965812 | 40.30228882376916 +/- 1.4992438684499447 | 40.36518263766215 +/- 1.9974925133894919 | 109.34334650977898 +/- 2.509101137048958 |
| 25% | BTD-FL | 6.861017519608453 +/- 0.1407076008026602 | 41.58133500695266 +/- 6.641782457112283 | 42.94666187171009 +/- 6.361295887146189 | 72.97653108555116 +/- 1.6920079606620906 |
| 50% | Industrial Local | 7.927051981780391 +/- 0.2332979563793728 | 33.88033096931136 +/- 0.7279115386514897 | 39.08223710583073 +/- 1.0546724327031263 | 85.27730781466111 +/- 5.250388054534253 |
| 50% | BTD-FL | 6.505022760233227 +/- 0.07824088734735754 | 33.98726504953107 +/- 1.843429033857374 | 35.535371907573214 +/- 1.1574595160692578 | 65.87531100354795 +/- 2.2083280168133492 |

## 4. Target-conditioned directed-benefit 机制分析

表 7 为工业目标的四个 donor directed calibration benefits；来源键为两个工业 summary 的 `btd_donor_behavior.by_seed.<seed>.benefits.<donor>`。selected donor 直接使用 summary 的 `selected_donor` 字段，未从热图最大值推断。

| History | Seed | Benefit 39 | Benefit 50 | Benefit 56 | Benefit 80 | Selected donor | Fallback |
|---|---:|---:|---:|---:|---:|---|---|
| 25% | 42 | 0.22704404679251647 | 0.06800634934900457 | 0.23225674588915424 | -0.04449526253628153 | 56_bus_semi_urban_reference_grid | false |
| 25% | 123 | 0.2181470819032599 | 0.10793306299401881 | 0.23138472548973268 | 0.17921654350988395 | 56_bus_semi_urban_reference_grid | false |
| 25% | 2026 | 0.2999401133463308 | 0.05152005864900721 | 0.3242365452189944 | 0.10465218686251597 | 56_bus_semi_urban_reference_grid | false |
| 50% | 42 | 0.2808615653736997 | 0.16627908193232332 | 0.25452304292910505 | 0.05552738150965696 | 39_bus_semi_urban_reference_grid | false |
| 50% | 123 | 0.2392883531781149 | 0.09484448253963716 | 0.20307682049656162 | 0.1822986519883357 | 39_bus_semi_urban_reference_grid | false |
| 50% | 2026 | 0.22192968516874065 | 0.0482711962175039 | 0.24196165113272763 | 0.21999318565078235 | 56_bus_semi_urban_reference_grid | false |

Industrial benefit sign counts are 25% positive/non-positive/fallback `11/1/0` and 50% `12/0/0`, from `btd_donor_behavior.positive_benefit_count`, `.non_positive_benefit_count`, and `.fallback_count`. Reference-grid counts are 25% `36/0/0` and 50% `35/1/0`, from `donor_selection_robustness.positive_directed_benefit_count`, `.non_positive_directed_benefit_count`, and `.fallback_occurrence_count`. Fallback was never triggered; its performance has not been empirically validated.

Seed-42, 25% BTD mechanism values are shown below. The final direct artifact key is `four_target_unweighted_macro.<split>.node_macro.mae`; the historical adapted-state result key is `four_scenario_macro.<split>.btd_fl.node_macro.mae`.

| Mechanism | Audit node-MAE | Validation node-MAE | Exact source artifact |
|---|---:|---:|---|
| Final direct raw temporal transfer | 0.00031355851678525095 | 0.0003437680937597074 | `results/federated/formal_25pct/btd_fl_direct_transfer_25pct_seed42.json` |
| No benefit selection | 0.0003167013266437048 | 0.000348316297728666 | `results/federated/formal_25pct/ablations/btd_no_benefit_selection_25pct_seed42.json` |
| Full-model transfer | 0.00031857846012325056 | 0.00034676604563052874 | `results/federated/formal_25pct/ablations/btd_full_model_transfer_25pct_seed42.json` |
| Historical adapted-state BTD | 0.0003133921560375219 | 0.0003448194006983876 | `results/federated/formal_25pct/btd_fl_25pct_seed42.json` |

No-benefit-selection 与 final BTD 不是完全受控的 WHO-only 消融：接受的 no-benefit artifact 复用了 adapted probe state，而 final direct BTD 使用 raw donor temporal state 并丢弃 adapted probe state。因此，该结果比较同时涉及 donor-selection 和 STATE 差异，不能做单因素因果解释。

## TEST 状态

Canonical TEST 保持锁定。本稿未读取 TEST observation，也未计算或报告任何 TEST metric。
