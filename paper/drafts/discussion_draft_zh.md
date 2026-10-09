# 讨论草稿（中文）

冻结结果支持一个范围明确的结论：在三个 seed 和两个 target history fraction 下，BTD-FL direct temporal transfer 在接受的 primary node-macro MAE 汇总中优于各 comparator。参考电网上的 canonical validation 属于 development validation，因为它参与了方法开发；工业结果来自不同运行域的 external industrial validation。两类数值分开报告，工业 `Load_kWh` 不转换为参考电网 MW，也不用于跨域比较绝对 MAE。

集中式证据首先显示时间建模的重要性：`temporal_residual_only` 的 node-macro MAE 为 `0.00032925806226992394`，低于 GRU anchor 的 `0.000344790270315459`，而 Graph WaveNet 与完整 PUC-RSTAttn V2 分别为 `0.0003329057723753775` 和 `0.0003317687407702807`。uniform spatial utility、无 physics utility prior 和 full V2 的 exact ablation values 为 `0.00033166683391395405`、`0.0003322749034044745`、`0.0003317687407702807`。这些值更适合表述为物理/空间贡献较弱且不稳定，而不是证明空间建模没有价值；三个 seed 的 federated 汇总也不构成统计显著性结论。

参考电网中，BTD-FL 在 25% 和 50% history 的 validation node-MAE 均在三个 seed 上胜过五个 comparator。25% 相对 Local/FedAvg/FedProx/FedPer/FedFomo-style 的平均相对改善为 `7.5293%/4.2969%/10.9143%/9.0690%/6.2293%`；50% 为 `4.3273%/6.1009%/13.2436%/8.2248%/3.9003%`。这些是三 seed 的 consistency/robustness 描述，不是显著性检验。FedFomo-style 是本文定义的 comparator 名称，不应写成 exact FedFomo。

工业域结果体现了外部域差异：BTD-FL 的 primary node-MAE 在 25% 为 `3.7056023919283034 +/- 0.09877240033274068`，在 50% 为 `3.412532453334036 +/- 0.05701939410339808`；相对 Industrial Local 的平均改善分别为 `18.5077%` 和 `13.8228%`。FedAvg/FedProx 在工业 validation 中的 MAE 明显高于 Industrial Local，说明把参考客户端参数直接用于该外部域可能产生 negative-transfer evidence。RMSE 和 grid-aggregate MAE 与 primary endpoint 方向一致，但 node WAPE 与 sMAPE 并非统一改善，应保留其不一致性。

directed benefit 是 target-conditioned calibration quantity，不能解释为因果 benefit，也不能把某个 donor 身份称为内在最优。工业 25% 有 11 条 positive edge、1 条 non-positive edge；50% 有 12 条 positive edge、0 条 non-positive edge；两个 fraction 都没有触发 zero-transfer fallback。没有 fallback 事件意味着 fallback 分支的性能尚未得到经验验证。

机制解释需要保持控制边界。final BTD 传输 selected donor 的 raw temporal state，丢弃 adapted probe state；full-model transfer 改变 WHAT；历史 BTD 改变 STATE；no-benefit-selection 改变 WHO 规则。然而，接受的 no-benefit-selection artifact 同时复用了 adapted probe state，因此它不是与 final BTD 完全隔离的 WHO-only ablation。论文应把它称为机制演化/组合消融，并明确这一混杂，而不是给出过强的单因素因果解释。

最后，当前证据没有 canonical TEST 结果。paper-ready manifest 的 `canonical_test_metrics_included` 为 `false`，checkpoint preflight 也不应把 JSON metric 当作 checkpoint 身份。任何未来 TEST 开启都必须遵循一次性、预注册的 checkpoint/矩阵协议，TEST 结果不能触发新的训练、重建或方法选择。
