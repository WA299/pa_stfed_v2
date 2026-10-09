# pa_stfed_v2

项目：基于时空注意力与联邦学习的配电网负荷预测

本仓库为 V2 实验主线。

原始数据存放于仓库外部，不提交 GitHub。

当前阶段：科学结果与训练协议冻结，等待用户在本地执行 checkpoint 重建和最终独立 TEST。

## 本地环境与统一启动入口

本仓库包含已冻结的模型、训练协议、验证结果和结项前的本地执行入口。原始数据放在仓库外部，不提交 Git；历史 `results/` 结果文件只读保存。

建议在 PyCharm 中选择项目专用 Python 环境（当前 CUDA 环境示例：`E:\pa_stfed_gpu_env\Scripts\python.exe`），并确认 PyTorch 能看到 CUDA：

```powershell
E:\pa_stfed_gpu_env\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

数据目录默认是 `E:\pa_stfed_data_v2\raw`，也可以通过命令行参数显式指定。

所有真实实验由用户在本地启动。默认 `status`、`preflight` 和 `validate` 不访问 TEST；重建和最终 TEST 都需要不同的显式授权。

```powershell
# 查看阶段、GPU、当前单元和安全进度
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\run_graduation_experiment.py --phase status

# 只做 180 单元来源映射预检（不训练、不读取 TEST）
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\run_graduation_experiment.py --phase preflight

# 用户明确授权后，重建并安全续跑 checkpoint（仅 TRAIN/FIT/CALIBRATION）
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\run_graduation_experiment.py --phase reconstruct --authorize-reconstruction --resume --data-root E:\pa_stfed_data_v2\raw --device cuda

# 重建完成后，必须提供完整 parity report 才能通过核验
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\run_graduation_experiment.py --phase validate

# 只有 protocol-lock 已由用户单独完成且明确授权，才可打开一次 TEST
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\run_graduation_experiment.py --phase test --authorize-test --resume --data-root E:\pa_stfed_data_v2\raw --device cuda
```

重建输出隔离在 `artifacts/frozen_pretest_checkpoints/`，最终 TEST 输出隔离在 `results/final_test_opening/`。程序会原子写入 checkpoint、sidecar、哈希、进度和日志；续跑只跳过完整性、来源、模型身份和 TEST guardrail 均通过的单元。任何缺失、损坏、来源不一致或 parity 不完整都会 fail-closed，不能用 `--force` 覆盖历史结果。

25%/seed42 的历史 BTD 重建输入严格使用 `results/federated/formal_25pct/btd_fl_25pct_seed42.json`；最终 direct-transfer checkpoint 的 parity 对照严格使用同目录下的 `btd_fl_direct_transfer_25pct_seed42.json`，两者用途不同。

## 预检与单元测试

在本地开发阶段可运行：

```powershell
E:\pa_stfed_gpu_env\Scripts\python.exe -m pytest tests/test_checkpoint_reconstruction.py -q --basetemp E:\pa_stfed_v2\.pytest-basetemp
```

该测试包含真实 `PUCRSTAttnV2ConditionalUtility` 的微型保存/加载与预测一致性检查，但不执行完整数据集训练、不读取 canonical TEST。默认重建命令只生成计划：

```powershell
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\reconstruct_frozen_checkpoints.py --help
```

## 早期 smoke baseline

如需单独运行历史 39-bus GRU validation-only smoke：

```powershell
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\run_gru_smoke_39bus.py
```

它不计算 TEST 指标，也不替换冻结结果。
