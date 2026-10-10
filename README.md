# pa_stfed_v2

项目：基于时空注意力与联邦学习的配电网负荷预测

本仓库为 V2 实验主线。

原始数据存放于仓库外部，不提交 GitHub。

当前阶段：科学结果与训练协议冻结，等待用户在本地执行 checkpoint 重建和最终独立 TEST。

## 本地环境与统一启动入口

本仓库包含已冻结的模型、训练协议、验证结果和结项前的本地执行入口。原始数据放在仓库外部，不提交 Git；历史 `results/` 结果文件只读保存。

建议使用 Python 3.10 或更高版本，并在 PyCharm 中选择项目专用环境（当前 CUDA 环境示例：`E:\pa_stfed_gpu_env\Scripts\python.exe`）。安装仓库依赖：

```powershell
E:\pa_stfed_gpu_env\Scripts\python.exe -m pip install -r requirements.txt
```

PyTorch 必须与本机 NVIDIA 驱动/CUDA 环境兼容。若默认 pip 源安装的 PyTorch 不适合本机，请按 PyTorch 官方安装选择器安装对应 CUDA wheel。核实环境：

```powershell
E:\pa_stfed_gpu_env\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

数据目录默认是 `E:\pa_stfed_data_v2\raw`，也可以通过命令行参数显式指定。

所有真实实验由用户在本地启动。默认 `status`、`preflight` 和 `validate` 不访问 TEST；重建和最终 TEST 使用不同的显式授权。重建仅使用冻结协议允许的 TRAIN/FIT/CALIBRATION 数据，并在 AUDIT/VALIDATION 上做一致性核验。

```powershell
# 查看阶段、GPU、当前单元和安全进度
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\run_graduation_experiment.py --phase status

# 只做 180 单元来源映射预检（不训练、不读取 TEST）
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\run_graduation_experiment.py --phase preflight

# 用户明确授权后，重建并安全续跑 checkpoint（仅 TRAIN/FIT/CALIBRATION）
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\run_graduation_experiment.py --phase reconstruct --authorize-reconstruction --resume --data-root E:\pa_stfed_data_v2\raw --device cuda

# 重建完成后，必须提供完整 parity report 才能通过核验
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\run_graduation_experiment.py --phase validate

# 创建不授予 TEST 权限的 protocol-lock 文件（需所有 checkpoint 和 parity 先通过）
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\run_graduation_experiment.py --phase protocol-lock-create

# 创建后仅提交 protocol-lock 文件，形成独立、可验证的 Git 锁提交
git add paper/preflight/test_protocol_lock.json
git commit -m "protocol: lock final TEST evaluation"

# 检查当前 HEAD、工作区、锁文件、冻结配置、checkpoint 集和 parity 摘要
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\run_graduation_experiment.py --phase protocol-lock-check

# 只有用户另行明确授权且上述核验通过后，才可打开一次 TEST
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\run_graduation_experiment.py --phase test --authorize-test --resume --data-root E:\pa_stfed_data_v2\raw --device cuda
```

protocol-lock 的 Git 身份由已跟踪锁文件所在提交、当前 HEAD 和文件 blob/content 哈希共同验证；锁提交必须是当前 HEAD，工作区不得有已跟踪改动，且锁文件经 Git 路径规范化后的 blob 必须与提交一致（兼容 Windows 换行转换）。未跟踪的 `code/`、`scripts/`、`configs/` 源文件也会被拒绝。锁文件本身永远不代表 TEST 授权，必须另行传入 `--authorize-test`。创建锁前必须先提交代码；若之后修改代码或冻结配置，需要重新完成核验和协议锁流程。

重建输出隔离在 `artifacts/frozen_pretest_checkpoints/`，最终 TEST 输出隔离在 `results/final_test_opening/`。程序会原子写入 checkpoint、sidecar、哈希、进度和日志；只有完整性、来源、模型身份和 TEST guardrail 均通过的 checkpoint 才能跳过，已有有效权重不会被重建覆盖。最终 TEST 续跑会核对每个旧结果的单元身份、checkpoint SHA256、protocol-lock Git 身份、TEST split 和完整来源信息；有效记录原样跳过，任何已存在但无效的记录都会停止，不会重新计算或覆盖。每个 TEST 单元保留评估起止时间、耗时、checkpoint/source/data/mapping/split 与协议锁来源信息。

最终统计固定为：参考配电网先在每个 seed 内对四网等权宏平均，再对三个 seed 计算均值和总体标准差；工业配电网按三个 seed 统计。180 个逐单元原始结果全部保留。参考网原始 P 指标按暂定 MW 口径保存，结项展示可将 MAE 乘以 1000 显示为 kW，但原始数值不变；工业负荷继续使用每小时 Load_kWh 口径。两个数据域的绝对误差不直接比较。

25%/seed42 的历史 BTD 重建输入严格使用 `results/federated/formal_25pct/btd_fl_25pct_seed42.json`；最终 direct-transfer checkpoint 的 parity 对照严格使用同目录下的 `btd_fl_direct_transfer_25pct_seed42.json`，两者用途不同。

## 预检与单元测试

在本地开发阶段可运行：

```powershell
E:\pa_stfed_gpu_env\Scripts\python.exe -m pytest tests/test_checkpoint_reconstruction.py -q --basetemp E:\pa_stfed_v2\.pytest-basetemp
E:\pa_stfed_gpu_env\Scripts\python.exe -m pytest tests/test_graduation_final_safety.py -q --basetemp E:\pa_stfed_v2\.pytest-basetemp-final-safety
```

测试包含真实 `PUCRSTAttnV2ConditionalUtility` 的微型保存/加载与预测一致性检查，以及合成统计、结果续跑、协议锁和 checkpoint 保护用例；不执行完整数据集训练、不读取 canonical TEST。默认重建命令只生成计划：

```powershell
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\reconstruct_frozen_checkpoints.py --help
```

## 早期 smoke baseline

如需单独运行历史 39-bus GRU validation-only smoke：

```powershell
E:\pa_stfed_gpu_env\Scripts\python.exe scripts\run_gru_smoke_39bus.py
```

它不计算 TEST 指标，也不替换冻结结果。
