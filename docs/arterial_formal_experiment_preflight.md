# arterial_1x6 正式实验预检

预检日期：2026-07-30；执行设备：`linux4090`；Git 基线：`e7705f77c76673321b5a977da713b2c2c9d2e5ce`。

## 结论

当前 shared DQN 实现和最小真实 SUMO 链路可用，完整测试为 `177 passed`。目前可以进入 Stage 0 资源与预算校准，但不能进入依赖正式历史数据的 Stage 2–4。Stage 1 的 20 个独立在线 collector run 尚未执行，现有历史档案均为 smoke 数据，不符合 HOA 的 400 episodes、5 collector seeds 和不可变聚合档案要求。

仓库工作树在本 Goal 开始前已包含大量未提交实现改动。本次工作保留这些改动，正式 run metadata 必须同时引用 Git commit、Git diff 快照和配置哈希，不能只用 commit 判断代码版本。

## 运行环境

- 仓库推荐环境：Python 3.9、Conda `colight`。
- `linux4090` 当前没有可发现的 Conda 安装，无法恢复 `colight`。
- 采用已验证的统一备用环境：Python 3.10.12（`/usr/bin/python`）、libsumo/traci/sumolib 1.27.1、PyTorch `2.2.0a0+81ea7a4`、CUDA 12.3。
- 系统没有 `sumo` CLI；本项目正式入口使用 Python `libsumo` 1.27.1。
- `libsumo` 依赖的 `libXrender.so.1` 不在系统动态库路径中。正式 run 必须设置：

  ```bash
  LD_LIBRARY_PATH=/tmp/sumo-runtime-libs/root/usr/lib/x86_64-linux-gnu
  ```

- 为防止多进程下 BLAS 线程超卖，每个 run 默认设置：

  ```bash
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  VECLIB_MAXIMUM_THREADS=1
  ```

- libsumo 导入时报告 pyarrow ABI 版本警告。当前完整测试和真实 smoke 均成功，但 Stage 0 应继续监控仿真异常；在同一正式矩阵中不得更换该环境。

系统快照输出到 `artifacts/system_profile/`，包括 `profile.json`、`profile.sha256`、`pip_freeze.txt` 和 `git_diff.patch`。每个正式 run 应引用 `profile.sha256`。

## 配置审计

1. 四个真实场景分别映射到：
   - `configs/sim/sumoarterial1x6_300_03.cfg` → `data/raw_data/arterial_1x6_sumo/arterial_1x6_300_0.3.sumocfg`
   - `configs/sim/sumoarterial1x6_300_06.cfg` → `data/raw_data/arterial_1x6_sumo/arterial_1x6_300_0.6.sumocfg`
   - `configs/sim/sumoarterial1x6_700_03.cfg` → `data/raw_data/arterial_1x6_sumo/arterial_1x6_700_0.3.sumocfg`
   - `configs/sim/sumoarterial1x6_700_06.cfg` → `data/raw_data/arterial_1x6_sumo/arterial_1x6_700_0.6.sumocfg`
2. 四个 simulator config 的 SUMO seed 均固定为 `0`。正式重复只改变 training seed。
3. shared DQN 正式状态为每路口 12 维进口 lane count 加 8 维相位 one-hot，共 20 维；不启用 position encoding、neighbor summary 或 lane-count normalization。
4. 正式 reward 为 `original`：每路口进口车道等待车辆数均值取负并乘 12；DQN 使用局部 reward，网络级日志取六路口平均。
5. action interval 为 10 秒；episode 和冻结 evaluation 均为 3600 秒，因此每 episode 理论为 360 decisions。
6. 每个 decision 生成 6 条局部 transition，稳定训练期执行 1 次 gradient update。
7. replay 容量为 30000；batch size 为 64。
8. optimizer 为 RMSprop；learning rate 为 0.001；discount factor 为 0.95；gradient clip 为 5.0。
9. target network 采用硬同步，`update_target_rate=10`。
10. epsilon 初值 0.1、每次训练更新乘 0.995、下限 0.01；场景切换支持 reset、continue 和 fixed-low，公平主比较必须统一模式。
11. online learning warm-up 为 1000 decisions。ORB 不足时不会用 HOA 填充 online 部分。
12. resumable checkpoint 包含 online/target network、optimizer、epsilon、完整 replay、计数器、Python/NumPy/PyTorch RNG、当前场景、stage 和可见历史集合。
13. 冻结 evaluation 保存并恢复训练状态，禁止梯度、remember/replay 写入和随机状态漂移；对应隔离测试已通过。
14. causal history 的可见集合由 stage 决定，Stage 1 为空，后续只包含先前场景；full history 从首阶段可见全部四场景。相关自动测试已通过。

## HOA 审计

当前找到的 history archive 仅包括开发 smoke 档案，例如 `300_0.6` 的 1 episode、4 decisions、24 transitions、collector seed 0。它们不得进入正式 HOA。

正式 HOA 当前状态：**不存在**。Stage 1 必须完成：

- 四场景各 seeds `0,1,2,3,4`；
- 每个 run 从随机初始化开始独立训练 400 episodes；
- 固定 SUMO seed 0；
- 全量保留 episode 1–400 transition；
- 聚合后生成每场景不可变 archive、manifest、文件哈希和统一 archive hash；
- semi/sequential seeds 固定为 `1000–1004`，不得与 collector seeds 混用。

## 尚需在 Stage 0 前后完成的支持项

- 生成并冻结本次系统环境快照。
- 建立正式实验 manifest 和 run-level 并发队列；当前 `arterial_run.py` 只能生成或执行单个 stage，不满足自动补充 worker、状态机、失败重试和断点恢复要求。
- 将 full-history 的目录、manifest 与日志标识统一为精确名称 `non_causal_full_history_upper_bound`；当前实现只有布尔字段 `full_history_noncausal_upper_bound`，不满足最终命名约束。
- 增加正式 HOA 聚合审计，覆盖 5 seeds、400 episodes、decision/transition 数、六路口分布、schema/hash、重复与 NaN/Inf。
- Stage 0 对 `300_0.6` 和 `700_0.6` 运行 pilot，并实测 1/2/4/必要时 8 并发；根据 CPU、内存、swap、I/O 和吞吐量选择 `max_workers`，不得预设固定值。
- Stage 0 根据 100–150 episode 学习曲线确定统一正式预算 B；Stage 1 collector 数据口径仍按附件冻结为 400 episodes。

## 已完成验证

- 自动测试：

  ```bash
  LD_LIBRARY_PATH=/tmp/sumo-runtime-libs/root/usr/lib/x86_64-linux-gnu \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
  ```

  结果：`177 passed in 29.18s`。

- 真实 SUMO smoke：`linux4090`，`300_0.6`，shared DQN，training seed 0，1 episode，40 simulation seconds，10 秒 action interval；训练、冻结 evaluation、checkpoint 和 24 条 history transition 均成功。完整命令：

  ```bash
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  LD_LIBRARY_PATH=/tmp/sumo-runtime-libs/root/usr/lib/x86_64-linux-gnu \
  python arterial_run.py --config configs/arterial/smoke.yml --stage 0 \
    --prefix arterial_preflight_runtime_20260730 \
    --output-overlay artifacts/arterial_experiments/preflight/smoke_stage0.yml \
    --execute
  ```

## 阶段门禁

- Stage 0：允许开始，但必须先使用统一环境快照并补齐资源采集/队列能力。
- Stage 1：在 Stage 0 确定安全并发数后允许开始；collector 预算固定 400 episodes。
- Stage 2–4：禁止开始，直到 Stage 1 HOA 完整性审计通过并冻结 archive hash。
