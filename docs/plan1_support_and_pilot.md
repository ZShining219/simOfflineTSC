# Plan 1 项目支持、传统基线与 Pilot 子计划

## 1. 状态与范围

- 所属总 Plan：`docs/plan0721.md`
- 当前状态：已通过
- 执行分支：`codex/milestone0-experiment-infrastructure`
- 目标：补齐 Plan 1 启动所需项目能力，完成传统基线确定性验证和四场景 DQN Pilot。
- 非目标：本子计划不执行 20 次正式 DQN 训练。

本子计划按功能模块执行。每个模块必须完成实现、针对性测试、真实验证、代码审核、中文 Git 提交和远端推送后，才能进入下一模块。

## 2. 已确认规则

### 2.1 Online replay 与 trajectory

Online DQN replay 的 tuple、容量、FIFO、均匀采样和训练逻辑保持不变。每个训练决策同时复制到独立的 episode 级 NPZ trajectory；附加科研字段不进入 Online replay、模型输入或 loss。

trajectory 每个训练 episode 生成一个不可覆盖的压缩 NPZ 分片，并维护 manifest、append-only index 和 validation 结果。每个决策一条 transition；当前 3600 simulation steps、10-step action interval 对应每 episode 360 条。

### 2.2 指标口径

- approximate delay 与 real delay 每个决策同时记录；real delay 必须是无副作用快照。
- waiting time：当前活跃车辆的 SUMO accumulated waiting time 平均值，无车辆为 0。
- unfinished vehicles：episode 结束时仍在路网中的车辆数。
- phase switching：相邻决策 action 不同计一次。
- throughput：已完成车辆数。
- reward 只用于同一控制器内部诊断，不跨 FixedTime、MaxPressure 和 DQN 排名。

### 2.3 评估与 checkpoint

通过 `trainer.evaluation_episodes` 显式配置评估节点。episode 0 是训练开始前的初始模型评估；evaluation checkpoint、resumable checkpoint 和评估运行使用同一节点。best checkpoint 按最低 average travel time 选择，并列时取较早节点。

Pilot 节点为 `[0, 10, 25, 50, 100]`。Pilot 通过后，正式节点冻结为 `[0, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400]`。

### 2.4 作图工具与目录

新增可复用 `tools/experiment_plotting/`，读取显式 run-list，不递归猜测有效运行。生成的表格、PNG 和 PDF 放在：

`data/output_data/analysis/plan1/<analysis_id>/`

大型 trajectory、checkpoint、运行日志、分析目录和生成图表不提交 Git。

## 3. 功能模块与 Git 门禁

### 模块 0：文档与执行基线

- 更新总 Plan 状态和子 Plan 索引；
- 创建本子计划；
- 验证 Markdown、链接和工作区 diff；
- 提交信息：`明确Plan 1项目支持与Pilot执行计划`。

### 模块 1：双口径指标与诊断基础

- 无副作用 real delay 快照；
- 双 delay、waiting、unfinished、action distribution、phase switching、replay size、target updates；
- 结构化指标 schema v2，并兼容读取 schema v1；
- 提交信息：`补全Plan 1双延误与交通诊断指标`。

### 模块 2：Replay 与 trajectory 双写

- episode NPZ 原子分片；
- manifest、index、SHA-256 和 validation；
- transition 数量、连续性、NaN、动作范围和 evaluation 零写入检查；
- Online replay 行为回归；
- 提交信息：`实现Plan 1训练轨迹分片与完整性校验`。

### 模块 3：固定评估与 checkpoint

- 显式评估节点；
- 初始 episode 0 评估；
- checkpoint 与评估同节点；
- final/best 选择和防重复最终评估；
- 提交信息：`增加Plan 1固定评估与检查点选择机制`。

### 模块 4：可复用作图工具

- run-list 解析和运行有效性校验；
- Plan 1 规范化 CSV；
- learning curve、双 delay、queue、throughput、loss、epsilon、replay、action 和 phase switching 图；
- PNG 与 PDF 双格式；
- 提交信息：`增加可复用实验结果作图工具`。

### 模块 5：集成验收

- 全量单元测试、语法检查和 `git diff --check`；
- DQN、FixedTime、MaxPressure 真实 SUMO smoke；
- 真实运行产物作图；
- 配置源文件无运行时写回；
- 提交信息：`完成Plan 1启动能力集成验收`。

## 4. 实验矩阵

### 4.1 传统基线

四个实际 network 分别运行 FixedTime 和 MaxPressure，每项重复两次：

`4 networks × 2 controllers × 2 repeats = 16 runs`

统一使用 training seed 0、SUMO fixed default、libsumo 和完整 3600-step。重复运行必须使用不同 prefix；若确定性检查不通过，先定位并修复，相关基线全部重跑。

### 4.2 DQN Pilot

实际 network：

- `sumohz1x1_config2`
- `sumohz1x1`
- `sumohz1x1_config4`
- `sumohz1x1_config3`

统一使用 training seed 0、100 episode、Pilot 评估节点、libsumo 和固定 SUMO 默认行为。四次运行除 network 外配置必须一致。

## 5. Pilot 验收

诊断应尽量完成运行并形成完整报告：

- NaN、非法动作、trajectory 数量或连续性错误、evaluation 污染、配置不一致：对应运行不能通过；
- 单动作占比过高、reward 与交通指标方向异常：只生成警告，人工复核；
- writer、配置归档或哈希失败等无法保证证据完整性的错误：运行直接失败。

四场景全部通过后：

1. 生成 Pilot 表格、PNG/PDF 和验收报告；
2. 更新本子计划与总 Plan；
3. 将 `configs/tsc/dqn.yml` 手动切换到 400 episode 和正式评估节点；
4. 记录并冻结正式配置 SHA-256；
5. 提交信息：`完成Plan 1 Pilot并冻结正式训练配置`。

若修复涉及 state、action、reward、网络结构、DQN 核心超参数、SUMO 环境、replay 或指标计算，四次 Pilot 全部作废并重跑。

## 6. Git 管理

- 每个模块验收后立即中文 commit 并 push 当前分支；
- 暂存时使用明确文件路径，不使用 `git add .`；
- 提交前检查 `git status`、staged diff、测试、`git diff --check`；
- 推送前 fetch 并确认远端未领先；
- 不 force push、不自动 rebase、不改写已推送历史；
- 远端出现未知提交时停止推送并报告；
- 代码、测试和对应验收文档进入同一模块提交。

## 7. 执行记录

### 模块 0：文档与执行基线

- 状态：已通过；
- 提交：`6bc3645 明确Plan 1项目支持与Pilot执行计划`；
- 远端：已推送 `origin/codex/milestone0-experiment-infrastructure`。

### 模块 1：双口径指标与诊断基础

- 状态：已通过；
- 提交：`dc6399e 补全Plan 1双延误与交通诊断指标`；
- 远端：已推送当前分支；
- 实现：real delay 无副作用快照、活跃车辆平均累计等待、路网内未完成车辆数、动作分布、切相频率、replay/target 诊断和结构化指标 schema v2；
- 兼容：schema v1 日志仍可验证和读取；
- 自动检查：Milestone 0 共 24 项通过，Plan 1 指标新增 2 项通过；
- 语法检查：`common/metrics.py`、`world/world_sumo.py`、`trainer/tsc_trainer.py`、`utils/logger.py` 和新增测试通过；
- 已知环境提示：旧 Gym 弃用提示不影响本模块 CPU/SUMO 路径。

### 模块 2：Replay 与 trajectory 双写

- 状态：已通过；
- 提交：`828f9e3 实现Plan 1训练轨迹分片与完整性校验`；
- 远端：已推送当前分支；
- 实现：保持 Online replay tuple 不变，在 trainer 层每个环境决策记录一条聚合 transition，按 episode 原子生成压缩 NPZ，并维护 manifest、append-only index、SHA-256 和 validation；
- 验收：每 episode 决策数、episode/global step、state/phase 链、NaN/Inf、动作范围和 terminated/truncated 语义均有自动校验；
- 评估隔离：EvaluationIsolationGuard 已纳入 trajectory 写入计数，evaluation transition count 固定为 0；
- 自动检查：Milestone 0 共 24 项通过；Plan 1 指标和 trajectory 共 5 项通过；
- replay 回归：`DQNAgent.remember()` 仍保存原 6 元训练 payload，未写入科研元数据。

### 模块 3：固定评估与 checkpoint

- 状态：已通过；
- 提交：`5036f93 增加Plan 1固定评估与检查点选择机制`；
- 远端：已推送当前分支；
- 实现：新增显式 `evaluation_episodes`，按已完成 episode 数调度训练前 episode 0、过程中节点和单次最终评估；evaluation/resumable checkpoint 与评估节点完全一致；
- best 规则：按 travel time 最低选择，并列取较早节点；结果写入运行目录 `evaluation/summary.json`；
- 自动检查：Milestone 0 共 24 项、Plan 1 共 7 项全部通过；
- 真实 SUMO smoke：`sumohz1x1`、DQN、seed 41、2 个训练 episode、700 training steps、评估节点 `[0,1,2]`；
- smoke 命令：`SUMO_HOME=/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo /home/dev/miniforge3/envs/colight/bin/python3.10 run.py -w sumo -a dqn -n sumohz1x1 --prefix p1m3_eval_smoke_20260722_1 --seed 41 --interface libsumo --delay_type apx`；
- smoke 结果：退出码 0；records 顺序为 EVALUATION、TRAIN、EVALUATION、TRAIN、FINAL_EVALUATION；三组两类 checkpoint 均存在；trajectory 为 2 episode/140 transitions、evaluation transitions=0、validation=true；
- 配置恢复：临时 smoke 参数已清除，`dqn.yml` 恢复 200 episode 开发配置和 `[0,10,25,50,100,150,200]` 节点；SUMO source cfg 哈希保持 `314f1915c...bb9dbd`。

### 模块 4：可复用作图工具

- 状态：已通过；
- 提交：`f25fba8 增加可复用实验结果作图工具`；
- 远端：已推送当前分支；
- 实现：新增 `tools/experiment_plotting/`，通过显式 CSV run-list 读取运行，不扫描或猜测输出目录；run-list 固定字段为 `role,agent,network,training_seed,run_dir,include`；
- 强校验：拒绝重复条目、失败运行、run-list 与运行身份不一致、配置归档哈希损坏、指标 schema/JSONL 损坏；DQN Pilot/正式运行额外要求 evaluation summary、有效 trajectory validation，并复核 trajectory manifest、index 连续性、分片 SHA-256 和计数；
- 配置比较：四场景 DQN Pilot/正式运行使用 `utils/run_config_compare.py` 比较，除 network、prefix、training seed 和对应路径外的差异会阻止分析；
- 表格：生成 `metrics.csv`、`action_distribution.csv`、`run_summary.csv`、`final_best_comparison.csv`、`first_100_auc.csv` 和 `config_comparison.json`；
- 图表：生成 travel time、双 delay、queue/throughput、reward/loss、epsilon/replay、action distribution、phase switching、final/best、interaction costs 和 first-100 AUC；每组同时输出 PNG/PDF；
- 输出隔离：固定写入 `data/output_data/analysis/plan1/<analysis_id>/`，包含 `inputs/`、`tables/`、`figures/` 和 `plotting_manifest.json`；已存在的 analysis id 不允许覆盖；
- 使用命令：`python -m tools.experiment_plotting plan1 --run-list <runs.csv> --analysis-id <analysis_id>`；
- 自动检查：Milestone 0 共 24 项、Plan 1 共 10 项全部通过，新增作图测试 3 项覆盖成功输出、重复/失败运行、损坏配置和缺失 trajectory；
- 真实产物验证：读取模块 3 的 `p1m3_eval_smoke_20260722_1`，成功规范化 5 条 schema v2 指标并生成 6 份表格、10 组 PNG/PDF；
- 验证命令：`PYTHONPATH=. /home/dev/miniforge3/envs/colight/bin/python3.10 -m tools.experiment_plotting plan1 --run-list /tmp/p1m4_smoke_runs.csv --analysis-id p1m4_real_smoke_20260722_2 --dpi 80`；
- 证据目录：`data/output_data/analysis/plan1/p1m4_real_smoke_20260722_2/`，仅本地保留，不提交 Git；
- 静态检查：新增模块和测试通过 `py_compile`，`git diff --check` 无错误；
- 已知环境提示：旧 Gym 和未使用的 PyG CUDA 扩展 ABI 警告不影响本模块 CPU/Matplotlib 路径。

### 模块 5：集成验收

- 状态：已通过；
- 提交：`8e20c40 完成Plan 1启动能力集成验收`；
- 远端：已推送当前分支；
- 自动检查：Milestone 0 共 24 项、Plan 1 共 11 项全部通过；新增传统基线配置回归 1 项；相关 Python 文件通过 `py_compile`，`git diff --check` 无错误；
- DQN 证据：复用模块 3 已通过的 `data/output_data/tsc/sumo_dqn/sumohz1x1/p1m3_eval_smoke_20260722_1/`，包含训练、固定评估、checkpoint、trajectory 和 schema v2 指标；
- FixedTime 命令：`SUMO_HOME=/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo /home/dev/miniforge3/envs/colight/bin/python3.10 run.py -w sumo -a fixedtime -n sumohz1x1 --prefix p1m5_fixedtime_smoke_20260722_1 --seed 41 --interface libsumo --delay_type apx`；
- FixedTime 结果：退出码 0，3600 simulation steps/360 decisions，episode 1，travel time `233.16492949110975`，real delay `170.36838048514238`，trajectory 不适用；
- MaxPressure 初次 smoke：`p1m5_maxpressure_smoke_20260722_1` 的仿真退出码为 0，但发现其继承 `base.yml` 的 `episodes: 200`，单次最终评估被错误标为 episode 200；该运行仅元数据错误，已作废且不作为后续证据；
- 修复：在 `configs/tsc/maxpressure.yml` 明确设置 `episodes: 1`，并增加 FixedTime/MaxPressure 均必须为单 episode 的配置回归测试；
- MaxPressure 有效命令：`SUMO_HOME=/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo /home/dev/miniforge3/envs/colight/bin/python3.10 run.py -w sumo -a maxpressure -n sumohz1x1 --prefix p1m5_maxpressure_smoke_20260722_2 --seed 41 --interface libsumo --delay_type apx`；
- MaxPressure 结果：退出码 0，3600 simulation steps/360 decisions，episode 1，travel time `80.51447435246318`，real delay `21.615554974699602`，trajectory 不适用；
- 混合作图：三类 agent 通过同一个显式 run-list 生成规范化表格和 10 组 PNG/PDF，证据目录为 `data/output_data/analysis/plan1/p1m5_integrated_smoke_20260722_2/`；
- 配置源核对：运行后 `base.yml`、`dqn.yml`、`fixedtime.yml` 和 `sumohz1x1.cfg` 哈希与运行前一致；`maxpressure.yml` 仅包含已审核的 `episodes: 1` 永久修复；
- 结论：Plan 1 启动所需项目能力全部通过，可以执行 16 次传统基线确定性验证。

### 传统基线：16 次确定性验证

- 状态：已通过；
- 提交：`a0c49c0 完成Plan 1传统基线确定性验证`；
- 远端：已推送当前分支；
- 矩阵：四个实际 network × FixedTime/MaxPressure × 两次重复，共 16 次；统一 training seed 0、SUMO fixed default、libsumo、3600 simulation steps、360 decisions；
- prefix 规则：`p1_baseline_<agent>_<network>_seed0_r<repeat>_20260722`；
- 通用命令：`SUMO_HOME=/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo /home/dev/miniforge3/envs/colight/bin/python3.10 run.py -w sumo -a <fixedtime|maxpressure> -n <network> --prefix <prefix> --seed 0 --interface libsumo --delay_type apx`；
- 完整 run-list：`data/output_data/analysis/plan1/p1_baseline_16_seed0_20260722/inputs/runs.csv`；
- 分析目录：`data/output_data/analysis/plan1/p1_baseline_16_seed0_20260722/`，包含 16 个运行的规范化表格、配置校验清单和 10 组 PNG/PDF；
- 完整性：16 个运行均为 `已完成`、exit code 0、episode 1、schema v2、3600 steps/360 decisions，配置归档哈希全部通过；
- 确定性：每个 network/controller 的 repeat 1 与 repeat 2 在删除 `wall_time_seconds` 后完整结构化 JSON 相等，8/8 组通过；
- 作废边界：模块 5 中 episode 元数据错误的 `p1m5_maxpressure_smoke_20260722_1` 不属于本 16 次矩阵，也不进入基线结论。

| network | agent | travel time | approximate delay | real delay | queue | throughput | waiting time | unfinished |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `sumohz1x1_config2` | FixedTime | 129.3201 | 3.1596 | 71.6354 | 25.6722 | 1340 | 34.3175 | 63 |
| `sumohz1x1_config2` | MaxPressure | 74.8343 | 3.1834 | 15.3985 | 5.0500 | 1382 | 2.2069 | 29 |
| `sumohz1x1` | FixedTime | 233.1649 | 4.3998 | 170.3684 | 68.7917 | 1631 | 44.9864 | 147 |
| `sumohz1x1` | MaxPressure | 80.5145 | 4.1474 | 21.6156 | 10.2611 | 1969 | 24.3469 | 49 |
| `sumohz1x1_config4` | FixedTime | 170.1606 | 3.4259 | 112.7313 | 40.2778 | 1407 | 39.4919 | 124 |
| `sumohz1x1_config4` | MaxPressure | 78.0718 | 3.5427 | 19.4514 | 7.7139 | 1615 | 19.6429 | 56 |
| `sumohz1x1_config3` | FixedTime | 132.0780 | 1.8659 | 73.4434 | 14.9278 | 705 | 34.8947 | 38 |
| `sumohz1x1_config3` | MaxPressure | 68.0784 | 1.3949 | 9.4207 | 1.5194 | 727 | 0.0625 | 16 |

基线结论：MaxPressure 在四个 network 上均降低 travel time、real delay、queue、waiting time 和 unfinished vehicles，并提高 throughput。`sumohz1x1_config2` 与 `sumohz1x1_config4` 的 approximate delay 略高于 FixedTime，因此后续必须同时报告 approximate/real delay，不得由单一 delay 口径替代综合交通指标。FixedTime 与 MaxPressure 的 reward 不做跨控制器比较或排名。

### DQN Pilot：四场景 seed 0、100 episode

- 状态：已通过；
- 提交信息：`完成Plan 1 Pilot并冻结正式训练配置`；
- 统一命令：`SUMO_HOME=/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo /home/dev/miniforge3/envs/colight/bin/python3.10 run.py -w sumo -a dqn -n <network> --prefix p1_pilot_dqn_<network>_seed0_100ep_20260722 --seed 0 --interface libsumo --delay_type apx`；
- Pilot 配置：100 episodes，评估节点 `[0,10,25,50,100]`，`configs/tsc/dqn.yml` 的 Pilot SHA-256 为 `5d3343ed0e5977adc7c7dc342d20f6b86e80e8eb8a13ce5b413327719ddc439c`；四个运行归档的 `dqn.yml` 哈希一致；
- 配置可比性：四个运行通过 `utils/run_config_compare.py`，除 network、prefix、training seed 和场景路径外无差异，runtime input/action dimension 一致；
- 运行完整性：四个运行均为 `已完成`、exit code 0，各 100 条 TRAIN + 4 条 EVALUATION + 1 条 FINAL_EVALUATION，共 105 条 schema v2 记录；
- trajectory：每场景 100 个 NPZ 分片、36,000 transitions、evaluation transitions 0；validation、index 连续性和所有分片 SHA-256 均通过；
- 评估/checkpoint：每场景评估节点均为 `[0,10,25,50,100]`，evaluation/resumable checkpoint 均为 5/5；四场景 best 均为 episode 100；
- 更新计数：每场景最终为 35,000 gradient updates、3,500 target updates、replay size/capacity `5000/5000`；
- 数值安全：结构化 JSON、trajectory 数值、动作范围和 terminal flags 均通过，无 NaN/Inf 或非法动作；
- 动作诊断：初始未训练模型在部分场景存在 0.95 以上单动作占比，但最终最大单动作占比降至 `0.2806～0.4306`，未形成持续动作坍缩；
- 方向诊断：四场景从 episode 0 到 100 的 travel time、real delay、queue 和 reward 均显著改善；最终 DQN travel time 和 real delay 均优于相同 network 的 MaxPressure 基线；
- 作图优化：学习曲线只将传统控制器绘制为 reference line，不再重复画 episode 1 点；相关自动测试继续通过；
- 最终分析目录：`data/output_data/analysis/plan1/p1_pilot_seed0_100ep_20260722_v2/`，其 `inputs/runs.csv` 同时列出四个 Pilot 和八个基线参考；
- 数据边界：Pilot trajectory 仅用于验证 writer 和诊断，不作为 Plan 2 正式 Offline DQN 数据；Plan 2 仍只允许使用 Plan 1 正式训练产生的 trajectory。

| network | initial travel time | final/best travel time | MaxPressure travel time | final real delay | final queue | throughput | max action share | total measured wall time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sumohz1x1_config2` | 211.4344 | 70.4699 | 74.8343 | 11.2681 | 3.3194 | 1379 | 0.2806 | 1068.69 |
| `sumohz1x1` | 209.0610 | 77.2457 | 80.5145 | 18.0280 | 8.1556 | 1974 | 0.3694 | 1315.38 |
| `sumohz1x1_config4` | 330.7812 | 73.7432 | 78.0718 | 14.6373 | 5.3111 | 1620 | 0.4306 | 1189.45 |
| `sumohz1x1_config3` | 296.0610 | 65.9477 | 68.0784 | 7.5021 | 1.0417 | 727 | 0.3056 | 751.13 |

### 正式训练配置冻结

- 状态：已冻结，尚未执行 20 次正式 DQN 训练；
- `episodes: 400`；
- `evaluation_episodes: [0,10,25,50,100,150,200,250,300,350,400]`；
- 其余 DQN、trainer、SUMO 和 network 配置保持 Pilot 口径；
- 正式 `configs/tsc/dqn.yml` SHA-256：`77a2ef0ccca2b6c5045da42d314b74862703d166380721a3f51635ecf2d31016`；
- 自动回归：`tests/test_plan1_baseline_config.py` 明确检查 400 episodes 和完整正式评估节点，防止配置意外漂移；
- 最终回归：Milestone 0 共 24 项、Plan 1 共 12 项全部通过，相关语法检查和 `git diff --check` 通过。

本子计划结论：Plan 1 启动能力、传统基线和四场景 Pilot 全部通过。下一工作包应另行拆分 Plan 1 正式 Online DQN（四场景 × 五个 training seed，共 20 次），不得直接复用 Pilot prefix 或将 Pilot trajectory 混入正式数据。

后续继续追加真实运行命令、产物地址、验收结论、异常和修复记录。
