# Plan 1 项目支持、传统基线与 Pilot 子计划

## 1. 状态与范围

- 所属总 Plan：`docs/plan0721.md`
- 当前状态：执行中
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

- 状态：已通过，待提交；
- 实现：real delay 无副作用快照、活跃车辆平均累计等待、路网内未完成车辆数、动作分布、切相频率、replay/target 诊断和结构化指标 schema v2；
- 兼容：schema v1 日志仍可验证和读取；
- 自动检查：Milestone 0 共 24 项通过，Plan 1 指标新增 2 项通过；
- 语法检查：`common/metrics.py`、`world/world_sumo.py`、`trainer/tsc_trainer.py`、`utils/logger.py` 和新增测试通过；
- 已知环境提示：旧 Gym 弃用提示不影响本模块 CPU/SUMO 路径。

后续继续追加真实运行命令、产物地址、验收结论、异常和修复记录。
