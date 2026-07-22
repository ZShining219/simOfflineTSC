# Plan 1 正式 Online DQN 子计划

## 1. 状态、目标与边界

- 所属总 Plan：`docs/plan0721.md`
- 当前状态：执行中
- 执行分支：`codex/milestone0-experiment-infrastructure`
- 目标：完成四个实际 network × 五个 training seed 的 20 次独立 Online DQN 训练，并形成 Plan 2 唯一允许使用的正式 trajectory 白名单。
- 正式矩阵：`sumohz1x1_config2`、`sumohz1x1`、`sumohz1x1_config4`、`sumohz1x1_config3` × seed `[0,1,2,3,4]`。
- 固定环境：SUMO、DQN、libsumo、`delay_type=apx`、SUMO fixed default、3600 simulation steps、action interval 10。
- 非目标：不优化既有 plotting 范围，不修改 state/action/reward、DQN 核心超参数、SUMO、replay 或 trajectory 定义。

传统基线继续有效。2026-07-22 的四场景旧 Pilot 仅保留为旧 evaluation/checkpoint 协议的历史工程证据，不再作为新正式协议的启动证据，其 trajectory 仍禁止进入 Plan 2。旧正式配置 SHA-256 `77a2ef0ccca2b6c5045da42d314b74862703d166380721a3f51635ecf2d31016` 已因协议调整失效；新 Pilot 通过后重新冻结正式配置和哈希。

## 2. 新评估与 checkpoint 协议

每个正式 run 训练 400 episodes；episode 0 在训练前评估，episode 1～400 在训练后立即评估：

- 400 条 `TRAIN`；
- episode 0～399 共 400 条 `EVALUATION`；
- episode 400 一条 `FINAL_EVALUATION`；
- 401 个轻量 evaluation checkpoint，仅保存 online Q-network；
- 11 个 resumable checkpoint，节点为 `[0,10,25,50,100,150,200,250,300,350,400]`；
- final episode 400 为正式主结果；best 从 401 个 evaluation 中按最低 travel time 选择，并列取较早 episode，仅作诊断。

每次 evaluation 必须保持模型、target、optimizer、epsilon、replay、trajectory、训练计数器以及 Python/NumPy/Torch CPU/CUDA RNG 不变。正式 run 每个产生 400 个 trajectory NPZ、144,000 training transitions；evaluation transition count 必须为 0。

## 3. 实现与 Git 门禁

代码优化按模块独立完成实现、针对性测试、Milestone 0/Plan 1 全量回归、语法和 `git diff --check`；通过后使用明确路径暂存、中文 commit，fetch 确认远端无未知提交后 push，才进入下一模块。

已完成模块：

1. `e1bb6c2 拆分Plan 1评估与恢复检查点语义`：拆分 evaluation/resumable 节点，增加 RNG 隔离与 summary v2；Milestone 0 25 项、Plan 1 12 项通过。
2. `68fb401 增强Plan 1逐回合评估分析与强验收`：增加新协议强验收、TRAIN/EVALUATION 双 AUC、单点/连续 5 次学习速度指标；Milestone 0 25 项、Plan 1 13 项通过。

训练或真实验证只允许逐条执行明确的 `run.py` 命令。禁止创建 launcher、batch、watch 或服务调用脚本，禁止调用现有服务启动训练，禁止一次性启动脚本堆积任务。并行只通过受控地直接启动若干独立 `run.py` 进程实现；每个进程必须有独立 prefix、会话和验收记录。

## 4. 启动门禁与 Pilot

正式配置冻结前必须完成：

1. 真实 SUMO 非干扰 A/B：`sumohz1x1`、同一新测试 seed、5 个训练 episodes；A 评估 `[0,5]`，B 评估 `[0,1,2,3,4,5]`，恢复点均为 `[0,5]`。
2. A/B 必须在 trajectory 全部数组、去除 wall time 的 TRAIN 记录、online/target tensor hash、optimizer、epsilon、replay、训练计数器和 RNG 状态上一致。任何不一致均停止新 Pilot。
3. A/B 通过后运行四场景 seed 0、100-episode 新 Pilot；每场景 100 条 TRAIN、101 次 evaluation、101 个 evaluation checkpoint、5 个 resumable checkpoint、100 个 trajectory 分片和 36,000 transitions。
4. 新 Pilot 通过后切换到 400 episodes，重新冻结配置 SHA-256，执行全量测试、配置归档检查和 Git 门禁。

若修复触及 state/action/reward、DQN 网络或核心超参数、SUMO、replay、trajectory/指标计算或 evaluation/checkpoint 语义，立即停止启动新 run，明确已有证据的作废范围，并重新判断 A/B、Pilot 和正式 run 的重跑范围。

## 5. 正式波次、命名与验收

资源波次固定为：

1. Wave 1：seed 0，四个 network，共 4 runs；全部强验收后继续。
2. Wave 2：seed 1、2，共 8 runs。
3. Wave 3：seed 3、4，共 8 runs。

prefix：`p1_formal_dqn_<network>_seed<seed>_400ep_20260722`。失败 run 不覆盖、不删除；重跑依次使用 `_r2`、`_r3`，最终显式 run-list 只收录有效替代运行。

每个正式 run 必须满足：完成状态和 exit code 0；配置归档及哈希有效；400 TRAIN、401 次连续 evaluation；401/11 两类 checkpoint 可加载且节点正确；400 trajectory 分片、144,000 transitions、evaluation 零写入；index/分片哈希/step 连续；无 NaN/Inf、非法动作或断链；terminal 语义正确；最终 replay `5000/5000`；gradient/target/epsilon/loss/wall time 完整；best 规则正确；原始 episode 400 `FINAL_EVALUATION.action_distribution` 完整。

动作坍缩与 reward/交通指标方向不设自动阈值。最终及末 20 次 evaluation 的完整动作分布、reward、travel time 和两类 delay 逐 run 人工复核并记录理由；人工警告不自动判失败。

## 6. 分析与交付

Evaluation 曲线是无探索策略性能主口径，TRAIN 曲线是含探索的辅助诊断；两者分别报告 first-100 AUC。FixedTime、MaxPressure 和自身 episode 400 travel time 的 110% 阈值分别报告首次单点达到与首次连续 5 次达到。

最终分析只读取显式 run-list：20 个 `role=formal` DQN 加 8 个已通过传统基线 reference；不包含 Pilot、失败或作废 run。travel time 为跨控制器主排名指标，approximate delay 和 real delay 同时报告，reward 不跨控制器排名。

Plan 2 白名单只包含通过全部强验收的 20 个正式 run trajectory。最终必须记录完整命令、新配置哈希、产物地址、定量指标、资源波次、异常、人工诊断、结论和作废边界。

## 7. 执行记录

后续按 A/B、新 Pilot、正式配置冻结、Wave 1～3、最终分析与交付顺序追加真实命令和证据，不预填未执行结果。
