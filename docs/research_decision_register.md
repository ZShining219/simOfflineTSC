# 集中科研决策表

## 1. 使用规则

本表只管理可能改变实验解释、算法语义、指标口径或预算的决定。普通工程实现不得加入本表并借此要求用户逐项确认。

字段固定为：决策编号、议题、项目当前行为、Plan要求、差距、科研影响、可选方案、推荐方案、用户决定、状态、影响里程碑、实现提交、验证证据。

状态只允许使用：

- 已发现：已发现问题，尚未形成完整选项；
- 待决定：选项和影响已明确，等待用户决定；
- 已决定：用户已确认但尚未实现；
- 已实现：已经实现并形成提交；
- 已验证：针对性测试和真实 smoke 已通过；
- 已推迟：明确推迟至后续里程碑；
- 不适用：审核后确认不适用于当前项目。

Codex 不得用“推荐方案”代替“用户决定”。没有明确决定时不得修改科研语义。一个决定可以阻塞相关功能，但不阻塞无关工程工作。每个已实现决定必须关联提交和验证证据。

## 2. 决策登记

### RD-001：Seed 语义与 SUMO seed 边界

- 决策编号：RD-001
- 议题：项目中未限定 seed 的含义及 SUMO seed 是否管理
- 项目当前行为：run.py --seed 设置训练侧 Python、NumPy、PyTorch 等随机性；SUMO 启动命令不显式传递 seed
- Plan要求：区分训练侧 seed 与 SUMO seed，不把不同 training seed 解释为不同微观交通实现
- 差距：README 尚未明确说明该边界
- 科研影响：错误解释 seed 会夸大重复实验覆盖的随机性与鲁棒性
- 可选方案：A. 增加 SUMO seed 管理；B. 保持现状并明确 training_seed 边界
- 推荐方案：B
- 用户决定：后续未明确限定的 seed 均为 training_seed，即 run.py --seed；不额外管理 SUMO seed；记录 sumo_seed_mode=fixed_default
- 状态：已验证
- 影响里程碑：Milestone 0 及全部后续实验
- 实现提交：本功能提交
- 验证证据：`docs/verification/milestone0/function3_reproducibility.md`；README 声明、两次同 seed 运行 manifest、模型哈希与随机序列一致性

### RD-002：Learning starts 边界条件

- 决策编号：RD-002
- 议题：learning_start=1000 时使用大于还是大于等于
- 项目当前行为：total_decision_num > learning_start
- Plan要求：计划文字使用达到 learning_starts 后开始训练，尚未明确边界包含关系
- 差距：原项目在第 1000 与第 1001 个 decision 附近的语义可能与常见解释不同
- 科研影响：改变首次更新时刻，影响 Pilot 曲线和复现实验
- 可选方案：A. 保留原项目严格大于；B. 改为大于等于并统一全部正式实验
- 推荐方案：正式实验前结合计数定义一次性确定
- 用户决定：尚未决定
- 状态：待决定
- 影响里程碑：Plan 1、Plan 3、Plan 4；Milestone 0 只记录现状
- 实现提交：无
- 验证证据：trainer/tsc_trainer.py 训练条件审计

### RD-003：Learning starts 前的动作策略

- 决策编号：RD-003
- 议题：开始更新前完全随机动作还是 epsilon-greedy
- 项目当前行为：learning starts 前直接调用 agent.sample，属于完全随机动作
- Plan要求：冻结 epsilon 探索与训练过程，但未明确预热阶段动作规则
- 差距：配置 epsilon 在预热阶段不生效
- 科研影响：改变早期数据覆盖和 Plan 2 的早期数据分布
- 可选方案：A. 保留完全随机预热；B. 全程使用 epsilon-greedy
- 推荐方案：若强调保持 LibSignal 基线，优先 A，并在配置定义中明确
- 用户决定：尚未决定
- 状态：待决定
- 影响里程碑：Plan 1、Plan 2 数据来源；Milestone 0 不修改
- 实现提交：无
- 验证证据：trainer/tsc_trainer.py 动作选择分支审计

### RD-004：Epsilon schedule 的单位与形式

- 决策编号：RD-004
- 议题：epsilon 按何种计数衰减
- 项目当前行为：每次 DQN 梯度更新后乘 0.995，最低 0.01
- Plan要求：冻结 epsilon start、end、decay，但尚未明确单位
- 差距：Plan 文本不足以复现当前 schedule
- 科研影响：探索强度影响 Online 性能、轨迹覆盖与 Offline 数据质量
- 可选方案：A. 保留按梯度更新乘法衰减；B. 改为按 decision step 的显式 schedule
- 推荐方案：正式实验前集中决定，不在 Milestone 0 修改
- 用户决定：尚未决定
- 状态：待决定
- 影响里程碑：Plan 1、Plan 3、Plan 4
- 实现提交：无
- 验证证据：agent/dqn.py epsilon 更新位置审计

### RD-005：Episode 截断与 bootstrap

- 决策编号：RD-005
- 议题：固定 3600 秒结束是否 bootstrap
- 项目当前行为：transition 未保存 terminated/truncated，TD target 始终 bootstrap
- Plan要求：轨迹要求分别保存 terminated 和 truncated，但没有冻结 target mask 语义
- 差距：后续 Online/Offline DQN 无法按终止类型一致计算 target
- 科研影响：直接改变 DQN 与 Offline DQN loss
- 可选方案：A. 保留始终 bootstrap；B. terminated 不 bootstrap、truncated bootstrap；C. 两者均不 bootstrap
- 推荐方案：Plan 1 轨迹实现前决定
- 用户决定：尚未决定
- 状态：已推迟
- 影响里程碑：Plan 1、Plan 2
- 实现提交：无
- 验证证据：agent/dqn.py replay 与 TD target 审计

### RD-006：Delay 主指标口径

- 决策编号：RD-006
- 议题：average delay 使用 approximate 还是 real
- 项目当前行为：由 run.py --delay_type 选择，默认 apx
- Plan要求：冻结 average delay 并说明 SUMO 字段来源
- 差距：尚未决定正式实验主口径
- 科研影响：直接影响主要结果和不同控制器比较
- 可选方案：A. apx 为主；B. real 为主；C. 二者都报但预先指定主指标
- 推荐方案：指标审计完成后集中决定
- 用户决定：尚未决定
- 状态：已推迟
- 影响里程碑：Plan 1 及后续报告；Milestone 0 只审计
- 实现提交：无
- 验证证据：common/metrics.py 与 world/world_sumo.py 指标来源审计

### RD-007：Checkpoint 主策略与 best 选择指标

- 决策编号：RD-007
- 议题：best checkpoint 按何种指标选择
- 项目当前行为：按固定 save_rate 保存 target network，不自动选择 best
- Plan要求：最终 checkpoint 为主，best 只作稳定性诊断
- 差距：best 的选择指标和并列处理尚未冻结；checkpoint 的 online/target 内容由 RD-008 单独决定
- 科研影响：选择规则可能造成选择性报告
- 可选方案：A. 按预设评估节点的 average travel time；B. 按 delay；C. 只保存但不自动选择
- 推荐方案：Plan 1 评估协议确定时决定
- 用户决定：尚未决定
- 状态：已推迟
- 影响里程碑：Plan 1
- 实现提交：无
- 验证证据：checkpoint 与评估机制审计

### RD-008：Checkpoint 的 online/target 语义

- 决策编号：RD-008
- 议题：evaluation checkpoint 与 resumable checkpoint 分别保存和使用哪套网络
- 项目当前行为：save_model 只保存 target network；动作和常规贪心评估由 online network 产生
- Plan要求：最终 checkpoint 代表当前策略，同时顺序实验需要完整恢复训练状态
- 差距：旧 target-only 文件既不能完整恢复训练，也不一定代表保存时 online 执行策略
- 科研影响：改变 evaluation 使用的网络会影响报告性能；缺少 target/optimizer 等状态会破坏恢复连续性
- 可选方案：A. 保持旧 target-only checkpoint；B. evaluation checkpoint 保存并评估 online，resumable checkpoint 保存 online、target 及完整训练状态；C. 两类文件均保存两套网络但另行决定评估网络
- 推荐方案：B
- 用户决定：采用 B。evaluation checkpoint 保存 online Q-network，并以 online Q-network 评估；resumable checkpoint 保存 online、target、optimizer、epsilon、训练计数器、随机状态和恢复所需 replay 状态
- 状态：已验证
- 影响里程碑：Milestone 0、Plan 1、Plan 3、Plan 4
- 实现提交：本功能提交
- 验证证据：`docs/verification/milestone0/function6_checkpoints.md`；evaluation online hash 一致；resume 后继续一次 optimizer update；旧 target-only 文件无 checkpoint_type

### RD-009：跨控制器 Reward 口径与报告边界

- 决策编号：RD-009
- 议题：DQN、FixedTime 和 MaxPressure 的 reward 是否可作为同口径指标直接比较
- 项目当前行为：DQN 使用 incoming lane waiting_count 的负平均值乘 12；FixedTime 和 MaxPressure 使用 incoming lane_count 的负平均值乘 12；Metrics 仅按 decision 聚合传入 reward
- Plan要求：记录 reward 曲线并保持现有算法 reward，但尚未明确跨控制器 reward 的报告边界
- 差距：同名 reward 实际来自不同底层交通量，直接横向比较会造成错误解释
- 科研影响：影响基线控制器与 DQN 的 reward 表格、曲线和结论；不影响 travel time、queue、delay、throughput 的独立报告
- 可选方案：A. reward 仅用于各控制器内部诊断，不做跨控制器比较；B. 后续统一 reward 定义并重跑全部相关实验；C. 同时报原始 reward 与另行定义的共同诊断 reward
- 推荐方案：在 Plan 1 正式报告协议确定时决定，不在 Milestone 0 改公式
- 用户决定：尚未决定
- 状态：待决定
- 影响里程碑：Plan 1 及后续跨控制器结果报告；Milestone 0 只审计并保持现状
- 实现提交：无
- 验证证据：`docs/verification/milestone0/function7_metric_audit.md`；agent/dqn.py、agent/fixedtime.py、agent/maxpressure.py reward_generator 审计

## 3. 同步要求

用户决定导致 plan0721.md 发生变化时，必须在同一功能周期同步本表。状态从“已决定”进入“已实现”后必须填写提交；进入“已验证”后必须填写验证命令或证据路径。sumo_seed_mode=fixed_default 是已确认边界，不再作为待决策功能。
