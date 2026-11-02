# hz1x1_offline_v1：四控制器异构历史经验纯离线基线

> 状态：正式训练运行中（8/162 已启动，其余由队列待派发）
> 发起日期：2026-10-23　｜　负责人/Agent：Z / Codex
> 最近更新：2026-10-23

## 关联

- campaign：`hz1x1_offline_v1`（实验配置修订 `mplight_r1`）
- run_id / 队列文件：`hz1ofm_*` / `artifacts/hz1x1_offline_v1/mplight_r1/run_queue.json`
- 数据集：`data/output_data/offline_datasets/hz1x1/v1_hetero_mplight/`
- 源 campaign：`hz1x1_sanity_v1`，d3qn / FRAP / SAC / MPLight 的 DONE run
- 代码/配置：`configs/offline_tsc/hz1x1_v1.yml`、`data/dataset/offline_trajectory_dataset.py`、`dev/hz1x1_offline/prepare.py`

## 1. 改动意图

将新加入且已完成轨迹校验的 MPLight 行为策略纳入离线基线。四种策略共享入口车道等待车辆数负均值奖励语义；d3qn 历史实现额外乘 12，因此离线训练视图对其奖励乘 `1/12`，FRAP、SAC、MPLight 乘 1。只在离线加载时缩放 reward，源 transitions 与 shard 不修改。

替代此前登记但未派发的三控制器 v1。旧 144 行保留并转 `SCOPE_DISCARD`，旧数据集保留，旧队列标记 dropped；新设计仍属 campaign `hz1x1_offline_v1`，使用独立 revision、dataset ID、run ID 与队列路径。

## 2. 改动模块

- `data/dataset/offline_trajectory_dataset.py`：确认 MPLight 奖励语义和单位。
- `configs/offline_tsc/hz1x1_v1.yml`：四控制器矩阵、奖励尺度、E5 与新 revision 身份。
- `dev/hz1x1_offline/prepare.py`：数据索引、队列、台账更新及零仿真可用性检查。
- `ledger/runs.jsonl`、`EXPERIMENTS.md`、`EXPERIMENTS/hz1x1_offline_v1.md`：旧方案作废、新方案登记。

## 3. 实验设计

- 数据：48 个完整在线 run（4 控制器 × 4 同构路口车流 × 3 行为 seeds），每 run 200 × 360 = 72k transitions；总计 3.456M transitions。四个网络配置的控制拓扑签名一致，差异是车流。
- 条件：E0 单 run；E1 多行为 seed；E2 分别比较 FRAP、SAC、MPLight 与 d3qn；E3 同路口四控制器混合；E4 d3qn 跨车流；E4-LOO 排除评估目标 A；E5 采用 A/d3qn、B/FRAP、C/SAC、D/MPLight 的四来源等权稀疏池。
- E5 每个来源占 25%，共 288k transitions。由于控制器与车流在 E5 中一一配对，E5 解释为双轴稀疏混合池压力测试，不用于单独归因控制器或车流效应；对应主效应由 E2/E3 与 E4 刻画。
- 9 条件 × 3 池复本 = 27 池；每池运行 batch_dqn、cql_dqn 各 3 个离线 seeds，共 162 个训练 run。每 run 144k updates、batch 64、81 个评估节点；训练期间不与仿真交互，评估在 A 上执行贪心 SUMO episode。
- 末端 `terminated` 或 `truncated` 均关闭 TD bootstrap。池内按 transition 均匀采样，不作质量筛选。

## 4. 实验结果

- 源数据核验：MPLight 12/12 run 均为 DONE；轨迹校验通过，覆盖 200 episodes、72,000 transitions/run、360 decisions/episode、8 actions。四个需求配置的控制拓扑哈希一致。
- 数据集索引构建通过：48 源 run、9,600 episode shards、3.456M transitions、27 pool manifests；27/27 池通过 manifest/transition 计数、reward view、动作与观测维度检查，并能抽取 batch。
- 队列包含 162 个训练任务，已登记；没有派发训练或启动 SUMO 实验，因此没有离线学习指标或科研结论。旧三控制器方案 144 行为 `SCOPE_DISCARD`，旧 queue 已标 dropped，原 `v1_hetero` 数据集保留。
- 2026-10-23 正式派发：干净代码快照 `cc8672d`；队列 runner tmux `hz1ofm_queue`，并行度 8；TensorBoard 投影器 tmux `hz1ofm_tb`，地址 `http://127.0.0.1:6010`。启动核验时 8/162 任务为 RUNNING，其余 154 个为 PENDING；SUMO run 状态文件已写入正式输出目录，TensorBoard 已从结构化离线记录投影 45,034 个标量点。实验尚未完成，不报告学习结果。

## 5. 结论与后续

- MPLight 奖励单位与 FRAP/SAC 对齐；d3qn 仅作 `1/12` 离线视图缩放，原始轨迹不变。
- 训练持续由队列 runner 管理；TensorBoard 从 `metrics/offline_records.jsonl` 派生事件文件，日志位于 `artifacts/hz1x1_offline_v1/mplight_r1/tb/`。正式结果和结论需在任务完成并通过结果核验后回填。
