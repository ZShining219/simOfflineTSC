# xiasha1_cross_eval_v1：v2 best-checkpoint 跨场景评估

> 状态：进行中（2026-10-22 准备派发 384 eval）
> 发起日期：2026-10-22 ｜ 负责人/Agent：Z / Claude Code
> 最近更新：2026-10-22

## 关联

- campaign：`xiasha1_cross_eval_v1`（8 臂 × 4 source × 3 train seeds ×
  4 target × 1 SUMO seed = 384 eval）
- 队列文件：`artifacts/xiasha1_cross_eval_v1/run_queue.json`
- 源 checkpoint：`xiasha1_online_v2` 的 288 个 DONE run，
  每个 run 的 `evaluation/summary.json` 选出的 best checkpoint
- 评估 overlay：`configs/tsc/eval_baseline_checkpoint.yml`
  （legacy pseudo_step，与 v2 训练口径一致）；Dueling/D3QN 另用
  `eval_baseline_checkpoint_{dueling,d3qn}.yml` 重建对应网络结构
- 产物目录：`data/output_data/tsc/sumo_<arm>/xiasha1_<target>/xi2x_*`

## 1. 改动意图

v2 是单场景专训：每个 run 固定一个需求包训练，并在同一包上评估。
本实验用 v2 的 best checkpoint 做**跨场景评估**，回答：

1. 单场景训练的控制器在未参与训练的需求包上性能下降多少？
2. 哪个臂的跨场景泛化最好/最差？
3. 跨场景性能下降主要来自 TT、throughput 还是 uninserted/queue？

这是评估层实验，不重训、不改模型；只加载冻结 checkpoint 在目标包上
跑一个 3600 s episode。

## 2. 改动模块

- `dev/xiasha1_online_v2/build_cross_scenario_eval_queue.py`：
  读取 v2 的 best checkpoint，生成 384 个 eval 任务。
- `dev/ledger/register_xiasha1_cross_eval.py`：登记 384 行 eval。
- `configs/tsc/eval_baseline_checkpoint.yml`：评估-only overlay（已有）。
- 新增本实验记录与台账索引。

## 3. 实验设计

### 3.1 source / target

- source = target = 4 个包：
  `asym_nonly`、`asym_ns`、`dense_extreme`、`fluct_burst`；
- 每个 source 的每个臂/训练 seed 的 best checkpoint，在 4 个 target 上
  各评估一次；
- source = target 的格子是**同场景控制**，其余 12 个格子是跨场景。

### 3.2 臂与预算

- 8 臂：DQN / DDQN / Dueling / D3QN / PressLight / FRAP / MPLight / SAC；
- 训练 seeds：0/1/2（来自 v2）；
- SUMO seed：10000（pilot 先固定一个；若需要再扩 10001/10002）；
- 每个 eval：1 episode × 3600 s，action_interval 10，legacy pseudo_step；
- 共 8 × 4 × 3 × 4 = **384 eval**。

### 3.3 指标与判定

- 每次 eval 取 `FINAL_EVALUATION` 的
  travel_time / throughput / unfinished_vehicles / queue / real_delay；
- `uninserted = loaded(target) - throughput - unfinished`；
- 泛化差：
  - 同臂、同 source、同 train seed 的跨场景均值 − 同场景控制均值；
  - 按臂汇总 ΔTT、Δthroughput、Δuninserted；
  - 同时报每个 source→target 的矩阵，避免只看总均值。
- 判定：跨场景性能下降越小，泛化越好；若某些臂在同场景好但跨场景
  崩，说明是场景专训。

## 4. 实验结果

**待收割。** 384 个 eval 派发后回填本节、台账与索引。

## 5. 结论与后续

**待收割。**
