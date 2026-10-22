# xiasha1_cross_eval_v1：v2 best-checkpoint 跨场景评估

> 状态：已完成（2026-10-22；384/384 DONE）
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

### 4.1 完成状态

- **384/384 eval DONE**，0 FAILED；2026-10-22 02:55–03:23 UTC。
- 第一轮 48 个 Dueling/D3QN eval 因评估 overlay 未设置
  `dueling/double_dqn` 架构开关而失败；补
  `eval_baseline_checkpoint_{dueling,d3qn}.yml` 后重跑成功，
  已成功的 144 个 eval 保留。
- 分析产物：
  `artifacts/xiasha1_cross_eval_v1/analysis/{eval_metrics.csv,summary.md}`。

### 4.2 目标相对的同场景 vs 跨场景

对每个 target T：in-domain = 在 T 上训练并在 T 上评估；
cross = 在其他 3 个 source 上训练、在 T 上评估。数值为 3 个训练
seeds / 3 个 source 的均值。

| 臂 | in TT | cross TT | ΔTT | in unins | cross unins | Δunins | in queue | cross queue | Δqueue | 平均 TT gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DQN | 13.36 | 13.97 | +0.61 | 115.8 | 57.2 | -58.6 | 3.44 | 3.85 | +0.42 | 4.8% |
| DDQN | 13.37 | 13.90 | +0.53 | 118.4 | 59.8 | -58.6 | 3.46 | 3.84 | +0.38 | 4.2% |
| Dueling | 13.37 | 13.87 | +0.50 | 114.3 | 59.8 | -54.5 | 3.46 | 3.81 | +0.35 | 3.9% |
| D3QN | 13.35 | 13.91 | +0.56 | 122.0 | 55.6 | -66.4 | 3.44 | 3.82 | +0.38 | 4.3% |
| PressLight | 13.90 | 15.33 | +1.44 | 232.3 | 121.4 | -110.9 | 3.66 | 4.36 | +0.71 | 12.0% |
| FRAP | 14.02 | 14.15 | +0.13 | 77.5 | 83.2 | +5.7 | 3.79 | 3.87 | +0.08 | 0.9% |
| MPLight | 13.95 | 14.17 | +0.21 | 81.5 | 75.1 | -6.4 | 3.78 | 3.90 | +0.12 | 1.6% |
| SAC | 13.46 | 14.44 | +0.98 | 134.9 | 71.3 | -63.6 | 3.50 | 4.12 | +0.62 | 7.6% |

**注意**：`Δuninserted` 不能直接当作泛化增益。DQN/SAC/PressLight 的
in-domain checkpoint 是按 travel_time 选的，本身就可能留下大量
未插入车辆；跨源模型（尤其训练自 `dense_extreme`）反而服务更多，
导致 Δunins 为负。TT 的 target-relative 比较更可解释。

### 4.3 源→目标迁移矩阵（目标相对 ΔTT / Δuninserted）

Cell = 在 source 训练、在 target 评估，减去 target 的 in-domain；
对角线为 0。TT 为主，uninserted 受 checkpoint 选择影响较大。

| source \ target | asym_nonly | asym_ns | dense_extreme | fluct_burst |
|---|---:|---:|---:|---:|
| asym_nonly | 0 / 0 | +0.83 / +8.6 | -0.02 / -123.5 | +1.09 / +5.6 |
| asym_ns | +0.20 / -129.6 | 0 / 0 | +0.04 / -133.1 | +0.15 / -0.5 |
| dense_extreme | +1.10 / -103.8 | +1.26 / +9.8 | 0 / 0 | +1.48 / +45.2 |
| fluct_burst | +0.56 / -129.4 | +0.56 / -0.3 | +0.19 / -69.0 | 0 / 0 |

TT 上比较明显的现象：

- 从 `dense_extreme` 迁移到 asym/fluct 包，TT 退化约 +1.1–1.5；
- 从 `asym_ns` 迁移到 `dense_extreme`，TT 几乎不变（+0.04），
  但 uninserted 变化很大；
- 从 `asym_nonly` 迁移到 `fluct_burst`，TT 退化 +1.09；
- 从 `fluct_burst` 迁移到两个 asym 包，TT 退化约 +0.56。

### 4.4 跨源绝对性能排名

同一组 target/source 下，平均跨源性能：

| 排名 | 臂 | cross TT | cross unins | cross queue | 平均 TT gap |
|---:|---|---:|---:|---:|---:|
| 1 | Dueling | 13.87 | 59.8 | 3.81 | 3.9% |
| 2 | DDQN | 13.90 | 59.8 | 3.84 | 4.2% |
| 3 | D3QN | 13.91 | 55.6 | 3.82 | 4.3% |
| 4 | DQN | 13.97 | 57.2 | 3.85 | 4.8% |
| 5 | FRAP | 14.15 | 83.2 | 3.87 | 0.9% |
| 6 | MPLight | 14.17 | 75.1 | 3.90 | 1.6% |
| 7 | SAC | 14.44 | 71.3 | 4.12 | 7.6% |
| 8 | PressLight | 15.33 | 121.4 | 4.36 | 12.0% |

两种排名口径不同：

- **绝对跨源 TT**：DQN 家族最好（13.87–13.97）；
- **相对退化最小**：FRAP 0.9%、MPLight 1.6%，因为它们的 in-domain
  TT 本来就较高（约 14.0），跨源只增加 0.13–0.21；
- **PressLight** 绝对和相对都最差，跨源 TT 退化 1.44（12.0%），
  在 `fluct_burst` 上最严重；
- **SAC** 绝对 14.44、相对 7.6%，不如 DQN 家族稳定。

### 4.5 判读

1. 单场景专训的 DQN 家族在未见需求包上 TT 退化约 **4–5%**，
   不是完全崩溃，但也不是零成本泛化。
2. FRAP/MPLight 的相对 TT 退化最小（<2%），但绝对跨源 TT 仍高于
   DQN 家族；这是“本身 in-domain 较差 + 跨源变化小”的组合，
   不能解读为它们泛化能力全面最好。
3. PressLight 跨场景最不稳定；SAC 次之。
4. uninserted/queue 的跨场景变化很大，且受 in-domain checkpoint
   选择（按 TT 选 best）影响；如果主指标包含服务量，应在
   checkpoint 选择阶段就纳入 uninserted，而不是只看 TT。
5. 本实验每个 target 只用了 1 个 SUMO seed（10000），3 个训练
   seeds；结论是 pilot 级别，扩到 3 个 SUMO seeds 后才能给最终
   泛化排名。

## 5. 结论与后续

**结论**：v2 的 best checkpoint 在未见需求包上表现出中等程度的
TT 泛化退化：DQN 家族约 4–5%，FRAP/MPLight 相对退化最小但绝对
TT 更高，SAC 约 7.6%，PressLight 约 12%。单场景专训不能保证
服务量/queue 的跨场景稳定；checkpoint 选择只看 TT 会放大这一偏差。

**后续**：

1. 扩到 SUMO seeds 10001/10002，确认排序稳定；
2. 用 TT + uninserted 的 Pareto 准则重新选 checkpoint，再做一次
   跨场景评估；
3. 若需要“通用控制器”结论，设计多场景训练 campaign，而不是
   继续增加单场景 seed。

## 勘误（2026-10-22，几何截断缺陷）

本文档全部指标采集自**截断几何**的 xiasha1 路网：进口车道仅 106.4m，
且路由 `departPos≈80`（车辆自距停车线 ~22-27m 处、以 10m/s 凭空插入——
源自 `tools/xiasha_sumo/converter.py` 相机视野重建遗留语义，经
`dev/demand_packs/build_xiasha_demands.py` 传播至全部 14 包）。

影响口径：
- 每车道可观测排队物理上限 ~3 辆，queue 均值/分位、waiting_time 长尾、
  uninserted 计数、travel_time 绝对值均失真；
- 结论中"臂间相对比较"仅在同几何内成立，**不可与修复后（280m、车道起点
  插入）的 v3 指标直接比较**；
- 台账 `ledger/runs.jsonl` 中全部 xiasha1* campaign 行已加
  `errata=geometry_truncated_v1v2`；正式对比口径以 `xiasha1_online_v3`
  为准。
