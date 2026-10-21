# xiasha1_online_v2：充分探索的纯在线 off-policy 基线

> 状态：进行中（2026-10-21 准备派发 288 run）
> 发起日期：2026-10-21 ｜ 负责人/Agent：Z / Claude Code
> 最近更新：2026-10-21

## 关联

- campaign：`xiasha1_online_v2`（8 臂 × 12 包 × 3 seeds = 288 train run）
- 队列文件：`artifacts/xiasha1_online_v2/run_queue.json`
  （artifacts 为 gitignore 仓外资产）
- overlay：`configs/tsc/xiasha1_online_v2/{dqn,ddqn,dueling,d3qn,
  presslight,frap,sac,mplight}.yml`
- 前序：`xiasha1_online_v1`（已收口：24 DONE / 12 ABORTED /
  252 SCOPE_DISCARD）；`xiasha1_explore_v1`（已暂停：slow 6 DONE /
  full 6 ABORTED）
- 产物目录：`data/output_data/tsc/sumo_<arm>/xiasha1_<pack>/xi2on_*`

## 1. 改动意图

v1 的 24 个 plain DQN run 在 ep 18–40 早收敛；`xiasha1_explore_v1`
的 slow 臂（ε 1.0→0.05，约 ep118 到下限）证明：

- 6/6 slow run 仍在 ep 36–37 收敛，远早于 epsilon 下限；
- final-20 TT 相对 native 仅 -0.04/-0.07，吞吐/未插入无稳定改善。

因此用户决定：不再续跑 explore 的 full 臂，把 online 基线整体改为
**full-horizon 充分探索**，从零重新训练，记为 `xiasha1_online_v2`。
v2 的目的不是再验证“slow 是否有用”，而是提供一个**统一充分探索
口径下的 8 臂正式基线**，并回答：

1. 在 full-horizon 探索下，各臂的早收敛是否仍存在；
2. 相比 v1 的 native 口径，各臂训练后三指标是否改变；
3. 不同控制器之间在同一充分探索口径下是否可比。

## 2. 改动模块

- `configs/tsc/xiasha1_online_v2/`：8 个臂 overlay。
  - DQN 家族（dqn/ddqn/dueling/d3qn）、PressLight、FRAP：
    ε 1.0→0.05，decay 0.99995，epsilon_min 0.05，
    learning_start 5000（约 ep180 到下限）；
  - SAC：不加 epsilon；Categorical 策略原生随机探索，
    `sac_temperature 0.2` 不变；
  - MPLight：不改 epsilon；pfrl 线性 ε 1.0→0.0，约 160 ep 覆盖
    大部分训练预算；
  - 所有臂统一 200 ep、每 ep 贪心评估 ep0–200、
    resumable {0,25,50,100,150,200}、`saveReplay: false`、
    legacy pseudo_step。
- `dev/xiasha1_online_v2/build_online_v2_queue.py`：生成 288-run 队列。
- `dev/ledger/register_xiasha1_online_v2.py`：幂等登记 288 行。
- `dev/ledger/backfill_xiasha1_explore.py`：暂停 explore 后的台账回填。
- 文档：本文件、README、v1/explore 记录更新、台账索引再生成。

## 3. 实验设计

### 3.1 臂与探索口径

| 臂 | 启动 | 观测 | 奖励 | 探索 |
| --- | --- | ---: | --- | --- |
| DQN | `-a dqn` | 16 | `-queue` | ε 1.0→0.05，约 ep180 |
| DDQN | `-a dqn` + `double_dqn` | 16 | `-queue` | 同上 |
| Dueling | `-a dqn` + `dueling` | 16 | `-queue` | 同上 |
| D3QN | `-a dqn` + 两开关 | 16 | `-queue` | 同上 |
| PressLight | `-a presslight` | 22 | `-pressure` | 同上 |
| FRAP | `-a frap` | 13 | `-queue` | 同上；`one_hot: false` |
| MPLight | `-a mplight` | 13 | `-queue` | pfrl 线性 ε 1.0→0.0 |
| SAC | `-a sac` | 16 | `-queue` | Categorical 策略，temperature 0.2 |

### 3.2 场景与预算

- 路网：xiasha1*1 单受控路口 J，4 个绿相位。
- 需求：12 个 1 h 包（741–2872 veh/h）。
- seeds：0/1/2；训练期 SUMO seed 固定默认。
- 预算：200 episodes；`steps=3600`、`action_interval=10`；
  每 episode 360 个决策，200 ep = 72k 决策。
- 评估：每 episode 一次贪心评估（ep0–ep200），共 201 点。
- 信号执行：legacy `pseudo_step`（不安装 `signal_control`）。

### 3.3 指标与判定

- 收敛：10-ep 滑动平均连续 20 次落在 final-20 均值 ±5%
  （同时报 ±3%）；记录 epsilon 到下限的 episode（SAC/MPLight
  不适用 epsilon 判据）。
- 性能：final-20 的 travel_time / throughput /
  unfinished_vehicles / uninserted / queue；best checkpoint 仅作诊断。
- 对照：v2 同臂 vs v1 同臂同 pack/seed（v1 只有 plain DQN 完整）；
  跨臂只比交通指标，不比 reward/loss。
- 与 v1 的可比性声明：v2 的探索 schedule 是**显式语义变更**，
  v2 与 v1 的同名指标不能混排；对比时必须标明口径。

### 3.4 派发

- 队列：`artifacts/xiasha1_online_v2/run_queue.json`；
- runner：`tools/run_experiment_queue.py`，计划 `--parallel 12
  --stagger 2`；预计 12 并发约 14 h（单 run ~35 min）。
- 正式 run 前：commit → 台账 REGISTERED → smoke → 派发。

## 4. 实验结果

**待收割。** 288 个 run 派发后回填本节、台账与索引。

## 5. 结论与后续

**待收割。** 不在数据完成前写结论。
