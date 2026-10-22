# xiasha1_online_v2：充分探索的纯在线 off-policy 基线

> 状态：已完成（2026-10-22；288/288 DONE）
> 发起日期：2026-10-21 ｜ 负责人/Agent：Z / Claude Code
> 最近更新：2026-10-22

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

### 4.1 完成状态

- 288/288 run **DONE**，0 FAILED；2026-10-21 12:22 UTC 启动，
  2026-10-22 02:15 UTC 收尾，约 14 h。
- 台账 288 行已全部 DONE，metrics/config_sha 已替换为 run_manifest
  实际值。

### 4.2 按臂汇总（36 run/臂，12 包 × 3 seeds）

| 臂 | final-20 TT | throughput | unfinished | uninserted | queue | queue P90* | queue P99.5* | waiting_time | conv5 中位 ep | ε 下限 ep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DQN | 13.30 | 1586 | 5.3 | 64.1 | 2.858 | 4.511 | 5.777 | 9.55 | 30.0 | 181 |
| DDQN | 13.26 | 1588 | 5.3 | 62.1 | 2.844 | 4.513 | 5.741 | 9.24 | 30.5 | 181 |
| Dueling | 13.26 | 1588 | 5.3 | 62.7 | 2.847 | 4.506 | 5.839 | 8.53 | 31.0 | 181 |
| D3QN | 13.24 | 1587 | 5.2 | 63.4 | 2.840 | 4.506 | 5.777 | 9.79 | 32.0 | 181 |
| PressLight | **13.96** | 1499 | 5.4 | **151.0** | 3.043 | **5.494** | **6.180** | **17.62** | 48.0 | 181 |
| FRAP | 14.06 | 1608 | 5.5 | **42.4** | 3.174 | 4.742 | 5.801 | 11.50 | 23.5 | 181 |
| MPLight | 14.04 | 1608 | 5.4 | **41.8** | 3.169 | 4.722 | 5.758 | 10.59 | 9.0 | — |
| SAC | 13.28 | 1578 | 5.5 | 71.6 | 2.861 | 4.546 | 5.944 | 11.10 | 13.0 | — |

\* P90/P99.5 是**最后 100 个评估 episode 的 episode 级平均 queue** 的分布，
不是单 episode 内逐决策的时间序列分位数（v2 未开 step log）。

判读：

- DQN 家族四臂几乎同水平（TT 差 ≤0.06，uninserted 差 ≤2）。
- PressLight 明显最差：TT 最高、uninserted 最高、queue 尾部最重、
  waiting_time 约 17.6（DQN 约 9.5）。
- FRAP / MPLight：TT 比 DQN 高约 0.75，但 uninserted 少约 22；
  queue 均值更高，但 P99.5 与 DQN 接近，属于“用 TT 换服务量”。
- SAC：TT 与 DQN 接近，但 uninserted +7.5、waiting_time +1.6，
  没有明显优势。

### 4.3 早收敛

- DQN 家族 conv5 中位 30–32，epsilon 到下限是 ep181；full-horizon
  探索没有消除早收敛。
- PressLight conv5 中位 48，仍远早于 ep181；但它从很高初始 TT 下降，
  属于真实学习后的平台。
- SAC conv5 中位 13；初始 TT 约 106 → 最终 13.3，学习很快。
- FRAP / MPLight 的 conv 数字不能直接解读：FRAP 初始→最终 TT
  约 16.45→14.06（改善 14.8%），MPLight 约 14.49→14.04（改善 3.3%）；
  初始策略已接近最终，5% 判据会把“初始不差”误判为早收敛。

### 4.4 相对 maxpressure（同 pack，12 包均值）

| 臂 | ΔTT（臂−MP） | TT 更优 pack | Δuninserted（臂−MP） | 未插入更优 pack |
|---|---:|---:|---:|---:|
| DQN | -0.42 | 11/12 | +23.5 | 1/12 |
| DDQN | -0.46 | 12/12 | +21.4 | 1/12 |
| Dueling | -0.46 | 11/12 | +22.1 | 2/12 |
| D3QN | -0.47 | 11/12 | +22.7 | 3/12 |
| PressLight | +0.24 | 5/12 | +110.3 | 1/12 |
| FRAP | +0.34 | 1/12 | +1.7 | 4/12 |
| MPLight | +0.33 | 1/12 | +1.2 | 4/12 |
| SAC | -0.43 | 10/12 | +31.0 | 1/12 |

DQN 家族 TT 看起来优于 MP，但 uninserted 在 11/12 包更差；`asym_nonly`
最典型：DQN TT 13.39 vs MP 14.13，但 uninserted 127.7 vs 4。**ATT/TT
的“优势”大部分来自少服务车辆**，不能单独作为胜出证据。

### 4.5 v2 充分探索 vs v1 native（仅 DQN，24 个配对）

- 平均 ΔTT（v2−v1）= **-0.101**，15/24 配对 v2 更低；
- 平均 Δuninserted = **-2.24**，14/24 配对 v2 更少；
- 分 pack 方向不一致：
  - v2 改善：`dense_extreme`（ΔTT -0.311）、`fluct_burst`（-0.343）、
    `fluct_sin`（-0.334）；
  - 基本不变或略差：`asym_nonly`（+0.036）、`asym_ns`（-0.040）、
    `asym_we`（+0.108）、`dense_high`（+0.040）、`fluct_ramp`（+0.033）。

结论：full-horizon 探索对波动/极端需求包有一点帮助，但对 asym 包
没有，整体不是一致提升。

### 4.6 场景与指标边界

- 每个 run 仍只训练一个需求包、在同一包上评估；这是单场景专训，
  不能回答跨场景泛化。
- ATT/TT 对服务差异不敏感：DQN 的低 queue/TT 部分来自 uninserted；
  评价应以 TT + throughput + unfinished/uninserted + queue 尾部同报。
- 真正的 W90/W99.5（单 episode 内逐决策 queue 分位数）当前未采集；
  需要 step-log 评估或 trainer 端增加分位数聚合。

## 5. 结论与后续

**结论**：

1. full-horizon 充分探索没有消除 DQN 家族的早收敛（conv 中位约 30，
   远早于 ε 下限 ep181）；早收敛不是 epsilon 衰减过快造成的。
2. 性能上，v2 相对 v1 native 只有小幅、分场景不一致的改善；不能
   声称“充分探索普遍提升性能”。
3. 控制器差异主要在服务量/queue 尾部，而不是平均 TT：
   PressLight 最差；FRAP/MPLight 用 TT 换 uninserted；SAC 与 DQN 接近。
4. 单场景专训 + ATT 指标不足以区分控制器能力；下一步应先做服务
   感知的 Pareto 汇总和跨场景评估，再决定是否上多路口场景。

**后续**：

1. 以 v2 的 best checkpoint 做跨场景评估（2–3 个未训练 pack ×
   8 臂 × 3 seeds），检验单场景专训的泛化。
2. 对选定 checkpoint 开 `eval_step_log`，补真正的 W90/W99.5 与
   queue 时间序列。
3. 若单路口仍无法区分控制器，再设计多路口/干线 campaign。

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
