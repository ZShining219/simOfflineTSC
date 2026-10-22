# xiasha1_explore_v1：充分探索下的 DQN 早收敛与性能验证

> 状态：已暂停（2026-10-21；slow 6 DONE，full 6 ABORTED；转 `xiasha1_online_v2`）
> 发起日期：2026-10-21 ｜ 负责人/Agent：Z / Claude Code
> 最近更新：2026-10-21

## 关联

- campaign：`xiasha1_explore_v1`（12 train run：2 包 × 3 seeds ×
  {slow, full}）
- 队列文件：`artifacts/xiasha1_explore_v1/run_queue.json`
  （artifacts 为 gitignore 仓外资产）
- native 参照：`xiasha1_online_v1` 的 6 个 DONE run
  `xi1on_dqn_{asym_nonly,asym_ns}_s{0,1,2}`
- overlay：`configs/tsc/xiasha1_online/dqn_slow.yml`、
  `dqn_full.yml`；冒烟 `dqn_explore_smoke.yml`
- code_commit：见本轮提交；config_sha 见台账行
- 产物目录：`data/output_data/tsc/sumo_dqn/xiasha1_<pack>/xi1exp_*`

## 1. 改动意图

上一轮 `xiasha1_online_v1` 的 24 个 plain DQN run 在 ep 18–40 就达到
5% 收敛，且 native epsilon 在 ep 4–5 已到下限。本实验只改探索 schedule，
保持其他条件不变，回答两个问题：

1. **充分探索后，评估曲线是否仍然早收敛？** 即收敛点是否仍明显早于
   epsilon 到达下限的 episode。
2. **充分探索是否提升控制器性能？** 相对 native DQN，final-20 的
   travel_time 是否下降、throughput / uninserted 是否改善；同时对照
   maxpressure 规则基线，避免只看 TT 忽略服务需求。

本实验不改变环境、动作、奖励、网络结构、训练预算或评估协议；不改变
native 臂，native 参照直接复用已完成的 6 个 run。

## 2. 改动模块

- `configs/tsc/xiasha1_online/dqn_slow.yml`：epsilon 1.0 →
  0.05，decay 0.99992，learning_start 5000（约 ep14 开始更新，
  epsilon 约 ep118 到下限）。
- `configs/tsc/xiasha1_online/dqn_full.yml`：同一组值，decay 0.99995，
  epsilon 约 ep180 到下限（覆盖几乎整个 200 ep 预算）。
- `configs/tsc/xiasha1_online/dqn_explore_smoke.yml`：2 ep 冒烟，
  learning_start=0，仅验证训练/epsilon/checkpoint 代码路径。
- `dev/xiasha1_online/build_schedule_pilot_queue.py`：生成 12-run 队列。
- `dev/xiasha1_online/analyze_schedule_pilot.py`：native/slow/full
  对比分析，输出 `artifacts/xiasha1_explore_v1/analysis/`。
- `dev/ledger/close_xiasha1_online.py`：上一轮 campaign 收口。
- `dev/INDEX.md`：登记上述一次性脚本。

## 3. 实验设计

### 3.1 场景与参照

- 路网：xiasha1*1，单受控路口 J，4 绿相位。
- 需求包：`asym_nonly`、`asym_ns`（上一轮 DQN 与 maxpressure 的
  两种典型关系：前者 TT 更好但未插入明显更多，后者几乎同服务量）。
- native 参照：`xi1on_dqn_*`，epsilon 0.1→0.01（decay 0.995，
  learning_start 1000，约 ep4 到下限）。

### 3.2 臂与预算

| 臂 | epsilon | decay | epsilon_min | learning_start | epsilon 到下限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 0.1 | 0.995 | 0.01 | 1000 | ~ep4 |
| slow | 1.0 | 0.99992 | 0.05 | 5000 | ~ep118 |
| full | 1.0 | 0.99995 | 0.05 | 5000 | ~ep180 |

- 新跑 12 个 run：`asym_nonly` / `asym_ns` × seeds 0/1/2 ×
  {slow, full}；
- 200 train episodes；每 episode 一次贪心评估（ep0–ep200）；
- 训练期 SUMO seed 固定默认；训练 seed 经 `--seed`；
- legacy `pseudo_step` 信号语义；`saveReplay: false`；
- 资源：CPU（`CUDA_VISIBLE_DEVICES=-1`），队列 `--parallel 6 --stagger 2`。

### 3.3 指标与判定

- 过程：EVALUATION 贪心曲线；收敛 = 10-ep 滑动平均连续 20 次落在
  final-20 均值 ±5%（同时报 ±3%）；
- 收敛相对探索：记录每 run 的 `epsilon_min_episode`，比较收敛 ep
  是否小于 epsilon 到下限的 ep；若小于，说明在仍有探索时曲线已稳定；
- 性能：final-20 的 travel_time / throughput / unfinished_vehicles /
  uninserted / queue；best checkpoint 的 episode 与 travel_time；
- 配对：同 pack/seed 的 slow-native、full-native 差值；3 seeds 报
  均值与方向一致性；
- 判定标准：
  - 问题 1：若 slow/full 的收敛点仍早于各自的 epsilon 下限（或
    明显早于 full 的 ep180），早收敛现象不能只用探索不足解释；
  - 问题 2：若 slow/full 的 final-20 TT 或 uninserted 相比 native
    有稳定改善（3 seeds 同方向），视为性能提升；否则无证据。

## 4. 实验结果

### 4.1 slow 臂（6 DONE）

slow 设定：ε 1.0→0.05，decay 0.99992，learning_start 5000；
epsilon 约 **ep118** 到下限。native 设定：ε 0.1→0.01，decay 0.995，
learning_start 1000；epsilon 约 **ep5** 到下限。

| pack | native TT | slow TT | ΔTT | native 未插入 | slow 未插入 | native 收敛 ep | slow 收敛 ep | epsilon 下限 ep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| asym_nonly | 13.35 | 13.31 | -0.04 | 144.9 | 144.1 | 31.3 | 36.7 | 5 → 118 |
| asym_ns | 11.65 | 11.59 | -0.07 | 1.1 | 1.3 | 28.0 | 37.3 | 5 → 118 |

- 6/6 slow run 的 5% 收敛点在 ep 36–37，**仍远早于 epsilon 下限 ep118**；
  6/6 也早于 epsilon 下限的 50%（ep59）。
- 扣掉 warmup：native 开始更新后约 25–28 ep 收敛，slow 约 22–23 ep；
  slow 没有因为探索更多而延长学习。
- 配对差：ΔTT 仅 -0.04 / -0.07，Δthroughput 约 0，Δuninserted
  -0.8 / +0.2（方向不一致）。first-100 AUC slow 约 2200 vs native
  1440–1580，说明高探索让前期更差，但最终 greedy 策略回到同一水平。
- 结论：**充分（slow）探索没有消除早收敛，也没有带来稳定性能提升。**

### 4.2 full 臂（6 ABORTED）

full 设定：ε 1.0→0.05，decay 0.99995，epsilon 约 ep180 到下限。
用户于 2026-10-21 决定暂停当前 explore 验证、转 `xiasha1_online_v2`；
6 个 full run 停止时约 ep143–154，partial 数据不作最终判断。台账记
ABORTED + 暂停原因，partial 目录保留。

### 4.3 判定

- 问题 1（是否仍早收敛）：**是**。slow 在 epsilon 仍高的阶段就达到
  5% 收敛，且收敛点在 epsilon 下限之前。
- 问题 2（是否有性能提升）：**没有稳定证据**。TT 差值在 ±0.07 内，
  吞吐/未插入方向不一致。
- 因此“探索 schedule 不足”不能解释 xiasha1 上 plain DQN 的早收敛；
  用户决定把 online 基线改为充分探索并重新从头跑，记为
  `xiasha1_online_v2`（见 `docs/experiments/2026-10-21_xiasha1_online_v2.md`）。

## 5. 结论与后续

**结论**：在 xiasha1 的 plain DQN 上，slow epsilon schedule 已足以
排除“探索不足导致早收敛”的解释；早收敛是当前状态/动作/网络/优化
配置下的策略快速稳定，而非 epsilon 衰减过快。性能也没有因慢探索
而提升。

**后续**：`xiasha1_explore_v1` 暂停，不再续跑 full；slow 的 6 个
DONE run 保留为 v2 设计依据。新的 `xiasha1_online_v2` 使用
full-horizon epsilon（ε1.0→0.05，约 ep180 到下限）对 8 臂 ×12 包 ×
3 seeds 重新训练；SAC/MPLight 保留各自原生探索机制。

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
