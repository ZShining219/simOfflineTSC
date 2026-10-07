# Methods 实现事实核验报告

> 核验范围：正式 Plan34、HA-SODQN E3/E4 训练代码、配置、manifest、运行记录和正式分析脚本。本文档为实现事实记录，不是论文 Methods 草稿。
>
> 纪律：本次核验只读，未修改代码、未重新训练、未补跑仿真、未联网。正式运行 manifest 指向的 commit 优先于当前工作区源码。

## 1. Executive summary

### 已确认

- 正式训练 reward 是入口 lane 的 `lane_waiting_count` 聚合值，取负后乘 12；training reward 与 evaluation travel time 是两个不同指标。
- state 是 8 个入口 lane-count 加 8 维当前 phase one-hot，共 16 维；action 有 8 个 green-phase index。
- horizon=3600 s，action interval=10 s，每 episode=360 decisions。
- HA 的 history ratio 是每次 gradient update 的 historical batch 配额，并同时作为 historical loss 的权重。
- batch_size=64 时，ratio 0.25/0.50/0.75 分别为 16/32/48 个 historical samples。
- online 与 historical 分别计算同一 DQN TD/MSE loss，再做加权和。
- historical archive 来自 Plan 2 Q1、episodes 1–100；P1C 只访问当前 stage 之前场景，P1F 访问全部场景。
- RAND、COV、CQ、CQA、DHOA 的实际 selection/sampling 规则可以由 `sequential/owp.py` 还原。
- frozen evaluation 为 epsilon=0、无 replay 写入、无 gradient update；`evaluation_retries=3` 是同一 evaluation cell 的最多三次执行尝试，不是三个 evaluation seeds。
- travel time 是已完成车辆的实际行程时间平均值。
- Q drift 是固定 probe 上完整 8-action Q 向量的 Euclidean L2 距离均值。
- HA CONT–DHOA R25 正式严格配对为 19 对；O3 seed1 因 Stage-1 checkpoint digest 不一致被排除。
- 缺失 checkpoint 在正式分析中排除，不插值。

### 与当前理解冲突

- “reward 与 travel time 有关”不符合正式训练实现。
- reward 不是 `-sum(queue)`，也不是 pressure 或 waiting-time total。
- DHOA 不是一个额外 Q-value selector；其实现直接从全部可见历史数据采样。
- P1C 不能确认是“causally available history”的正式英文缩写。
- P1F 在 stage 1 即可访问当前及未来场景历史数据。
- HA 从 episode-0 initial-state checkpoint 开始；Plan34 从 Plan 1 episode-100 parent checkpoint 继续，二者训练起点不同。
- HA 配置使用 `normalized_travel_time_aulc_0_100`，Plan34 配置使用 `normalized_travel_time_auc_0_100`，虽然底层都是 trapezoidal integration。

### 仍未确认

- `DHOA` 的正式英文全称。
- `P1C`/`P1F` 中 C/F 的正式英文展开。
- 当前项目没有唯一名为 R3/R4 的正式目录；实际证据分布在 Plan34、HA E3/E4 和分析 bundle。
- 没有找到独立的 Fig4 ratio-sensitivity 正式脚本，不能确认其是否严格使用 5 个 matched seeds。
- 没有确认正式分析实际使用 Holm 或其他 multiple-comparison correction。
- 本地只能确认使用 hz1x1/Hangzhou 文件并声明基于 LibSignal，不能确认原始公开 benchmark 的完整出处。

## 2. Core Methods fact table

| Methods item | confirmed implementation | evidence path/line | confidence |
|---|---|---|---|
| Training reward | `LaneVehicleGenerator(['lane_waiting_count'], in_only=True, average='all', negative=True)`，再乘 12 | `sequential/agent.py:140-149, 380-382` | CONFIRMED |
| Waiting count | 实际代码根据 `waiting_times` 中的 `wait>0` 计数；不是仅按当前速度判断 | `world/world_sumo.py:275-290` | CONFIRMED |
| Reward aggregation | 每 road 先 lane mean，再 road mean，取负 | `generator/lane_vehicle.py:158-172` | CONFIRMED |
| Decision reward | 10 个一秒 reward 的 arithmetic mean | `sequential/trainer.py:57-84` | CONFIRMED |
| Terminal reward | 无特殊 terminal reward | `sequential/trainer.py:74-84` | CONFIRMED |
| Terminal bootstrap | `terminated/truncated` 保存但 TD target 不做 terminal mask | formal `e7705f7:sequential/ha_agent.py:184-198`; `build_checkpoint_probe.py:291-293` | CONFIRMED |
| State | 8 lane-count + 8 phase one-hot = 16 | `archive_root_manifest.json` 的 `observation_dim=16/state_dim=8/phase_dim=8`; `sequential/agent.py:431-450` | CONFIRMED |
| Lane count | incoming lanes 的 SUMO lane vehicle count，检测范围默认 200 m | `sequential/agent.py:140-143`; `world/world_sumo.py:309-327, 738-750` | CONFIRMED |
| State normalization | 未执行 normalization | `sequential/agent.py:431-461` | CONFIRMED |
| Action | 8 个 green-phase index | `archive_root_manifest.json` 的 `action_dim=8/green_action_count=8` | CONFIRMED |
| Yellow | phase 切换时自动生成 interstitial yellow phase | `world/world_sumo.py:127-132, 340-361` | CONFIRMED |
| Action mask | 正式 DQN path 未见 action mask | `sequential/agent.py:389-405`; `sequential/evaluator.py:149-160` | CONFIRMED |
| Timing | 3600 s / 10 s = 360 decisions | `configs/sequential/ha_sodqn_b100.yml:15-23`; `sequential/evaluator.py:394-397` | CONFIRMED |
| Online replay | persistent FIFO deque，capacity=5000 | `sequential/core.py:96-108` | CONFIRMED |
| Historical archive | Plan 2 Q1，episodes 1–100 | `sequential/historical_archive.py:41-87, 222-233` | CONFIRMED |
| Ratio/update | historical quota 与 historical loss weight | `sequential/ha_agent.py:224-253` | CONFIRMED |
| Warm-up | local episode≤10 时关闭历史采样，同时收集 alignment observations | `sequential/ha_agent.py:153-168, 224-233` | CONFIRMED |
| Target network | 每 10 次成功 gradient update 同步 | `sequential/ha_agent.py:260-268`; `sequential/core.py:63-93` | CONFIRMED |
| Frozen evaluation | epsilon=0，无 replay/update | `sequential/evaluator.py:86-100, 217-239` | CONFIRMED |
| Travel time | completed vehicles 的实际行程时间均值 | `world/world_sumo.py:540-546, 680-696`; `common/metrics.py:162-170` | CONFIRMED |
| Q drift | 完整 Q 向量的 Euclidean L2，先 state-level 再均值 | `build_checkpoint_probe.py:193-201` | CONFIRMED |
| Probe | 每 scene×behavior seed×probe type=128，uniform without replacement | `build_checkpoint_probe.py:32, 114-150` | CONFIRMED |
| HA statistics | two-sided SciPy Wilcoxon signed-rank | `complete_mechanism_bundle.py:313-321, 543-555` | CONFIRMED |
| Plan34 statistics | exact sign-flip + paired bootstrap | `sequential/analysis.py:329-367` | CONFIRMED |

## 3. Reward definition

设 incoming roads 为 (r=1,ldots,R)，road (r) 的 lanes 为 (L_r)，时刻 (t) 的 lane waiting count 为 (w_{r,l}(t))，则单秒 reward 为：

\[
r_t=-12\cdot\frac{1}{R}\sum_{r=1}^{R}\left(\frac{1}{|L_r|}\sum_{l\in L_r}w_{r,l}(t)\right).
\]

一个 action interval 包含 10 个一秒仿真步，因此 transition reward 为：

\[
R_k=\frac{1}{10}\sum_{j=1}^{10}r_{k,j}.
\]

reward 没有 normalization、clipping 或 terminal-specific 分支。环境 horizon 到达时 transition 标记为 `truncated`，但正式 DQN target 仍为：

\[
y=r+\gamma\max_a Q_{target}(s',a).
\]

training reward 与 evaluation travel time 必须分开报告：前者是 lane waiting-count reward，后者是 SUMO 已完成车辆的实际行程时间均值。

## 4. Semi-offline update pseudocode

```text
initialize online model, target model, optimizer, epsilon
initialize persistent online FIFO replay(capacity=5000)

for each stage:
    set current network
    compute visible archive:
        P1C -> networks before current stage
        P1F -> all archive networks

    for each episode:
        reset SUMO
        for each decision until 3600 s:
            observe state and current phase
            choose random action while global decision step <= learning_start;
            otherwise choose argmax_a Q_online(state, phase_one_hot)

            execute action for 10 one-second steps
            reward = mean of the 10 per-second rewards
            observe next_state and next_phase
            append transition to online FIFO

            if replay size >= 64 and replay size > 1000:
                if historical sampling is enabled:
                    historical_count = round(64 * offline_ratio)
                else:
                    historical_count = 0
            online_count = 64 - historical_count

            sample online_count from online FIFO
            compute online full-Q MSE TD loss

            if historical_count > 0:
                sample historical_count from historical sampler
                compute the same full-Q MSE TD loss
                loss = (1-ratio)*online_loss + ratio*historical_loss
            else:
                loss = online_loss

            optimize once
            decay epsilon
            update target network at every 10th successful gradient update
```

`batch_size=64` 时：

| ratio | historical count | online count |
|---:|---:|---:|
| 0.25 | 16 | 48 |
| 0.50 | 32 | 32 |
| 0.75 | 48 | 16 |

两类样本不是 concat 后计算单一 batch loss，而是分别计算 branch loss 后加权。每个 ready environment decision 对应一次 gradient update；正式 audit 的 144000 transitions 与 143000 updates 差额来自 `learning_start=1000`。

## 5. Historical method definitions

| method | input | selection rule | output | unique mechanism | source code |
|---|---|---|---|---|---|
| RAND | visible historical transitions | 从 visible ordinal 中均匀随机抽取 `min(owp_capacity, candidate_count)` | 固定 working pool | random subset | `sequential/owp.py:204-212` |
| COV | transitions；前 8 个 state features、phase argmax、action、scene、behavior seed | 按 scene、seed、total/max/imbalance lane bins、phase、action 构造 coverage cell，lexicographic round-robin | 固定 working pool | coverage-balanced allocation | `sequential/owp.py:15-25, 50-78, 106-129, 213-216` |
| CQ | 与 COV 相同 | episode mean reward 与 local reward 的 z-score 组合后，在 coverage cell 内 score 降序 round-robin | 固定 working pool | coverage + quality | `sequential/owp.py:81-103, 213-218` |
| CQA | 与 CQ 相同，另使用 warm-up online state | 75% 由 global coverage/quality 选择，剩余 25% 按与 warm-up state 的 standardized Euclidean distance 选择 | 固定 working pool | coverage/quality + alignment | `sequential/owp.py:219-241` |
| DHOA | 全部 visible historical transitions | 不建立固定子集，直接从 visible view 抽样 | visible population 内动态 batch | full-history sampling | `sequential/owp.py:204-205, 242-260` |

CQ quality score：

\[
q_i=0.75z(\text{episode mean reward}_i)+0.25z(\text{transition reward}_i).
\]

CQA 的 alignment feature 是 state 前 8 个 lane-count；online target 是前 8 个 lane-count 在 warm-up observations 中的均值；距离为 candidate-population z-score 后的 Euclidean distance。配置字段为 `cqa_global_fraction=0.75`、`cqa_alignment_fraction=0.25`，实现中使用同值 literal。

`owp_capacity=5000` 是 historical working-pool capacity；它与 online replay capacity=5000、batch_size=64、offline ratio 是四个不同概念。

代码没有定义 DHOA 的正式英文全称，不能据缩写补全。

## 6. P1C/P1F semantics

可确认行为语义，但无法确认正式全称。源码规则为：

```python
visible = ordered_networks[:stage_index - 1] if mode == 'P1C' else archive_networks
```

证据：`sequential/historical_archive.py:244-281`。

以 O1 `S4 -> S1 -> S3 -> S2` 为例：

| stage | current scene | P1C 可访问 | P1F 可访问 |
|---:|---|---|---|
| 1 | S4 | 空 | S4、S1、S3、S2 |
| 2 | S1 | S4 | S4、S1、S3、S2 |
| 3 | S3 | S4、S1 | S4、S1、S3、S2 |
| 4 | S2 | S4、S1、S3 | S4、S1、S3、S2 |

P1C/P1F 先改变 visible archive 范围，再由 RAND/COV/CQ/CQA/DHOA 建立或使用 sampler；archive root 本身不随 stage 更新。每个 network 的 Q1 archive 为 episodes 1–100，四个 network 共 720000 transitions。

## 7. Sequential-training protocol

### Scene mapping

| scene | config |
|---|---|
| S1 | `sumohz1x1_config2` |
| S2 | `sumohz1x1` |
| S3 | `sumohz1x1_config4` |
| S4 | `sumohz1x1_config3` |

证据：`README.md:59-62`、正式 bundle `bundle_common.py:30-38`。

### Orders and budgets

| order | sequence |
|---|---|
| O1 | S4 → S1 → S3 → S2 |
| O2 | S2 → S3 → S1 → S4 |
| O3 | S1 → S4 → S2 → S3 |
| O4 | S3 → S2 → S4 → S1 |

正式 HA manifest：`configs/sequential/ha_sodqn_b100.yml:3-23`；每 stage 100 episodes，seeds=0–4。

### 继承关系

**Plan34：** 从 Plan 1 episode-100 parent checkpoint 导入 model、target、optimizer、epsilon、RNG 和 parent state。`clear` 在 stage 切换清空 replay；`fifo` 保留 persistent FIFO；`fifo_matched_wait` 保留 FIFO，但 readiness 使用当前 stage 新写入数量。证据：`sequential/core.py:110-125`、`sequential/agent.py:346-360, 542-579`。

**HA-SODQN/CONT/DHOA：** 从 Plan 1 episode-0 initial checkpoint 开始；model、target、optimizer、epsilon 被加载，counters 必须为零，online replay 重新初始化为空；stage 切换不清空 online replay。CONT 的身份是 `archive_mode=NONE, method=CONT, offline_ratio=0`，DHOA 等方法只额外启用 historical sampling。证据：`sequential/ha_agent.py:68-118, 134-160` 及 formal `child_run_manifest.json`。

严格 CONT–DHOA 配对的 order、seed、initialization、Stage-1 checkpoint、budget、optimizer、epsilon schedule、batch size、replay capacity 和 interaction count 相同；O3 seed1 被排除。证据：`build_ha_pair_index.py:37-70`、`build_mechanism_completion.py:475-482`。

### Frozen evaluation

- `InferenceOnlyDQN`，epsilon=0；
- 不写 replay，不进行 gradient update；
- 3600 s / 360 decisions；
- request 中 `evaluation_seed=null`；
- 相同 checkpoint/network/protocol 使用 shared evaluation cache；
- `evaluation_retries=3` 是最多三次 attempt，首个成功 attempt 被提交。

证据：`sequential/evaluator.py:86-100, 217-239, 349-445`。

## 8. Metric definitions

### Travel time

车辆进入时记录进入时间，车辆到达时记录 `current_time-entry_time`；最终对 `world.vehicles` 中已完成车辆求平均。证据：`world/world_sumo.py:540-546, 680-696`。evaluation summary 调用 `metric.real_average_travel_time()`，证据：`sequential/evaluator.py:304-318`。

### Historical travel-time degradation / forgetting

HA 代码定义：

\[
F_{abs}=TT_{final}-TT_{own\ stage\ end},
\qquad
F_{rel}=\frac{TT_{final}-TT_{own\ stage\ end}}{TT_{own\ stage\ end}}.
\]

证据：`sequential/ha_analysis.py:165-179`。reference 是该 scene 自己训练完成时的 stage-end checkpoint，不是 Plan 1 baseline。

Plan34 的 normalized travel time 则是：

\[
TT_{norm}=TT_{eval}/TT_{Plan1,scene\ episode400\ mean}.
\]

证据：`sequential/analysis.py:201-221`、`build_plan_comparison_tables.py:175-182`。两种 denominator 不同。

### Retention

HA 保存：

\[
RetentionRatio=TT_{own\ stage\ end}/TT_{final}.
\]

证据：`sequential/ha_analysis.py:170-178`。当前代码没有名为 `retention_benefit` 的正式原始字段，也没有确认一个独立的 `degradation_CONT - degradation_Semi` 指标。

### Adaptation AULC

HA 先计算 `travel_time / Plan1 scene reference`，再执行：

```python
np.trapz(series, dx=1.0) / (len(series)-1)
```

证据：`sequential/ha_analysis.py:114-120, 130-155`。

- x-axis：local episode；
- y-axis：normalized travel time；
- reference：该 scene 的 Plan 1 五个 seed 的 episode-400 final travel-time mean；
- 越低越好；
- `AULC` 与 `AUC` 在当前项目主要是字段命名差异，底层均为 trapezoidal area / horizon。

### Historical Q drift

\[
d(s_i)=\left\|Q_t(s_i,\cdot)-Q_{ref}(s_i,\cdot)\right\|_2.
\]

先对每个 state 求 8-action 向量的 Euclidean norm，再在 probe states 上求 mean。reference 在 probe scene 被引入 sequence 的 stage-end episode-100 checkpoint 固定。证据：`build_checkpoint_probe.py:193-201, 270-301`。

`cumulative_q_drift_mean_l2` 表示相对于固定 reference 的当前差异，不是跨 episode 累加；相邻 checkpoint 差异另存为 `adjacent_q_drift_mean_l2`。

### Policy disagreement

\[
D_{policy}=\frac{1}{N}\sum_i\mathbf{1}[\arg\max_aQ_t(s_i,a)\ne\arg\max_aQ_{ref}(s_i,a)].
\]

证据：`build_checkpoint_probe.py:298-301`。

### Archive / held-out probes

- archive probes：episodes 1–100；
- held-out probes：episodes 101–400；
- held-out 不进入 historical replay；
- 每 scene × behavior seed × probe type=128；
- uniform without replacement；
- 总数=5120。

证据：`build_checkpoint_probe.py:32, 114-150`、`validate_bundle.py:88-98`。

## 9. Statistical rules

- Plan34 的 paired unit 是 same `order_id × training_seed`；代码使用 paired bootstrap 和 exact sign-flip，不是 Wilcoxon。证据：`sequential/analysis.py:329-367`。
- HA mechanism bundle 对保存的 paired differences 使用 SciPy 默认 two-sided Wilcoxon signed-rank。证据：`complete_mechanism_bundle.py:313-321, 543-555`。
- HA CONT–DHOA R25 primary strict pairs=19；O3 seed1 因 Stage-1 end checkpoint digest 不一致排除。
- checkpoint-level observations 不作为独立实验 n；主配对单位仍是 order×seed。
- 缺失 checkpoint/cell 直接排除，不做 nearest-checkpoint substitution 或 interpolation。
- 当前 formal analysis pathway 未确认实际调用 multiple-comparison correction；`holm_adjust()` 存在但未确认被正式结果调用。
- Fig4 25% vs 50% 的独立正式统计脚本未找到，不能确认其精确 pair count。

## 10. Manuscript-relevant parameters

建议正文明确：SUMO/libsumo、四个 scene/config 映射、O1–O4 order、3600 s horizon、10 s decision interval、360 decisions、16 维 state、8 个 green actions、yellow insertion、精确 reward、training/evaluation 指标区分、Plan 2 Q1 episodes 1–100 archive、P1C/P1F visibility、RAND/COV/CQ/CQA/DHOA 规则、每 stage 100 episodes、seeds 0–4、batch=64、history ratios、loss aggregation、learning_start=1000、target interval=10、frozen evaluation protocol 和 `evaluation_retries=3` 的语义。

可放 Supplementary：online replay capacity=5000、OWP capacity=5000、learning rate=0.001、gamma=0.95、epsilon decay=0.995、epsilon min=0.01、grad clip=5.0、RMSprop 参数、SHA-256 RNG derivation、CQA fractions、alignment warm-up=10、evaluation seed null/fixed default。

## 11. Unresolved items

1. DHOA 正式英文全称未定义。
2. P1C/P1F 的正式英文展开未定义。
3. R3/R4 没有唯一正式目录映射。
4. 当前 checkout 存在未提交修改；E4 manifest 指向 `cdcd3d736...`，E3 指向 `e7705f77...`。论文事实应以对应 formal manifest commit 和 frozen artifact 为准。
5. `lane_waiting_count` 的注释与实际 `waiting_times` 实现有表述差异。
6. 没有确认独立 all-red phase 是否作为单独 phase 插入。
7. 没有找到 Fig4 ratio-sensitivity 的独立正式实现，不能确认其 matched-seed 数。
8. 没有确认 `retention_benefit` 的正式原始公式；当前实现保存的是 forgetting/retention 字段。
9. 没有确认 multiple-comparison correction 实际应用。
10. 本地无法证明 hz1x1 文件的完整原始公开 benchmark 出处；只能确认项目声明基于 LibSignal 并使用这些 Hangzhou 文件。
