# 半离线跨算法实验计划

## 文档状态

- 状态：进行中
- 建立日期：2026-09-09
- 最后更新：2026-09-09
- 适用范围：HA-SODQN 半离线机制的 DQN、Double DQN、Dueling Double DQN 跨算法比较
- 维护规则：每完成一项代码实现、资产准备、Pilot、正式实验或分析，都要在本文件更新状态、证据位置和变更记录。

本文是当前跨算法实验的执行依据。已有 HA-SODQN E0～E4 结果作为 Independent DQN 的历史参考；本计划不修改已经冻结的 HA-SODQN 正式协议。历史资产本身也属于本实验设计的一部分，由 A0～A5 资产链生成和验收，不把它视为无法解释的外部目录。

## 1. 研究意图

连续交通场景切换会使强化学习策略偏向当前场景，并造成旧场景性能遗忘。半离线训练的意图是在保持当前在线交互的同时，引入运行开始前固定的 Plan 1 历史轨迹，研究历史数据是否能够改善旧场景保持，以及这种收益是否依赖于 value-based 算法的 target operator 或 Q 网络结构。

半离线不是纯 Offline 训练，也不是普通 Sequential Online DQN：

- ORB 保存当前运行产生的在线 transition；
- HOA 只读引用 Plan 1 Episode 1～100 的历史 transition；
- 每次 gradient update 按预设比例混合在线样本和历史样本；
- 当前运行不得写入 HOA 或修改 Plan 1 源数据。

论文中的主要问题是：

1. 历史轨迹混合能否降低连续场景训练中的旧场景遗忘？
2. 历史利用是否会增加当前场景适应代价？
3. DQN、Double DQN 和 Dueling Double DQN 对半离线历史利用的收益是否不同？

## 2. 当前研究边界

当前正式半离线证据来自 Independent DQN + HA-SODQN。E4 已完成 4 个 Order、5 个 training seed、5 个条件，共 100 个逻辑身份。

现有结果支持的表述是：历史利用在当前 DQN 协议下通常降低旧场景 average/worst forgetting、提高 retention，同时带来一定 current-scene adaptation cost。该结论不能直接外推到 Double DQN、Dueling DQN、PPO 或其他 TSC agent。

PPO 不纳入本轮 replay-based 跨算法矩阵。PPO 的 on-policy 数据语义与当前 HOA transition 不兼容；若研究 PPO，应另建 Online PPO 或 Behavior Cloning → PPO 协议。

## 3. 冻结实验协议

### 3.0 历史资产生产链

跨算法实验使用的历史数据和初始状态按以下顺序设计。A0～A5 的计划、路径和校验摘要由 `configs/sequential/ha_cross_algorithm_assets_v1.yml` 和 `sequential/historical_assets.py` 统一描述；这些大文件留在 `data/output_data/`，不提交到 Git。

| 阶段 | 产物 | 设计约束 |
| --- | --- | --- |
| A0 | 20 个历史行为源运行 | 4 个场景 × 5 个 behavior seed；每个只运行 100 episodes；保留 episode-0 resumable checkpoint 和 Episode 1–100 trajectory |
| A1 | 20 行 source whitelist | 每个 network/behavior seed 恰好一行，固定 source run 路径和身份 |
| A2 | Plan 2 Q1 只读索引 | 每个场景一个 Q1 manifest，范围为 Episode 1–100，保留 shard/hash/count 证据 |
| A3 | behavior seed reproduction audit | 用最小复现运行判断同 seed trajectory 是否逐 transition 相同，结果决定 archive seed 排除规则 |
| A4 | HOA archive root manifest | 固定 P1C/P1F 的可见性来源、数据集 hash、transition counts 和 archive digest |
| A5 | 初始状态目录 | Independent/Double DQN 使用同一审计过的 episode-0 source；Dueling 通过等价 Q 转换生成独立目录和 hash |

资产计划只定义生成链，不伪造 trajectory 或 checkpoint。可以先生成和校验计划：

```bash
python sequential_run.py build-cross-algorithm-assets-plan \
  --config configs/sequential/ha_cross_algorithm_assets_v1.yml \
  --output data/output_data/ha_sodqn/engineering_cross_algorithm/assets_plan.json

python sequential_run.py validate-cross-algorithm-assets-plan \
  --plan data/output_data/ha_sodqn/engineering_cross_algorithm/assets_plan.json
```

A0 完成后，按既有 Plan 1 runner 生成 20 个源运行；A1–A4 依次使用 `offline_run.py prepare-plan2`、`build-ha-reproduction-audit`、`build-ha-archive`，A5 使用 `build-ha-initial-catalog` 和 `build-dueling-initial-catalog`。只有 `validate-cross-algorithm-assets-plan --require-existing`、各源 validator、Q1 validator、行为审计和 archive hash 均通过，才允许构建 12-run Pilot manifest。

A2 对本轮 100 episode 源运行必须显式传入 `--expected-episodes 100`；A5 构建初始目录时显式传入 `--source-episodes 100`。旧的 400 episode Plan 1 资产可以复用，但不再是本轮历史资产的生成要求。

A0 的单源命令模板固定为：

```bash
SUMO_HOME=/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo \
/home/dev/miniforge3/envs/colight/bin/python3.10 run.py \
  -w sumo -a dqn -n <network> \
  --prefix p1_formal_dqn_<network>_seed<seed>_100ep_cross_assets \
  --seed <seed> --episodes 100 --interface libsumo --delay_type apx
```

四个 network 和五个 seed 需要逐一完成并写入 A1 whitelist。若已有通过强验收的 Plan 1 正式 20-run 目录，直接复用其 source run 并重新生成路径绑定的 A1–A5 manifest；不重复生成同一行为资产，也不把 Pilot、失败运行或评估运行加入 archive。

### 3.1 场景和顺序

使用四个 SUMO 杭州单路口场景：

| 代称 | network |
| --- | --- |
| S1 | `sumohz1x1_config2` |
| S2 | `sumohz1x1` |
| S3 | `sumohz1x1_config4` |
| S4 | `sumohz1x1_config3` |

使用已有 O1～O4 顺序：

| Order | 场景顺序 |
| --- | --- |
| O1 | S4 → S1 → S3 → S2 |
| O2 | S2 → S3 → S1 → S4 |
| O3 | S1 → S4 → S2 → S3 |
| O4 | S3 → S2 → S4 → S1 |

### 3.2 训练预算和状态动作语义

- 正式运行：4 stages，每个 stage 100 episodes；
- 每个 stage：3600 simulator steps、action interval 10；
- 每个运行：400 episodes、144,000 decisions、143,000 gradient updates；
- batch size：64；learning start：1000；ORB capacity：5000；
- target update interval：10；
- `sumo_seed_mode=fixed_default`；training seeds：0～4；
- observation：8 维 incoming lane count + 8 维 phase one-hot，共 16 维；
- action：8 个绿灯相位动作；
- reward：沿用当前 HA-SODQN 的 incoming lane waiting count 负均值及其缩放语义。

跨算法比较必须复用同一 observation/action/reward schema、场景文件、评估节点和统计单位。

### 3.3 历史数据和可见性

- HOA 来源：Plan 1 每个场景、behavior seeds 0～4、Episode 1～100；
- 四个场景合计约 720,000 transitions；
- P1C：stage k 只能使用已经完成的旧场景历史，是主因果设置；
- P1F：从 T1 开始可见全部历史场景，包含当前和未来场景，只作非因果参考；
- alignment warm-up：带 HOA 的方法前 10 个 episodes 仍使用纯在线更新；
- R25/R50/R75：每个 64 样本 batch 中分别包含 16/32/48 个历史样本。

## 4. 跨算法矩阵

### 4.1 算法

本轮 value-based 主线包含：

1. `independent_dqn`：当前正式 HA-SODQN 算法；
2. `double_dqn`：online Q 选择动作、target Q 取值；
3. `dueling_double_dqn`：Dueling Q 网络 + Double DQN target。

算法间保持场景、数据、训练预算、历史样本比例、评估和 seed 相同。允许算法单独冻结网络参数量、optimizer state、target semantics 和合理 learning rate；不得根据正式测试结果反调超参数。

### 4.2 条件

| 条件 | 作用 |
| --- | --- |
| CONT-FIFO | 纯在线顺序训练基线 |
| P1C-DHOA-R25 | 因果历史可见性、动态 HOA 采样、历史比例 25% |
| P1C-CQ-R75 | 因果历史可见性、coverage/quality 采样、历史比例 75% |
| P1C-CQA-R75 | CQ + 当前 stage 状态分布对齐、历史比例 75% |

### 4.3 两阶段规模

工程 Pilot：

```text
3 algorithms × 4 conditions = 12 runs
O2 × training seed 0 × 12 episodes/stage
```

正式验证：

```text
3 algorithms × 4 orders × 5 training seeds × 4 conditions
= 240 logical runs
```

已有 Independent DQN E4 结果可以作为历史参考。若新算法的 manifest、evaluator、checkpoint identity 或统计协议不能通过兼容审计，则 Independent DQN 也必须按新协议重跑，不能直接拼接结果。

## 5. 评价指标和统计方法

### 5.1 主要指标

- current adaptation：normalized travel-time AULC；
- old-scene retention：average forgetting、worst forgetting；
- retention：旧场景保持率；
- traffic outcomes：travel time、queue、real delay、throughput。

### 5.2 配对和效应

每个 `Order × training seed` 是一个配对单位。报告两层效应：

- 算法内：每个 HA condition 相对于同算法 CONT-FIFO 的 paired effect；
- 算法间：比较三种算法的 paired improvement，不直接依据原始绝对指标给算法排名。

统计报告使用 paired effect、bootstrap confidence interval 和 sign-flip permutation test。机制诊断可报告 Q displacement、online/offline loss、实际历史采样比例和 replay 来源构成，但不能将关联证据表述为反馈因果证明。

## 6. 工程实现进度

| 工作项 | 状态 | 证据 |
| --- | --- | --- |
| 算法 identity：DQN/DDQN/Dueling DDQN | 已完成 | `sequential/agent.py` |
| DDQN target operator | 已完成 | `sequential/agent.py`、`sequential/ha_agent.py` |
| Dueling Q 网络 | 已完成 | `sequential/agent.py` |
| checkpoint/resume 算法隔离 | 已完成 | `sequential/agent.py`、`sequential/ha_agent.py` |
| HA runtime 传递 algorithm identity | 已完成 | `sequential/runtime.py` |
| 跨算法 Pilot 配置 | 已完成 | `configs/sequential/ha_cross_algorithm_v1.yml` |
| 12-run Pilot manifest builder/validator | 已完成 | `sequential/cross_algorithm.py`、`sequential/cli.py` |
| 240-run 正式 manifest builder/validator | 已完成 | `sequential/cross_algorithm.py`、`sequential/cli.py` |
| 跨算法结果分析接口 | 已完成 | `sequential/cross_analysis.py`、`sequential/cli.py` |
| Dueling episode-0 初始化转换器 | 已完成 | `sequential/initial_state.py` |
| 单元测试和受影响 Sequential 测试 | 已完成 | `tests/test_sequential_algorithms.py`、`tests/test_cross_algorithm_manifest.py` |
| 历史资产 A0–A5 设计与计划校验 | 已完成 | `configs/sequential/ha_cross_algorithm_assets_v1.yml`、`sequential/historical_assets.py`、`tests/test_cross_algorithm_assets.py` |
| Plan 1/HA 历史资产实际生成 | 未完成 | 需执行 A0 源运行和 A1–A4 索引/审计链 |
| Dueling initial-state catalog 实际生成 | 未完成 | 需先完成 A5 的 Independent episode-0 catalog |
| 12-run Pilot smoke/正式运行 | 未完成 | 等待历史资产和资源门禁 |
| 跨算法结果汇总和论文图表 | 未完成 | 需先完成 Pilot |
| 240-run 正式矩阵 | 未完成 | 需通过 Pilot、恢复和统计门禁 |

当前代码修改尚未提交。工作区中原有的 SUMO 转换、图片和其他实验产物改动未被清理或覆盖。

## 7. 执行入口

历史资产设计完成后，实际执行顺序为：

0. 生成并验证资产计划；按 A0–A5 完成源运行、Q1 索引、行为审计、archive 和两个初始状态目录：

```bash
python sequential_run.py build-cross-algorithm-assets-plan \
  --config configs/sequential/ha_cross_algorithm_assets_v1.yml \
  --output data/output_data/ha_sodqn/engineering_cross_algorithm/assets_plan.json
python sequential_run.py validate-cross-algorithm-assets-plan \
  --plan data/output_data/ha_sodqn/engineering_cross_algorithm/assets_plan.json \
  --require-existing
```

1. 生成 Dueling initial-state catalog：

```bash
python sequential_run.py build-dueling-initial-catalog \
  --source-catalog data/output_data/ha_sodqn/engineering_20260726/initial_state_catalog.json \
  --output-root data/output_data/ha_sodqn/engineering_20260726/dueling_initial \
  --output data/output_data/ha_sodqn/engineering_20260726/initial_state_catalog_dueling.json
```

2. 构建并验证 12-run Pilot manifest：

```bash
python sequential_run.py build-cross-algorithm-pilot \
  --config configs/sequential/ha_cross_algorithm_v1.yml \
  --output data/output_data/ha_sodqn/engineering/cross_algorithm_pilot.json

python sequential_run.py validate-cross-algorithm-pilot \
  --plan data/output_data/ha_sodqn/engineering/cross_algorithm_pilot.json
```

3. 历史资产、Pilot 和审计通过后，生成并验证 240-run 正式 manifest：

```bash
python sequential_run.py build-cross-algorithm-formal \
  --config configs/sequential/ha_cross_algorithm_v1.yml \
  --output data/output_data/ha_sodqn/engineering/cross_algorithm_formal.json

python sequential_run.py validate-cross-algorithm-formal \
  --plan data/output_data/ha_sodqn/engineering/cross_algorithm_formal.json
```

4. Pilot 运行前必须确认：

- archive root manifest、Plan 1 episode-0 checkpoint 和三个 initial-state catalog 均存在；
- 每个 catalog 的 hash 与 manifest 一致；
- Independent DQN、Double DQN、Dueling Double DQN 的 checkpoint identity 不混用；
- smoke 和单运行门禁通过；
- 磁盘、内存、SUMO 并发资源满足 12 个 Pilot 的预算。

5. Pilot 完成后，先检查每个算法的 CONT 与三个 HA 条件，再决定是否授权正式 240-run 矩阵。

6. 运行完成并通过 HA audit 后，使用同一分析入口生成算法内和算法间配对效应：

```bash
python sequential_run.py analyze-cross-algorithm \
  --manifest data/output_data/ha_sodqn/engineering/cross_algorithm_formal.json \
  --output-root data/output_data/ha_sodqn/cross_algorithm_formal \
  --whitelist data/output_data/ha_sodqn/analysis/cross_algorithm_audit.json \
  --output-dir data/output_data/ha_sodqn/analysis/cross_algorithm_formal
```

分析输出包括 `runs.csv`、`adaptation_curves.csv`、`algorithm_effects.csv`、`algorithm_pair_effects.csv` 和 `cross_algorithm_analysis.json`。前两类效应分别回答“每个算法内历史利用是否有效”和“不同算法的 paired improvement 是否不同”。

## 8. 主要风险和解释边界

- P1F 含未来场景历史，不能作为无泄漏因果结论；
- 不同算法的原始绝对 travel time 不能替代配对 improvement 比较；
- Dueling 初始化必须保持独立且可审计，不能直接加载 DQN 不兼容的 state dict；
- 当前历史轨迹没有每次 update 的完整 minibatch transition trace，机制分析不能恢复完整采样因果链；
- reward 只在同一算法/控制器内部作为训练诊断，不作为不同控制器的统一排名指标；
- Pilot 通过只表示工程链路和最小行为验证通过，不代表正式科研结论成立；
- 240-run 完成只表示实验矩阵完成，仍需经过数据完整性、配对统计和论文表述审查。

## 9. 维护记录

| 日期 | 变更 |
| --- | --- |
| 2026-09-09 | 建立本计划；冻结跨算法研究问题、协议、12-run Pilot 和 240-run 正式矩阵。 |
| 2026-09-09 | 完成算法 identity、DDQN target、Dueling 网络、checkpoint 隔离、跨算法 manifest/CLI 和 Dueling 初始化转换器。 |
| 2026-09-09 | 通过 HA 17 项、launcher 20 项、recovery 6 项、算法 4 项和跨算法/分析 2 项测试；尚未运行真实 Pilot。 |
| 2026-09-09 | 补齐 240-run 正式 manifest builder/validator，并增加正式矩阵配置校验；真实实验仍等待历史资产。 |
| 2026-09-09 | 增加跨算法分析入口，输出算法内条件效应和算法间 paired improvement；真实结果分析仍等待 Pilot/正式运行。 |
| 2026-09-09 | 将历史资产纳入正式设计，定义 A0–A5 生产链、20 个 source identity、资产计划配置、构建/校验 CLI 和 2 项资产链测试；历史源运行预算修正为每个 100 episodes，保留 400 episode 兼容入口；尚未生成真实 trajectory/checkpoint。 |
