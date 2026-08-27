# 半离线实验总结与跨算法复现实用指南

## 1. 文档用途与结论摘要

本文面向两类工作：理解当前项目中已经执行的半离线科学实验，以及在其他服务器上复用同一仓库、场景和历史数据开展 DQN、Double DQN、其他 value-based agent 或 PPO 的跨算法验证。

截至 2026-08-27，仓库内真正完成工程实现和正式实验的半离线主线是 **HA-SODQN**。它使用 SUMO 四个单路口场景顺序训练，在每个 gradient update 中混合当前在线回放和 Plan 1 静态历史档案。完成审计 `E0_E4_completion_audit_20260730.json` 的 `valid=true`，E0～E4 分别覆盖 1、16、20、50、100 个实验身份；最终 E4 为 4 个顺序 × 5 个训练 seed × 5 个条件，共 100 个身份。

当前实现只能直接运行 Independent DQN 语义：target 为 `r + gamma * max_a Q_target(s', a)`。Double DQN、Dueling DQN 和 PPO 均不能只改 YAML 后复用 HA-SODQN；它们需要新增算法适配和对应验证。推荐先实现 Double DQN，再实现 Dueling Double DQN。PPO 应作为单独的 on-policy/离线预训练研究线，不能把当前 DQN replay 方案直接套到 PPO 上。

Git 只同步代码、配置、四个 SUMO 场景及工具。Plan 1 trajectory、episode-0 checkpoint、Plan 2 数据集索引和全部正式结果不在 Git 中。新服务器仅执行 `git clone/pull` 不能直接启动 HA-SODQN，必须另行迁移历史数据资产；若仓库绝对路径不同，还必须在目标路径重建 path-bound run-list、dataset index 和 manifest。

## 2. “半离线”在本项目中的准确含义

本项目存在四类容易混淆的实验：

| 实验族 | 训练期间环境交互 | 历史数据 | 当前状态 | 正确入口 |
| --- | ---: | --- | --- | --- |
| Plan 1 Online DQN | 有 | 无 | 20 个正式运行已完成；也是历史数据和初始化来源 | `run.py` |
| Plan 2 pure Offline DQN | 0 | Plan 1 冻结 trajectory | Batch-DQN/CQL-DQN 共 160 个正式运行已完成 | `offline_run.py` |
| Plan 3/4 Sequential DQN | 有 | 当前顺序运行生成的 replay | clear/FIFO/fifo_matched_wait；用于分析切换和遗忘 | `sequential_run.py` |
| HA-SODQN 半离线 | 有 | Plan 1 Episode 1–100 静态档案 | E0～E4 正式实验已完成 | `sequential_run.py` 的 HA 子命令 |

此外，`CS-HR` 原型位于 `sequential/hybrid.py` 和 `sequential/hybrid_agent.py`。它冻结当前顺序运行中已经完成的 stage 数据作为历史池；HA-SODQN 的 HOA 则只来自运行开始前就固定的 Plan 1 数据。两者的数据因果边界不同，不应合并成同一方法。

## 3. 场景、顺序和状态动作语义

四个正式场景均为 SUMO 单路口、8 个绿色相位动作：

| 代称 | network | SUMO 输入目录 |
| --- | --- | --- |
| S1 | `sumohz1x1_config2` | `data/raw_data/hangzhou_1x1_qc-yn_18041608_1h/` |
| S2 | `sumohz1x1` | `data/raw_data/hangzhou_1x1_bc-tyc_18041610_1h/` |
| S3 | `sumohz1x1_config4` | `data/raw_data/hangzhou_1x1_sb-sx_18041607_1h/` |
| S4 | `sumohz1x1_config3` | `data/raw_data/hangzhou_1x1_kn-hz_18041608_1h/` |

正式顺序由 `configs/sequential/ha_sodqn_b100.yml` 冻结：

| Order | 场景顺序 |
| --- | --- |
| O1 | S4 → S1 → S3 → S2 |
| O2 | S2 → S3 → S1 → S4 |
| O3 | S1 → S4 → S2 → S3 |
| O4 | S3 → S2 → S4 → S1 |

每个 observation 为 16 维：前 8 维是进入车道的 lane count，后 8 维是当前绿灯 action index 的 one-hot。动作空间为 8。reward 来自进入车道等待车辆数的负均值，并使用现有缩放语义。跨算法实验必须复用相同的 feature/action/reward schema 和哈希门禁，不能让某个算法使用不同观测或 reward 后仍称为同协议对比。

## 4. HA-SODQN 科学实验设计

### 4.1 数据与初始化

- HOA（Historical Offline Archive）只读引用 Plan 1 每个场景、behavior seeds 0–4、Episodes 1–100 的 trajectory。
- 每个场景有 180,000 transitions，四场景共 720,000 transitions。
- 每条顺序运行从该 Order 首场景、同 training seed 的 Plan 1 episode-0 resumable checkpoint 初始化。
- episode-0 checkpoint 包含 online/target model、optimizer、epsilon、训练计数器和 RNG 状态；ORB 初始为空。
- behavior seed 重复性审计没有证明同 seed trajectory 可逐 transition 复现，因此正式 archive 未排除与 training seed 相同的 behavior seed。

### 4.2 顺序训练预算

每个运行包含 T1～T4 四个 stage，每个 stage 100 episodes：

```yaml
stage_episodes: [100, 100, 100, 100]
steps: 3600
action_interval: 10
learning_start: 1000
buffer_size: 5000
batch_size: 64
update_model_rate: 1
target_update_interval: 10
learning_rate: 0.001
gamma: 0.95
epsilon_decay: 0.995
epsilon_min: 0.01
grad_clip: 5.0
sumo_seed_mode: fixed_default
training_seeds: [0, 1, 2, 3, 4]
```

一次完整运行共 400 episodes、144,000 decisions、143,000 gradient updates、14,300 target updates，并执行 409 个冻结评估 cell。加入历史数据不得增加 gradient update 数。

### 4.3 ORB、HOA、OWP 和可见性

- ORB（Online Replay Buffer）：容量 5,000，保存当前半离线运行产生的 transition，跨 stage 持续 FIFO。
- HOA：Plan 1 静态只读历史档案，当前运行不能写入。
- DHOA：直接从当前可见 HOA 动态采样，不建立固定工作池。
- OWP（Offline Working Pool）：RAND/COV/CQ/CQA 每个 stage 构建一次、stage 内冻结，默认容量 5,000。
- `P1C`：stage k 只能使用该顺序中 stage 1..k-1 的历史场景。T1 的 HOA 为空，必须退化为纯 Online。这是主因果设置。
- `P1F`：从 T1 起可见四个场景全部 Plan 1 历史，包含当前和未来场景，只能作为非因果 full-history reference。

### 4.4 五种历史采样方法

| 方法 | 定义 |
| --- | --- |
| DHOA | 在所有当前可见历史 transition 上动态抽样 |
| RAND | 无放回、确定性随机选择固定 OWP |
| COV | 按 scene/seed 平衡和 state/phase/action coverage 分层构建 OWP |
| CQ | 保持 coverage 配额，在 cell 内优先选择质量分数较高的样本 |
| CQA | 75% CQ 全局保持部分 + 25% 与当前 stage 前 10 episodes 状态分布对齐的部分 |

所有带 HOA 的方法都使用 10 episodes alignment warm-up，warm-up 内仍正常纯 Online 更新。离线比例 R25/R50/R75 分别对应每个 64 样本 batch 的 16/32/48 个离线样本；总损失为：

```text
loss = (1 - rho) * loss_online + rho * loss_offline
```

### 4.5 正式 E0～E4 矩阵

| 阶段 | 目的 | 规模和选择 |
| --- | --- | --- |
| E0 | 完整链路门禁 | O2/seed0/P1C-DHOA-R50，1 个身份 |
| E1 | 五方法 × 三 ratio 筛选 | O2/seed0，15 个 P1C + CONT，共 16 |
| E2 | 候选多 seed | O2/seeds 0–4；CONT、DHOA-R25、CQ-R75、CQA-R75，共 20 |
| E3 | P1C/P1F 可见性对照 | O2/seeds 0–4，5 methods × 2 archive modes × R50，共 50 |
| E4 | 跨顺序正式验证 | O1–O4 × seeds 0–4 × 5 条件，共 100 |

E4 的五个条件为：

1. CONT-FIFO；
2. P1C-DHOA-R25；
3. P1C-CQ-R75；
4. P1C-CQA-R75；
5. P1F-CQA-R50，仅作为非因果参考。

E4 复用了 O2 的 25 个既有身份，新运行 75 个身份。E0～E4 按阶段计数合计 187，但存在阶段间身份复用，不能将 187 解释为 187 条独立训练轨迹。

## 5. 当前结果和可支持的科研表述

### 5.1 完整性结论

- E0～E4 completion audit 的所有 stage checks 通过，E4 恰好包含 100 个预注册身份。
- 纳入统计的运行具有完整物理产物；失败前驱被保留但从有效统计中排除。
- P1C 离线样本来源检查未发现当前/未来场景泄漏。
- 每个正式运行的 update budget 一致：144,000 decisions、143,000 gradient updates、14,300 target updates。
- 固定 OWP 方法在 stage 内冻结；DHOA 明确记录为动态池。

### 5.2 E4 主要量化证据

`paired_effects.csv` 中 `effect_mean = CONT AULC - HA AULC`。normalized travel-time AULC 越低越好，因此下表的负值表示 HA 的当前场景适应 AULC 更高，即存在适应代价，而不是改善：

| 条件 | effect mean | 95% CI | paired units | sign-flip p |
| --- | ---: | --- | ---: | ---: |
| P1C-CQ-R75 | -0.01595 | [-0.01825, -0.01344] | 20 | 1.91e-6 |
| P1C-CQA-R75 | -0.01367 | [-0.01566, -0.01140] | 20 | 3.81e-6 |
| P1C-DHOA-R25 | -0.00229 | [-0.00432, -0.0000013] | 20 | 0.06085 |
| P1F-CQA-R50 | -0.00745 | [-0.01397, -0.00053] | 20 | 0.05175 |

由 E4 `runs.csv` 按相同 Order/seed 与 CONT 配对计算，四个历史条件的 mean average-forgetting 均降低约 23.24～24.43，mean worst-forgetting 均降低约 33.09～35.18，mean retention 提高约 0.092～0.107；与此同时，mean normalized travel-time AULC 增加约 0.0023～0.0160。当前证据支持的核心结论是：在该 DQN 协议下，历史利用在配对均值上大幅改善旧场景保持，但总体存在小幅当前场景适应代价，且不同方法/ratio 的权衡不同。上述保持指标是描述性配对均值，不额外声称本文未给出的显著性检验。

补充机制分析在 19 个严格同 Order/seed/初始化/Stage-1-checkpoint 的 DHOA-CONT 配对上支持以下描述：DHOA 与更小的旧场景瞬时性能下降和更小的 Q displacement 同时出现。O3 seed1 因 Stage-1 checkpoint digest 不同被排除。该证据是受控关联和时间关系，不足以证明反馈因果机制、普遍降低 Bellman residual 或让所有在线 trajectory 收敛。

### 5.3 必须保留的限制

- P1F 从 T1 看到当前和未来场景，不得解释为无泄漏因果实验，也不是有保证的经验上界。
- 正式运行设置 `trace_replay_samples=false`，无法恢复每次 update 的完整 minibatch transition 身份。
- 现有轨迹缺少用于 sampled state 的完整 phase-inclusive 原始样本，部分 requested checkpoint/evaluation cell 缺失；分析没有插值替代。
- 现有结果证明的是当前 Independent DQN 实现上的 HA 机制表现，不能自动外推到 Double DQN、PPO 或其他 agent。

## 6. 当前算法支持边界

| 算法 | 仓库已有普通 Online agent | 可直接运行 HA-SODQN | 说明 |
| --- | --- | --- | --- |
| Independent DQN | 是 | 是 | 当前正式半离线算法 |
| Double DQN | 否 | 否 | 当前 target 使用 target network 上的直接 max，需要新增 online argmax + target gather |
| Dueling DQN/DDQN | 否 | 否 | 需要新增网络结构、checkpoint identity 和 evaluator 构建逻辑 |
| Batch-DQN/CQL-DQN | 是，仅 Plan 2 pure Offline | 否 | `offline_run.py` 的算法和 trainer 与 sequential/HA 隔离 |
| PPO | 是，普通 Online | 否 | 当前 PPO 不是 HA runtime/checkpoint/evaluator 支持对象；算法也是 on-policy |
| FRAP/PressLight/CoLight | 是 | 否 | 虽含 value update，但状态/网络结构和多智能体语义未接入 HA runtime |

因此，“跨算法验证”当前是下一阶段开发目标，不是一个已经存在、只需换 `agent:` 字段的配置选项。

## 7. 推荐的跨算法实验设计

### 7.1 value-based 主验证线

第一批推荐三种算法：DQN、Double DQN、Dueling Double DQN。三者共同使用完全相同的场景顺序、history archive、online interaction budget、batch 配额、evaluation 和 seed；只改变 Q 网络结构或 target operator。

建议采用两阶段矩阵：

1. 工程 pilot：O2、seed0、每 stage 12 episodes，覆盖每个算法的 CONT、P1C-DHOA-R25、P1C-CQ-R75、P1C-CQA-R75，共 `3 × 4 = 12` runs。
2. 正式验证：pilot 和恢复等价门禁通过后，O1–O4、seeds 0–4、相同四条件，共 `3 × 4 × 5 × 4 = 240` 个身份。当前 DQN 已有 E4 结果可以作为历史参考，但只有当新 manifest 的代码、evaluation 和 identity 兼容审计通过时才允许复用；否则同协议重跑。

算法比较至少报告两个效应层：

- 同一算法内：HA condition 相对该算法自己的 CONT；
- 算法间：比较上述 paired improvement，而不是直接比较两个算法的原始绝对指标。

必须给 algorithm 一个独立 manifest identity 字段。logical ID 至少包含 `Algorithm/Archive/Method/Ratio/Order/Seed`，避免不同算法覆盖同名运行。

### 7.2 公平性冻结项

以下字段必须跨算法相同：

- SUMO network 文件和 `fixed_default` seed mode；
- O1–O4、seeds 0–4、400 episodes/run、144,000 decisions；
- observation/action/reward schema；
- ORB capacity 5,000、batch 64、learning start 1,000；
- P1C visibility 和行为 seed 规则；
- DHOA/OWP transition ID 序列的 RNG 独立性；
- 每 decision 最多一次 update，且总 update budget 相同；
- lower-triangle evaluator、指标、bootstrap/sign-flip 单位。

需要按算法单独冻结而不强求数值相同的字段包括网络参数量、optimizer state、目标更新语义和合理的 learning rate。若调参，应使用独立 tuning 场景/seed 或预注册搜索，不能根据正式 O1–O4 测试结果回调超参数。

### 7.3 Double DQN 的最小工程改动边界

Double DQN target 应改为：

```text
next_action = argmax_a Q_online(next_state, a)
target = reward + gamma * Q_target(next_state, next_action)
```

需要同步覆盖 online/offline 两支 loss、checkpoint/recovery、inference evaluator、manifest `algorithm_id/target_operator`、数值审计和单元测试。验收至少包括：构造 online/target argmax 不一致的固定 batch，证明 DQN 与 DDQN target 产生预期差异；resume 与不中断运行最终状态等价；不同算法不共享不兼容的 episode-0 checkpoint。

### 7.4 PPO 的独立研究线

当前 archive 没有冻结行为策略的 action log-prob，PPO 也依赖 on-policy rollout。直接把 HOA transition 放进 PPO minibatch 会破坏 clipped policy objective 的数据语义。

可执行的两个选择是：

- Online PPO 顺序基线：只做 CONT/clear 等在线顺序实验，用于比较算法的 continual adaptation，但不声称验证 HA replay。
- Behavior Cloning → PPO：先用可见 HOA 的 `(state, action)` 做监督预训练，再在每个 stage 用纯 on-policy PPO 微调。这可以称为半离线迁移，但机制、预算和研究问题不同，必须单独注册协议，不能与 DQN 的 R25/R50/R75 replay ratio 当作同一变量。

若需要真正的 off-policy actor-critic 历史利用，应另行选择 AWAC/IQL/CQL actor-critic、V-trace 等方法，并补充行为策略概率或明确估计假设。建议在 value-based 跨算法验证完成后再设计这一条线。

## 8. 新服务器迁移清单

### 8.1 Git 同步的内容

```bash
git clone https://github.com/ZShining219/simOfflineTSC.git
cd simOfflineTSC
git switch codex/milestone0-experiment-infrastructure
git pull --ff-only
git rev-parse HEAD
```

正式实验应记录 branch、commit SHA 和 remote。不要使用未 push 的本地 commit 构建正式 manifest。四个场景的 `.sumocfg/.net.xml/.rou.xml` 等文件已受 Git 跟踪；可用以下命令确认：

```bash
git status --short
git ls-files data/raw_data/hangzhou_1x1_bc-tyc_18041610_1h
git ls-files data/raw_data/hangzhou_1x1_qc-yn_18041608_1h
git ls-files data/raw_data/hangzhou_1x1_sb-sx_18041607_1h
git ls-files data/raw_data/hangzhou_1x1_kn-hz_18041608_1h
```

### 8.2 环境

仓库协作基线是名为 `colight` 的 Conda 环境和 Python 3.9。原项目文档给出的基线为 PyTorch 1.11.0；当前服务器的既有 `colight` 环境实际为 Python 3.10.18、PyTorch 1.13.1+cu116、Gym 0.26.2、NumPy 1.26.4。两者不一致，因此新正式跨算法实验应先选定并冻结一套环境，不要把“能 import”当成与历史正式运行完全等价。

基础安装：

```bash
conda activate colight
python -m pip install -r requirements.txt
# 只有运行 Plan 2 pure Offline/d3rlpy 时才需要：
python -m pip install --no-deps -r requirements-offline.txt
```

另行安装与 CUDA 匹配的 PyTorch 和 SUMO。验证：

```bash
python --version
python -c "import torch, gym, numpy, pandas, yaml; print(torch.__version__, gym.__version__, numpy.__version__)"
sumo --version
python -c "import libsumo, traci, sumolib; print('SUMO Python API OK')"
python sequential_run.py --help
```

应把 `conda list --explicit` 或 environment export、SUMO version、GPU/driver 信息放入实验外部资产包或 run metadata，不把机器相关环境文件未经审核直接覆盖仓库配置。

### 8.3 Git 不同步、必须另行迁移的资产

至少需要：

1. Plan 1 正式 20-run 白名单；
2. 白名单对应的 20 个 source run，至少包含 run/config/trajectory 和 episode-0 resumable checkpoint；
3. Plan 2 schema-v2 数据集目录，包括 root manifest、Q1 manifests、`source_runs.csv`、`source_shards.jsonl`；目标绝对路径不同时也可以只迁移 Plan 1 source runs 后在目标机重建；
4. behavior seed reproduction audit 结果，或在目标服务器重新执行该 audit；
5. 如果要复核旧结论，再额外迁移 HA formal outputs、validation 和 analysis；新实验本身不依赖旧 HA 结果目录。

默认目标路径应使用配置中的 `data/output_data/...`：

```text
data/output_data/analysis/plan1/p1_formal_20_trajectory_whitelist_20260722.csv
data/output_data/tsc/sumo_dqn/<network>/<plan1-run>/
data/output_data/offline_datasets/plan2/plan2_formal_v2_a3a66e8_20260723/
data/output_data/ha_sodqn/engineering_<date>/
```

当前工作站的大型结果曾被重定位到根目录 `output_data/`，但旧 manifest 内记录的仍是 `/projects/simOfflineTSC/data/output_data/` 绝对路径。若目标 checkout 也固定为 `/projects/simOfflineTSC`，可迁移完整源资产并先用 validator 核对；若目标绝对路径不同，不要直接复制旧 archive/initial catalog 后手工搜索替换，应从 Plan 1 source runs 重建 run-list、Plan 2 index、archive 和 initial catalog，以便重新计算绝对路径和哈希。

推荐用 `rsync`、对象存储或共享文件系统迁移数据，不通过 Git 提交 trajectory/checkpoint。迁移前后都生成 SHA256 清单并核对。示意命令中的主机和目录必须替换为实际值：

```bash
rsync -a --info=progress2 SOURCE_HOST:/path/to/data/output_data/tsc/ data/output_data/tsc/
rsync -a --info=progress2 SOURCE_HOST:/path/to/data/output_data/offline_datasets/ data/output_data/offline_datasets/
rsync -a --info=progress2 SOURCE_HOST:/path/to/whitelist.csv data/output_data/analysis/plan1/
```

### 8.4 在目标服务器重建路径绑定资产

以下流程同时适用于不同 checkout 路径和新的跨算法数据身份。先把已迁移的旧白名单改写为目标服务器绝对路径；CSV 的 network/seed 列必须保持不变：

```bash
mkdir -p data/output_data/analysis/plan1
awk -v root="$(pwd)" 'BEGIN{FS=OFS=","} NR==1{print; next} {sub("^/projects/simOfflineTSC", root, $1); print}' \
  data/output_data/analysis/plan1/p1_formal_20_trajectory_whitelist_20260722.csv \
  > data/output_data/analysis/plan1/p1_formal_20_trajectory_whitelist_target.csv

python offline_run.py prepare-plan2 \
  --run-list data/output_data/analysis/plan1/p1_formal_20_trajectory_whitelist_target.csv \
  --dataset-id plan2_cross_algorithm_target \
  --output-root data/output_data/offline_datasets/plan2 \
  --source-root "$(pwd)/data/output_data/tsc/sumo_dqn" \
  --source-root-id plan1_formal_target

for network in sumohz1x1_config2 sumohz1x1 sumohz1x1_config4 sumohz1x1_config3; do
  python offline_run.py validate-dataset \
    --manifest "data/output_data/offline_datasets/plan2/plan2_cross_algorithm_target/datasets/${network}/Q1/manifest.json"
done
```

如果旧白名单前缀不是 `/projects/simOfflineTSC`，应将 `awk` 中的旧前缀替换为 CSV 实际前缀；生成后先执行 `head` 和 `wc -l` 核对，共应有表头加 20 行。`prepare-plan2` 会验证 4 networks × 5 behavior seeds、400 episodes、shard SHA 和语义兼容性。

接着建立目标服务器本地配置。它放在已忽略的输出目录，不修改冻结的 `configs/sequential/ha_sodqn_b100.yml`：

```bash
mkdir -p data/output_data/ha_sodqn/engineering_target
cp configs/sequential/ha_sodqn_b100.yml \
  data/output_data/ha_sodqn/engineering_target/ha_sodqn_target.yml
```

在 `ha_sodqn_target.yml` 中只修改以下三个路径，其他冻结实验参数保持不变：

```yaml
archive:
  dataset_root: data/output_data/offline_datasets/plan2/plan2_cross_algorithm_target
  root_manifest: data/output_data/ha_sodqn/engineering_target/archive_root_manifest.json
  initial_state_catalog: data/output_data/ha_sodqn/engineering_target/initial_state_catalog.json
```

然后依次重建 initial catalog、behavior audit 和 archive manifest：

```bash
conda activate colight

python sequential_run.py build-ha-initial-catalog \
  --whitelist data/output_data/analysis/plan1/p1_formal_20_trajectory_whitelist_target.csv \
  --config data/output_data/ha_sodqn/engineering_target/ha_sodqn_target.yml \
  --output data/output_data/ha_sodqn/engineering_target/initial_state_catalog.json

python sequential_run.py build-ha-reproduction-audit \
  --config data/output_data/ha_sodqn/engineering_target/ha_sodqn_target.yml \
  --order O2 \
  --training-seed 0 \
  --output data/output_data/ha_sodqn/engineering_target/behavior_seed_audit_manifest.json

python sequential_run.py launch \
  --manifest data/output_data/ha_sodqn/engineering_target/behavior_seed_audit_manifest.json \
  --output-root data/output_data/ha_sodqn/behavior_seed_audit_target \
  --max-child 4

python sequential_run.py compare-ha-reproduction-audit \
  --source-run data/output_data/tsc/sumo_dqn/sumohz1x1/p1_formal_dqn_sumohz1x1_seed0_400ep_20260722_r2 \
  --attempt-dir data/output_data/ha_sodqn/behavior_seed_audit_target/CONT-FIFO-O2-SD0/attempts/attempt_1 \
  --episode-count 4 \
  --output data/output_data/ha_sodqn/engineering_target/behavior_seed_audit.json

python sequential_run.py build-ha-archive \
  --dataset-root data/output_data/offline_datasets/plan2/plan2_cross_algorithm_target \
  --audit-report data/output_data/ha_sodqn/engineering_target/behavior_seed_audit.json \
  --output data/output_data/ha_sodqn/engineering_target/archive_root_manifest.json
```

示例中的 Plan 1 S2/seed0 source run 名称来自当前正式白名单；若迁移包使用不同但经审计有效的重跑目录，应以目标服务器白名单中的对应行替换。audit manifest 的 mode 为 `audit` 且自身已授权，不需要 `--authorize-formal`。

不要伪造一个只有布尔值的 audit 文件。`build-ha-archive` 会校验 audit kind、确定性结果和 SHA256；正式 protocol 还需要保留该证据链。

### 8.5 先 smoke，再 formal

当前 DQN 线路的参考命令：

```bash
python sequential_run.py build-ha-plan \
  --stage smoke \
  --config data/output_data/ha_sodqn/engineering_target/ha_sodqn_target.yml \
  --output data/output_data/ha_sodqn/engineering_target/smoke_manifest.json

python sequential_run.py validate-ha-plan \
  --plan data/output_data/ha_sodqn/engineering_target/smoke_manifest.json

python sequential_run.py launch \
  --manifest data/output_data/ha_sodqn/engineering_target/smoke_manifest.json \
  --output-root data/output_data/ha_sodqn/smoke_target \
  --max-child 4

python sequential_run.py status \
  --manifest data/output_data/ha_sodqn/engineering_target/smoke_manifest.json \
  --output-root data/output_data/ha_sodqn/smoke_target

python sequential_run.py audit-ha \
  --plan data/output_data/ha_sodqn/engineering_target/smoke_manifest.json \
  --output-root data/output_data/ha_sodqn/smoke_target \
  --output data/output_data/ha_sodqn/engineering_target/smoke_audit.json
```

正式 manifest 含冻结 git commit。`validate-ha-plan` 会拒绝当前 checkout 与 manifest commit 不一致的情况。正式阶段 `launch` 还需要显式 `--authorize-formal`。不要为了越过门禁手改 manifest、completed flag 或 checkpoint。

## 9. 跨算法实现后的验收清单

在其他服务器开始正式矩阵前，逐项满足：

- [ ] 代码 commit 已 push，目标服务器 checkout 与 manifest SHA 一致；
- [ ] 四个 SUMO 场景文件由 Git 提供且工作树干净；
- [ ] Python/PyTorch/SUMO/CUDA 版本已记录；
- [ ] Plan 1 source runs、Q1 数据集和目标服务器重建 manifest 的 SHA 校验通过；
- [ ] 每种算法有独立 `algorithm_id`、checkpoint schema 和 evaluator 构造；
- [ ] DQN/DDQN target operator 的固定 batch 单元测试通过；
- [ ] action RNG、online replay RNG、HOA/OWP RNG、evaluation RNG 相互隔离；
- [ ] P1C T1 offline count 为 0，T2–T4 无当前/未来场景样本；
- [ ] 各算法 CONT 与 HA condition 的 decisions/update 数完全一致；
- [ ] smoke 覆盖 CONT、DHOA、固定 OWP、resume 和 lower-triangle evaluation；
- [ ] 真实 SUMO 完整 100×4 单 run 通过 validator 后才启动正式矩阵；
- [ ] 分析以同 Order/seed 配对，报告算法内 HA improvement 和算法间 improvement difference；
- [ ] PPO 若纳入，使用独立协议和解释，不冒充 DQN replay 的等价替换。

## 10. 权威代码、配置和证据入口

- 半离线冻结配置：`configs/sequential/ha_sodqn_b100.yml`
- 总体设计与 E0～E4 预注册：`docs/plan0726.md`
- CLI：`sequential/cli.py`、`sequential_run.py`
- HA agent 和 mixed loss：`sequential/ha_agent.py`
- 静态历史档案与 P1C/P1F：`sequential/historical_archive.py`
- OWP：`sequential/owp.py`
- manifest：`sequential/ha_manifest.py`
- 顺序 runtime/checkpoint/launcher：`sequential/runtime.py`、`sequential/checkpoint.py`、`sequential/launcher.py`
- validator/analysis：`sequential/validation.py`、`sequential/ha_analysis.py`
- 当前 DQN target 证据：`sequential/agent.py` 和 `sequential/ha_agent.py` 中的 target 计算；均为 plain DQN max-target，而不是 Double DQN。
- 本机正式完成审计：`output_data/ha_sodqn/engineering_20260726/E0_E4_completion_audit_20260730.json`（不在 Git）
- 本机 E4 效应表：`output_data/ha_sodqn/analysis_e7705f7/E4/paired_effects.csv`（不在 Git）
- 补充机制证据与限制：`output_data/analysis/ANALYSIS_BUNDLE_V2_MECHANISM_COMPLETION/99_REPORT/`（不在 Git）
- E4 manifest 冻结训练 commit：`cdcd3d7`；恢复审计使用后续 commit `664ac2a`；最终 E0～E4 completion audit 记录的代码 head 为 `125f9d7`。三者均位于当前分支历史中，不能用当前 HEAD 字符串替代旧运行实际记录。

本文记录的是当前已验证状态和下一阶段跨算法建议。新增算法实现完成后，应更新“当前算法支持边界”、实际命令、冻结环境、正式 manifest identity 和验证证据；不要让建议矩阵长期被误读成已执行实验。
