# Plan5：跨算法机制定位与交通上下文干预实验

## 0. 文档状态与适用范围

本文是 `Plan5 Cross-Algorithm Mechanism Localization` 的唯一正式设计与执行规范，

执行优先级如下：

1. 用户在执行过程中的最新明确指令；
2. 仓库根目录 `AGENTS.md`；
3. 本文；
4. `docs/semi_offline_cross_algorithm_reproduction.md` 及其他历史材料。

低优先级材料与本文冲突时，以本文为准。执行者不得根据实验结果反向修改本文冻结的科学语义。本文只授权第一阶段的 DDQN、PPO 和 CTXDDQN；任何条件扩展都保持 `HOLD`。

本文形成时的仓库基线为：

```text
machine = linux4090
branch = codex/milestone0-experiment-infrastructure
baseline HEAD = ebfe04a
remote policy = no push
```

正式运行必须记录实际 source commit，不能把上述基线 SHA 当作实现完成后的 source commit。

---

## 1. 科学任务与非目标

既有 DQN 数据继续承担：

- 独立单场景学习；
- 四阶段顺序学习；
- 多 order、多 training seed；
- replay 策略和 historical replay；
- Bellman-target conflict；
- checkpoint probe；
- historical policy displacement；
- transient failure；
- HA intervention。

Plan5 只回答两个新问题：

### 问题 A：算法边界

现有持续适应失效主要属于：

1. vanilla DQN；
2. 更广泛的 value-based RL；
3. 更一般的 shared neural controller continual adaptation。

### 问题 B：信息边界

当前瞬时 observation 缺少 causal traffic-demand context，是否是跨需求干扰的重要来源。

Plan5 不是算法排行榜，不比较谁的绝对交通性能最好，也不复制 Plan1～Plan4 或 HA-SODQN 的完整矩阵。

---

## 2. 实验身份、目录与覆盖保护

实验族：

```text
Plan5
Plan5 Cross-Algorithm Mechanism Localization
```

正式配置入口：

```text
configs/sequential/plan5_cross_algorithm_b100.yml
```

正式输出根目录：

```text
data/output_data/cross_algorithm/plan5_b100/
```

第一阶段算法 ID：

```text
DQN       canonical existing reference; no new training
DDQN      Double DQN
PPO       discrete-action PPO
CTXDDQN   Context-Augmented Double DQN
```

以下 ID 仅保留为 Phase2 后候选，不得在本轮启动：

```text
DDDQN
HISTDQN
DDQN_DHOA
```

任何 Plan5 路径不得覆盖 `Plan1`、`Plan2`、`Plan3`、`Plan4`、`HA` 或既有 runtime/manifest。重复 logical run ID、已有完成产物、配置 hash 不同但目录相同，均必须 fail closed；禁止静默覆盖或续写。

正式 run ID 模式固定为：

```text
P5-ANCHOR-DDQN-S3-SD0
P5-ANCHOR-CTXDDQN-S3-SD0
P5-ANCHOR-PPO-S3-SD0
P5-DDQN-H34-SD0
P5-CTXDDQN-L32-SD0
P5-PPO-H43-SD0
```

一个 logical run 的唯一身份至少包含：

```text
plan_id, algorithm_id, run_type, scene/transition_id,
training_seed, config_sha256, source_commit
```

失败后的 retry 使用同一 logical run ID 和递增 attempt ID，不创建一个伪装成新 seed 的 run。

---

## 3. DQN canonical reference 边界

Plan5 不重新训练 DQN。

优先只读使用：

```text
Plan1 episode-100 specialist checkpoints
Plan3 b100 Clear Stage2 results
existing mechanism/checkpoint-probe derived assets
```

执行者必须先生成：

```text
canonical_reference_manifest.json
```

其中每项资产只能标记为：

```text
available = 本机存在，且身份、schema 和 SHA 校验通过
external  = 已知存在于外部正式资产，但本机没有实体
missing   = 没有可验证的正式资产
```

当前 `linux4090` 的 Plan1 白名单入口为：

```text
data/output_data/analysis/plan1/p1_formal_20_trajectory_whitelist_20260722.csv
```

episode-100 checkpoint 候选中存在每场景重复 seed0 的可能性，只能按该 canonical 20-run whitelist 解析，不能凭文件名选取 24 个候选中的任意文件。当前本机没有可假定存在的完整 Plan3/4 结果包；缺失时标记 `external` 或 `missing`，不得重跑 DQN 填洞，也不得使用邻近 checkpoint 替代。

现有 Plan3 transition 最多与 `L23/L32` 局部对齐，不能为新的 `H34/H43` 提供 matched DQN effect。最终跨算法表必须显式显示 DQN 数据可用性，不能把不可比的历史数据补成完整 Plan5 DQN 矩阵。

---

## 4. 固定交通环境

场景映射：

| Scene | Network | 角色 |
|---|---|---|
| S1 | `sumohz1x1_config2` | canonical scene |
| S2 | `sumohz1x1` | low-conflict pair source/target |
| S3 | `sumohz1x1_config4` | same-start anchor |
| S4 | `sumohz1x1_config3` | high-conflict pair source/target |

固定时间协议：

```text
simulation duration = 3600 s
SUMO step = 1 s
decision interval = 10 s
decisions per episode = 360
training episodes per anchor/continuation = 100
```

固定共同语义：

- 同一 topology/control semantics；
- 8 incoming lanes；
- 8 discrete green actions；
- 现有 dynamic clearance；
- 同一 vehicle type、route、phase、lane mapping 和 simulation duration；
- 记录 S2 已知 junction-type 差异，但不修改场景。

基础 observation：

```text
raw16 = 8 lane counts + 8 current-phase one-hot
dtype = float32
normalization = none
clipping = none
```

action：

```text
8 discrete green actions
```

reward：

```text
-mean(incoming-lane waiting count) * 12
```

PPO 不得启用旧实现中的 reward clipping/normalization，也不得改为连续动作。

---

## 5. 随机性和正式环境

正式 training seeds：

```text
0, 1, 2, 3, 4
```

training seed 控制 Python、NumPy、PyTorch、网络初始化、探索/动作采样、replay/rollout minibatch sampling 以及算法 RNG。每条 run 必须保存完整 RNG state 和恢复元数据。

SUMO 保持既有固定默认模式：

```text
SUMO --seed not present
SUMO --random false
```

每条 run 必须保存完整 resolved SUMO command，并验证其中没有 `--seed` 和 `--random`。Plan5 不引入 SUMO-seed factorial experiment。

Plan5 正式环境冻结为：

```text
Conda environment name = colight
Python = 3.10.18
PyTorch = 1.13.1+cu116
Gym = 0.26.2
NumPy = 1.26.4
PFRL = 0.4.0
SUMO = 1.27.1
formal training/evaluation device = CPU
```

这是 Plan5 的实验环境锁，覆盖 `AGENTS.md` 中用于原始项目匹配的 Python 3.9 一般建议，但不改变仓库对其他实验的环境约定。Phase0 必须输出可复建的环境清单、package/version probe、SUMO Python API probe 和 SHA；仅 import 成功不构成环境 gate 通过。

---

## 6. 固定 transition 与推断边界

Plan5 不使用 O1～O4 作为主要身份。四个 transition 在正式结果产生前冻结为：

| ID | Stage1 | Stage2 | 角色 |
|---|---|---|---|
| H34 | S3 | S4 | representative high-conflict |
| H43 | S4 | S3 | high-conflict reverse |
| L23 | S2 | S3 | representative low-conflict |
| L32 | S3 | S2 | low-conflict reverse |

high/low 分组来自既有 DQN mechanism evidence，不得根据 Plan5 结果重选。证据位置：

```text
data/output_data/analysis/plan1/
  s1_s4_adaptation_diagnostics_20260723/reports/final_diagnostic_report.md
```

### 6.1 Primary same-start contrast

每个算法内的核心比较是：

```text
H34 vs L32
```

对同一 `algorithm × training_seed`，两条 continuation 必须引用完全相同的 S3 Stage1 checkpoint：

```text
checkpoint path identical
checkpoint file SHA256 identical
model state identical
optimizer state identical
algorithm state identical
RNG state identical
```

manifest builder 和 validator 必须逐字段断言；任一不同均使 paired unit 无效，不能降级为普通 comparison。

### 6.2 Secondary same-target contrast

```text
H43: S4 -> S3
L23: S2 -> S3
```

二者 Stage2 target demand 相同，但 Stage1 controller 不同。该组只能用于描述相同新需求下不同历史 controller 的适应差异，不能使用 same-start 因果措辞。

### 6.3 跨算法边界

严格 same-start 只指算法内部 H34/L32，不适用于任何跨算法 comparison。

DDQN 与 CTXDDQN 的正式表述固定为：

> 在相同 Double-DQN 学习框架、匹配初始化和训练协议下，引入 causal demand context 后持续稳定性效应如何变化。

不得称为 Stage2 same-start、same learned controller 或单组件 Stage2 ablation。

---

## 7. Stage1 anchors 与 Stage2 规模

每个新增算法训练：

```text
Stage1: 4 scenes × 5 seeds × 100 episodes = 20 anchors
Stage2: 4 transitions × 5 seeds × 100 episodes = 20 continuations
```

三个算法总计：

```text
60 anchors + 60 continuations
= 120 100-episode units
= 12,000 formal training episodes
= 4.32 million training control decisions
```

PPO calibration 另计，最多 1,800 training episodes。DQN 新训练为 0。

每个 Stage1 anchor 同时承担：

1. single-scene capability check；
2. Stage2 source checkpoint；
3. target-scene specialist reference；
4. current-adaptation normalization reference。

Stage1 每个 episode 0～100 都做独立 frozen current-scene evaluation。Stage1 episode100 full resumable checkpoint 是 Stage2 唯一合法 source；checkpoint 缺失或损坏时不得使用 episode99、邻近 seed 或重建近似状态替代。

---

## 8. DDQN

DDQN 是 vanilla-DQN target construction 的近邻机制控制。

固定网络与训练参数：

```text
network = 16 -> 20 -> 20 -> 8
hidden activation = ReLU
optimizer = RMSprop
learning rate = 0.001
gamma = 0.95
batch size = 64
replay capacity = 5000
learning start = 1000 decisions
gradient clip = 5
target sync = every 10 successful gradient updates
epsilon schedule = existing formal DQN schedule
```

唯一核心算法变化为 Double-DQN target：

```text
a_star = argmax_a Q_online(next_state, a)
y = reward + gamma * Q_target(next_state, a_star)
```

terminal/truncation bootstrap 必须延续正式 DQN 语义，不得借 DDQN 实现调整其他超参数、update budget、replay sampling 或 target sync 计数。

Stage2 `Clear`：

```text
retain = online network, target network, optimizer, epsilon,
         counters, scheduler state, all RNG state
clear  = replay buffer
```

clear 后 replay size 必须为 0；learning-start 的判定继续使用保留的 global counters 和既有 Clear 语义，不擅自重置 epsilon 或全局步数。

---

## 9. CTXDDQN 与受控嵌套初始化

CTXDDQN 是 causal context representation intervention：

```text
input = raw16 + ctx4 = 20 dimensions
network = 20 -> 20 -> 20 -> 8
target = Double-DQN target
```

除输入维度和 causal feature 外，hidden layers、optimizer、replay、epsilon、target update、预算和训练协议与 DDQN 相同。

### 9.1 Nested initialization

每个 `scene × seed` 必须先构造 DDQN 初始化，再由它确定性构造配对 CTXDDQN 初始化：

1. CTXDDQN 第一层 raw16 对应的 16 列复制 DDQN 第一层权重；
2. CTXDDQN 新增 ctx4 对应的 4 列全部初始化为精确零；
3. 第一层 bias、第二 hidden layer、output layer全部复制；
4. online 和 target network 都按同一规则构造；
5. 两个 optimizer 使用相同算法与超参数，初始 optimizer state 均为空；
6. 保存 paired initialization manifest、每层参数 digest 和固定 raw16 输入上的输出相等证明。

在 `ctx4=任意有限值` 且 context columns 为零的初始化时刻，两者对相同 raw16 的初始 Q 函数必须逐元素相等到 exact/validator 指定的数值容差；单元测试使用 `atol=0, rtol=0`，若底层序列化造成不可避免差异则必须先修复构造方式，不能放宽科学门禁。

ctx4 从 Stage1 第一条 interaction 起参与 CTXDDQN 学习。因此 DDQN 和 CTXDDQN 的 Stage1 episode100 anchors 已经不同，不是同一个 learned controller。

### 9.2 Stage2 Clear

```text
retain = online network, target network, optimizer, epsilon,
         counters, scheduler state, all RNG state
clear  = replay buffer, context event/history buffer
```

context history 在每个新 episode 和新 scene reset 时清空，cold start 固定为 zero padding。

---

## 10. Causal traffic context

四维 context 顺序固定为：

```text
ctx4(t) = [North, South, East, West]
```

每维定义为：

```text
number of actually observed network-entry arrivals in (t-60, t] / 60 seconds
```

窗口始终除以固定 60 秒。episode 前 60 秒没有历史的部分按零事件 padding，不按已观察秒数改分母。

arrival 必须来自车辆实际进入当前控制网络的在线事件。方向映射必须由 boundary edge 的几何进入方向确定并冻结为 manifest；不得按 scene/network 名称、route label 或人工场景 ID 映射。车辆在网络内部换 edge 不得重复计数。

禁止输入或间接推导：

```text
scene ID
network label
future route records
future depart schedule
future flow statistics
完整 route 文件未来统计
```

Phase0 必须通过四个 causal gates：

1. `context(t)` 只依赖时间 `<=t` 的事件；
2. 修改 `t` 之后的 route/depart records 不改变 `context(t)`；
3. episode reset 后 history 为空且首个 context 为 zero-padding 结果；
4. 从已记录的过去 60 秒 entry events 离线重算，与在线输出逐 decision 完全一致。

任一失败时：

```text
CTXDDQN formal training = STOP
```

不得自动换成 queue、scene ID、future flow 或其他 proxy。

---

## 11. PPO 正式定义

PPO 是跨学习范式诊断，不是 DDQN 的单组件 ablation。

实现边界：

- 使用 PFRL 0.4.0 PPO core，通过新的 Plan5 adapter 接入共同 runtime；
- 不使用存在已知语义/字段问题的 `agent/ppo.py`；
- 不直接原样复用 `agent/ppo_pfrl.py`；可复用经审计的局部实现，但必须由 Plan5 adapter 提供 checkpoint、resume、evaluator 和 manifest 契约。

正式 PPO：

```text
input = raw16
action distribution = 8-action Categorical
training action = categorical sample
frozen evaluation action = categorical mode / argmax
actor-critic = shared trunk 64 -> 64, ReLU
initialization = LeCun
optimizer = Adam, eps=1e-5
gamma = 0.95
GAE lambda = 0.95
clip ratio = 0.2
value coefficient = 1.0
value clip = 0.2
entropy coefficient = selected by calibration
learning rate = selected by calibration
standardize advantages = true
max grad norm = 0.5
observation normalizer = none
rollout length = 360
minibatch size = 90
PPO epochs per rollout = 4
device = CPU
```

必须记录总参数量和 actor/critic/shared parameter ownership。

### 11.1 Episode 和 bootstrap 语义

一个 360-decision episode 正好构成一个 rollout。3600 秒到达仿真上限属于 truncation，不是环境 terminal，必须用 `V(next_state)` bootstrap。每个 episode 结束完成 PPO update 后断言 rollout buffer 为空；stage boundary 也必须为空。

### 11.2 Stage2 continuation

```text
retain = actor-critic, optimizer, counters, running statistics,
         algorithm state, all RNG state
clear  = empty current rollout buffer only
```

禁止重置 actor、critic、optimizer 或从头训练。因为 Stage1 episode100 已完成 update，合法 checkpoint 中 rollout 应为空；非空 checkpoint 不能进入 Stage2。

PFRL 默认 save 不满足 Plan5。full resumable checkpoint 至少包含：

```text
model and optimizer
rollout memory
batch_last_state/action/episode
n_updates and all counters
training/evaluation statistics
Python/NumPy/PyTorch/PFRL RNG state
run/stage/config identity
canonical training-state digest
```

---

## 12. PPO calibration 与冻结

calibration 只使用：

```text
scene = S2
training seeds = 100, 101, 102
learning rate = {1e-4, 2.5e-4, 5e-4}
entropy coefficient = {0.001, 0.01}
```

共 6 candidates × 3 seeds，最多 18 个 100-episode runs。其余 PPO 参数使用第 11 节固定值。

每条 calibration run 必须有 episode0 及 episode1～100 frozen evaluation。每个 candidate 的确定性选择顺序如下，所有比较均以较小为优：

1. 对每条 run 计算 episode91～100 frozen travel time 的 median；candidate score 为三个 seed run medians 的 median；
2. 若 score 数值完全相同，比较三个 run medians 的 IQR；
3. 若仍相同，比较三个 seeds 的 Current Adaptation AULC median；
4. 若仍相同，选择更小 learning rate；
5. 若 learning rate 也相同，选择更小 entropy coefficient，作为最终确定性 tie-break。

进入正式 anchor 前，胜出 candidate 必须在至少 2/3 calibration seeds 上同时满足：

```text
TT_late = median TT over episodes 91..100
TT_late < TT_FixedTime
(TT_episode0 - TT_late) / TT_episode0 >= 0.10
```

否则：

```text
PPO formal anchors and Stage2 = HOLD
```

不得扩大搜索空间或改用 formal transition 结果调参。胜出配置写入输出资产：

```text
ppo_formal_config.json
ppo_calibration_summary.csv
```

并冻结 SHA256。正式 run manifest 只引用该 hash；正式结果产生后不得修改。

---

## 13. Anchor readiness gates

Phase0 使用 Plan5 common evaluator 为每个 scene 生成 `plan5_fixedtime_reference.json`。FixedTime 使用相同 scene、reward、3600 秒、decision interval、clearance、SUMO fixed-default command 和 travel-time 公式；每个 scene 独立运行两次，两个结果及 trajectory digest 必须完全一致，取该共同值为 `TT_FixedTime`。不允许从不同 evaluator/protocol 的历史表中抄值。任一 scene 的重复运行不一致时，所有依赖 FixedTime 的 PPO calibration/anchor gate 均为 `HOLD`。

DDQN 和 CTXDDQN 的 20 anchors 必须全部满足：

```text
completed = true
no NaN/Inf
checkpoint schema/hash valid
episode100 full resume valid
frozen evaluation isolation valid
fixed probe inference valid
episode91-100 specialist metrics present
```

PPO 除上述通用条件外，每个 scene 的 5 个 formal seeds 中至少 4 个必须同时满足：

```text
TT_late < TT_FixedTime
(TT_episode0 - TT_late) / TT_episode0 >= 0.10
```

任一 scene 未达到 4/5：

```text
PPO formal sequential Stage2 = HOLD
```

这表示 PPO anchor protocol 在当前预算下不可比，不是 Outcome A～E，也不得通过继续调参、删除 seed 或增加 episode 修复。

“usable traffic control”只用于 anchor gate 的上述预注册判据，不得事后增加主观筛选。

---

## 14. `plan5_fixed_probe_v1`

正式训练前必须创建并冻结不可变 probe：

```text
probe_id = plan5_fixed_probe_v1
```

behavior source 为 canonical whitelist 中四个 scene、training seeds 0～4 的 Plan1 episode100 DQN specialists。每个 specialist 执行一条完全冻结的 360-decision evaluation trajectory：

```text
4 scenes × 5 specialists × 360 rows = 7,200 rows
```

split：

```text
Plan1 seeds 0..3 = main probe
Plan1 seed 4 = controller-trajectory held-out probe
```

seed4 不能称为 traffic-seed held-out，因为 SUMO 仍使用 fixed-default 行为。

每行至少保存：

```text
probe_id
split
scene/network
source Plan1 training seed
simulation time/decision index
raw16
ctx4
ctx arrival counts before division
source action
source checkpoint path and SHA256
source trajectory SHA256
resolved SUMO command digest
```

生成时在线记录 raw16、实际 boundary arrival events 和 ctx4。probe 不参与任何 Plan5 training。必须输出：

```text
probe_manifest.json
probe_main_sha256
probe_heldout_sha256
```

formal training 开始后不得重新生成、追加或修改。资产缺失或 hash 改变时停止，不得用 Plan5 policy 新轨迹替代。

输入规则：

```text
DDQN/PPO = raw16
CTXDDQN = raw16 + ctx4
```

Historical Policy Disagreement 只使用 transition source-scene rows：

```text
H34/L32 -> S3 probe
H43     -> S4 probe
L23     -> S2 probe
```

main probe 是主结果；held-out probe 仅复核，不与 main 合并扩大样本量。

### 14.1 Exact raw-state context distinction

在跨 scene probe rows 中建立 `raw16` 完全相同的 row pairs。判断 context 是否不同必须比较四方向 arrival **整数 count vector**，不能比较浮点 rate 的近似相等。

报告：

```text
pair-weighted distinguished fraction
unique-raw-state-weighted distinguished fraction
```

probe row 或 matched pair 不是独立统计样本，不进入 seed-level inferential sample size。

---

## 15. Frozen evaluation 与 checkpoint 协议

### 15.1 Schedule

Stage1 current scene：

```text
episode 0..100, every episode
```

Stage2 current target scene：

```text
episode 0..100, every episode
```

Stage2 historical source scene：

```text
episode = 0,1,2,3,5,10,15,20,25,30,40,50,60,75,100
```

Stage2 episode100 额外评估全部四 scenes。除 source 和 target 外的两个 scene 只属于 secondary generalization，不进入 primary transition inference。

Episode0 evaluation 必须发生在 Stage2 第一次 environment interaction/optimizer update 之前。

### 15.2 Isolation

每次 frozen evaluation：

```text
separate process
deterministic greedy/mode action
no exploration or categorical sampling
no optimizer update
no replay/rollout/context mutation
no epsilon/counter/statistics mutation
no training RNG mutation
```

evaluation 前后比较完整 `training_state_digest`，必须完全一致。digest 覆盖模型、target/critic、optimizer、replay 或 rollout、epsilon、counters、statistics、context state 和所有 RNG。评估进程只读 evaluation snapshot；必须同时验证 snapshot、parent checkpoint 和训练状态未被写回。

### 15.3 Checkpoints

- episode0～100 每个 current evaluation 点保存带 SHA 的 lightweight evaluation snapshot；
- episode0、historical schedule nodes 和 episode100 保存 full resumable checkpoint；
- PPO full checkpoint 即使 rollout 按协议为空，也必须包含第 11 节列出的 rollout 字段；
- 所有 checkpoint 原子写入，进入 `checkpoint_index`；
- resume 必须从 full resumable checkpoint 恢复，不能从 evaluation snapshot 恢复；
- failure/outlier checkpoints 不删除，状态标记为 failed/invalid/recovered。

---

## 16. Stage2 performance metrics

正式 statistical unit：

```text
algorithm × transition × training_seed
```

episode、probe row 和 evaluation decision 都不是独立样本。

### 16.1 Historical degradation

对 source scene：

```text
D(e) = (TT_old(e) - TT_old(0)) / TT_old(0)
Worst Historical Degradation = max_e D(e)
Episode-100 Degradation = D(100)
Recovery Gap = Worst Historical Degradation - D(100)
```

最大值只在预注册 historical evaluation schedule 上计算。若任一必需节点无有效 frozen evaluation，该 run 对这些汇总指标无效，不能插值。

### 16.2 Current adaptation

同一 `algorithm × target scene × seed` 的 Stage1 specialist reference：

```text
TT_specialist = median frozen TT over Stage1 episodes 91..100
R(e) = TT_current(e) / TT_specialist
Current Adaptation AULC = trapezoidal_integral(R(e), e=0..100) / 100
Episode100 normalized performance = R(100)
```

```text
time-to-reference = first episode e such that
R(e), R(e+1), ..., R(e+4) are all <= 1.10
```

仅允许 `e<=96` 成为首次到达点；没有五个连续 episode 满足时记录为 right-censored `>100`，不能写成 100。

Current AULC 越小表示在相对于本算法 specialist 的意义下适应越好。跨算法绝对 TT 可以描述，但不是机制 headline。

统一 headline metrics：

```text
Worst Historical Degradation
Current Adaptation AULC
```

同时报告 queue、delay、throughput 和 reward，但不替代预注册 headline。

---

## 17. Policy displacement 与内部指标

所有 policy displacement 相对于同一 continuation 的 Stage2 episode0 policy。

Historical Greedy-Policy Disagreement：

```text
PD(e) = mean over source-scene probe rows of
        I[argmax_policy_e(input) != argmax_policy_0(input)]
```

其中：

```text
DDQN/CTXDDQN: argmax Q
PPO: argmax categorical probability
```

在 episode0 和 historical schedule nodes 上计算，报告 `max PD`、`PD(100)` 和 timeline。main 与 held-out 分开报告。

DDQN/CTXDDQN 内部指标：

```text
Q vector
Q drift L1/L2 from episode0
top1-top2 Q margin
greedy disagreement
Bellman residual
action entropy/frequency
```

PPO 内部指标：

```text
pi(a|s)
argmax action
policy entropy
TV distance from episode0
JS divergence from episode0
critic V(s)
critic-value drift
```

禁止直接比较 absolute DQN-family Q drift 与 PPO critic V drift。跨算法统一比较只使用外部 performance 和 policy disagreement 等同定义指标。

CTXDDQN 额外报告 exact raw-state pairs 被 ctx count vector 区分的比例，以及不同 context 下 action preference 的描述性变化；不要求 Bellman target difference 变为零。

---

## 18. 统计与推断

固定分析参数：

```text
bootstrap resamples = 10,000
analysis seed = 20260827
confidence interval = percentile 95%
```

每个 algorithm 的 primary inference：

```text
H34 vs L32, five paired training seeds
```

对以下指标逐 seed 计算 `H34 - L32`：

```text
Worst Historical Degradation
Episode-100 Degradation
max Policy Disagreement
Current Adaptation AULC
```

输出五个原始 paired effects、mean、median、direction consistency 和 paired-seed cluster bootstrap CI。bootstrap 每次从 5 个 seed pairs 中有放回抽取 5 对，保持一对内 H34/L32 绑定。`n=5`，传统显著性检验不得作为 headline。

Secondary inference 使用 H43 和 L23 描述 same-target/reverse pattern。可以描述全部 10 high-direction units 与 10 low-direction units，但 pooled 10-vs-10 Mann–Whitney 不作为 headline。

跨算法描述性 comparison 的 bootstrap 单位为完整 training-seed cluster：一次抽中某 seed 时保留该算法该 seed 的所有可用 transitions。DDQN 与 CTXDDQN 报告连续 effect、CI 和 seed consistency，不设置二元 confirmation threshold。

只有两个 traffic-demand pairs，所有报告必须保留这一外推限制。不得把 episode、probe row 或 repeated evaluation 当作增加统计样本量。

---

## 19. CTXDDQN 解释规则

CTXDDQN 的机制假设是：

```text
same raw observation o -> context ambiguous
(o, c1) != (o, c2) -> model can represent distinct conditions
```

正确检查：

1. exact raw-state matches 被 ctx4 区分的比例；
2. historical degradation；
3. historical policy displacement；
4. current adaptation。

若结果改善，只允许表述：

> causal traffic-demand context reduces the ambiguity associated with instantaneous observation and is accompanied by improved continual stability.

不得自动表述 partial observability fully solved、context 是唯一根因或严格因果中介已经证明。若没有改善，只能削弱 representation-only explanation，不能证明 context 完全无关。

---

## 20. Outcome A～E 的地位

以下仅是结果解释模板，不是通过/失败门槛，也不产生 `reproduced=true/false` 标签：

```text
A: DDQN and PPO both show similar instability
   -> phenomenon may extend beyond vanilla value-based DQN.

B: DDQN shows it while PPO is descriptively more stable
   -> value-based/replay/bootstrapped dynamics may be an important boundary.

C: CTXDDQN shows improved continual stability
   -> causal context ambiguity remains a supported source.

D: CTXDDQN does not show improvement
   -> representation-only explanation is weakened.

E: DDQN itself shows little corresponding instability
   -> revisit the narrow boundary around vanilla target/optimization dynamics.
```

不得为 Outcome C 或其他 Outcome 设置事后效果阈值。结果可能同时部分符合多个模板；最终报告必须呈现连续指标、CI、seed consistency、失败和异常值。

---

## 21. Phase0：基础设施与门禁

Phase0 不得启动 formal batch。按顺序完成：

1. environment restore：恢复 `colight`、SUMO executable 和 Python APIs，冻结版本；
2. source provenance：记录 branch/commit/worktree，得到 clean working tree proof 和 Git bundle；
3. DQN canonical reference manifest；
4. Plan5 config/schema、run identity、manifest、launcher 和 validator；
5. DDQN target/operator、checkpoint、resume 和 fixed-probe inference；
6. CTX online arrival feature、geometric direction mapping、nested initialization 和 causal gates；
7. PPO adapter、Categorical policy、rollout/update、custom checkpoint/resume；
8. common frozen evaluator、training-state digest 和 fixed-probe logger；
9. `plan5_fixed_probe_v1` 生成、校验和 hash freeze；
10. analysis pipeline 和所有最终表的 schema；
11. targeted unit/integration tests；
12. one-run smoke、resume equivalence、evaluation isolation；
13. 1/4/8 concurrency resource gate。

正式 source identity 要求 tracked worktree clean。若 Phase0 实现产生代码修改，而用户未授权本地 commit，必须在 formal runs 前暂停请求授权；获准后只允许中文 commit，不 push。Git bundle 保存到 Plan5 输出 provenance 目录，不提交到 Git。

### 21.1 必测场景

- 固定 batch 中 online argmax 与 target argmax 不同，断言 DDQN target 数值正确且与 vanilla target 产生预期差异；
- DDQN/CTXDDQN nested initialization 在固定 raw16 batch 上初始 Q exact equal；
- context 四项 causal tests；
- PPO 360-step rollout、truncation bootstrap、4 epochs/minibatch scheduling 和 rollout-empty assertion；
- DDQN、CTXDDQN、PPO uninterrupted 与 checkpoint-resume 最终 canonical state/metrics 等价；
- evaluation 前后 full training-state digest 完全相同；
- manifest 拒绝重复 ID、错误 checkpoint SHA、错误 transition source、H34/L32 source 不同和缺字段；
- probe 行数、split、source whitelist、raw16/ctx4/count vectors 和 hashes 完整；
- 分析器使用 synthetic fixtures 验证 D(e)、AULC、time-to-reference、PD 和 paired cluster bootstrap。

### 21.2 Smoke

固定 smoke scene 为 S2，smoke-only training seed 为 999，产物写入独立 `smoke/`，不得进入 formal manifest。DDQN/CTXDDQN 各运行 3 episodes/1080 decisions，使用正式 `learning_start=1000`，断言至少 1 次 gradient update 和 1 次 target sync；PPO 运行 1 episode/360 decisions，断言完成 1 个 rollout 和 4-epoch update，结束后 rollout 为空。

resume-equivalence 另使用 seed 998：DDQN/CTXDDQN 比较连续 4 episodes 与 episode2 checkpoint 恢复后继续到 episode4；PPO 比较连续 2 episodes 与 episode1 checkpoint 恢复后继续到 episode2。CPU 下 canonical training-state digest、counters 和逐 episode metrics 必须完全一致。记录完整命令、network、seed、episodes、steps、退出码、更新数、SUMO command、产物路径和 hashes。

### 21.3 并发 gate

分别使用 DDQN S2 3-episode smoke profile 和互不相同的非正式 seeds 运行 concurrency 1、4、8。正式并发取通过 gate 的最高档，目标为 8。每档必须满足：

```text
all child runs complete and validate
no run/config/output collision
semantic hashes and expected counters valid
peak host RAM <= 80% of physical RAM
swap growth <= 1 GiB
free disk after projected formal output >= 150 GiB
median per-run wall-time slowdown <= 2.5x single-run baseline
```

8 未通过则降到 4；4 未通过则降到 1。不得为保持并发 8 放宽实验语义或隔离要求。

---

## 22. Phase1：Anchors

严格顺序：

```text
DDQN 20 anchors -> validate
CTXDDQN 20 anchors -> validate
PPO calibration -> freeze config -> PPO 20 anchors -> validate
```

不得让某算法 Stage2 与另一算法未验收的 anchors 大规模混跑。DDQN 与 CTXDDQN 的 paired initialization identities 必须在启动两组 anchor 前共同冻结。

任一 run failure 必须保留失败 attempt 和日志。允许从最近合法 full resumable checkpoint 恢复同一 logical run；恢复后 validator 必须证明与不中断协议一致。不得删除 outlier、改 seed 或从邻近 checkpoint 重新命名。

算法未通过第 13 节 readiness gate 时，仅该算法进入 `HOLD`；不受依赖的其他算法可继续。任何 HOLD 都必须进入 validation report，而不是静默缩小矩阵。

---

## 23. Phase2：Core transition experiment

对每个已通过 anchor gate 的算法执行：

```text
H34 × seeds 0..4
H43 × seeds 0..4
L23 × seeds 0..4
L32 × seeds 0..4
```

每算法 20 continuations。构建 manifest 时先验证 H34/L32 的 exact same S3 source checkpoint path 和 SHA，再允许 launch。

每个 algorithm 的 20 runs 全部进入终态并通过或明确标记 invalid/failed 后，才生成该算法正式 summary。缺失 run 不得用其他 seed、checkpoint 或插值补齐。

三算法 Phase2 完成或按 gate 明确 HOLD 后，生成完整分析包并：

```text
STOP ALL EXPANSION
```

不得自动启动 DDDQN、HISTDQN、DDQN+DHOA、same-start RNG forks、更多 seeds、更多 transitions 或 PPO historical replay。

---

## 24. 正式输出包

输出根目录至少包含：

```text
plan5_run_manifest.csv
plan5_anchor_summary.csv
plan5_transition_summary.csv
plan5_historical_timeline.csv
plan5_current_adaptation.csv
plan5_policy_displacement.csv
plan5_internal_metrics.csv
plan5_same_start_H34_L32.csv
plan5_secondary_H43_L23.csv
plan5_validation_report.json
plan5_resource_report.json
canonical_reference_manifest.json
plan5_fixedtime_reference.json
ppo_formal_config.json
ppo_calibration_summary.csv
probe/plan5_fixed_probe_v1/
resolved_configs/
checkpoint_indexes/
provenance/
```

每个 run 必须提供：

```text
resolved config and hash
source commit and worktree proof
training manifest
complete SUMO command
training seed and RNG metadata
checkpoint index and hashes
evaluation manifest
probe manifest reference
training dynamics
action distribution
state summary
run-level summary
validation status and failure lineage
```

JSON/CSV schema 必须在 Phase0 测试中固定。结果、checkpoint、trajectory、环境包、Git bundle 和大日志留在 `data/output_data`，不得提交 Git。代码仓库只保留必要源码、配置、小型测试和文档。

---

## 25. 资源预算与本机支持结论

`linux4090` 审计资源：

```text
GPU = RTX 4060 Ti 8 GiB (Plan5 formal runs do not use it)
CPU = Intel i7-13700K, 16 cores / 24 threads
RAM = 62 GiB
free disk at design time ~= 4.6 TiB
existing data/output_data ~= 109 GiB
```

预计：

```text
formal training = 12,000 episodes
formal frozen evaluation ~= 13,140 episodes
PPO calibration <= 1,800 training episodes
8-way formal wall time estimate = 11..16 hours
PPO calibration estimate = 2..4 hours
additional disk reserve = 100..150 GiB
```

结论：硬件容量支持本 Plan，但正式并发必须由 Phase0 的 1/4/8 实测 gate 决定；上述时间只是排程估算，不是验收条件。

---

## 26. 暂停、失败与不可变规则

必须暂停并报告的条件：

- 需要改变 observation、reward、action、transition、seed、预算或指标定义；
- CTX causal gate 任一失败；
- PPO calibration 或 anchor gate 触发 HOLD；
- canonical Plan1 source identity/hash 无法验证；
- H34/L32 没有引用 exact same algorithm-internal S3 checkpoint；
- checkpoint/resume 或 evaluation isolation 不成立；
- 正式实现需要 commit 但尚未获得用户授权；
- 用户已有修改与 Plan5 必需改动直接冲突；
- 继续操作需要 push、不可恢复操作或条件扩展。

失败 run、outlier、recovery lineage 和无效结果必须保留。只有物理损坏且已有 SHA 证明的临时文件可以隔离，不能删除证据。正式分析不得因结果方向不好而排除 run。

禁止：

- 重跑 DQN bridge、Plan1 400 episodes 或完整 Plan3/4；
- 修改 SUMO scene、reward 或 action 来适配 PPO；
- 给 CTXDDQN 输入 scene ID 或未来信息；
- 看 formal transition 结果后调整 PPO；
- 为 PPO 发明 DHOA 或历史 replay；
- 把 missing checkpoint 替换为邻近 checkpoint；
- 删除 failure/outlier/recovery runs；
- 自动提交、自动 push、force push 或修改已推送历史；
- Phase2 后自动扩展。

---

## 27. Phase0～Phase2 完成定义

只有同时满足以下条件，才可报告 Plan5 第一阶段完成：

1. formal-ready environment 和 source provenance 通过；
2. DDQN、PPO、CTXDDQN 实现及所有适用 Phase0 tests 通过；
3. immutable `plan5_fixed_probe_v1` 生成并校验；
4. 所有未 HOLD 算法的 20 anchors 通过 readiness gate；
5. 所有未 HOLD 算法的 20 continuations 进入可审计终态；
6. frozen evaluations、checkpoints、resume、same-start 和 state isolation 全部验证；
7. 第 24 节输出包完整，缺失/HOLD/failed 明示；
8. Outcome A～E 仅按解释模板讨论，没有事后阈值和选择性删除；
9. tracked worktree 状态、提交授权状态和未 push 状态明确；
10. 已执行 `STOP ALL EXPANSION`，所有条件实验仍为 HOLD。

Plan5 的最终问题是：

```text
Traffic-demand context
        -> Observation ambiguity
        -> Value-learning formulation
        -> Shared neural updates
        -> Historical policy reorganization
        -> Transient control degradation
```

DDQN 定位 vanilla DQN 与更广 value-based learning 的边界；PPO 定位 value-based 与一般 neural policy learning 的边界；CTXDDQN 在匹配初始化和 Double-DQN 框架下直接干预 causal traffic-context information。
