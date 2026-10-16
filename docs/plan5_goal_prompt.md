# Codex CLI Goal Prompt：执行 Plan5 Phase0～Phase2

将本文件“任务正文”整体交给另一个 Codex CLI agent。该 agent 应在仓库根目录 `/workspace/projects/simOfflineTSC` 工作。

## 任务正文

Dear Codex，

你的目标是在 `linux4090` 上完整实现并执行 Plan5 的 Phase0～Phase2，生成可审计的跨算法机制定位实验包，然后强制停止所有扩展。

### 一、最高执行依据

开始前完整阅读：

```text
AGENTS.md
docs/plan.md
```

二者共同构成权威执行规范。优先级为：用户最新明确指令、`AGENTS.md`、`docs/plan.md`、其他仓库材料。`docs/semi_offline_cross_algorithm_reproduction.md` 只提供历史背景，和 `docs/plan.md` 冲突时以 `docs/plan.md` 为准。

`docs/plan.md` 中算法、环境、transition、初始化、context、PPO calibration、checkpoint、probe、指标、bootstrap、gate、HOLD 和 STOP 语义均已冻结。不要重新设计、简化或根据结果修改。

本 goal 对 Plan5 范围内的代码、配置、小型测试和本地运行适配提供实施授权，但不替代 `AGENTS.md` 要求的 `linux4090` 分阶段执行确认。开始 Phase0、Phase1、Phase2 前，分别向用户说明即将执行的范围、预计运行和产物，并取得明确确认；交付本 prompt 本身不视为任一阶段的运行确认。

### 二、绝对边界

只执行：

```text
Phase0: environment + infrastructure + gates
Phase1: DDQN anchors, CTXDDQN anchors, PPO calibration and anchors
Phase2: DDQN, CTXDDQN and PPO core transitions + analysis package
```

第一阶段算法只有：

```text
DDQN
PPO
CTXDDQN
```

禁止：

```text
new DQN training
DDDQN
HISTDQN
DDQN + DHOA
PPO historical replay
same-start RNG forks
extra seeds or transitions
full Plan1/Plan3/Plan4 reruns
post-formal hyperparameter tuning
automatic remote push
```

Phase2 输出包完成后执行 `STOP ALL EXPANSION`，等待用户的新指令。

### 三、Git 与本地工作保护

1. 开始只读审核时记录当前 branch、HEAD、remote tracking、worktree 状态和已有用户修改。
2. 当前预期分支为 `codex/milestone0-experiment-infrastructure`，设计时 HEAD 为 `ebfe04a`，本地比远端领先 2 commits；以实际只读结果为准。
3. 不 pull、rebase、reset、checkout 覆盖、stash/pop、丢弃或改写已有用户修改。发现与 Plan5 直接重叠的用户修改时，先理解并兼容；无法安全兼容才暂停。
4. 不 push，不 force push，不修改远端历史。
5. 不自动 commit。Phase0 实现完成后，正式训练要求 clean tracked worktree 和可引用 source commit；若必须创建本地提交，先取得用户明确授权，提交标题和正文必须为中文。即使获准 commit，也不得 push。
6. 正式 source commit 冻结后生成 Git bundle，放入 `data/output_data/cross_algorithm/plan5_b100/provenance/`，不得提交 bundle。
7. 不向 Git 添加 checkpoint、trajectory、完整日志、environment archive、结果 CSV/JSON 大包或其他生成产物。

### 四、开始方式和进度沟通

先做不改变项目状态的只读审计，核对：

```text
repo/worktree/source identity
available Plan1 whitelist and checkpoints
available/missing/external DQN reference assets
current sequential runtime and evaluator
existing DQN and PPO implementations
SUMO/Conda/Python/PyTorch/PFRL availability
disk, RAM, CPU and process state
```

然后给用户一份 Phase0 启动说明并请求确认。未经确认，不安装依赖、不修改代码/配置、不运行 SUMO smoke 或训练。

持续执行期间每完成一个可验证单元就简短报告：当前完成项、验证证据、下一项和任何 HOLD/blocker。不要让用户只看到长时间无状态输出。

当遇到失败时，先定位根因、保留证据并在原 logical run 的新 attempt 上恢复；不要删除失败目录或用新 identity 掩盖失败。只有 `docs/plan.md` 列出的科研语义、权限、source identity、causal gate、same-start、checkpoint/resume 或 evaluation-isolation blocker 才暂停等待用户。

### 五、Phase0 实现

在获得 Phase0 确认后，依照 `docs/plan.md` 第 21 节完成以下工作。

#### 1. Formal environment

恢复并冻结：

```text
Conda env = colight
Python = 3.10.18
PyTorch = 1.13.1+cu116
Gym = 0.26.2
NumPy = 1.26.4
PFRL = 0.4.0
SUMO = 1.27.1
formal device = CPU
```

记录可复建环境、package/version probe、SUMO binary 和 Python APIs。不要把“能够 import”当作 gate 全部通过。所有机器相关适配保持本地化、最小化，并记录到 provenance/resource report。

#### 2. Plan5 runtime and identities

优先扩展现有 `sequential/` 公共基础设施，保持 `common/registry.py` 注册架构和 `world/` 的 SUMO 特有边界。新增清晰隔离的 Plan5 config/schema、manifest builder、launcher、run state、checkpoint index、validator、analyzer 和 CLI 子命令；不要复制一套无法与现有恢复/评估设施互操作的一次性训练脚本。

正式配置入口：

```text
configs/sequential/plan5_cross_algorithm_b100.yml
```

正式输出：

```text
data/output_data/cross_algorithm/plan5_b100/
```

实现 fail-closed identity 和覆盖保护。相同 logical run 的恢复使用 attempt lineage，不创建伪新 run。

#### 3. Canonical DQN references

从 canonical Plan1 20-run whitelist 解析 episode100 specialists，生成 `canonical_reference_manifest.json`，逐资产标注 `available/external/missing`。存在重复 seed0 candidate 时只能由 whitelist 选择。不要训练 DQN，也不要用邻近 checkpoint 补缺失资产。

#### 4. DDQN

实现 `docs/plan.md` 冻结的 `16→20→20→8` DDQN。仅改变 target operator：online network 选择 next action，target network 评价该 action。保持正式 DQN 其余训练语义、Clear、计数、replay、epsilon、optimizer 和 checkpoint 行为。

#### 5. CTXDDQN

实现 `raw16+ctx4` 的 Double-DQN。ctx4 必须来自过去 60 秒实际 boundary entry events，按 `[N,S,E,W]` 几何方向映射，固定分母 60 秒，episode cold start 零 padding。

实现 DDQN→CTXDDQN nested initialization：复制 raw16 columns、bias 和后续层，新增 context columns 精确置零；online/target 都匹配；生成 paired initialization manifest 和 digests。初始固定 raw16 batch 的 Q 输出必须 exact equal。

实现四项 causal tests。任一失败立即将 CTXDDQN formal training 标记 `STOP`，不得改用 queue、scene ID、network label、future route/depart/flow 或任何替代 proxy。

CTXDDQN Stage2 使用 Clear：保留网络、target、optimizer、epsilon、counters 和 RNG；清 replay，并在 episode/scene reset 清 context history。

#### 6. PPO

通过新的 Plan5 adapter 使用 PFRL 0.4.0 core。不要使用 `agent/ppo.py`，不要直接把 `agent/ppo_pfrl.py` 当作正式资产。实现 raw16、8-action Categorical、shared 64×64 ReLU actor-critic、LeCun initialization 和 `docs/plan.md` 的全部冻结参数。

360 decisions 是一个 rollout；3600 秒结束是 truncation，必须 bootstrap `V(next_state)`。每 episode update 后 rollout 必须为空。实现 full custom resumable checkpoint，包含 PFRL 默认 save 未覆盖的 rollout/batch-last/update/statistics/RNG/training state。

PPO Stage2 保留 actor-critic、optimizer、counters、statistics、algorithm state 和 RNG，只清空本应为空的 rollout；不得重置模型或 optimizer。

#### 7. Common evaluator, checkpoint and probe

实现三算法共同 frozen evaluator和 fixed-probe inference：

```text
separate evaluation process
full training-state digest before/after identical
evaluation snapshot each episode 0..100
full resume checkpoints at required nodes
source/current/all-scene schedules exactly as plan
```

生成并冻结 `plan5_fixed_probe_v1`：canonical Plan1 specialists、4 scenes×5 seeds×360=7,200 rows；seeds0–3 main，seed4 controller-trajectory held-out。保存 raw16、ctx4、原始 arrival counts、source identities 和全部 hashes。formal training 开始后不得重建或修改。

按 Plan 第 13 节用 common evaluator 为四个 scenes 各执行两次 deterministic FixedTime reference；重复结果和 trajectory digest 必须完全一致，冻结为 `plan5_fixedtime_reference.json`。不得从其他 evaluator 的历史汇总表抄 FixedTime gate 值。

#### 8. Analysis and schema

在 formal results 前冻结所有最终 CSV/JSON schema，实现并用 synthetic fixtures 验证：historical degradation、AULC、time-to-reference、policy disagreement、exact raw-state context distinction、paired H34/L32 effects 和 cluster bootstrap。

bootstrap 固定为：

```text
10,000 resamples
analysis seed = 20260827
percentile 95% CI
```

#### 9. Tests, smoke and resource gate

完成 `docs/plan.md` 第 21 节全部 unit/integration tests。按该节冻结的 S2、seed999 和 episode 预算分别执行 DDQN、CTXDDQN、PPO 真实 SUMO smoke；按 seed998 的中断点执行 resume equivalence，并验证真实 optimizer update、target/update schedule、checkpoint/resume 和 evaluation isolation。

执行 concurrency 1/4/8 gate；只选择通过完整资源与隔离标准的最高并发，目标为 8。将完整命令、退出码、更新计数、峰值资源、输出路径和验证结果写入 `plan5_resource_report.json`。

Phase0 未全部通过时不得启动 formal anchors。

### 六、Phase1 Anchors

Phase0 gate 完成后，先报告实现、测试、smoke、资源结论、source/worktree 状态和预计成本，请求用户确认 Phase1。

获准后严格按顺序：

```text
1. DDQN: 4 scenes × seeds0..4 × 100 episodes = 20 anchors
2. validate all DDQN anchors
3. CTXDDQN: 4 scenes × seeds0..4 × 100 episodes = 20 anchors
4. validate all CTXDDQN anchors
5. PPO calibration on S2, seeds100..102, 6 candidates
6. freeze/hash ppo_formal_config.json
7. PPO: 4 scenes × seeds0..4 × 100 episodes = 20 anchors
8. validate all PPO anchors and per-scene 4/5 gate
```

PPO calibration 最多 18×100 episodes，完全按 `docs/plan.md` 的 score、IQR、AULC、LR、entropy tie-break 和 2/3 eligibility gate 选择。失败时 `PPO formal anchors and Stage2 = HOLD`，不扩大搜索。

PPO 任一 scene 未达到 formal 4/5 anchor gate 时，`PPO Stage2 = HOLD`。DDQN/CTXDDQN 需要 20/20 anchors 完整通过各自 gate。一个算法 HOLD 不阻塞其他不依赖算法，但必须保留并报告完整失败证据。

不要把 anchors 与 Stage2 batch 混跑。所有 Stage2 source checkpoint 必须是同算法、同 seed、对应 scene 的 episode100 full resumable checkpoint。

### 七、Phase2 Core Transitions

Phase1 完成后先输出 anchor summaries、gate status、failure/outlier list、实际耗时和剩余资源，请求用户确认 Phase2。

对每个未 HOLD 算法执行：

```text
H34 = S3 -> S4, seeds0..4
H43 = S4 -> S3, seeds0..4
L23 = S2 -> S3, seeds0..4
L32 = S3 -> S2, seeds0..4
```

每算法 20 continuations，每条 100 episodes。

在 manifest build 和 launch 两处都强制断言：同一算法/seed 的 H34 与 L32 使用完全相同的 S3 checkpoint path、file SHA、model、optimizer、algorithm state 和 RNG state。严格 same-start 只存在于此算法内部 comparison。

DDQN 与 CTXDDQN 不是 Stage2 same-start。跨算法正式措辞只能是：

> 在相同 Double-DQN 学习框架、匹配初始化和训练协议下，引入 causal demand context 后持续稳定性效应如何变化。

执行所有 frozen evaluations、probe inference、checkpoint 和 validation schedule。缺失 cell 不插值；失败 run 不删除；missing checkpoint 不替代。

### 八、正式分析和推断纪律

statistical unit 是 `algorithm × transition × training_seed`。不要把 episode、probe row 或 decision 当独立样本。

Primary：每算法 H34 vs L32 的五个 paired seeds，输出每个原始 effect、mean、median、direction consistency 和 paired cluster bootstrap CI。

Secondary：H43/L23 same-target pattern 和全部 high/low directional 描述。不要把 pooled Mann–Whitney 作为 headline。

跨算法 bootstrap 按完整 training-seed cluster 重采样并保留所有 transitions。DDQN vs CTXDDQN 只报告连续 effect、CI 和 seed consistency，不设置 binary confirmation threshold。

Outcome A～E 只是解释模板：

```text
no pass/fail threshold
no reproduced/not-reproduced classification
no Outcome C confirmation threshold
no selective run removal
```

CTXDDQN 改善时也不得声称 partial observability fully solved。不要直接比较 PPO V drift 和 DQN-family Q drift 的绝对量。

### 九、必须返回的输出包

至少生成并验证：

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
all resolved configs
all checkpoint indexes and hashes
probe manifest and hashes
source/environment provenance
all failure/outlier/recovery lineages
```

所有生成结果留在：

```text
data/output_data/cross_algorithm/plan5_b100/
```

不要提交这些大产物到 Git。

### 十、完成与停止

只有 `docs/plan.md` 第 27 节完成条件全部满足，才报告 Phase0～Phase2 完成。若某算法按预注册 gate 进入 HOLD，完整报告 HOLD 和原因，不把它伪装成完整成功。

生成最终分析包后：

```text
STOP ALL EXPANSION
```

不得启动任何条件实验。最终回复必须清楚列出：

```text
implemented capabilities and scientific invariants
environment/source identity
commands and validation results
actual run counts by complete/failed/HOLD
anchor gates and PPO calibration choice
same-start validation
headline continuous metrics and uncertainty
all output paths
Git worktree/local commits, if any
explicit confirmation that nothing was pushed
remaining risks and HOLD items
```

不要以“结果符合预期”作为成功条件。成功标准是协议被忠实执行、证据完整、失败保留、推断边界正确，并在 Phase2 后停止。
