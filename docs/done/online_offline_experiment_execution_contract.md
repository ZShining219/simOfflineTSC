# Online–Offline 实验实施契约

## 1. 目标与适用范围

本契约指导 Codex CLI 基于当前 LibSignal 项目，实现 plan0721.md 所需的可复现 Online、Offline 与 Sequential 实验能力。当前执行范围仅为 Milestone 0：公共实验基础设施。

Milestone 0 不执行 100 episode Pilot、20 次正式 Online 训练、Offline 训练或顺序训练，也不产出论文实验结论。

Milestone 0 的行为基线固定为创建工作分支时最新 origin/main 的完整提交 SHA，记为 baseline_commit。后续 origin/main 更新不自动改变本次 Milestone 的行为基线。所有“项目当前行为”和“保持当前训练数值语义”均指 baseline_commit 在 Milestone 0 修改前的行为。

本次 Milestone 0 在创建 `codex/milestone0-experiment-infrastructure` 分支前已执行 `git fetch origin`，固定 `baseline_commit=73d860bb3924ec15c30433a8f8b7af17787baeff`。

## 2. 规则与决策优先级

科研语义按以下顺序解释：

1. 用户最新明确决定；
2. research_decision_register.md 中状态为“已决定”“已实现”或“已验证”的决定；
3. plan0721.md；
4. 本契约中的默认原则。

工程执行按以下顺序约束：

1. 系统和用户指令；
2. 仓库 AGENTS.md；
3. 本契约；
4. milestone0_acceptance_checklist.md；
5. codex_cli_milestone0_prompt.md。

不得静默解决规则冲突。科研语义冲突必须登记到集中科研决策表并暂停受影响功能；工程冲突遵循更高优先级规则，并在功能完成报告中说明。

## 3. 已确认边界

### 3.1 场景

S1–S4 只用于 Plan 和 Goal 解释实验设计，不作为命令行参数、配置键、代码映射或输出目录名。实际执行始终使用项目已有 network 名称。

### 3.2 Seed

- 后续未加限定词的 seed 均指 training_seed，即 run.py --seed；
- training_seed 控制 LibSignal 现有训练侧随机性；
- 当前项目不额外管理 SUMO seed，也不向 SUMO 增加 seed 参数；
- sumo_seed_mode = fixed_default 只作为事实记录，不实现为可调功能；
- 该边界必须在项目 README 中明确说明。

### 3.3 配置入口

- 不为 run.py 增加额外配置路径参数；
- 继续使用 configs/tsc/base.yml 与 configs/tsc/{agent}.yml；
- DQN 专属参数优先在 configs/tsc/dqn.yml 覆盖；
- 每次运行自动归档源配置、最终配置和运行时模型信息。

### 3.4 Checkpoint

- evaluation checkpoint 保存 online Q-network，并以 online Q-network 执行评估；
- resumable checkpoint 保存 online Q-network、target network、optimizer、epsilon、训练计数器、随机状态和恢复所需 replay 状态；
- Milestone 0 不实现 best checkpoint 选择，也不改变 Plan 1 的 best 指标决定；
- checkpoint 格式必须带明确 schema/version，不得把旧 target-only 文件误标为新的 evaluation checkpoint。

## 4. Milestone 0 功能顺序

以下编号是推荐执行顺序。每项必须独立完成闭环；当前项未通过时不得进入依赖它的功能：

1. 配置与运行时模型归档，以及 simulator 源配置只读化；
2. 运行身份与状态管理；
3. seed 记录与可重复性检查；
4. 结构化训练与评估日志；
5. 训练与评估隔离；
6. checkpoint 语义与完整性；
7. 指标能力审计与基础诊断；
8. 跨运行配置一致性校验；
9. Milestone 0 整体回归。

依赖关系为：功能 1 是功能 2、3、8 的前置条件；功能 2 是所有会产生运行状态证据的后续功能前置条件；功能 4 是功能 5 和最终整体回归的前置条件；功能 6 的验收依赖功能 2；功能 7 的只读审计可独立执行，但指标公式修改受科研决策门约束；功能 9 依赖所有未阻塞功能均已通过。

某功能因科研决策阻塞时，可以继续执行经审计确认不依赖该决定的后续功能。跳过时必须在验收清单记录依赖判断和科研决策编号。

功能 1 必须保证 configs/sim 下的源 cfg 在运行期间保持只读。每次运行基于源 cfg 生成运行目录内的 resolved simulator cfg，项目和 SUMO 只使用该运行专属文件，不得写回共享源 cfg。该内部优化不增加 run.py 配置路径参数，也不改变仿真或算法参数。

## 5. 默认授权

在不改变科研语义的前提下，Codex 可以自主：

- 审计和修改项目代码；
- 修改现有 YAML，并在测试结束后恢复临时参数；
- 新增必要工具、校验器和小型自动检查；
- 改进归档、manifest、哈希、结构化日志、状态记录和错误提示；
- 运行 FixedTime、MaxPressure 和 DQN 最小 SUMO smoke；
- 检查真实生成文件，而不只检查退出码；
- 修复测试发现的 bug 并重新验证；
- 清理由自身测试造成的配置和 simulator cfg 副作用；
- 按功能创建中文提交并推送指定远端分支；
- 更新验收证据和科研决策记录。

Plan 已明确要求的工程能力无需重复询问。普通文件命名、模块拆分和测试组织由 Codex 自主决定。

## 6. 科研语义保护门

Milestone 0 默认严格保持当前训练数值语义。未经用户明确决定，不得修改：

- state、reward 或 action mapping；
- DQN loss 与 terminal/truncated bootstrap；
- epsilon schedule；
- learning-start 条件和开始训练前的动作策略；
- train frequency、gradient steps、target update frequency；
- replay replacement 或 sampling；
- 指标公式、主指标口径或实验预算。

发现冲突时必须记录项目当前行为与代码证据，说明科研影响，写入集中科研决策表，并只暂停依赖该决定的部分。不得以推荐方案代替用户决定。

## 7. 单功能执行闭环

每个功能开始前输出功能执行卡：

    功能：
    当前项目行为：
    Plan要求：
    本次差距：
    允许修改：
    禁止修改：
    验证方案：
    预期产物：
    是否存在科研决策阻塞：

随后连续执行：

    审核现状 → 明确意图和不变量 → 实现 → 针对性检查
    → 真实最小 smoke → 产物内容校验 → bug 审查
    → 修复与复验 → 完整 diff 审查 → 功能提交 → 推送
    → 验收/决策记录同步

验证失败时不得提交失败实现。可安全修复的问题应继续修复；同一阻塞连续三轮仍无法解决时才报告阻塞。

这里“一轮”指针对同一根因提出一个实质不同的修复方案，完成实现，并执行足以验证该根因的复验。重复执行同一失败命令、未改变方案的重试或外部状态未变化的重复尝试，不计为新一轮。

## 8. 验证与证据

每个功能至少满足：记录优化前行为证据；有针对性自动检查；涉及运行行为时执行真实 SUMO smoke；验证相关失败路径；检查生成文件内容；执行适用语法/导入检查和 git diff --check；确认没有临时 YAML 或 simulator cfg 变化；检查完整 diff 并说明算法语义边界；在验收清单记录命令、结果、产物和提交。

验证按风险分三级：纯文档或只读校验器不强制 SUMO；影响 Runner、Trainer、Agent、World、日志或 checkpoint 时，至少执行一个相关 agent 的真实 SUMO smoke；Milestone 最终回归必须分别执行 FixedTime、MaxPressure 和 DQN 真实 SUMO smoke。

Smoke profile 固定为：

- 基础启动 smoke：1 episode、100 simulation steps，用于 Runner、SUMO、归档、状态和日志；
- 训练更新 smoke：使用临时 YAML，保证 replay_size 不小于 batch_size，且至少发生 1 次 gradient update 和 1 次 target update；临时 learning_start 不得小于 batch_size，除非代码已有显式保护；
- checkpoint 恢复 smoke：实际保存、加载，并在恢复后继续至少 1 次 optimizer update；
- 传统控制器完整环境 smoke：FixedTime 和 MaxPressure 各 1 episode、3600 simulation steps；
- DQN 最终回归使用受控短预算，不执行 Pilot，但必须覆盖真实更新路径。

运行身份文件固定为运行根目录下的 run_manifest.json 和 run_status.json。run_id 固定为相对运行路径 task/world_agent/network/prefix，不另行生成随机 UUID。config_hash 固定为归档后 config/resolved_config.yaml 文件字节的 SHA-256。run_manifest.json 保存不可变身份字段；run_status.json 保存可变状态字段。

状态只允许“已创建”“运行中”“已完成”“失败”，并原子写入。独占创建运行目录后写“已创建”；调用 task.run 前写“运行中”；task.run 正常返回且必需产物已完成 flush/校验后写“已完成”；目录创建后的受控异常写“失败”。created_at_utc 在“已创建”时写入；started_at_utc 在“运行中”时写入；finished_at_utc 只在“已完成”或“失败”时写入。未发生的时间和未知 exit_code 写 null，不得写空字符串或伪造 0。正常完成 exit_code 为 0；捕获并导致进程失败的异常记录非零值和 error_type/error_message；进程被强制终止而无法更新时保持“运行中”，不得视为完成。

两文件合计的最小字段为：schema_version、run_id、task、agent、world、network、prefix、training_seed、sumo_seed_mode、baseline_commit、created_at_utc、started_at_utc、finished_at_utc、status、exit_code、error_type、error_message、config_hash。只有“已完成”可进入结果汇总；Milestone 0 不支持在失败目录内 resume，重试必须使用新 prefix；错误信息不得包含敏感环境变量。

结构化日志文件固定为运行根目录下的 metrics/records.jsonl，UTF-8 编码，每行一个完整 JSON 对象，append-only。最小字段为：schema_version、record_type、agent、network、training_seed、episode、simulation_step、decision_step、global_decision_step、gradient_updates、travel_time、reward_mean、reward_sum、queue、delay、throughput、loss_mean、epsilon、wall_time_seconds。record_type 只允许 TRAIN、EVALUATION、FINAL_EVALUATION。simulation_step 指 episode 内已执行的 SUMO simulation step；decision_step 是 episode 内已执行的动作决策次数；global_decision_step 跨训练 episode 连续且评估不得推进；gradient_updates 只在真实 optimizer.step 成功返回后增加；wall_time_seconds 是当前 record_type 对应阶段的墙钟耗时。缺失或对非学习控制器不适用的指标写 null，不得伪造为 0。保留现有文本日志兼容性。

Milestone 0 的评估隔离只验证：评估不调用 agent.remember；评估前后 replay 长度和当前 dataset 写入计数不变；评估路径具有明确 record_type；online/target、optimizer、epsilon 和 gradient counter 不变。Plan 1 实现 append-only trajectory 后，再验收 evaluation transition count 等于 0。

同 training_seed 的可重复性检查限定为同一 baseline_commit、network 和配置下：初始 online model state hash 相同；初始 target model state hash 相同；online 与 target 初始一致；首个受控 NumPy 随机动作序列相同；首个受控 Python random sample 序列相同。不要求不同硬件位级一致或完整 400 episode 结果完全一致。

模型 state hash 不得直接对 torch.save 文件求哈希。必须按 state_dict 键名排序，依次编码键名、dtype、shape 和 contiguous CPU tensor 原始字节，再计算 SHA-256。随机序列检查必须在隔离进程中执行，或完整保存并恢复 RNG 状态，不得消耗正式运行的 RNG 序列。

跨运行配置一致性比较必须先规范化。Plan 1 DQN 运行允许变化的字段仅为：command.network、command.prefix、command.seed、world.combined_file、world.roadnetFile、world.flowFile、world.convertroadnetFile、world.convertflowFile、配置记录时间、运行目录和时间戳。trainer、model、logger schema、interface、delay_type、运行时 hidden_layers、input_dim、action_dim、optimizer 和 loss 必须一致。四场景的运行时 input_dim 或 action_dim 不一致时直接失败，不得加入允许变化列表。哈希文件自身不参与内容一致性比较。

Checkpoint 最小 schema 固定为：schema_version、checkpoint_type、episode、global_decision_step、gradient_updates、config_hash 和 agents。evaluation 类型的每个 agent 至少保存 rank 与 online_model_state_dict；resumable 类型的每个 agent 至少保存 rank、online_model_state_dict、target_model_state_dict、optimizer_state_dict、epsilon 和 replay_state，并在顶层保存 Python random、NumPy、PyTorch CPU 与可用 CUDA RNG 状态以及全部训练计数器。checkpoint_type 只允许 evaluation 或 resumable。建议路径为 checkpoints/evaluation/episode_XXXX.pt 和 checkpoints/resumable/episode_XXXX.pt；若实现采用其他文件名，目录和 checkpoint_type 仍必须明确区分，且不得覆盖旧 model 目录文件。

大型运行产物继续位于被 Git 忽略的 data/output_data/。不得提交模型、LMDB、完整日志或大型 SUMO 产物；应提交足以复跑的命令和小型摘要证据。

## 9. Git 管理

- 从最新 origin/main 建立并使用 codex/milestone0-experiment-infrastructure；
- 一个逻辑功能至少一个可独立回退的提交；
- 每个功能验证通过后立即推送该分支；
- 不 force push，不改写已推送历史，不自动合并 main；
- 推送后发现 bug 时使用新的修复提交；
- 四份规范文档及其同步的 plan0721.md 修改作为规范提交；功能代码使用后续独立提交。规范提交不得混入 agent、trainer、world、run.py 或 utils 等功能代码。

启动 Git 流程固定为：记录当前 HEAD、origin/main 和工作区 diff；执行 git fetch origin；若当前 HEAD 等于最新 origin/main，则在保留工作区修改的前提下创建分支；若不相等，先只读检查当前修改能否干净应用到新基线，有冲突时停止，不自动 stash/pop、rebase、reset、丢弃修改或强制解决。目标分支已存在时，本地与远端历史一致则继续；历史不一致则停止，不删除或强制重建分支。

提交使用中文标题和详细正文，正文必须包含：优化前、优化意图、主要修改、验证、行为边界。

## 10. 暂停与完成

只在以下情况暂停：需要用户决定科研语义；需要新权限；同一阻塞连续三轮；外部依赖持续不可用；用户修改直接冲突；继续执行将启动正式大规模实验或不可恢复操作。

Milestone 0 完成需满足：所有未阻塞验收项均“已通过”；阻塞项均关联科研决策编号；每个功能独立提交并推送；整体回归和最终 diff review 完成；生成总结；不自动合并 main。
