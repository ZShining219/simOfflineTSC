# Codex CLI：Milestone 0 执行 Prompt

将下方“任务正文”整体交给 Codex CLI。执行前确认当前目录为本仓库根目录，且可以访问 origin 远端和 colight 环境。

## 任务正文

目标：

按照以下文件完成 Milestone 0：公共实验基础设施：

- AGENTS.md
- docs/done/online_offline_experiment_execution_contract.md
- docs/done/milestone0_acceptance_checklist.md
- docs/done/research_decision_register.md
- docs/plan0721.md

这些文件共同构成执行依据。严格遵守其中的优先级、科研语义保护门、验证要求和 Git 规则。不要重新设计已确认规则。

背景与边界：

1. 本项目基于 LibSignal；目标是在尽量保留现有 Online DQN 数值语义的前提下，为后续 Online、Offline 和 Sequential 实验建立可靠基础设施。
2. S1–S4 只用于文档说明。所有实际执行、目录和记录使用项目现有 network 名称。
3. 不增加 run.py 配置路径参数；继续使用 configs/tsc/base.yml 与 configs/tsc/{agent}.yml。
4. 后续未限定的 seed 均为 training_seed，即 run.py --seed。不得新增或管理 SUMO seed；只记录 sumo_seed_mode=fixed_default。
5. 不运行 100 episode Pilot、20 次正式训练、Offline 训练或 Sequential 正式实验。
6. 不自动合并 main。

启动与 Git 要求：

1. 检查工作区、当前分支、origin/main 和现有用户修改。
2. 当前若存在已确认但尚未提交的运行时模型归档修改，必须保护这些修改；不得丢弃或覆盖。
3. 从最新 origin/main 创建并切换到 codex/milestone0-experiment-infrastructure。若需要在保留工作区修改的情况下切换，先做只读检查并采用不会丢失修改的方法。
4. 先将四份 Milestone 0 执行材料以及为同步已确认规则而产生的 docs/plan0721.md 修改作为独立规范提交，提交标题建议为“建立Milestone 0实验基础设施执行规范”，验证并推送。该提交不得包含 agent、trainer、world、run.py 或 utils 等功能代码。
5. 再把现有配置与运行时模型归档作为第一个功能单元重新审核、测试、提交和推送。
6. 一个逻辑功能至少一个独立提交。每个功能完成后立即推送该分支。不 force push，不修改已推送历史。

Git 启动步骤不得自行简化：先记录当前 HEAD、origin/main 和工作区 diff，再执行 git fetch origin。将分支创建时最新 origin/main 完整 SHA 记录为 baseline_commit。当前 HEAD 与最新 origin/main 不一致时，先只读检查现有修改能否干净应用；有冲突则停止，不自动 stash/pop、rebase、reset、丢弃修改或强制解决。目标分支已存在时，只有本地与远端历史一致才继续，否则停止。

执行方法：

依次完成实施契约第 4 节列出的九个功能单元。每个功能开始时先输出：

    功能：
    当前项目行为：
    Plan要求：
    本次差距：
    允许修改：
    禁止修改：
    验证方案：
    预期产物：
    是否存在科研决策阻塞：

没有科研决策阻塞时，不等待逐项批准，连续完成：

    审核现状
    → 明确优化意图和必须保持的不变量
    → 实现
    → 针对性自动检查
    → 真实最小 SUMO smoke
    → 检查实际生成内容
    → 主动查找 bug、边界错误和回归
    → 修复并重新验证
    → 检查完整 diff 和工作区
    → 更新验收清单与科研决策表
    → 中文功能提交
    → 推送远端分支
    → 确认本地提交与远端一致
    → 进入下一未阻塞功能

科研决策规则：

1. 将发现严格分类为：原项目既有行为、Plan 要求的新能力、行为保持型工程优化、需要用户确认的科研语义。
2. 对 Plan 新能力和行为保持型工程优化自主实施。
3. 不得自行改变 state、reward、action、DQN loss、terminal/truncated bootstrap、epsilon、learning-start、更新频率、replay sampling、指标公式或实验预算。
4. 发现上述问题时，在 docs/done/research_decision_register.md 使用中文字段和中文状态登记，暂停依赖部分，继续其他不受影响的功能。
5. 不得用“推荐方案”代替“用户决定”。
6. 已决定采用 checkpoint B 方案：evaluation checkpoint 保存并评估 online Q-network；resumable checkpoint 保存 online、target、optimizer、epsilon、训练计数器、随机状态和恢复所需 replay 状态。不得继续用旧 target-only 文件代表新的 evaluation checkpoint。

验证要求：

1. 每个功能至少有针对性自动检查。纯文档或只读校验器不强制 SUMO；影响 Runner、Trainer、Agent、World、日志或 checkpoint 时至少有一个相关真实 SUMO smoke；最终回归必须覆盖 FixedTime、MaxPressure 和 DQN。
2. smoke 必须记录 agent、network、training_seed、episodes、simulation steps、完整命令、退出码、关键指标、产物路径、字段/哈希校验和已知警告。
3. 不能只看退出码，必须读取和断言配置、日志、checkpoint、状态或其他目标产物内容。
4. 验证相关失败路径，如重复 prefix、缺失字段、损坏哈希、失败状态或恢复错误。
5. 临时修改 dqn.yml 缩短 smoke 时，结束后必须恢复。在功能 1 完成前复现原项目行为时，若运行重写 simulator cfg，也必须恢复该基线测试副作用；功能 1 完成后，任何运行再次修改 configs/sim 源 cfg 都应判定为验收失败，而不是依赖事后恢复。
6. 执行适用的 Python 语法/导入检查、git diff --check 和完整 diff review。
7. 不提交 data/output_data、模型、LMDB、完整日志或大型实验产物；提交可复跑命令、小型测试和必要摘要。
8. 任一验收失败不得进入下一依赖功能。可安全修复时继续修复；同一阻塞连续三轮仍无法解决时才报告。

固定 smoke profile：基础启动使用 1 episode、100 simulation steps；训练更新 smoke 必须保证 replay_size 不小于 batch_size，并断言至少 1 次 gradient update 和 1 次 target update；checkpoint smoke 必须实际保存、加载并继续至少 1 次 optimizer update；FixedTime 和 MaxPressure 最终回归各使用 1 episode、3600 simulation steps；DQN 使用受控短预算覆盖真实更新，不执行 Pilot。

“一轮修复”是针对同一根因提出一个实质不同方案、完成实现并执行足以验证该根因的复验。重复同一失败命令或外部状态未变化的重试不计为新一轮。

功能 1 必须移除对共享 configs/sim 源 cfg 的运行时写回：基于源 cfg 生成运行目录内的 resolved simulator cfg，并让项目/SUMO 使用该运行专属文件。验证源 cfg 运行前后哈希一致，并验证不同 prefix 并发解析不串扰。

运行状态、结构化日志、评估隔离、可重复性和跨运行配置比较必须逐字段遵守实施契约中的固定 schema 与允许变化列表，不得自行缩减字段或扩大允许差异。Milestone 0 只验证评估不调用 remember、replay/dataset 写入计数不变；正式 trajectory transition count 留到 Plan 1。

提交要求：

每个功能提交使用中文标题和详细正文，正文包含：

    优化前：
    优化意图：
    主要修改：
    验证：
    行为边界：

提交后报告短哈希、标题、验证命令、关键结果和推送目标。需要补充上一提交哈希时，创建只更新证据的记录同步提交，不混入新功能代码。

暂停条件：

只在以下情况暂停：需要用户决定科研语义；需要新权限；同一阻塞连续三轮；外部依赖持续不可用；现有用户修改与目标功能直接冲突；继续执行将启动正式大规模实验或不可恢复操作。无关科研决策不得阻塞其他功能。

完成条件：

1. docs/done/milestone0_acceptance_checklist.md 中所有未阻塞项均为“已通过”，且有命令、结果、产物和提交证据。
2. 被阻塞项均关联明确的科研决策编号。
3. 每个功能均已独立提交并推送 codex/milestone0-experiment-infrastructure。
4. FixedTime、MaxPressure 和 DQN 整体回归通过。
5. 无临时 YAML、simulator cfg 或非目标工作区变化。
6. 完成最终 diff review 和 Milestone 0 总结。
7. 不自动合并 main，不启动后续 Pilot 或正式实验。

最终报告必须包含：

- 原项目差距与其代码证据；
- 各功能实现及其意图；
- 是否改变算法语义；
- 验证命令和结果；
- 每个提交及远端分支；
- 已知警告与剩余风险；
- 待用户决定的科研事项；
- 下一 Milestone 的进入条件。
