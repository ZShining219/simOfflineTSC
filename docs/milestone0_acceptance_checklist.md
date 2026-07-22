# Milestone 0 验收清单

## 1. 使用规则

本清单是 Milestone 0 的完成门。每项必须填写状态、验证命令、关键结果、关联提交和备注。

允许的状态只有：

- 未开始
- 进行中
- 已通过
- 未通过
- 被科研决策阻塞
- 不适用

不能只填写“已通过”而不提供证据。真实 smoke 必须记录 agent、network、training_seed、episodes 和 simulation steps。失败后修复的项目应保留简短失败原因与复验结果。

大型原始产物不提交 Git；证据应包含可复跑命令、关键结果和相对产物路径。需要长期保存的小型摘要放入 docs/verification/milestone0/。

## 2. 功能验收

| 编号 | 验收项 | 状态 | 验证命令 | 关键结果/证据 | 关联提交 | 备注 |
|---|---|---|---|---|---|---|
| M0-01 | 源 YAML、simulator source/resolved 和最终配置自动归档 | 已通过 | 见 `docs/verification/milestone0/function1_config_archive.md` | FixedTime、DQN 真实产物含全部配置快照 | 本功能提交 |  |
| M0-02 | 运行时模型、target、optimizer、loss 或传统控制器参数被实际对象归档 | 已通过 | 同上 | DQN 16→20→20→8/RMSprop/MSE；FixedTime `t_fixed=30` | 本功能提交 |  |
| M0-03 | 配置与运行时描述均进入 SHA-256 清单并通过回读校验 | 已通过 | unittest + 真实产物 `verify_config_archive` | 正常、损坏和缺失文件路径均验证 | 本功能提交 |  |
| M0-04 | 相同 network 与 prefix 不会静默覆盖或混写 | 已通过 | unittest `test_duplicate_prefix_is_rejected` | 明确要求使用新 prefix | 本功能提交 |  |
| M0-04A | configs/sim 源 cfg 运行前后哈希一致，SUMO 使用运行专属 resolved cfg | 已通过 | unittest 并发检查 + `sha256sum configs/sim/sumohz1x1.cfg` | 两 prefix 隔离；前后均为 `314f…9dbd` | 本功能提交 |  |
| M0-04B | baseline_commit 被固定为分支创建时最新 origin/main 完整 SHA | 已通过 | `git rev-parse` + 实施契约 | `73d860bb3924ec15c30433a8f8b7af17787baeff` | `1eb3bd9` | 后续 origin/main 更新不改变本次基线 |
| M0-05 | 运行身份、开始/结束时间、状态和失败原因可追溯 | 已通过 | 见 `docs/verification/milestone0/function2_run_state.md` | 完成与失败真实 Runner 路径均逐字段读取断言 | 本功能提交 |  |
| M0-06 | 失败运行不会被误判为正式完成结果 | 已通过 | 受控任务异常 smoke + 状态机 unittest | 失败状态、非零退出码和异常类型明确；终态不可转完成 | 本功能提交 |  |
| M0-07 | README 明确未限定 seed 即 training_seed，且不管理 SUMO seed | 未开始 |  |  |  | 对应决策 RD-001 |
| M0-08 | 运行产物记录 training_seed 与 sumo_seed_mode=fixed_default | 未开始 |  |  |  | 不增加 SUMO seed 参数 |
| M0-09 | 同 training_seed 的初始化模型和关键随机边界可检查 | 未开始 |  |  |  | 不承诺不同硬件完全位级一致 |
| M0-10 | TRAIN 与 TEST 指标具有机器可读、来源明确的结构化日志 | 未开始 |  |  |  | 保留现有文本日志兼容性 |
| M0-11 | 结构化日志明确 episode、decision/global counter 和单位 | 未开始 |  |  |  | 不改变训练调度 |
| M0-12 | 评估不改变 online/target、optimizer、epsilon、replay 和 gradient counter | 未开始 |  |  |  | 依赖项必须做评估前后快照检查 |
| M0-13 | 评估不调用 remember，且 replay 长度和当前 dataset 写入计数不变 | 未开始 |  |  |  | 正式 trajectory transition count 留到 Plan 1 验收 |
| M0-14 | evaluation checkpoint 保存并评估 online Q-network；resumable checkpoint 语义明确 | 未开始 |  |  |  | 对应决策 RD-008 |
| M0-15 | resumable checkpoint 保存恢复所需训练状态并可做最小恢复验证 | 未开始 |  |  |  | 不改变当前更新算法 |
| M0-16 | 当前 travel time、delay、queue、throughput、reward 口径有代码证据 | 未开始 |  |  |  | 缺失或歧义登记科研决策表 |
| M0-17 | 未经用户决定不修改指标公式或主指标口径 | 未开始 |  |  |  |  |
| M0-18 | 跨 network/seed 配置一致性工具按契约允许字段规范化比较 | 未开始 |  |  |  | input_dim/action_dim 不一致直接失败 |
| M0-19 | FixedTime 最小真实 SUMO smoke 通过 | 未开始 |  |  |  | 记录完整命令和指标 |
| M0-20 | MaxPressure 最小真实 SUMO smoke 通过 | 未开始 |  |  |  | 记录完整命令和指标 |
| M0-21 | DQN 最小真实 SUMO smoke 通过 | 未开始 |  |  |  | 临时 YAML 必须恢复 |
| M0-22 | 全部针对性语法、导入、哈希和失败路径检查通过 | 未开始 |  |  |  |  |
| M0-23 | smoke 后无临时 YAML、simulator cfg 或其他非目标变化 | 未开始 |  |  |  |  |
| M0-24 | 每个功能均有独立中文提交并已推送指定远端分支 | 未开始 |  |  |  | 不 force push |
| M0-25 | Milestone 0 最终 diff review 和整体回归通过 | 未开始 |  |  |  | 不自动合并 main |

## 3. 固定验收语义

- run_manifest.json 与 run_status.json 必须满足实施契约规定的字段、状态机和原子写入要求；
- 训练更新 smoke 必须断言 gradient_updates 大于 0，并至少发生一次 target update；
- checkpoint 恢复 smoke 必须实际加载并继续至少一次 optimizer update；
- 纯文档或只读校验器不强制 SUMO；影响运行链路的功能必须执行相关真实 SUMO smoke；
- 某项被科研决策阻塞但继续其他功能时，必须在备注中写明依赖判断和决策编号。

## 4. 单次 smoke 证据模板

    日期：
    功能编号：
    agent：
    network：
    training_seed：
    episodes：
    simulation steps：
    完整命令：
    退出码：
    关键指标：
    关键产物相对路径：
    字段/哈希校验结果：
    已知警告：
    失败与复验记录：
    关联提交：
    远端分支：codex/milestone0-experiment-infrastructure

## 5. 提交同步规则

当前功能的测试与文档证据应随功能提交更新。提交哈希可在紧随其后的“记录同步提交”中补充；记录同步提交只能更新证据，不混入下一功能代码。任一验收项失败时不得进入下一依赖功能。
