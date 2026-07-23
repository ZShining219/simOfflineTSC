# 阶段性执行文档

本目录用于存放与用户讨论后形成的 plan、goal，以及其他非永久但需要在任务执行期间持续参考的文件。

此类文件应清晰记录适用任务、目标、执行依据和当前状态。内容失效后应及时更新或清理，避免将过期信息继续作为执行依据。

## 当前状态索引

| 内容 | 状态 | 说明 |
|---|---|---|
| Milestone 0 公共实验基础设施 | 已完成并归档 | 实现、回归和验收已完成，见 `done/milestone0_summary.md` |
| 本轮 M1 优化与科研决策整理 | 已关闭并归档 | 仅表示本轮规则审核和决策确认结束，见 `done/m1_optimization_closure.md` |
| Plan 0 环境、算法和场景冻结 | 未执行 | `plan0721.md` 中的后续执行计划，不得因 M1 优化关闭而视为完成 |
| Plan 1 启动能力与 Pilot | 已完成 | append-only trajectory、固定评估/checkpoint 和作图能力已通过，见 `plan1_support_and_pilot.md` |
| Plan 1 四场景正式 Online DQN | 已通过 | 20 次正式运行和 Plan 2 trajectory 白名单均已验收，见 `plan1_formal_online_dqn.md` |
| Plan 2 纯 Offline DQN 工程支持 | 已实现，正式实验未启动 | 算法梯度、schema v2、resume、SUMO 清理、严格汇总及 8/16 并发 I/O 门禁已通过，见 `plan2_offline_support.md` |
| Plan 3～Plan 4 | 未执行 | 继续受 `plan0721.md` 的依赖关系和启动门约束 |

状态解释：

- “已关闭”用于表示一次讨论、审核或优化工作不再继续扩展；
- “已完成”只用于已经实现并通过验收的里程碑；
- “未执行”表示文档仍是计划或执行依据，不能作为已产生实验结果的证据。

## 目录约定

- `docs/` 根目录只保留仍在使用、尚未执行或需要继续维护的 plan、goal 和执行依据；
- 已完成或已关闭、且不再作为当前活动执行入口的材料统一移动到 `docs/done/`；
- 归档时必须同步更新仓库内引用，保留原有子目录结构，避免证据链断裂；
- 如果归档内容需要重新执行，应先恢复为活动状态并移出 `docs/done/`，不能直接把归档记录当作新的执行依据。

## 主要文档

- `plan0721.md`：Online、Offline 和 Sequential 实验总体计划及持续状态；
- `plan2_offline_support.md`：Plan 2 工程架构、入口隔离、数据/训练/恢复/汇总协议和验证记录；
- `done/research_decision_register.md`：已归档的科研语义决定及其状态；
- `done/online_offline_experiment_execution_contract.md`：已归档的 Milestone 0 实施与语义保护契约；
- `done/milestone0_summary.md`：Milestone 0 已完成内容和后续进入条件；
- `done/m1_optimization_closure.md`：本轮 M1 优化关闭边界；
- `done/verification/milestone0/`：Milestone 0 功能验证证据。
