# M1 优化工作关闭记录

## 1. 状态

- 工作项：本轮 M1 优化与科研决策整理
- 状态：已关闭
- 关闭日期：2026-07-22
- 适用分支：`codex/milestone0-experiment-infrastructure`

“已关闭”仅表示本轮优化审核、选项讲解和用户决策同步已经结束，不表示 `docs/plan0721.md` 中的 Plan 0、Plan 1 或任何后续实验已经执行。

## 2. 本轮已完成内容

- 审核并确认 RD-002 至 RD-007；
- 保持现有 learning-start、随机预热、epsilon 衰减和始终 bootstrap 语义；
- 确认 approximate delay 为主、real delay 为辅助；
- 确认 final checkpoint 为主结果，诊断性 best 按最低 average travel time 选择，并列时取较早节点；
- 将 RD-009 跨控制器 reward 口径问题延后到后续实验启动前专项讨论；
- 将上述决定同步到 `research_decision_register.md`、`plan0721.md` 和 `milestone0_summary.md`。

## 3. 明确未执行内容

- 未执行 Plan 0 场景审计和正式配置冻结；
- 未实现或验收 Plan 1 append-only trajectory；
- 未实现 Plan 1 同时记录 approximate/real delay 的正式结果链路；
- 未实现 Plan 1 best checkpoint 自动选择和报告链路；
- 未运行四场景 100 episode Pilot；
- 未运行 20 次正式 Online DQN 训练；
- 未运行 Offline 或 Sequential 实验；
- 未决定 RD-009，也未统一或修改任何控制器 reward。

因此，本记录不能作为 Plan 0/Plan 1 验收、Pilot 通过或正式实验完成的证据。

## 4. 后续重新开启条件

准备启动后续实验时，应重新开启执行工作，并至少完成：

1. 按 `plan0721.md` 执行 Plan 0 审计和配置冻结；
2. 落实并验收 RD-002 至 RD-007 对应的运行、轨迹、指标和 checkpoint 规则；
3. 在实验拉起前专项讨论并决定 RD-009；
4. 满足 Plan 1 启动门后，才允许运行 Pilot；
5. Pilot 验收通过后，才允许正式训练。

## 5. 关联文档

- `docs/plan0721.md`
- `docs/research_decision_register.md`
- `docs/milestone0_summary.md`
- `docs/online_offline_experiment_execution_contract.md`
