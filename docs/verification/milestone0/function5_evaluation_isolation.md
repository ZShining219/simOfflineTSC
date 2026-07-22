# 功能 5：训练与评估隔离验证

- 日期：2026-07-22
- 自动检查：设置 `SUMO_HOME` 后运行全部 Milestone 0 unittest，14 项通过；守卫覆盖无突变通过、`remember` 调用即时失败、模型参数突变失败和 gradient counter 突变失败。
- 保护快照：每个 agent 的 online/target state hash、optimizer state_dict、epsilon、replay 长度，以及 trainer dataset 写入计数、gradient_updates、global_decision_step。
- `remember` 检测：评估期间临时安装阻断包装器；任何调用立即抛错，退出时恢复原绑定。

## 真实 DQN 隔离 smoke

- agent/network/training_seed：`dqn` / `sumohz1x1` / `29`
- episodes/simulation steps：1 / 训练 700 + episode 内评估 100 + 最终评估 100
- 命令：`SUMO_HOME=/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo /home/dev/miniforge3/envs/colight/bin/python3.10 -c '<构造 Runner，run 后逐字段断言 evaluation_isolation_checks>'`
- 退出码：0
- 训练结果：70 decisions、6 gradient updates、replay 长度 70；training travel time 145.652985、reward -63.985714、queue 43.328571、delay 0.473537、throughput 268。
- 两次评估审计：record_type 分别为 `EVALUATION`、`FINAL_EVALUATION`；remember_calls 均为 0；replay_lengths 均为 `[70]`；dataset_writes 均为 0；gradient_updates 均为 6；global_decision_step 均为 70。
- JSONL 顺序为 `TRAIN`、`EVALUATION`、`FINAL_EVALUATION`，两条评估记录均未推进 global/gradient counter。
- 守卫还逐深度比较 online/target、optimizer 和 epsilon；真实运行无突变，否则 Runner 会失败且不能标记“已完成”。
- 产物：`data/output_data/tsc/sumo_dqn/sumohz1x1/m0f5_eval_isolation_20260722/`。

临时 DQN YAML 已恢复；simulator source cfg 哈希不变。Milestone 0 未实现或声称 trajectory transition count 验收，该项保留至 Plan 1。
