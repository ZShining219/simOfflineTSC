# 功能 8：跨运行配置一致性验证

- 日期：2026-07-22
- 工具：`python scripts/compare_run_configs.py <run1> <run2> [...]`；以第一个运行为 reference，兼容返回 0，任何非法差异返回 1 并列出逐字段路径。
- 闭合规范化：只屏蔽 command.network/prefix/seed、world 的 combined/roadnet/flow/convertroadnet/convertflow、CityFlow 运行日志路径、config_record 时间和由 network 派生的 simulator source 元数据；不读取或比较 `config_hashes.json`。
- runtime model：去除 seed 派生的模型 state hash 与随机探针后，仍完整比较 agent 数、rank、hidden layers、model/target 结构、optimizer、loss 和 action_dim；input_dim/action_dim 先做显式硬失败。
- 自动检查：Milestone 0 共 22 项 unittest 通过；允许变化通过；trainer.batch_size 非法变化失败；input_dim mismatch 返回专用错误。
- 同 network 实际验证：`m0f3_same_seed_a_20260722` 与 `m0f3_same_seed_b_20260722` 比较兼容，差异为空。

## 四场景真实 DQN 归档比较

- network：`sumohz1x1`、`sumohz1x1_config2`、`sumohz1x1_config3`、`sumohz1x1_config4`。
- agent/training_seed/episodes/steps：DQN / 37 / 1 / 训练 100 + 最终评估 100。
- prefix：均为 `m0f8_compare_20260722`；四次退出码均为 0，状态均为“已完成”。
- 完整比较命令：`python scripts/compare_run_configs.py data/output_data/tsc/sumo_dqn/sumohz1x1/m0f8_compare_20260722 data/output_data/tsc/sumo_dqn/sumohz1x1_config2/m0f8_compare_20260722 data/output_data/tsc/sumo_dqn/sumohz1x1_config3/m0f8_compare_20260722 data/output_data/tsc/sumo_dqn/sumohz1x1_config4/m0f8_compare_20260722`。
- 结果：总体 `compatible=true`，三个 candidate 的 differences 均为空；四场景 runtime input_dim=16、action_dim=8、hidden_layers=[20,20]，optimizer=RMSprop、loss=MSELoss。
- 关键 smoke 指标：训练 travel time 分别为 67.5556、72.3333、85.0、72.1429；短评估中 config3/config4 尚无完成车辆，travel time=0，与 Function 7 已审计口径一致，不作为性能结论。

## Simulator cfg 规范化

三个 alternate cfg 原先以字符串写 `"True"/"False"` 且缺少描述性 `network`；统一为 JSON boolean 和 `network="hz1x1"`。`world_sumo.py` 对 gui 明确同时识别字符串/boolean，`--no-warnings` 最终也均转为同一字符串，因此该修正不改变 SUMO 命令语义。

四个 source cfg 在各自运行前后哈希一致；临时 DQN YAML 已恢复。比较器没有扩大科研字段 allowlist。
