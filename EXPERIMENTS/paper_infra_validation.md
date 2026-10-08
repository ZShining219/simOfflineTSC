# 论文基础设施验证批次

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：paper_robustness/sumo_events/dashboard 示例等 pytest 级工程验证。
- plan_id/队列：—
- 设计稿/证据位置：tests/ + data/output_data/{paper_robustness,sumo_events,training_dashboard_examples}
- 结论：19 个批次根全部 DONE（内部 run 单元见各批次 notes）。
- 登记单元 19（train/eval/smoke/batch/calibration = 0/0/0/19/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `BATCH/colight_event150` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/paper_robustness/colight_event150` | 工程验证批次：内部 run 单元 173（完成 166，失败 3，其他 4）；pytest/预检类微执行 |
| `BATCH/colight_phase_fix_20260920` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/paper_robustness/colight_phase_fix_20260920` | 工程验证批次：内部 run 单元 20（完成 20，失败 0，其他 0）；pytest/预检类微执行 |
| `BATCH/event_force_injection_20260920` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/paper_robustness/event_force_injection_20260920` | 工程验证批次：内部 run 单元 30（完成 23，失败 0，其他 7）；pytest/预检类微执行 |
| `BATCH/event_gap_fix_20260920` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/paper_robustness/event_gap_fix_20260920` | 工程验证批次：内部 run 单元 38（完成 23，失败 0，其他 15）；pytest/预检类微执行 |
| `BATCH/event_preflight_20260920` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/paper_robustness/event_preflight_20260920` | 工程验证批次：内部 run 单元 63（完成 46，失败 0，其他 17）；pytest/预检类微执行 |
| `BATCH/frap_live_verified` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/training_dashboard_examples/frap_live_verified` | 工程验证批次：内部 run 单元 1（完成 1，失败 0，其他 0）；pytest/预检类微执行 |
| `BATCH/p0_20260920` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/paper_robustness/p0_20260920` | 工程验证批次：内部 run 单元 42（完成 39，失败 3，其他 0）；pytest/预检类微执行 |
| `BATCH/paper_agents_readiness_20260920_v2` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/sumo_events/paper_agents_readiness_20260920_v2` | 工程验证批次：内部 run 单元 4（完成 0，失败 4，其他 0）；pytest/预检类微执行 |
| `BATCH/paper_agents_readiness_20260920_v3` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/sumo_events/paper_agents_readiness_20260920_v3` | 工程验证批次：内部 run 单元 4（完成 4，失败 0，其他 0）；pytest/预检类微执行 |
| `BATCH/reward_monitor_lifecycle_20260920` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/sumo_events/reward_monitor_lifecycle_20260920` | 工程验证批次：内部 run 单元 1（完成 0，失败 1，其他 0）；pytest/预检类微执行 |
| `BATCH/reward_monitor_validation_20260920_v1` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/sumo_events/reward_monitor_validation_20260920_v1` | 工程验证批次：内部 run 单元 4（完成 3，失败 1，其他 0）；pytest/预检类微执行 |
| `BATCH/reward_monitor_validation_20260920_v2` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/sumo_events/reward_monitor_validation_20260920_v2` | 工程验证批次：内部 run 单元 8（完成 7，失败 1，其他 0）；pytest/预检类微执行 |
| `BATCH/reward_monitor_validation_20260920_v3` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/sumo_events/reward_monitor_validation_20260920_v3` | 工程验证批次：内部 run 单元 4（完成 3，失败 1，其他 0）；pytest/预检类微执行 |
| `BATCH/reward_monitor_validation_20260920_v4` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/sumo_events/reward_monitor_validation_20260920_v4` | 工程验证批次：内部 run 单元 4（完成 3，失败 1，其他 0）；pytest/预检类微执行 |
| `BATCH/runner_extension_20260920` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/paper_robustness/runner_extension_20260920` | 工程验证批次：内部 run 单元 114（完成 70，失败 1，其他 43）；pytest/预检类微执行 |
| `BATCH/strong_event150` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/paper_robustness/strong_event150` | 工程验证批次：内部 run 单元 101（完成 93，失败 3，其他 5）；pytest/预检类微执行 |
| `BATCH/strong_event500` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/paper_robustness/strong_event500` | 工程验证批次：内部 run 单元 7（完成 0，失败 2，其他 5）；pytest/预检类微执行 |
| `BATCH/strong_event_pilot_20260920` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/paper_robustness/strong_event_pilot_20260920` | 工程验证批次：内部 run 单元 8（完成 6，失败 0，其他 2）；pytest/预检类微执行 |
| `BATCH/strong_event_stage_a_20260920` | —/hz4x4/s7 | — | DONE | — | — | `data/output_data/paper_robustness/strong_event_stage_a_20260920` | 工程验证批次：内部 run 单元 4（完成 0，失败 0，其他 4）；pytest/预检类微执行 |
