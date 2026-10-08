# Milestone0 实验基础设施烟测

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：m0f*/verify_sumo 系列验证 run.py+sumo+dqn 链路可用。
- plan_id/队列：milestone0
- 设计稿/证据位置：docs/plan.md
- 结论：12 个 smoke 全部 DONE。
- 登记单元 12（train/eval/smoke/batch/calibration = 0/0/12/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `m0f3_same_seed_a_20260722` | dqn/sumohz1x1/s19 | — | DONE | 2026-07-22 | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/m0f3_same_seed_a_20260722` |  |
| `m0f3_same_seed_b_20260722` | dqn/sumohz1x1/s19 | — | DONE | 2026-07-22 | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/m0f3_same_seed_b_20260722` |  |
| `m0f4_dqn_counter_assert_20260722` | dqn/sumohz1x1/s23 | 1 | DONE | 2026-07-22 | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/m0f4_dqn_counter_assert_20260722` |  |
| `m0f4_dqn_updates_20260722` | dqn/sumohz1x1/s23 | 1 | DONE | 2026-07-22 | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/m0f4_dqn_updates_20260722` |  |
| `m0f5_eval_isolation_20260722` | dqn/sumohz1x1/s29 | 1 | DONE | 2026-07-22 | 63.1/9/— | `data/output_data/tsc/sumo_dqn/sumohz1x1/m0f5_eval_isolation_20260722` |  |
| `m0f6_checkpoint_resume_20260722` | dqn/sumohz1x1/s31 | 1 | DONE | 2026-07-22 | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/m0f6_checkpoint_resume_20260722` |  |
| `m0f6_checkpoint_resume_v2_20260722` | dqn/sumohz1x1/s31 | 1 | DONE | 2026-07-22 | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/m0f6_checkpoint_resume_v2_20260722` |  |
| `m0f9_dqn_updates_20260722` | dqn/sumohz1x1/s41 | 1 | DONE | 2026-07-22 | 66.9/7/— | `data/output_data/tsc/sumo_dqn/sumohz1x1/m0f9_dqn_updates_20260722` |  |
| `sumo_dqn__sumohz1x1__m0f8_compare_20260722` | dqn/sumohz1x1/s37 | 1 | DONE | 2026-07-22 | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/m0f8_compare_20260722` | alias→sumo_dqn__sumohz1x1_config3__m0f8_compar; 同名隔离/残留副本，逻辑同一 run→sumo_dqn__sumohz1x1_config3__m0f8_compare_20260722 |
| `sumo_dqn__sumohz1x1_config2__m0f8_compare_20260722` | dqn/sumohz1x1_config2/s37 | 1 | DONE | 2026-07-22 | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/m0f8_compare_20260722` | alias→sumo_dqn__sumohz1x1_config3__m0f8_compar; 同名隔离/残留副本，逻辑同一 run→sumo_dqn__sumohz1x1_config3__m0f8_compare_20260722 |
| `sumo_dqn__sumohz1x1_config3__m0f8_compare_20260722` | dqn/sumohz1x1_config3/s37 | 1 | DONE | 2026-07-22 | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/m0f8_compare_20260722` |  |
| `sumo_dqn__sumohz1x1_config4__m0f8_compare_20260722` | dqn/sumohz1x1_config4/s37 | 1 | DONE | 2026-07-22 | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/m0f8_compare_20260722` | alias→sumo_dqn__sumohz1x1_config3__m0f8_compar; 同名隔离/残留副本，逻辑同一 run→sumo_dqn__sumohz1x1_config3__m0f8_compare_20260722 |
