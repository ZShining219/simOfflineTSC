# 杂项一次性探针

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：readiness/inspect/frap_colight 适配探针等一次性验证。
- plan_id/队列：—
- 设计稿/证据位置：data/output_data/tsc 下散件
- 结论：1 DONE、5 FAILED、1 SCOPE_DISCARD（inspect_mplight，2026-10-07 裁决：基线已被 baseline_eval 覆盖）。
- 登记单元 7（train/eval/smoke/batch/calibration = 0/0/7/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `frap_adaptation_probe2_20260920` | frap_adaptation_probe2_20260920/hz4x4/s7 | 2 | DONE | 2026-09-20 | 161.1/29/213 | `data/output_data/tsc/sumo_frap/hz4x4/frap_adaptation_probe2_20260920` |  |
| `frap_adaptation_probe_20260920` | frap_adaptation_probe_20260920/hz4x4/s7 | — | FAILED | — | — | `data/output_data/tsc/sumo_frap/hz4x4/frap_adaptation_probe_20260920` |  |
| `inspect_mplight` | inspect_mplight/hz4x4/s7 | — | SCOPE_DISCARD | — | — | `data/output_data/tsc/sumo_mplight/hz4x4/inspect_mplight` | 2026-10-07 裁决：mplight 基线已由 baseline_eval 90 记录覆盖，独立探针废弃 |
| `rewards_dashboard_probe_20260920` | rewards_dashboard_probe_20260920/hz4x4/s7 | 1 | FAILED | 2026-09-20 | 193.9/19/223 | `data/output_data/tsc/sumo_paper_frap/hz4x4/rewards_dashboard_probe_20260920` |  |
| `sumo_colight__hz4x4__readiness` | readiness/hz4x4/s7 | — | FAILED | — | — | `data/output_data/tsc/sumo_colight/hz4x4/readiness` |  |
| `sumo_frap__hz4x4__readiness` | readiness/hz4x4/s7 | — | FAILED | — | — | `data/output_data/tsc/sumo_frap/hz4x4/readiness` |  |
| `test` | test/hz4x4/s0 | — | FAILED | — | — | `data/output_data/tsc/sumo_fixedtime/hz4x4/test` |  |
