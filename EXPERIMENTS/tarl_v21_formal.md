# TARL v21 正式批

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：v21 修正后 TARL 正式对照批。
- plan_id/队列：reproduction/tarl_tsc v21 formal
- 设计稿/证据位置：reproduction/tarl_tsc/scripts/*v21*
- 结论：15 个正式训练 run 全部 DONE。
- 登记单元 15（train/eval/smoke/batch/calibration = 15/0/0/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `tarl_v21_formal_20260915_attention_seed17` | tarl_v21_formal_20260915_attention_seed17/hz4x4/s17 | 300 | DONE | 2026-09-15 | 331.6/2740/243 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v21_formal_20260915_attentio` |  |
| `tarl_v21_formal_20260915_attention_seed27` | tarl_v21_formal_20260915_attention_seed27/hz4x4/s27 | 300 | DONE | 2026-09-15 | 331.5/2737/246 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v21_formal_20260915_attentio` |  |
| `tarl_v21_formal_20260915_attention_seed37` | tarl_v21_formal_20260915_attention_seed37/hz4x4/s37 | 300 | DONE | 2026-09-15 | 332.0/2740/243 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v21_formal_20260915_attentio` |  |
| `tarl_v21_formal_20260915_attention_seed47` | tarl_v21_formal_20260915_attention_seed47/hz4x4/s47 | 300 | DONE | 2026-09-15 | 331.2/2733/250 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v21_formal_20260915_attentio` |  |
| `tarl_v21_formal_20260915_attention_seed7` | tarl_v21_formal_20260915_attention_seed7/hz4x4/s7 | 300 | DONE | 2026-09-15 | 332.6/2748/235 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v21_formal_20260915_attentio` |  |
| `tarl_v21_formal_20260915_gat_seed17` | tarl_v21_formal_20260915_gat_seed17/hz4x4/s17 | 300 | DONE | 2026-09-15 | 330.7/2740/243 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v21_formal_20260915_gat_seed17` |  |
| `tarl_v21_formal_20260915_gat_seed27` | tarl_v21_formal_20260915_gat_seed27/hz4x4/s27 | 300 | DONE | 2026-09-15 | 331.7/2736/247 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v21_formal_20260915_gat_seed27` |  |
| `tarl_v21_formal_20260915_gat_seed37` | tarl_v21_formal_20260915_gat_seed37/hz4x4/s37 | 300 | DONE | 2026-09-15 | 332.2/2739/244 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v21_formal_20260915_gat_seed37` |  |
| `tarl_v21_formal_20260915_gat_seed47` | tarl_v21_formal_20260915_gat_seed47/hz4x4/s47 | 300 | DONE | 2026-09-15 | 332.1/2735/248 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v21_formal_20260915_gat_seed47` |  |
| `tarl_v21_formal_20260915_gat_seed7` | tarl_v21_formal_20260915_gat_seed7/hz4x4/s7 | 300 | DONE | 2026-09-15 | 331.8/2744/239 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v21_formal_20260915_gat_seed7` |  |
| `tarl_v21_formal_20260915_gating_seed17` | tarl_v21_formal_20260915_gating_seed17/hz4x4/s17 | 300 | DONE | 2026-09-15 | 330.9/2738/245 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v21_formal_20260915_gating_seed` |  |
| `tarl_v21_formal_20260915_gating_seed27` | tarl_v21_formal_20260915_gating_seed27/hz4x4/s27 | 300 | DONE | 2026-09-15 | 331.7/2739/244 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v21_formal_20260915_gating_seed` |  |
| `tarl_v21_formal_20260915_gating_seed37` | tarl_v21_formal_20260915_gating_seed37/hz4x4/s37 | 300 | DONE | 2026-09-15 | 331.5/2736/247 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v21_formal_20260915_gating_seed` |  |
| `tarl_v21_formal_20260915_gating_seed47` | tarl_v21_formal_20260915_gating_seed47/hz4x4/s47 | 300 | DONE | 2026-09-15 | 332.0/2736/247 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v21_formal_20260915_gating_seed` |  |
| `tarl_v21_formal_20260915_gating_seed7` | tarl_v21_formal_20260915_gating_seed7/hz4x4/s7 | 300 | DONE | 2026-09-15 | 332.8/2747/236 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v21_formal_20260915_gating_seed` |  |
