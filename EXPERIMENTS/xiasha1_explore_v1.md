# xiasha1 充分探索验证（native / slow / full epsilon schedule）

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：在相同 xiasha1 需求包、200ep 预算和评估协议下，只改变 DQN epsilon schedule：若充分探索后评估曲线仍早于 epsilon 下限收敛，则早收敛不能只用探索不足解释；若 slow/full 的 final-20 TT 或 uninserted 相对 native 稳定改善，则充分探索提升控制器性能。
- plan_id/队列：xiasha1_explore_v1
- 设计稿/证据位置：docs/experiments/2026-10-21_xiasha1_explore_validation.md；configs/tsc/xiasha1_online/dqn_{slow,full}.yml；2 包（asym_nonly/asym_ns）×3 seeds×2 schedules=12 run；native 参照=xiasha1_online_v1 的 6 个 DONE xi1on_dqn_*；同 200ep+每ep贪心评估+legacy pseudo_step+saveReplay=false。
- 结论：已暂停（2026-10-21）：slow 6 DONE（5% 收敛 ep36–37，仍早于 epsilon 下限 ep118；ΔTT -0.04/-0.07，无稳定提升），full 6 ABORTED（ep143–154）；转 xiasha1_online_v2。
- 登记单元 12（train/eval/smoke/batch/calibration = 12/0/0/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `xi1exp_dqn_full_asym_nonly_s0` | dqn_full/xiasha1_asym_nonly/s0 | 151 | ABORTED | 2026-10-21 | — | `data/output_data/tsc/sumo_dqn/xiasha1_asym_nonly/xi1exp_dqn_full_asym_nonly_s0` | REGISTERED 2026-10-21: 充分探索验证；native 参照=xi1on_dqn_*；slow=epsilon 1.0->0.05 decay |
| `xi1exp_dqn_full_asym_nonly_s1` | dqn_full/xiasha1_asym_nonly/s1 | 151 | ABORTED | 2026-10-21 | — | `data/output_data/tsc/sumo_dqn/xiasha1_asym_nonly/xi1exp_dqn_full_asym_nonly_s1` | REGISTERED 2026-10-21: 充分探索验证；native 参照=xi1on_dqn_*；slow=epsilon 1.0->0.05 decay |
| `xi1exp_dqn_full_asym_nonly_s2` | dqn_full/xiasha1_asym_nonly/s2 | 148 | ABORTED | 2026-10-21 | — | `data/output_data/tsc/sumo_dqn/xiasha1_asym_nonly/xi1exp_dqn_full_asym_nonly_s2` | REGISTERED 2026-10-21: 充分探索验证；native 参照=xi1on_dqn_*；slow=epsilon 1.0->0.05 decay |
| `xi1exp_dqn_full_asym_ns_s0` | dqn_full/xiasha1_asym_ns/s0 | 154 | ABORTED | 2026-10-21 | — | `data/output_data/tsc/sumo_dqn/xiasha1_asym_ns/xi1exp_dqn_full_asym_ns_s0` | REGISTERED 2026-10-21: 充分探索验证；native 参照=xi1on_dqn_*；slow=epsilon 1.0->0.05 decay |
| `xi1exp_dqn_full_asym_ns_s1` | dqn_full/xiasha1_asym_ns/s1 | 147 | ABORTED | 2026-10-21 | — | `data/output_data/tsc/sumo_dqn/xiasha1_asym_ns/xi1exp_dqn_full_asym_ns_s1` | REGISTERED 2026-10-21: 充分探索验证；native 参照=xi1on_dqn_*；slow=epsilon 1.0->0.05 decay |
| `xi1exp_dqn_full_asym_ns_s2` | dqn_full/xiasha1_asym_ns/s2 | 143 | ABORTED | 2026-10-21 | — | `data/output_data/tsc/sumo_dqn/xiasha1_asym_ns/xi1exp_dqn_full_asym_ns_s2` | REGISTERED 2026-10-21: 充分探索验证；native 参照=xi1on_dqn_*；slow=epsilon 1.0->0.05 decay |
| `xi1exp_dqn_slow_asym_nonly_s0` | dqn_slow/xiasha1_asym_nonly/s0 | 200 | DONE | 2026-10-21 | 13.1/1842/5 | `data/output_data/tsc/sumo_dqn/xiasha1_asym_nonly/xi1exp_dqn_slow_asym_nonly_s0` | REGISTERED 2026-10-21: 充分探索验证；native 参照=xi1on_dqn_*；slow=epsilon 1.0->0.05 decay |
| `xi1exp_dqn_slow_asym_nonly_s1` | dqn_slow/xiasha1_asym_nonly/s1 | 200 | DONE | 2026-10-21 | 13.1/1831/6 | `data/output_data/tsc/sumo_dqn/xiasha1_asym_nonly/xi1exp_dqn_slow_asym_nonly_s1` | REGISTERED 2026-10-21: 充分探索验证；native 参照=xi1on_dqn_*；slow=epsilon 1.0->0.05 decay |
| `xi1exp_dqn_slow_asym_nonly_s2` | dqn_slow/xiasha1_asym_nonly/s2 | 200 | DONE | 2026-10-21 | 13.1/1831/6 | `data/output_data/tsc/sumo_dqn/xiasha1_asym_nonly/xi1exp_dqn_slow_asym_nonly_s2` | REGISTERED 2026-10-21: 充分探索验证；native 参照=xi1on_dqn_*；slow=epsilon 1.0->0.05 decay |
| `xi1exp_dqn_slow_asym_ns_s0` | dqn_slow/xiasha1_asym_ns/s0 | 200 | DONE | 2026-10-21 | 11.3/1849/6 | `data/output_data/tsc/sumo_dqn/xiasha1_asym_ns/xi1exp_dqn_slow_asym_ns_s0` | REGISTERED 2026-10-21: 充分探索验证；native 参照=xi1on_dqn_*；slow=epsilon 1.0->0.05 decay |
| `xi1exp_dqn_slow_asym_ns_s1` | dqn_slow/xiasha1_asym_ns/s1 | 200 | DONE | 2026-10-21 | 11.2/1849/6 | `data/output_data/tsc/sumo_dqn/xiasha1_asym_ns/xi1exp_dqn_slow_asym_ns_s1` | REGISTERED 2026-10-21: 充分探索验证；native 参照=xi1on_dqn_*；slow=epsilon 1.0->0.05 decay |
| `xi1exp_dqn_slow_asym_ns_s2` | dqn_slow/xiasha1_asym_ns/s2 | 200 | DONE | 2026-10-21 | 11.3/1849/6 | `data/output_data/tsc/sumo_dqn/xiasha1_asym_ns/xi1exp_dqn_slow_asym_ns_s2` | REGISTERED 2026-10-21: 充分探索验证；native 参照=xi1on_dqn_*；slow=epsilon 1.0->0.05 decay |
