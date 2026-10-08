# Plan1 半离线 DQN 基线（hz1x1 系列）

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：sumohz1x1 系列网 DQN 400ep 正式批建立半离线基线。
- plan_id/队列：plan1_dqn 20260722
- 设计稿/证据位置：data/output_data/tsc/sumo_dqn/sumohz1x1*/p1_*
- 结论：p1_formal 初批 4 个 run 中止后由同身份 *_r2 重跑取代（SUPERSEDED）；pilot_v3 4 run ABORTED（状态残留）；正式批完成。
- 登记单元 35（train/eval/smoke/batch/calibration = 32/3/0/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `p1_formal_dqn_sumohz1x1_config2_seed0_400ep_20260722` | dqn/sumohz1x1_config2/s0 | 179 | SUPERSEDED | 2026-07-22 | 70.7/1382/29 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/p1_formal_dqn_sumohz1x1_config2_` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 179 条; 由同身份重跑 p1_formal_dqn_sumoh |
| `p1_formal_dqn_sumohz1x1_config2_seed0_400ep_20260722_r2` | dqn/sumohz1x1_config2/s0 | 400 | DONE | 2026-07-22 | 70.4/1381/30 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/p1_formal_dqn_sumohz1x1_config2_` |  |
| `p1_formal_dqn_sumohz1x1_config2_seed1_400ep_20260722` | dqn/sumohz1x1_config2/s1 | 400 | DONE | 2026-07-22 | 70.9/1382/29 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/p1_formal_dqn_sumohz1x1_config2_` |  |
| `p1_formal_dqn_sumohz1x1_config2_seed2_400ep_20260722` | dqn/sumohz1x1_config2/s2 | 400 | DONE | 2026-07-22 | 70.6/1380/31 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/p1_formal_dqn_sumohz1x1_config2_` |  |
| `p1_formal_dqn_sumohz1x1_config2_seed3_400ep_20260722` | dqn/sumohz1x1_config2/s3 | 400 | DONE | 2026-07-22 | 69.5/1382/29 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/p1_formal_dqn_sumohz1x1_config2_` |  |
| `p1_formal_dqn_sumohz1x1_config2_seed4_400ep_20260722` | dqn/sumohz1x1_config2/s4 | 400 | DONE | 2026-07-22 | 70.2/1383/28 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/p1_formal_dqn_sumohz1x1_config2_` |  |
| `p1_formal_dqn_sumohz1x1_config3_seed0_400ep_20260722` | dqn/sumohz1x1_config3/s0 | 248 | SUPERSEDED | 2026-07-22 | 65.5/726/17 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/p1_formal_dqn_sumohz1x1_config3_` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 248 条; 由同身份重跑 p1_formal_dqn_sumoh |
| `p1_formal_dqn_sumohz1x1_config3_seed0_400ep_20260722_r2` | dqn/sumohz1x1_config3/s0 | 400 | DONE | 2026-07-22 | 65.3/727/16 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/p1_formal_dqn_sumohz1x1_config3_` |  |
| `p1_formal_dqn_sumohz1x1_config3_seed1_400ep_20260722` | dqn/sumohz1x1_config3/s1 | 400 | DONE | 2026-07-22 | 65.5/727/16 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/p1_formal_dqn_sumohz1x1_config3_` |  |
| `p1_formal_dqn_sumohz1x1_config3_seed2_400ep_20260722` | dqn/sumohz1x1_config3/s2 | 400 | DONE | 2026-07-22 | 65.2/727/16 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/p1_formal_dqn_sumohz1x1_config3_` |  |
| `p1_formal_dqn_sumohz1x1_config3_seed3_400ep_20260722` | dqn/sumohz1x1_config3/s3 | 400 | DONE | 2026-07-22 | 65.2/727/16 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/p1_formal_dqn_sumohz1x1_config3_` |  |
| `p1_formal_dqn_sumohz1x1_config3_seed4_400ep_20260722` | dqn/sumohz1x1_config3/s4 | 400 | DONE | 2026-07-22 | 65.5/727/16 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/p1_formal_dqn_sumohz1x1_config3_` |  |
| `p1_formal_dqn_sumohz1x1_config4_seed0_400ep_20260722` | dqn/sumohz1x1_config4/s0 | 164 | SUPERSEDED | 2026-07-22 | 75.7/1620/51 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/p1_formal_dqn_sumohz1x1_config4_` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 164 条; 由同身份重跑 p1_formal_dqn_sumoh |
| `p1_formal_dqn_sumohz1x1_config4_seed0_400ep_20260722_r2` | dqn/sumohz1x1_config4/s0 | 400 | DONE | 2026-07-22 | 72.8/1620/51 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/p1_formal_dqn_sumohz1x1_config4_` |  |
| `p1_formal_dqn_sumohz1x1_config4_seed1_400ep_20260722` | dqn/sumohz1x1_config4/s1 | 400 | DONE | 2026-07-22 | 73.5/1620/51 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/p1_formal_dqn_sumohz1x1_config4_` |  |
| `p1_formal_dqn_sumohz1x1_config4_seed2_400ep_20260722` | dqn/sumohz1x1_config4/s2 | 400 | DONE | 2026-07-22 | 73.2/1619/52 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/p1_formal_dqn_sumohz1x1_config4_` |  |
| `p1_formal_dqn_sumohz1x1_config4_seed3_400ep_20260722` | dqn/sumohz1x1_config4/s3 | 400 | DONE | 2026-07-22 | 72.5/1619/52 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/p1_formal_dqn_sumohz1x1_config4_` |  |
| `p1_formal_dqn_sumohz1x1_config4_seed4_400ep_20260722` | dqn/sumohz1x1_config4/s4 | 400 | DONE | 2026-07-22 | 74.1/1617/54 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/p1_formal_dqn_sumohz1x1_config4_` |  |
| `p1_formal_dqn_sumohz1x1_seed0_400ep_20260722` | dqn/sumohz1x1/s0 | 139 | SUPERSEDED | 2026-07-22 | 77.8/1973/45 | `data/output_data/tsc/sumo_dqn/sumohz1x1/p1_formal_dqn_sumohz1x1_seed0_400ep_2026` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 139 条; 由同身份重跑 p1_formal_dqn_sumoh |
| `p1_formal_dqn_sumohz1x1_seed0_400ep_20260722_r2` | dqn/sumohz1x1/s0 | 400 | DONE | 2026-07-22 | 77.4/1975/43 | `data/output_data/tsc/sumo_dqn/sumohz1x1/p1_formal_dqn_sumohz1x1_seed0_400ep_2026` |  |
| `p1_formal_dqn_sumohz1x1_seed1_400ep_20260722` | dqn/sumohz1x1/s1 | 400 | DONE | 2026-07-22 | 76.6/1977/41 | `data/output_data/tsc/sumo_dqn/sumohz1x1/p1_formal_dqn_sumohz1x1_seed1_400ep_2026` |  |
| `p1_formal_dqn_sumohz1x1_seed2_400ep_20260722` | dqn/sumohz1x1/s2 | 400 | DONE | 2026-07-22 | 76.4/1977/41 | `data/output_data/tsc/sumo_dqn/sumohz1x1/p1_formal_dqn_sumohz1x1_seed2_400ep_2026` |  |
| `p1_formal_dqn_sumohz1x1_seed3_400ep_20260722` | dqn/sumohz1x1/s3 | 400 | DONE | 2026-07-22 | 77.4/1976/42 | `data/output_data/tsc/sumo_dqn/sumohz1x1/p1_formal_dqn_sumohz1x1_seed3_400ep_2026` |  |
| `p1_formal_dqn_sumohz1x1_seed4_400ep_20260722` | dqn/sumohz1x1/s4 | 400 | DONE | 2026-07-22 | 77.8/1966/52 | `data/output_data/tsc/sumo_dqn/sumohz1x1/p1_formal_dqn_sumohz1x1_seed4_400ep_2026` |  |
| `p1_pilot_dqn_sumohz1x1_config2_seed0_100ep_20260722` | dqn/sumohz1x1_config2/s0 | 100 | DONE | 2026-07-22 | 71.5/1382/29 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/p1_pilot_dqn_sumohz1x1_config2_s` |  |
| `p1_pilot_dqn_sumohz1x1_config3_seed0_100ep_20260722` | dqn/sumohz1x1_config3/s0 | 100 | DONE | 2026-07-22 | 66.0/727/16 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/p1_pilot_dqn_sumohz1x1_config3_s` |  |
| `p1_pilot_dqn_sumohz1x1_config4_seed0_100ep_20260722` | dqn/sumohz1x1_config4/s0 | 100 | DONE | 2026-07-22 | 77.5/1615/56 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/p1_pilot_dqn_sumohz1x1_config4_s` |  |
| `p1_pilot_dqn_sumohz1x1_seed0_100ep_20260722` | dqn/sumohz1x1/s0 | 100 | DONE | 2026-07-22 | 82.1/1972/46 | `data/output_data/tsc/sumo_dqn/sumohz1x1/p1_pilot_dqn_sumohz1x1_seed0_100ep_20260` |  |
| `p1_pilot_v3_dqn_sumohz1x1_config2_seed0_100ep_20260722` | dqn/sumohz1x1_config2/s0 | 5 | ABORTED | 2026-07-22 | 193.0/638/187 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/p1_pilot_v3_dqn_sumohz1x1_config` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 5 条 |
| `p1_pilot_v3_dqn_sumohz1x1_config3_seed0_100ep_20260722` | dqn/sumohz1x1_config3/s0 | 9 | ABORTED | 2026-07-22 | 101.1/702/41 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/p1_pilot_v3_dqn_sumohz1x1_config` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 9 条 |
| `p1_pilot_v3_dqn_sumohz1x1_config4_seed0_100ep_20260722` | dqn/sumohz1x1_config4/s0 | 5 | ABORTED | 2026-07-22 | 226.1/654/224 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/p1_pilot_v3_dqn_sumohz1x1_config` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 5 条 |
| `p1_pilot_v3_dqn_sumohz1x1_seed0_100ep_20260722` | dqn/sumohz1x1/s0 | 4 | ABORTED | 2026-07-22 | 264.0/982/284 | `data/output_data/tsc/sumo_dqn/sumohz1x1/p1_pilot_v3_dqn_sumohz1x1_seed0_100ep_20` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 4 条 |

评估 attempt 共 3 行，按包/来源聚合：

| 评估包/来源 | n DONE/FAILED/ABORTED | 条件集 | eval seeds |
|---|---|---|---|
| `dqn` | 3/0/0 | — | — |
