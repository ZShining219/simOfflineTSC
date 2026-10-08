# ATT-ENTITY-002 SGA/CoLight 文本事件 200ep 矩阵

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：SGA 变体（sga_colight/concat_colight/sga_mplight）注入事件文本应优于无文本基线（event vs normal 条件）。
- plan_id/队列：sga_concat_colight_200_v1 eval_v1/eval_v2
- 设计稿/证据位置：artifacts/sga_concat_colight_200_v1/run_queue.json + eval_v1/eval_v2 包目录
- 结论：wiki review v4 完成；30 个训练计划位中 24 个有效训练（部分种子中止/失败留档），48 评估包；H2 获支持、H3 未获支持；含闭环 grid、反事实、v2 回归等探针。
- 登记单元 2291（train/eval/smoke/batch/calibration = 59/2200/32/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `colight_event_smoke_s17` | colight_event_smoke_s17/hz4x4/s17 | 2 | DONE | 2026-09-23 | — | `data/output_data/tsc/sumo_colight/hz4x4/colight_event_smoke_s17` |  |
| `colight_event_smoke_v1` | colight_event_smoke_v1/hz4x4/s7 | 2 | DONE | 2026-09-23 | — | `data/output_data/tsc/sumo_colight/hz4x4/colight_event_smoke_v1` |  |
| `concat_colight_text_event_smoke_v1` | concat_colight_text_event_smoke_v1/hz4x4/s7 | 2 | DONE | 2026-09-24 | 186.9/15/227 | `data/output_data/tsc/sumo_concat_colight/hz4x4/concat_colight_text_event_smoke_v` |  |
| `concat_colight_text_normal_smoke_v1` | concat_colight_text_normal_smoke_v1/hz4x4/s7 | 2 | DONE | 2026-09-24 | — | `data/output_data/tsc/sumo_concat_colight/hz4x4/concat_colight_text_normal_smoke_` |  |
| `mplight_event_smoke` | mplight_event_smoke/hz4x4/s7 | — | FAILED | 2026-09-23 | — | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_smoke` |  |
| `mplight_event_smoke_s17` | mplight_event_smoke_s17/hz4x4/s17 | 2 | DONE | 2026-09-23 | — | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_smoke_s17` |  |
| `mplight_event_smoke_v10` | mplight_event_smoke_v10/hz4x4/s7 | 2 | DONE | 2026-09-23 | — | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_smoke_v10` |  |
| `mplight_event_smoke_v2` | mplight_event_smoke_v2/hz4x4/s7 | — | FAILED | 2026-09-23 | — | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_smoke_v2` |  |
| `mplight_event_smoke_v3` | mplight_event_smoke_v3/hz4x4/s7 | — | FAILED | 2026-09-23 | — | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_smoke_v3` |  |
| `mplight_event_smoke_v4` | mplight_event_smoke_v4/hz4x4/s7 | — | FAILED | 2026-09-23 | — | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_smoke_v4` |  |
| `mplight_event_smoke_v5` | mplight_event_smoke_v5/hz4x4/s7 | — | FAILED | 2026-09-23 | — | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_smoke_v5` |  |
| `mplight_event_smoke_v6` | mplight_event_smoke_v6/hz4x4/s7 | 2 | FAILED | 2026-09-23 | — | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_smoke_v6` |  |
| `mplight_event_smoke_v7` | mplight_event_smoke_v7/hz4x4/s7 | 2 | FAILED | 2026-09-23 | — | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_smoke_v7` |  |
| `mplight_event_smoke_v8` | mplight_event_smoke_v8/hz4x4/s7 | 2 | FAILED | 2026-09-23 | — | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_smoke_v8` |  |
| `mplight_event_smoke_v9` | mplight_event_smoke_v9/hz4x4/s7 | 2 | FAILED | 2026-09-23 | — | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_smoke_v9` |  |
| `sga_colight_event_smoke` | sga_colight_event_smoke/hz4x4/s7 | 2 | DONE | 2026-09-23 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_event_smoke` |  |
| `sga_colight_event_smoke_s17` | sga_colight_event_smoke_s17/hz4x4/s17 | 2 | DONE | 2026-09-23 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_event_smoke_s17` |  |
| `sga_colight_event_smoke_v2` | sga_colight_event_smoke_v2/hz4x4/s7 | 2 | DONE | 2026-09-23 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_event_smoke_v2` |  |
| `sga_colight_text_event_smoke_v3` | sga_colight_text_event_smoke_v3/hz4x4/s7 | 2 | DONE | 2026-09-24 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_event_smoke_v3` |  |
| `sga_colight_text_log_smoke_s7` | sga_colight_text_log_smoke_s7/hz4x4/s7 | 2 | DONE | 2026-09-24 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_log_smoke_s7` |  |
| `sga_colight_text_normal_smoke_v1` | sga_colight_text_normal_smoke_v1/hz4x4/s7 | — | FAILED | 2026-09-24 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_normal_smoke_v1` |  |
| `sga_colight_text_normal_smoke_v2` | sga_colight_text_normal_smoke_v2/hz4x4/s7 | 2 | DONE | 2026-09-24 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_normal_smoke_v2` |  |
| `sga_colight_text_reset_smoke` | sga_colight_text_reset_smoke/hz4x4/s7 | — | FAILED | 2026-09-24 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_reset_smoke` |  |
| `sga_colight_text_reset_smoke_v2` | sga_colight_text_reset_smoke_v2/hz4x4/s7 | 2 | DONE | 2026-09-24 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_reset_smoke_v2` |  |
| `sga_event_evalfix_smoke` | sga_event_evalfix_smoke/hz4x4/s7 | — | FAILED | — | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_event_evalfix_smoke` |  |
| `sga_event_evalfix_smoke_v2` | sga_event_evalfix_smoke_v2/hz4x4/s7 | — | FAILED | 2026-09-24 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_event_evalfix_smoke_v2` |  |
| `sga_event_evalfix_smoke_v3` | sga_event_evalfix_smoke_v3/hz4x4/s7 | — | FAILED | 2026-09-24 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_event_evalfix_smoke_v3` |  |
| `sga_event_evalfix_smoke_v4` | sga_event_evalfix_smoke_v4/hz4x4/s7 | 2 | DONE | 2026-09-24 | 197.3/13/229 | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_event_evalfix_smoke_v4` |  |
| `sga_mplight_event_smoke_s17` | sga_mplight_event_smoke_s17/hz4x4/s17 | 2 | DONE | 2026-09-23 | — | `data/output_data/tsc/sumo_sga_mplight/hz4x4/sga_mplight_event_smoke_s17` |  |
| `sga_mplight_event_smoke_s7` | sga_mplight_event_smoke_s7/hz4x4/s7 | — | FAILED | 2026-09-23 | — | `data/output_data/tsc/sumo_sga_mplight/hz4x4/sga_mplight_event_smoke_s7` |  |
| `sga_mplight_event_smoke_s7v2` | sga_mplight_event_smoke_s7v2/hz4x4/s7 | 2 | DONE | 2026-09-23 | — | `data/output_data/tsc/sumo_sga_mplight/hz4x4/sga_mplight_event_smoke_s7v2` |  |
| `sga_sigmoid_text_smoke_s7` | sga_sigmoid_text_smoke_s7/hz4x4/s7 | 2 | DONE | 2026-09-24 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_sigmoid_text_smoke_s7` |  |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | colight_event/hz4x4/s27 | 20 | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/colight_ev` | alias→sumo_colight__hz4x4__colight_event_200_s; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_colight__hz4x4__colight_event_200_s27 |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | colight_event/hz4x4/s37 | 1 | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/colight_ev` | alias→sumo_colight__hz4x4__colight_event_200_s; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_colight__hz4x4__colight_event_200_s37 |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | colight_event/hz4x4/s47 | 1 | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/colight_ev` | alias→sumo_colight__hz4x4__colight_event_200_s; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_colight__hz4x4__colight_event_200_s47 |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | colight_normal/hz4x4/s27 | 19 | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/colight_no` | alias→sumo_colight__hz4x4__colight_normal_200_; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_colight__hz4x4__colight_normal_200_s27 |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | colight_normal/hz4x4/s37 | 1 | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/colight_no` | alias→sumo_colight__hz4x4__colight_normal_200_; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_colight__hz4x4__colight_normal_200_s37 |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | colight_normal/hz4x4/s47 | 0 | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/colight_no` | alias→sumo_colight__hz4x4__colight_normal_200_; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_colight__hz4x4__colight_normal_200_s47 |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | concat_colight_text_event/hz4x4/s27 | 0 | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/concat_col` | alias→sumo_concat_colight__hz4x4__concat_colig; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_concat_colight__hz4x4__concat_colight_text_ |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | concat_colight_text_event/hz4x4/s37 | 0 | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/concat_col` | alias→sumo_concat_colight__hz4x4__concat_colig; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_concat_colight__hz4x4__concat_colight_text_ |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | concat_colight_text_event/hz4x4/s47 | 0 | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/concat_col` | alias→sumo_concat_colight__hz4x4__concat_colig; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_concat_colight__hz4x4__concat_colight_text_ |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | concat_colight_text_normal/hz4x4/s27 | — | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/concat_col` | alias→sumo_concat_colight__hz4x4__concat_colig; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_concat_colight__hz4x4__concat_colight_text_ |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | concat_colight_text_normal/hz4x4/s37 | — | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/concat_col` | alias→sumo_concat_colight__hz4x4__concat_colig; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_concat_colight__hz4x4__concat_colight_text_ |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | concat_colight_text_normal/hz4x4/s47 | — | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/concat_col` | alias→sumo_concat_colight__hz4x4__concat_colig; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_concat_colight__hz4x4__concat_colight_text_ |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | sga_colight_text_event/hz4x4/s27 | 0 | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/sga_coligh` | alias→sumo_sga_colight__hz4x4__sga_colight_tex; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_sga_colight__hz4x4__sga_colight_text_event_ |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | sga_colight_text_event/hz4x4/s37 | 0 | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/sga_coligh` | alias→sumo_sga_colight__hz4x4__sga_colight_tex; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_sga_colight__hz4x4__sga_colight_text_event_ |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | sga_colight_text_event/hz4x4/s47 | — | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/sga_coligh` | alias→sumo_sga_colight__hz4x4__sga_colight_tex; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_sga_colight__hz4x4__sga_colight_text_event_ |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | sga_colight_text_normal/hz4x4/s27 | 0 | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/sga_coligh` | alias→sumo_sga_colight__hz4x4__sga_colight_tex; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_sga_colight__hz4x4__sga_colight_text_normal |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | sga_colight_text_normal/hz4x4/s37 | — | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/sga_coligh` | alias→sumo_sga_colight__hz4x4__sga_colight_tex; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_sga_colight__hz4x4__sga_colight_text_normal |
| `artifacts__sga_concat_colight_200_v1__quarantine__pre_pause_` | sga_colight_text_normal/hz4x4/s47 | — | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/pre_pause_s27_launches/sga_coligh` | alias→sumo_sga_colight__hz4x4__sga_colight_tex; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_sga_colight__hz4x4__sga_colight_text_normal |
| `colight_adaptation_probe_20260920` | colight_adaptation_probe_20260920/hz4x4/s7 | 2 | DONE | 2026-09-20 | 167.8/21/221 | `data/output_data/tsc/sumo_colight/hz4x4/colight_adaptation_probe_20260920` |  |
| `colight_event_200_s17` | colight_event_200_s17/hz4x4/s17 | 200 | DONE | 2026-09-24 | 336.2/2739/244 | `data/output_data/tsc/sumo_colight/hz4x4/colight_event_200_s17` |  |
| `colight_event_200_s7` | colight_event_200_s7/hz4x4/s7 | 200 | DONE | 2026-09-24 | 336.4/2741/242 | `data/output_data/tsc/sumo_colight/hz4x4/colight_event_200_s7` |  |
| `colight_normal_200_s17` | colight_normal/hz4x4/s17 | 29 | FAILED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/misresolved_includes_20260924/col` | quarantined |
| `colight_normal_200_s7` | colight_normal/hz4x4/s7 | 30 | FAILED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/misresolved_includes_20260924/col` | quarantined |
| `concat_colight_text_event_200_s17` | concat_colight_text_event_200_s17/hz4x4/s17 | 200 | DONE | 2026-09-24 | 361.4/2710/273 | `data/output_data/tsc/sumo_concat_colight/hz4x4/concat_colight_text_event_200_s17` |  |
| `concat_colight_text_event_200_s7` | concat_colight_text_event/hz4x4/s7 | — | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/dropped_stragglers_20260924/conca` | quarantined |
| `concat_colight_text_normal_200_s17` | concat_colight_text_normal_200_s17/hz4x4/s17 | 200 | DONE | 2026-09-24 | 409.5/1984/882 | `data/output_data/tsc/sumo_concat_colight/hz4x4/concat_colight_text_normal_200_s1` |  |
| `concat_colight_text_normal_200_s7` | concat_colight_text_normal/hz4x4/s7 | 16 | FAILED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/misresolved_includes_20260924/con` | quarantined |
| `sga_colight_text_150_module_s7` | sga_colight_text_150_module_s7/hz4x4/s7 | 150 | DONE | 2026-09-24 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_150_module_s7` |  |
| `sga_colight_text_150_s7` | sga_colight_text_150_s7/hz4x4/s7 | — | FAILED | 2026-09-24 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_150_s7` |  |
| `sga_colight_text_150_s7_v2` | sga_colight_text_150_s7_v2/hz4x4/s7 | 16 | ABORTED | 2026-09-24 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_150_s7_v2` |  |
| `sga_colight_text_event_200_s17` | sga_colight_text_event_200_s17/hz4x4/s17 | 200 | DONE | 2026-09-24 | 335.8/2735/248 | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_event_200_s17` |  |
| `sga_colight_text_event_200_s7` | sga_colight_text_event/hz4x4/s7 | — | ABORTED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/dropped_stragglers_20260924/sga_c` | quarantined |
| `sga_colight_text_normal_200_s17` | sga_colight_text_normal_200_s17/hz4x4/s17 | 200 | DONE | 2026-09-24 | 332.8/2739/244 | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_normal_200_s17` |  |
| `sga_colight_text_normal_200_s7` | sga_colight_text_normal/hz4x4/s7 | 16 | FAILED | 2026-09-24 | — | `artifacts/sga_concat_colight_200_v1/quarantine/misresolved_includes_20260924/sga` | quarantined |
| `sga_concat_colight_200_v1__colight_normal_200_s17` | colight_normal_200_s17/hz4x4/s17 | — | FAILED | — | — | `—` | 队列登记无对应产物目录(prefix=colight_normal_200_s17); 目标目录不存在（计划输出位置仅作记录） |
| `sga_concat_colight_200_v1__colight_normal_200_s7` | colight_normal_200_s7/hz4x4/s7 | — | FAILED | — | — | `—` | 队列登记无对应产物目录(prefix=colight_normal_200_s7); 目标目录不存在（计划输出位置仅作记录） |
| `sga_concat_colight_200_v1__concat_colight_text_event_200_s7` | concat_colight_text_event_200_s7/hz4x4/s7 | — | FAILED | — | — | `—` | 队列登记无对应产物目录(prefix=concat_colight_text_event_200_s7); 目标目录不存在（计划输出位置仅作记录） |
| `sga_concat_colight_200_v1__concat_colight_text_normal_200_s7` | concat_colight_text_normal_200_s7/hz4x4/s7 | — | FAILED | — | — | `—` | 队列登记无对应产物目录(prefix=concat_colight_text_normal_200_s7); 目标目录不存在（计划输出位置仅作记录） |
| `sga_concat_colight_200_v1__sga_colight_text_event_200_s7` | sga_colight_text_event_200_s7/hz4x4/s7 | — | FAILED | — | — | `—` | 队列登记无对应产物目录(prefix=sga_colight_text_event_200_s7); 目标目录不存在（计划输出位置仅作记录） |
| `sga_concat_colight_200_v1__sga_colight_text_normal_200_s7` | sga_colight_text_normal_200_s7/hz4x4/s7 | — | FAILED | — | — | `—` | 队列登记无对应产物目录(prefix=sga_colight_text_normal_200_s7); 目标目录不存在（计划输出位置仅作记录） |
| `sga_sigmoid_text_150_s7` | sga_sigmoid_text_150_s7/hz4x4/s7 | 150 | DONE | 2026-09-24 | — | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_sigmoid_text_150_s7` |  |
| `sumo_colight__hz4x4__colight_event_200_s27` | colight_event_200_s27/hz4x4/s27 | 200 | DONE | 2026-09-24 | 334.7/2742/241 | `data/output_data/tsc/sumo_colight/hz4x4/colight_event_200_s27` |  |
| `sumo_colight__hz4x4__colight_event_200_s37` | colight_event_200_s37/hz4x4/s37 | 200 | DONE | 2026-09-24 | 337.0/2741/242 | `data/output_data/tsc/sumo_colight/hz4x4/colight_event_200_s37` |  |
| `sumo_colight__hz4x4__colight_event_200_s47` | colight_event_200_s47/hz4x4/s47 | 200 | DONE | 2026-09-24 | 337.3/2740/243 | `data/output_data/tsc/sumo_colight/hz4x4/colight_event_200_s47` |  |
| `sumo_colight__hz4x4__colight_normal_200_s27` | colight_normal_200_s27/hz4x4/s27 | 200 | DONE | 2026-09-24 | 334.4/2743/240 | `data/output_data/tsc/sumo_colight/hz4x4/colight_normal_200_s27` |  |
| `sumo_colight__hz4x4__colight_normal_200_s37` | colight_normal_200_s37/hz4x4/s37 | 200 | DONE | 2026-09-24 | 332.9/2736/247 | `data/output_data/tsc/sumo_colight/hz4x4/colight_normal_200_s37` |  |
| `sumo_colight__hz4x4__colight_normal_200_s47` | colight_normal_200_s47/hz4x4/s47 | 200 | DONE | 2026-09-24 | 333.7/2740/243 | `data/output_data/tsc/sumo_colight/hz4x4/colight_normal_200_s47` |  |
| `sumo_concat_colight__hz4x4__concat_colight_text_event_200_s2` | concat_colight_text_event_200_s27/hz4x4/s27 | 200 | DONE | 2026-09-24 | 354.6/2108/788 | `data/output_data/tsc/sumo_concat_colight/hz4x4/concat_colight_text_event_200_s27` |  |
| `sumo_concat_colight__hz4x4__concat_colight_text_event_200_s3` | concat_colight_text_event_200_s37/hz4x4/s37 | 200 | DONE | 2026-09-24 | 388.9/1557/1264 | `data/output_data/tsc/sumo_concat_colight/hz4x4/concat_colight_text_event_200_s37` |  |
| `sumo_concat_colight__hz4x4__concat_colight_text_event_200_s4` | concat_colight_text_event_200_s47/hz4x4/s47 | 200 | DONE | 2026-09-24 | 439.7/2073/817 | `data/output_data/tsc/sumo_concat_colight/hz4x4/concat_colight_text_event_200_s47` |  |
| `sumo_concat_colight__hz4x4__concat_colight_text_normal_200_s` | concat_colight_text_normal_200_s27/hz4x4/s27 | 200 | DONE | 2026-09-24 | 358.3/2069/858 | `data/output_data/tsc/sumo_concat_colight/hz4x4/concat_colight_text_normal_200_s2` |  |
| `sumo_concat_colight__hz4x4__concat_colight_text_normal_200_s` | concat_colight_text_normal_200_s37/hz4x4/s37 | 200 | DONE | 2026-09-24 | 537.2/2037/938 | `data/output_data/tsc/sumo_concat_colight/hz4x4/concat_colight_text_normal_200_s3` |  |
| `sumo_concat_colight__hz4x4__concat_colight_text_normal_200_s` | concat_colight_text_normal_200_s47/hz4x4/s47 | 200 | DONE | 2026-09-24 | 406.3/1673/1303 | `data/output_data/tsc/sumo_concat_colight/hz4x4/concat_colight_text_normal_200_s4` |  |
| `sumo_sga_colight__hz4x4__sga_colight_text_event_200_s27` | sga_colight_text_event_200_s27/hz4x4/s27 | 200 | DONE | 2026-09-24 | 335.4/2738/245 | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_event_200_s27` |  |
| `sumo_sga_colight__hz4x4__sga_colight_text_event_200_s37` | sga_colight_text_event_200_s37/hz4x4/s37 | 200 | DONE | 2026-09-24 | 335.2/2734/249 | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_event_200_s37` |  |
| `sumo_sga_colight__hz4x4__sga_colight_text_event_200_s47` | sga_colight_text_event_200_s47/hz4x4/s47 | 200 | DONE | 2026-09-24 | 336.4/2738/245 | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_event_200_s47` |  |
| `sumo_sga_colight__hz4x4__sga_colight_text_normal_200_s27` | sga_colight_text_normal_200_s27/hz4x4/s27 | 200 | DONE | 2026-09-24 | 335.4/2737/246 | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_normal_200_s27` |  |
| `sumo_sga_colight__hz4x4__sga_colight_text_normal_200_s37` | sga_colight_text_normal_200_s37/hz4x4/s37 | 200 | DONE | 2026-09-24 | 333.9/2739/244 | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_normal_200_s37` |  |
| `sumo_sga_colight__hz4x4__sga_colight_text_normal_200_s47` | sga_colight_text_normal_200_s47/hz4x4/s47 | 200 | DONE | 2026-09-24 | 333.8/2740/243 | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_normal_200_s47` |  |

评估 attempt 共 2200 行，按包/来源聚合：

| 评估包/来源 | n DONE/FAILED/ABORTED | 条件集 | eval seeds |
|---|---|---|---|
| `colight_event_200_s17` | 96/0/1 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `colight_event_200_s27` | 96/0/1 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `colight_event_200_s37` | 80/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `colight_event_200_s47` | 80/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `colight_event_200_s7` | 96/0/1 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `colight_normal_200_s27` | 96/0/1 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `colight_normal_200_s37` | 80/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `colight_normal_200_s47` | 80/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `concat_colight_text_event_200_s17` | 102/0/13 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `concat_colight_text_event_200_s27` | 91/0/5 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `concat_colight_text_event_200_s37` | 90/0/2 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `concat_colight_text_event_200_s47` | 80/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `concat_colight_text_normal_200_s17` | 100/0/3 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `concat_colight_text_normal_200_s27` | 91/0/2 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `concat_colight_text_normal_200_s37` | 80/0/7 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `concat_colight_text_normal_200_s47` | 80/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `sga_cf2_s27_ep200` | 1/0/0 | — | — |
| `sga_closed_loop_correct_s7` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400007_correct` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400007_wrong_location` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400007_zero_g` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400017_correct` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400017_wrong_location` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400017_zero_g` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400027_correct` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400027_wrong_location` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400027_zero_g` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400037_correct` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400037_wrong_location` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400037_zero_g` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400047_correct` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400047_wrong_location` | 1/0/0 | — | — |
| `sga_closed_loop_grid_t400047_zero_g` | 1/0/0 | — | — |
| `sga_closed_loop_nog_s7` | 1/0/0 | — | — |
| `sga_closed_loop_wrong_s7` | 1/0/0 | — | — |
| `sga_colight_sga_v2_s17` | 1/0/0 | — | — |
| `sga_colight_sga_v2_s7` | 1/0/0 | — | — |
| `sga_colight_text_event_200_s17` | 93/0/1 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `sga_colight_text_event_200_s27` | 80/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `sga_colight_text_event_200_s37` | 80/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `sga_colight_text_event_200_s47` | 80/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `sga_colight_text_normal_200_s17` | 93/0/1 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `sga_colight_text_normal_200_s27` | 80/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `sga_colight_text_normal_200_s37` | 80/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `sga_colight_text_normal_200_s47` | 80/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007,400017,400027,400037,400047 |
| `sga_counterfactual_eval_s7` | 1/0/0 | — | — |
| `sga_counterfactual_sigmoid150_s7` | 1/0/0 | — | — |
| `sga_counterfactual_sigmoid_s7` | 1/0/0 | — | — |
| `sga_location_grid_s17` | 1/0/0 | — | — |
| `sga_location_grid_s27` | 1/0/0 | — | — |
| `sga_location_grid_s37` | 1/0/0 | — | — |
| `sga_location_grid_s47` | 1/0/0 | — | — |
| `sga_location_grid_t400007` | 1/0/0 | — | — |
| `sga_location_grid_t400017` | 1/0/0 | — | — |
| `sga_location_grid_t400027` | 1/0/0 | — | — |
| `sga_location_grid_t400037` | 1/0/0 | — | — |
| `sga_location_grid_t400047` | 1/0/0 | — | — |
| `sga_mplight_sga_v2_s17` | 1/0/0 | — | — |
| `sga_mplight_sga_v2_s7` | 1/0/0 | — | — |
| `sga_v1_regress_v1` | 1/0/0 | — | — |
| `sga_wrong_location_sigmoid150_s7` | 1/0/0 | — | — |
| `sgacf_concat_colight_text_event_200_s17_t400007` | 1/0/0 | — | — |
| `sgacf_concat_colight_text_event_200_s47_t400007` | 1/0/0 | — | — |
| `sgacf_concat_colight_text_normal_200_s27_t400007` | 1/0/0 | — | — |
| `sgacf_concat_s27_ep200` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s17_t400007_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s17_t400027_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s27_ep0000_t400007_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s27_ep0050_t400007_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s27_ep0100_t400007_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s27_ep0150_t400007_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s27_t400007_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s27_t400007_wrong_location` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s27_t400007_zero_g` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s27_t400027_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s27_w1500_all_t400007_corre` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s27_w1500_all_t400007_wrong` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s27_w1500_all_t400007_zero_` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s37_t400007_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s37_t400027_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s47_ep0150_t400007_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s47_t400007_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s47_t400007_wrong_location` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s47_t400007_zero_g` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s47_t400027_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s47_w1500_all_t400007_corre` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s47_w1500_all_t400007_wrong` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_event_200_s47_w1500_all_t400007_zero_` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_normal_200_s17_t400007_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_normal_200_s17_t400027_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_normal_200_s27_ep0150_t400007_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_normal_200_s27_t400007_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_normal_200_s27_t400007_wrong_location` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_normal_200_s27_t400007_zero_g` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_normal_200_s27_t400027_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_normal_200_s27_w1500_all_t400007_corr` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_normal_200_s27_w1500_all_t400007_wron` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_normal_200_s27_w1500_all_t400007_zero` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_normal_200_s37_t400007_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_normal_200_s37_t400027_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_normal_200_s47_t400007_correct` | 1/0/0 | — | — |
| `sgacf_sga_colight_text_normal_200_s47_t400027_correct` | 1/0/0 | — | — |
