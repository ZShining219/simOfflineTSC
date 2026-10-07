# ATT-ENTITY-004 结构化事件输入 + 语义辅助（注意力线当前主线）

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：事件信息以结构化 z_task 槽位 + 语义辅助头进入 TARL 注意力，应同时提升场景区分与决策优化（H: 结构化+aux > 纯文本注入）。
- plan_id/队列：att004_wave1/fix/struct/formal/ablation/holdout/e200 波次队列
- 设计稿/证据位置：artifacts/att_entity_004/{run_queue_att004_*.json, run_state_*/}；configs/tsc/att_entity_004/generated/*.yml；artifacts/att_entity_004/{ABLATION_ARMS_DESIGN,DIVIDE_CONQUER_DESIGN,SGA_REPAIR_DESIGN}.md
- 结论：fixG/fixR/fixD/crossq/ablA/ablation/a2s/a3s/holdout 波次均完成并留档；stg2 波次登记未执行、2026-10-07 裁决转 SCOPE_DISCARD，stg1/ctl 部分臂 DROPPED（SCOPE_DISCARD）；SGA-GB 梯度线经 pilot 判定为负结果后停止（SGA_GB_PILOT_RESULT.md）；三指标+条件矩阵证据在 data/output_data/analysis/att_entity_004/ 与 baseline_cmp.md。
- 登记单元 1978（train/eval/smoke/batch/calibration = 83/1862/33/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `att004_abl_smoke_concat` | abl_smoke_concat/hz4x4/s7 | 2 | DONE | 2026-10-03 | 204.2/617/1440 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/att004_abl_smoke_concat` |  |
| `att004_abl_smoke_selfattn` | abl_smoke_selfattn/hz4x4/s7 | 2 | DONE | 2026-10-03 | 361.8/562/1538 | `data/output_data/tsc/sumo_tarl_selfattn/hz4x4/att004_abl_smoke_selfattn` |  |
| `att004_atsmoke_ctrl` | atsmoke_ctrl/hz4x4/s7 | 1 | DONE | 2026-10-05 | 661.8/2331/652 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_atsmoke_ctrl` |  |
| `att004_bert_regress` | bert_regress/hz4x4/s7 | 4 | DONE | 2026-10-01 | 160.6/22/220 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_bert_regress` |  |
| `att004_crossq_smoke` | crossq_smoke/hz4x4/s7 | 1 | DONE | 2026-10-05 | 661.8/2331/652 | `data/output_data/tsc/sumo_tarl_crossq/hz4x4/att004_crossq_smoke` |  |
| `att004_h1x1_smoke_a2` | h1x1_smoke_a2/sumohz1x1/s7 | 2 | DONE | 2026-10-03 | 59.8/89/267 | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/att004_h1x1_smoke_a2` |  |
| `att004_h1x1_smoke_a3s` | h1x1_smoke_a3s/sumohz1x1/s7 | — | FAILED | — | — | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/att004_h1x1_smoke_a3s` |  |
| `att004_h1x1_smoke_a3s_b` | h1x1_smoke_a3s_b/sumohz1x1/s7 | — | FAILED | 2026-10-03 | — | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/att004_h1x1_smoke_a3s_b` |  |
| `att004_h1x1_smoke_a3s_c` | h1x1_smoke_a3s_c/sumohz1x1/s7 | — | FAILED | 2026-10-03 | — | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/att004_h1x1_smoke_a3s_c` |  |
| `att004_h1x1_smoke_a3s_d` | h1x1_smoke_a3s_d/sumohz1x1/s7 | — | FAILED | 2026-10-03 | — | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/att004_h1x1_smoke_a3s_d` |  |
| `att004_h1x1_smoke_a3s_e` | h1x1_smoke_a3s_e/sumohz1x1/s7 | 2 | DONE | 2026-10-03 | 61.0/576/238 | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/att004_h1x1_smoke_a3s_e` |  |
| `att004_h1x1_smoke_cls` | h1x1_smoke_cls/sumohz1x1/s7 | 0 | DONE | 2026-10-03 | — | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/att004_h1x1_smoke_cls` |  |
| `att004_h1x1_smoke_learn` | h1x1_smoke_learn/sumohz1x1/s7 | 2 | DONE | 2026-10-03 | 61.0/576/238 | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/att004_h1x1_smoke_learn` |  |
| `att004_h1x1_smoke_pb600` | h1x1_smoke_pb600/sumohz1x1/s7 | — | FAILED | 2026-10-03 | — | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/att004_h1x1_smoke_pb600` |  |
| `att004_h1x1_smoke_pb750` | h1x1_smoke_pb750/sumohz1x1/s7 | — | FAILED | 2026-10-03 | — | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/att004_h1x1_smoke_pb750` |  |
| `att004_h1x1_smoke_probe` | h1x1_smoke_probe/sumohz1x1/s7 | — | FAILED | 2026-10-03 | — | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/att004_h1x1_smoke_probe` |  |
| `att004_h1x1_smoke_probe2` | h1x1_smoke_probe2/sumohz1x1/s7 | — | FAILED | 2026-10-03 | — | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/att004_h1x1_smoke_probe2` |  |
| `att004_h1x1_smoke_probe3` | h1x1_smoke_probe3/sumohz1x1/s7 | 0 | DONE | 2026-10-03 | — | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/att004_h1x1_smoke_probe3` |  |
| `att004_h1x1_smoke_probe4` | h1x1_smoke_probe4/sumohz1x1/s7 | 0 | DONE | 2026-10-03 | — | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/att004_h1x1_smoke_probe4` |  |
| `att004_smoke_fixD` | smoke_fixD/hz4x4/s7 | 2 | DONE | 2026-10-04 | 660.9/2340/643 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_smoke_fixD` |  |
| `att004_smoke_fixG` | smoke_fixG/hz4x4/s7 | 2 | DONE | 2026-10-04 | 661.8/2331/652 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_smoke_fixG` |  |
| `att004_smoke_gate` | smoke_gate/hz4x4/s7 | 4 | DONE | 2026-10-01 | 160.6/22/220 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_smoke_gate` |  |
| `att004_smoke_sgagb` | smoke_sgagb/hz4x4/s7 | 4 | DONE | 2026-10-01 | 160.6/22/220 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_smoke_sgagb` |  |
| `att004_structaux_smoke` | structaux_smoke/hz4x4/s7 | 4 | DONE | 2026-10-01 | 167.8/21/221 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_structaux_smoke` |  |
| `att004_structaux_smoke2` | structaux_smoke2/hz4x4/s7 | 4 | DONE | 2026-10-01 | 167.8/21/221 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_structaux_smoke2` |  |
| `h1x1_cf_smoke` | h1x1_smoke/sumohz1x1/s7 | 2 | DONE | 2026-10-03 | 61.1/576/238 | `data/output_data/tsc/sumo_tarl_attention/sumohz1x1/h1x1_cf_smoke` |  |
| `smoke_dumplib_a3s` | a3s/hz4x4/s7 | 1 | DONE | 2026-10-03 | 360.1/2737/246 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/smoke_dumplib_a3s` |  |
| `smoke_dumplib_a3s2` | a3s2/hz4x4/s7 | 1 | DONE | 2026-10-03 | 360.1/2737/246 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/smoke_dumplib_a3s2` |  |
| `smoke_dumplib_cl2` | cl2/hz4x4/s7 | 1 | DONE | 2026-10-03 | 350.7/2748/235 | `data/output_data/tsc/sumo_colight/hz4x4/smoke_dumplib_cl2` |  |
| `smoke_dumplib_ft` | ft/hz4x4/s7 | — | FAILED | — | — | `data/output_data/tsc/sumo_fixedtime/hz4x4/smoke_dumplib_ft` |  |
| `smoke_dumplib_ft2` | ft2/hz4x4/s7 | 0 | DONE | 2026-10-03 | — | `data/output_data/tsc/sumo_fixedtime/hz4x4/smoke_dumplib_ft2` |  |
| `smoke_dumplib_ft3` | ft3/hz4x4/s7 | 1 | FAILED | 2026-10-03 | 458.7/2603/373 | `data/output_data/tsc/sumo_fixedtime/hz4x4/smoke_dumplib_ft3` |  |
| `smoke_dumplib_ft4` | ft4/hz4x4/s7 | 1 | DONE | 2026-10-03 | 458.7/2603/373 | `data/output_data/tsc/sumo_fixedtime/hz4x4/smoke_dumplib_ft4` |  |
| `att004__a2_ctl_s27` | att004_ctl_s27/hz4x4/s27 | — | SCOPE_DISCARD | — | — | `—` | 队列登记无对应产物目录(prefix=att004_ctl_s27); 队列标记 DROPPED=未派发; 目标目录不存在（计划输出位置仅作记录） |
| `att004__a2_ctl_s47` | att004_ctl_s47/hz4x4/s47 | — | SCOPE_DISCARD | — | — | `—` | 队列登记无对应产物目录(prefix=att004_ctl_s47); 队列标记 DROPPED=未派发; 目标目录不存在（计划输出位置仅作记录） |
| `att004__a2_stg1_s17` | att004_stg1_s17/hz4x4/s17 | — | SCOPE_DISCARD | — | — | `—` | 队列登记无对应产物目录(prefix=att004_stg1_s17); 队列标记 DROPPED=未派发; 目标目录不存在（计划输出位置仅作记录） |
| `att004__a2_stg1_s37` | att004_stg1_s37/hz4x4/s37 | — | SCOPE_DISCARD | — | — | `—` | 队列登记无对应产物目录(prefix=att004_stg1_s37); 队列标记 DROPPED=未派发; 目标目录不存在（计划输出位置仅作记录） |
| `att004__a2_stg1_s47` | att004_stg1_s47/hz4x4/s47 | — | SCOPE_DISCARD | — | — | `—` | 队列登记无对应产物目录(prefix=att004_stg1_s47); 队列标记 DROPPED=未派发; 目标目录不存在（计划输出位置仅作记录） |
| `att004__a2_stg2_s17` | att004_stg2_s17/hz4x4/s17 | — | SCOPE_DISCARD | — | — | `—` | 队列登记无对应产物目录(prefix=att004_stg2_s17); 目标目录不存在（计划输出位置仅作记录） ／ 2026-10-07 裁决：a2s 臂 s |
| `att004__a2_stg2_s27` | att004_stg2_s27/hz4x4/s27 | — | SCOPE_DISCARD | — | — | `—` | 队列登记无对应产物目录(prefix=att004_stg2_s27); 目标目录不存在（计划输出位置仅作记录） ／ 2026-10-07 裁决：a2s 臂 s |
| `att004__a2_stg2_s37` | att004_stg2_s37/hz4x4/s37 | — | SCOPE_DISCARD | — | — | `—` | 队列登记无对应产物目录(prefix=att004_stg2_s37); 目标目录不存在（计划输出位置仅作记录） ／ 2026-10-07 裁决：a2s 臂 s |
| `att004__a2_stg2_s47` | att004_stg2_s47/hz4x4/s47 | — | SCOPE_DISCARD | — | — | `—` | 队列登记无对应产物目录(prefix=att004_stg2_s47); 目标目录不存在（计划输出位置仅作记录） ／ 2026-10-07 裁决：a2s 臂 s |
| `att004__a2_stg2_s7` | att004_stg2_s7/hz4x4/s7 | — | SCOPE_DISCARD | — | — | `—` | 队列登记无对应产物目录(prefix=att004_stg2_s7); 目标目录不存在（计划输出位置仅作记录） ／ 2026-10-07 裁决：a2s 臂 st |
| `att004_a2s_s17` | a2s/hz4x4/s17 | 100 | DONE | 2026-10-01 | 367.1/2717/266 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_a2s_s17` |  |
| `att004_a2s_s27` | a2s/hz4x4/s27 | 100 | DONE | 2026-10-02 | 395.7/2551/432 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_a2s_s27` |  |
| `att004_a2s_s37` | a2s/hz4x4/s37 | 100 | DONE | 2026-10-02 | 392.9/2698/285 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_a2s_s37` |  |
| `att004_a2s_s47` | a2s/hz4x4/s47 | 100 | DONE | 2026-10-02 | 365.5/2724/259 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_a2s_s47` |  |
| `att004_a2s_s7` | a2s/hz4x4/s7 | 100 | DONE | 2026-10-01 | 389.7/2418/565 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_a2s_s7` |  |
| `att004_a3s_s17` | a3s/hz4x4/s17 | 100 | DONE | 2026-10-01 | 410.6/2496/487 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_a3s_s17` |  |
| `att004_a3s_s27` | a3s/hz4x4/s27 | 100 | DONE | 2026-10-02 | 373.0/2720/263 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_a3s_s27` |  |
| `att004_a3s_s37` | a3s/hz4x4/s37 | 100 | DONE | 2026-10-02 | 378.7/2604/379 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_a3s_s37` |  |
| `att004_a3s_s47` | a3s/hz4x4/s47 | 100 | DONE | 2026-10-02 | 363.4/2728/255 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_a3s_s47` |  |
| `att004_a3s_s7` | a3s/hz4x4/s7 | 100 | DONE | 2026-10-01 | 365.3/2729/254 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_a3s_s7` |  |
| `att004_ablA_concats_s17` | ablA_concats/hz4x4/s17 | 100 | DONE | 2026-10-04 | 365.0/2724/259 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/att004_ablA_concats_s17` |  |
| `att004_ablA_concats_s27` | ablA_concats/hz4x4/s27 | 60 | ABORTED | 2026-10-04 | 360.3/2729/254 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/att004_ablA_concats_s27` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 60 条 |
| `att004_ablA_concats_s37` | ablA_concats/hz4x4/s37 | 82 | ABORTED | 2026-10-04 | 359.7/2729/254 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/att004_ablA_concats_s37` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 82 条 |
| `att004_ablA_concats_s47` | ablA_concats/hz4x4/s47 | 73 | ABORTED | 2026-10-04 | 356.7/2727/256 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/att004_ablA_concats_s47` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 73 条 |
| `att004_ablA_concats_s7` | ablA_concats/hz4x4/s7 | 100 | DONE | 2026-10-04 | 357.6/2727/256 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/att004_ablA_concats_s7` |  |
| `att004_ablA_selfattns_s17` | ablA_selfattns/hz4x4/s17 | 100 | DONE | 2026-10-04 | 365.8/2732/251 | `data/output_data/tsc/sumo_tarl_selfattn/hz4x4/att004_ablA_selfattns_s17` |  |
| `att004_ablA_selfattns_s27` | ablA_selfattns/hz4x4/s27 | 32 | ABORTED | 2026-10-04 | 458.6/2418/565 | `data/output_data/tsc/sumo_tarl_selfattn/hz4x4/att004_ablA_selfattns_s27` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 32 条 |
| `att004_ablA_selfattns_s37` | ablA_selfattns/hz4x4/s37 | 55 | ABORTED | 2026-10-04 | 361.8/2725/258 | `data/output_data/tsc/sumo_tarl_selfattn/hz4x4/att004_ablA_selfattns_s37` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 55 条 |
| `att004_ablA_selfattns_s47` | ablA_selfattns/hz4x4/s47 | 37 | ABORTED | 2026-10-04 | 450.5/2461/522 | `data/output_data/tsc/sumo_tarl_selfattn/hz4x4/att004_ablA_selfattns_s47` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 37 条 |
| `att004_ablA_selfattns_s7` | ablA_selfattns/hz4x4/s7 | 100 | DONE | 2026-10-04 | 363.4/2730/253 | `data/output_data/tsc/sumo_tarl_selfattn/hz4x4/att004_ablA_selfattns_s7` |  |
| `att004_abl_concats_s17` | abl_concats/hz4x4/s17 | 100 | DONE | 2026-10-03 | 378.3/2712/271 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/att004_abl_concats_s17` |  |
| `att004_abl_concats_s7` | abl_concats/hz4x4/s7 | 100 | DONE | 2026-10-03 | 360.2/2726/257 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/att004_abl_concats_s7` |  |
| `att004_abl_selfattns_s17` | abl_selfattns/hz4x4/s17 | 100 | DONE | 2026-10-03 | 371.8/2724/259 | `data/output_data/tsc/sumo_tarl_selfattn/hz4x4/att004_abl_selfattns_s17` |  |
| `att004_abl_selfattns_s7` | abl_selfattns/hz4x4/s7 | 100 | DONE | 2026-10-03 | 368.8/2722/261 | `data/output_data/tsc/sumo_tarl_selfattn/hz4x4/att004_abl_selfattns_s7` |  |
| `att004_ctl_s17` | ctl/hz4x4/s17 | 100 | DONE | 2026-10-01 | 367.6/2565/418 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_ctl_s17` |  |
| `att004_ctl_s37` | ctl/hz4x4/s37 | 100 | DONE | 2026-10-01 | 354.5/2705/278 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_ctl_s37` |  |
| `att004_ctl_s7` | ctl/hz4x4/s7 | 100 | DONE | 2026-10-01 | 391.3/2327/656 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_ctl_s7` |  |
| `att004_e200_a3s_s17` | e200_a3s/hz4x4/s17 | 100 | DONE | 2026-10-02 | 404.2/2342/641 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_e200_a3s_s17` |  |
| `att004_e200_a3s_s7` | e200_a3s/hz4x4/s7 | 100 | DONE | 2026-10-02 | 357.1/2722/261 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_e200_a3s_s7` |  |
| `att004_fixD_s17` | fixD/hz4x4/s17 | 100 | DONE | 2026-10-04 | 355.9/2731/252 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixD_s17` |  |
| `att004_fixD_s27` | fixD/hz4x4/s27 | 15 | ABORTED | 2026-10-04 | 709.5/2320/663 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixD_s27` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 15 条 |
| `att004_fixD_s37` | fixD/hz4x4/s37 | 17 | ABORTED | 2026-10-04 | 787.3/2178/787 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixD_s37` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 17 条 |
| `att004_fixD_s47` | fixD/hz4x4/s47 | 17 | ABORTED | 2026-10-04 | 715.3/2305/678 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixD_s47` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 17 条 |
| `att004_fixD_s7` | fixD/hz4x4/s7 | 100 | DONE | 2026-10-04 | 357.9/2734/249 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixD_s7` |  |
| `att004_fixG_s17` | fixG/hz4x4/s17 | 200 | DONE | 2026-10-04 | 354.6/2730/253 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixG_s17` |  |
| `att004_fixG_s17_aborted100` | fixG_s17_aborted100/hz4x4/s17 | 4 | ABORTED | 2026-10-04 | 675.7/2353/630 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixG_s17_aborted100` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 4 条; 由 fixG 200ep 正式波次 att004_fix |
| `att004_fixG_s27` | fixG/hz4x4/s27 | 200 | DONE | 2026-10-04 | 353.5/2729/254 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixG_s27` |  |
| `att004_fixG_s27_aborted100` | fixG_s27_aborted100/hz4x4/s27 | 4 | ABORTED | 2026-10-04 | 709.4/2318/665 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixG_s27_aborted100` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 4 条; 由 fixG 200ep 正式波次 att004_fix |
| `att004_fixG_s37` | fixG/hz4x4/s37 | 200 | DONE | 2026-10-04 | 353.7/2732/251 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixG_s37` |  |
| `att004_fixG_s37_aborted100` | fixG_s37_aborted100/hz4x4/s37 | 4 | ABORTED | 2026-10-04 | 796.7/2205/778 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixG_s37_aborted100` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 4 条; 由 fixG 200ep 正式波次 att004_fix |
| `att004_fixG_s47` | fixG/hz4x4/s47 | 200 | DONE | 2026-10-04 | 354.4/2732/251 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixG_s47` |  |
| `att004_fixG_s47_aborted100` | fixG_s47_aborted100/hz4x4/s47 | 4 | ABORTED | 2026-10-04 | 720.8/2319/664 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixG_s47_aborted100` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 4 条; 由 fixG 200ep 正式波次 att004_fix |
| `att004_fixG_s7` | fixG/hz4x4/s7 | 200 | DONE | 2026-10-04 | 355.2/2734/249 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixG_s7` |  |
| `att004_fixG_s7_aborted100` | fixG_s7_aborted100/hz4x4/s7 | 4 | ABORTED | 2026-10-04 | 661.8/2331/652 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixG_s7_aborted100` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 4 条; 由 fixG 200ep 正式波次 att004_fix |
| `att004_fixR_s17` | fixR/hz4x4/s17 | 100 | DONE | 2026-10-04 | 359.0/2733/250 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixR_s17` |  |
| `att004_fixR_s27` | fixR/hz4x4/s27 | 17 | ABORTED | 2026-10-04 | 709.5/2320/663 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixR_s27` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 17 条 |
| `att004_fixR_s37` | fixR/hz4x4/s37 | 17 | ABORTED | 2026-10-04 | 787.3/2178/787 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixR_s37` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 17 条 |
| `att004_fixR_s47` | fixR/hz4x4/s47 | 10 | ABORTED | 2026-10-04 | 715.3/2305/678 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixR_s47` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 10 条 |
| `att004_fixR_s7` | fixR/hz4x4/s7 | 100 | DONE | 2026-10-04 | 359.9/2729/254 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_fixR_s7` |  |
| `att004_holdout_t3_a3s_s17` | holdout_t3_a3s/hz4x4/s17 | 100 | DONE | 2026-10-03 | 362.2/2727/256 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_holdout_t3_a3s_s17` |  |
| `att004_holdout_t3_a3s_s7` | holdout_t3_a3s/hz4x4/s7 | 100 | DONE | 2026-10-03 | 360.9/2733/250 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_holdout_t3_a3s_s7` |  |
| `att004_holdout_t4_a3s_s17` | holdout_t4_a3s/hz4x4/s17 | 100 | DONE | 2026-10-03 | 364.8/2727/256 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_holdout_t4_a3s_s17` |  |
| `att004_holdout_t4_a3s_s7` | holdout_t4_a3s/hz4x4/s7 | 100 | DONE | 2026-10-03 | 364.2/2727/256 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_holdout_t4_a3s_s7` |  |
| `att004_l05_a3s_s17` | l05_a3s/hz4x4/s17 | 15 | ABORTED | 2026-10-03 | 425.2/970/1300 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_l05_a3s_s17` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 15 条 |
| `att004_l05_a3s_s27` | l05_a3s/hz4x4/s27 | 15 | ABORTED | 2026-10-03 | 528.4/483/1858 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_l05_a3s_s27` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 15 条 |
| `att004_l05_a3s_s7` | l05_a3s/hz4x4/s7 | 15 | ABORTED | 2026-10-03 | 349.7/1113/1740 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_l05_a3s_s7` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 15 条 |
| `att004_l2_a3s_s17` | l2_a3s/hz4x4/s17 | 15 | ABORTED | 2026-10-03 | 425.2/970/1300 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_l2_a3s_s17` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 15 条 |
| `att004_l2_a3s_s27` | l2_a3s/hz4x4/s27 | 15 | ABORTED | 2026-10-03 | 528.4/483/1858 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_l2_a3s_s27` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 15 条 |
| `att004_l2_a3s_s7` | l2_a3s/hz4x4/s7 | 15 | ABORTED | 2026-10-03 | 349.7/1113/1740 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_l2_a3s_s7` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 15 条 |
| `att004_sgagb_s17` | sgagb/hz4x4/s17 | 100 | DONE | 2026-10-01 | 360.5/2725/258 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_sgagb_s17` |  |
| `att004_sgagb_s7` | sgagb/hz4x4/s7 | 100 | DONE | 2026-10-01 | 362.0/2728/255 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_sgagb_s7` |  |
| `att004_stg1_s27` | stg1/hz4x4/s27 | 111 | ABORTED | 2026-10-01 | 356.7/2716/267 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_stg1_s27` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 111 条 |
| `att004_stg1_s7` | stg1/hz4x4/s7 | 109 | ABORTED | 2026-10-01 | 356.2/2720/263 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_stg1_s7` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 109 条 |
| `sumo_tarl_crossq__hz4x4___aborted_premature_20261005__att004` | crossq/hz4x4/s17 | 2 | ABORTED | 2026-10-05 | 675.7/2353/630 | `data/output_data/tsc/sumo_tarl_crossq/hz4x4/_aborted_premature_20261005/att004_c` | alias→sumo_tarl_crossq__hz4x4__att004_crossq_s; quarantined under _aborted_premature_20261005; 隔离桶语义=中止残留; 同名隔离/残留副本，逻辑同一 run→su |
| `sumo_tarl_crossq__hz4x4___aborted_premature_20261005__att004` | crossq/hz4x4/s27 | 3 | ABORTED | 2026-10-05 | 709.4/2318/665 | `data/output_data/tsc/sumo_tarl_crossq/hz4x4/_aborted_premature_20261005/att004_c` | alias→sumo_tarl_crossq__hz4x4__att004_crossq_s; quarantined under _aborted_premature_20261005; 隔离桶语义=中止残留; 同名隔离/残留副本，逻辑同一 run→su |
| `sumo_tarl_crossq__hz4x4___aborted_premature_20261005__att004` | crossq/hz4x4/s37 | 3 | ABORTED | 2026-10-05 | 796.7/2205/778 | `data/output_data/tsc/sumo_tarl_crossq/hz4x4/_aborted_premature_20261005/att004_c` | alias→sumo_tarl_crossq__hz4x4__att004_crossq_s; quarantined under _aborted_premature_20261005; 隔离桶语义=中止残留; 同名隔离/残留副本，逻辑同一 run→su |
| `sumo_tarl_crossq__hz4x4___aborted_premature_20261005__att004` | crossq/hz4x4/s47 | 3 | ABORTED | 2026-10-05 | 720.8/2319/664 | `data/output_data/tsc/sumo_tarl_crossq/hz4x4/_aborted_premature_20261005/att004_c` | alias→sumo_tarl_crossq__hz4x4__att004_crossq_s; quarantined under _aborted_premature_20261005; 隔离桶语义=中止残留; 同名隔离/残留副本，逻辑同一 run→su |
| `sumo_tarl_crossq__hz4x4___aborted_premature_20261005__att004` | crossq/hz4x4/s7 | 2 | ABORTED | 2026-10-05 | 661.8/2331/652 | `data/output_data/tsc/sumo_tarl_crossq/hz4x4/_aborted_premature_20261005/att004_c` | alias→sumo_tarl_crossq__hz4x4__att004_crossq_s; quarantined under _aborted_premature_20261005; 隔离桶语义=中止残留; 同名隔离/残留副本，逻辑同一 run→su |
| `sumo_tarl_crossq__hz4x4__att004_crossq_s17` | crossq/hz4x4/s17 | 200 | DONE | 2026-10-05 | 354.1/2733/250 | `data/output_data/tsc/sumo_tarl_crossq/hz4x4/att004_crossq_s17` |  |
| `sumo_tarl_crossq__hz4x4__att004_crossq_s27` | crossq/hz4x4/s27 | 200 | DONE | 2026-10-05 | 354.1/2734/249 | `data/output_data/tsc/sumo_tarl_crossq/hz4x4/att004_crossq_s27` |  |
| `sumo_tarl_crossq__hz4x4__att004_crossq_s37` | crossq/hz4x4/s37 | 200 | DONE | 2026-10-05 | 352.7/2736/247 | `data/output_data/tsc/sumo_tarl_crossq/hz4x4/att004_crossq_s37` |  |
| `sumo_tarl_crossq__hz4x4__att004_crossq_s47` | crossq/hz4x4/s47 | 200 | DONE | 2026-10-05 | 355.8/2733/250 | `data/output_data/tsc/sumo_tarl_crossq/hz4x4/att004_crossq_s47` |  |
| `sumo_tarl_crossq__hz4x4__att004_crossq_s7` | crossq/hz4x4/s7 | 200 | DONE | 2026-10-05 | 354.4/2731/252 | `data/output_data/tsc/sumo_tarl_crossq/hz4x4/att004_crossq_s7` |  |

评估 attempt 共 1862 行，按包/来源聚合：

| 评估包/来源 | n DONE/FAILED/ABORTED | 条件集 | eval seeds |
|---|---|---|---|
| `a2s` | 49/0/0 | — | — |
| `a3s` | 122/0/0 | — | — |
| `a3s_s17` | 45/0/0 | — | — |
| `a3s_s7` | 45/0/0 | — | — |
| `ablA_concats_s17` | 45/0/0 | — | — |
| `ablA_concats_s27` | 28/2/0 | — | — |
| `ablA_concats_s37` | 30/0/0 | — | — |
| `ablA_concats_s47` | 30/0/0 | — | — |
| `ablA_concats_s7` | 45/0/0 | — | — |
| `ablA_selfattns_s17` | 45/0/0 | — | — |
| `ablA_selfattns_s27` | 14/1/0 | — | — |
| `ablA_selfattns_s37` | 30/0/0 | — | — |
| `ablA_selfattns_s47` | 15/0/0 | — | — |
| `ablA_selfattns_s7` | 45/0/0 | — | — |
| `abl_concats_s17` | 15/0/0 | — | — |
| `abl_concats_s7` | 15/0/0 | — | — |
| `abl_selfattns_s17` | 15/0/0 | — | — |
| `abl_selfattns_s7` | 15/0/0 | — | — |
| `att004_fixD_s17` | 0/1/0 | — | — |
| `att004_fixD_s7` | 0/1/0 | — | — |
| `att004_fixR_s17` | 6/1/0 | — | — |
| `att004_fixR_s7` | 15/0/0 | — | — |
| `colight` | 72/0/0 | — | — |
| `crossq_s17` | 60/0/0 | — | — |
| `crossq_s27` | 60/0/0 | — | — |
| `crossq_s37` | 60/0/0 | — | — |
| `crossq_s47` | 60/0/0 | — | — |
| `crossq_s7` | 60/0/0 | — | — |
| `fixD_s17` | 45/0/0 | — | — |
| `fixD_s7` | 45/0/0 | — | — |
| `fixG_s17` | 60/18/0 | — | — |
| `fixG_s27` | 60/20/0 | — | — |
| `fixG_s37` | 60/19/0 | — | — |
| `fixG_s47` | 60/35/0 | — | — |
| `fixG_s7` | 60/18/0 | — | — |
| `fixR_s17` | 45/0/0 | — | — |
| `fixR_s7` | 45/0/0 | — | — |
| `fixedtime` | 36/0/0 | — | — |
| `ft_ev` | 3/0/0 | — | — |
| `ft_noev` | 6/0/0 | — | — |
| `ft_noevB` | 6/0/0 | — | — |
| `ho_t3_s17` | 15/0/0 | — | — |
| `ho_t3_s7` | 15/0/0 | — | — |
| `ho_t4_s17` | 15/0/0 | — | — |
| `ho_t4_s7` | 15/0/0 | — | — |
| `maxpressure` | 36/0/0 | — | — |
| `mplight` | 72/0/0 | — | — |
| `s17_can` | 2/0/0 | — | — |
| `s17_emp` | 1/0/0 | — | — |
| `smoke_colight` | 1/0/0 | — | — |
| `smoke_fixed` | 1/0/0 | — | — |
| `smoke_mplight` | 1/0/0 | — | — |
