# EXPERIMENTS — 实验台账（人读索引）

> 追溯登记 2026-10-07（治理分支 gov/experiment-governance）。
> 事实源 = `ledger/runs.jsonl`（每 run 一行机读，本文件为其人读投影）。
> 口径：`run_kind=train|eval|smoke|batch|calibration`；eval 行为单次冻结评估 attempt，`behavior_source` 指回训练 run 包名；`alias_of` 非空者不作独立样本统计。
> 机器一律代号 34/73；路径均为仓内相对路径。

<!-- PANEL-BEGIN -->
## 当前面板（2026-10-07，campaign 收官时刷新）

- **当前有效设计栈**：`agent/scene_attention.py` + `agent/tarl.py`——per-node z_task 结构化槽位 + SemanticAuxHead 辅助监督 + 反事实探针四层判定；配置族 `configs/tsc/att_entity_004/`；参考臂 `a3s`，最新修复线 `fixG`（收官待判读）。
- **代码位置**：`codex/milestone0-experiment-infrastructure` 分支（收编点 c8be158..32d3e18，已与主干合并 70f3dbf）；main=治理层+半离线基线。
- **代号地图**（历史代际，明细在各 campaign 节）：
  - `tarl_reproduction`/`tarl_v21_formal` = TARL 论文复现基线（完结，仅对照用）
  - `att_entity_002` = SGA/concat/MPLight 文本注入矩阵（完结：H2 支持、H3 否）
  - `att_entity_003` = TARL 对齐/传感矩阵 flx（完结）
  - `att_entity_004` = **当前主线**（z_task+aux+反事实）；`sga_gb` 梯度门线 pilot 负结果已停
  - `arterial_1x6`/`plan5_b100` = 半离线线（技术点3，独立于注意力线）
- **最新结论**：z_task 语义保真达标、held-out 证实文本携场景身份（t34_devin）；"canonical≈empty"修正为"事件窗内小幅净负"（t34_cc，配对 n=59）；road_closure 覆盖缺口待裁决修复。
- **接管测试**：t34_devin / t34_cc 均 DONE（test/t34_* 分支）。
- **证据完整性**：台账 6516 行，evidence_lost=0。
<!-- PANEL-END -->

## 0. 总览

登记 run 总数：**6516**（train 571、eval attempt 5771、smoke 127、calibration 18、batch 29）

| campaign | DONE | FAILED | ABORTED | SUPERSEDED | REGISTERED | SCOPE_DISCARD | LEGACY_UNCLEAR | 合计 |
|---|---|---|---|---|---|---|---|---|
| att_entity_004 | 1812 | 126 | 30 | 0 | 0 | 10 | 0 | 1978 |
| att_entity_003 | 1372 | 67 | 0 | 0 | 0 | 0 | 0 | 1439 |
| att_entity_002 | 2206 | 26 | 59 | 0 | 0 | 0 | 0 | 2291 |
| arterial_1x6 | 501 | 8 | 0 | 0 | 0 | 0 | 0 | 509 |
| tarl_reproduction | 132 | 11 | 0 | 0 | 0 | 0 | 0 | 143 |
| tarl_v21_formal | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 15 |
| plan5_b100 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 68 |
| plan1_dqn | 27 | 0 | 4 | 4 | 0 | 0 | 0 | 35 |
| milestone0_infra | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |
| paper_infra_validation | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 19 |
| misc_probes | 1 | 5 | 0 | 0 | 0 | 1 | 0 | 7 |
| **合计** | 6165 | 243 | 93 | 4 | 0 | 11 | 0 | **6516** |

状态语义见 GOVERNANCE.md §2。LEGACY_UNCLEAR=0 表示全部可识别 run 均通过队列/状态文件/目录归属证据还原出 campaign；个别 run 的臂内身份仍以命名约定为据，已在 notes 标注处保持保守。

## ATT-ENTITY-004 结构化事件输入 + 语义辅助（注意力线当前主线）

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

## ATT-ENTITY-003 TARL 对齐/传感矩阵（flx/tarlp/stage2-5）

- 假设：文本-实体注意力对齐训练（align_trans/align_trans_meta）与随机事件分布随机化，应改善 TARL 在 hz4x4 事件场景下的鲁棒决策。
- plan_id/队列：att003 stage2/stage3(stage3_pilot/stage3_pilotB/main)/stage4/stage5_tarl
- 设计稿/证据位置：artifacts/att_entity_003/run_queue_stage{2,3*,4,5_tarl}.json；eval 包 artifacts/att_entity_003/{eval_v1,eval_stage4,eval_stage5_tarl,stage3_eval_v1}
- 结论：TARL 矩阵完成：20 训练 run + 20×14 评估包；文本反事实探针完成；终报与 wiki review v11 归档。随机事件臂 *_random_200（colight/sga_colight_text/sga_flx_colight_text/mplight/flx_align）属本 campaign。
- 登记单元 1439（train/eval/smoke/batch/calibration = 70/1354/15/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `sga_flx_align_smoke2_s7` | sga_flx_align_smoke2_s7/hz4x4/s7 | 3 | DONE | 2026-09-29 | — | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_align_smoke2_s7` |  |
| `sga_flx_align_smoke_s7` | sga_flx_align_smoke_s7/hz4x4/s7 | 3 | DONE | 2026-09-29 | — | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_align_smoke_s7` |  |
| `sga_flx_align_tm_smoke_s7` | sga_flx_align_tm_smoke_s7/hz4x4/s7 | 3 | DONE | 2026-09-29 | — | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_align_tm_smoke_s7` |  |
| `sga_flx_align_trans_smoke_s7` | sga_flx_align_trans_smoke_s7/hz4x4/s7 | 3 | DONE | 2026-09-29 | — | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_align_trans_smoke_s7` |  |
| `sga_flx_livefire_v1` | sga_flx_livefire_v1/hz4x4/s7 | — | FAILED | 2026-09-28 | — | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_livefire_v1` |  |
| `sga_flx_livefire_v2` | sga_flx_livefire_v2/hz4x4/s7 | 1 | DONE | 2026-09-28 | — | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_livefire_v2` |  |
| `sga_flx_livefire_v3` | sga_flx_livefire_v3/hz4x4/s7 | — | FAILED | 2026-09-28 | — | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_livefire_v3` |  |
| `sga_flx_livefire_v4` | sga_flx_livefire_v4/hz4x4/s7 | 1 | DONE | 2026-09-28 | — | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_livefire_v4` |  |
| `sga_flx_smoke_v1` | sga_flx_smoke_v1/hz4x4/s7 | — | FAILED | — | — | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_smoke_v1` |  |
| `sga_flx_smoke_v2` | sga_flx_smoke_v2/hz4x4/s7 | 3 | DONE | 2026-09-28 | — | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_smoke_v2` |  |
| `tarlcf_smoke` | smoke/hz4x4/s7 | 1 | DONE | 2026-09-29 | — | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarlcf_smoke` |  |
| `tarlcf_smoke3` | smoke3/hz4x4/s7 | 1 | DONE | 2026-09-29 | — | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarlcf_smoke3` |  |
| `tarlp_att_smoke` | att_smoke/hz4x4/s7 | 4 | DONE | 2026-09-29 | 160.6/22/220 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarlp_att_smoke` |  |
| `tarlrepr_att7_smoke` | att7_smoke/hz4x4/s400007 | 1 | DONE | 2026-09-30 | — | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarlrepr_att7_smoke` |  |
| `tarlutil_gatg7_emp_smoke` | gatg7_emp_smoke/hz4x4/s400007 | 1 | DONE | 2026-09-30 | — | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarlutil_gatg7_emp_smoke` |  |
| `align_l0.1_s17` | align_l0.1_s17/hz4x4/s17 | 200 | DONE | 2026-09-29 | 350.6/2741/242 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_l0.1_s17` |  |
| `align_l0.1_s7` | align_l0.1_s7/hz4x4/s7 | 200 | DONE | 2026-09-29 | 350.4/2738/245 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_l0.1_s7` |  |
| `align_l1.0_s17` | align_l1.0_s17/hz4x4/s17 | 200 | DONE | 2026-09-29 | 350.1/2739/244 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_l1.0_s17` |  |
| `align_l1.0_s7` | align_l1.0_s7/hz4x4/s7 | 200 | DONE | 2026-09-29 | 352.0/2537/446 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_l1.0_s7` |  |
| `align_trans_l0.1_s17` | align_trans_l0.1_s17/hz4x4/s17 | 200 | DONE | 2026-09-29 | 349.5/2738/245 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_trans_l0.1_s17` |  |
| `align_trans_l0.1_s27` | align_trans_l0.1_s27/hz4x4/s27 | 200 | DONE | 2026-09-29 | 349.5/2738/245 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_trans_l0.1_s27` |  |
| `align_trans_l0.1_s37` | align_trans_l0.1_s37/hz4x4/s37 | 200 | DONE | 2026-09-29 | 349.7/2738/245 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_trans_l0.1_s37` |  |
| `align_trans_l0.1_s47` | align_trans_l0.1_s47/hz4x4/s47 | 200 | DONE | 2026-09-29 | 335.7/2072/889 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_trans_l0.1_s47` |  |
| `align_trans_l0.1_s7` | align_trans_l0.1_s7/hz4x4/s7 | 200 | DONE | 2026-09-29 | 350.4/2734/249 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_trans_l0.1_s7` |  |
| `align_trans_l1.0_s17` | align_trans_l1.0_s17/hz4x4/s17 | 200 | DONE | 2026-09-29 | 350.8/2739/244 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_trans_l1.0_s17` |  |
| `align_trans_l1.0_s7` | align_trans_l1.0_s7/hz4x4/s7 | 200 | DONE | 2026-09-29 | 352.1/2740/243 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_trans_l1.0_s7` |  |
| `align_trans_meta_l0.1_s17` | align_trans_meta_l0.1_s17/hz4x4/s17 | 200 | DONE | 2026-09-29 | 352.9/2734/249 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_trans_meta_l0.1_s17` |  |
| `align_trans_meta_l0.1_s27` | align_trans_meta_l0.1_s27/hz4x4/s27 | 200 | DONE | 2026-09-29 | 351.8/2741/242 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_trans_meta_l0.1_s27` |  |
| `align_trans_meta_l0.1_s37` | align_trans_meta_l0.1_s37/hz4x4/s37 | 200 | DONE | 2026-09-29 | 385.1/2521/462 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_trans_meta_l0.1_s37` |  |
| `align_trans_meta_l0.1_s47` | align_trans_meta_l0.1_s47/hz4x4/s47 | 200 | DONE | 2026-09-29 | 350.1/2737/246 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_trans_meta_l0.1_s47` |  |
| `align_trans_meta_l0.1_s7` | align_trans_meta_l0.1_s7/hz4x4/s7 | 200 | DONE | 2026-09-29 | 351.3/2741/242 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/align_trans_meta_l0.1_s7` |  |
| `colight_event_random_200_s17` | colight_event_random_200_s17/hz4x4/s17 | 200 | DONE | 2026-09-28 | 349.4/2736/247 | `data/output_data/tsc/sumo_colight/hz4x4/colight_event_random_200_s17` |  |
| `colight_event_random_200_s27` | colight_event_random_200_s27/hz4x4/s27 | 200 | DONE | 2026-09-28 | 348.6/2740/243 | `data/output_data/tsc/sumo_colight/hz4x4/colight_event_random_200_s27` |  |
| `colight_event_random_200_s37` | colight_event_random_200_s37/hz4x4/s37 | 200 | DONE | 2026-09-28 | 326.7/2302/681 | `data/output_data/tsc/sumo_colight/hz4x4/colight_event_random_200_s37` |  |
| `colight_event_random_200_s47` | colight_event_random_200_s47/hz4x4/s47 | 200 | DONE | 2026-09-28 | 340.3/2443/540 | `data/output_data/tsc/sumo_colight/hz4x4/colight_event_random_200_s47` |  |
| `colight_event_random_200_s7` | colight_event_random_200_s7/hz4x4/s7 | 200 | DONE | 2026-09-28 | 351.2/2738/245 | `data/output_data/tsc/sumo_colight/hz4x4/colight_event_random_200_s7` |  |
| `flx_align_l0.1_event_random_200_s17` | flx_align_l0.1_event_random_200_s17/hz4x4/s17 | 200 | DONE | 2026-09-29 | 350.6/2741/242 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/flx_align_l0.1_event_random_200_` |  |
| `flx_align_l0.1_event_random_200_s27` | flx_align_l0.1_event_random_200_s27/hz4x4/s27 | 200 | DONE | 2026-09-29 | 336.8/2025/958 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/flx_align_l0.1_event_random_200_` |  |
| `flx_align_l0.1_event_random_200_s37` | flx_align_l0.1_event_random_200_s37/hz4x4/s37 | 200 | DONE | 2026-09-29 | 349.5/2739/244 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/flx_align_l0.1_event_random_200_` |  |
| `flx_align_l0.1_event_random_200_s47` | flx_align_l0.1_event_random_200_s47/hz4x4/s47 | 200 | DONE | 2026-09-29 | 354.3/2736/247 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/flx_align_l0.1_event_random_200_` |  |
| `flx_align_l0.1_event_random_200_s7` | flx_align_l0.1_event_random_200_s7/hz4x4/s7 | 200 | DONE | 2026-09-29 | 350.4/2738/245 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/flx_align_l0.1_event_random_200_` |  |
| `mplight_event_random_200_s17` | mplight_event_random_200_s17/hz4x4/s17 | 200 | DONE | 2026-09-29 | 348.4/2739/244 | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_random_200_s17` |  |
| `mplight_event_random_200_s27` | mplight_event_random_200_s27/hz4x4/s27 | 200 | DONE | 2026-09-29 | 348.9/2736/247 | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_random_200_s27` |  |
| `mplight_event_random_200_s37` | mplight_event_random_200_s37/hz4x4/s37 | 200 | DONE | 2026-09-29 | 348.9/2738/245 | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_random_200_s37` |  |
| `mplight_event_random_200_s47` | mplight_event_random_200_s47/hz4x4/s47 | 200 | DONE | 2026-09-29 | 347.7/2628/355 | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_random_200_s47` |  |
| `mplight_event_random_200_s7` | mplight_event_random_200_s7/hz4x4/s7 | 200 | DONE | 2026-09-29 | 348.1/2739/244 | `data/output_data/tsc/sumo_mplight/hz4x4/mplight_event_random_200_s7` |  |
| `sga_colight_text_event_random_200_s17` | sga_colight_text_event_random_200_s17/hz4x4/s17 | 200 | DONE | 2026-09-28 | 353.6/2731/252 | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_event_random_200_s1` |  |
| `sga_colight_text_event_random_200_s27` | sga_colight_text_event_random_200_s27/hz4x4/s27 | 200 | DONE | 2026-09-28 | 349.9/2739/244 | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_event_random_200_s2` |  |
| `sga_colight_text_event_random_200_s37` | sga_colight_text_event_random_200_s37/hz4x4/s37 | 200 | DONE | 2026-09-28 | 350.7/2733/250 | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_event_random_200_s3` |  |
| `sga_colight_text_event_random_200_s47` | sga_colight_text_event_random_200_s47/hz4x4/s47 | 200 | DONE | 2026-09-28 | 349.3/2734/249 | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_event_random_200_s4` |  |
| `sga_colight_text_event_random_200_s7` | sga_colight_text_event_random_200_s7/hz4x4/s7 | 200 | DONE | 2026-09-28 | 349.8/2739/244 | `data/output_data/tsc/sumo_sga_colight/hz4x4/sga_colight_text_event_random_200_s7` |  |
| `sga_flx_colight_text_event_200_s17` | sga_flx_colight_text_event_200_s17/hz4x4/s17 | 200 | DONE | 2026-09-28 | 335.8/2739/244 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_colight_text_event_200_s` |  |
| `sga_flx_colight_text_event_200_s27` | sga_flx_colight_text_event_200_s27/hz4x4/s27 | 200 | DONE | 2026-09-28 | 335.6/2739/244 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_colight_text_event_200_s` |  |
| `sga_flx_colight_text_event_200_s37` | sga_flx_colight_text_event_200_s37/hz4x4/s37 | 200 | DONE | 2026-09-28 | 336.1/2741/242 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_colight_text_event_200_s` |  |
| `sga_flx_colight_text_event_200_s47` | sga_flx_colight_text_event_200_s47/hz4x4/s47 | 200 | DONE | 2026-09-28 | 335.3/2738/245 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_colight_text_event_200_s` |  |
| `sga_flx_colight_text_event_200_s7` | sga_flx_colight_text_event_200_s7/hz4x4/s7 | 200 | DONE | 2026-09-28 | 335.5/2740/243 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_colight_text_event_200_s` |  |
| `sga_flx_colight_text_event_random_200_s17` | sga_flx_colight_text_event_random_200_s17/hz4x4/s17 | 200 | DONE | 2026-09-28 | 351.2/2739/244 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_colight_text_event_rando` |  |
| `sga_flx_colight_text_event_random_200_s27` | sga_flx_colight_text_event_random_200_s27/hz4x4/s27 | 200 | DONE | 2026-09-28 | 349.4/2737/246 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_colight_text_event_rando` |  |
| `sga_flx_colight_text_event_random_200_s37` | sga_flx_colight_text_event_random_200_s37/hz4x4/s37 | 200 | DONE | 2026-09-28 | 349.5/2739/244 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_colight_text_event_rando` |  |
| `sga_flx_colight_text_event_random_200_s47` | sga_flx_colight_text_event_random_200_s47/hz4x4/s47 | 200 | DONE | 2026-09-28 | 348.4/2626/357 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_colight_text_event_rando` |  |
| `sga_flx_colight_text_event_random_200_s7` | sga_flx_colight_text_event_random_200_s7/hz4x4/s7 | 200 | DONE | 2026-09-28 | 349.6/2739/244 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_colight_text_event_rando` |  |
| `sga_flx_diag60_s17` | sga_flx_diag60_s17/hz4x4/s17 | 60 | DONE | 2026-09-28 | 350.4/2736/247 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_diag60_s17` |  |
| `sga_flx_diag60_s27` | sga_flx_diag60_s27/hz4x4/s27 | 60 | DONE | 2026-09-28 | 350.0/2742/241 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_diag60_s27` |  |
| `sga_flx_diag60_s7` | sga_flx_diag60_s7/hz4x4/s7 | 60 | DONE | 2026-09-28 | 353.9/2738/245 | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_diag60_s7` |  |
| `sga_flx_noalign_regr_s7` | sga_flx_noalign_regr_s7/hz4x4/s7 | 3 | DONE | 2026-09-29 | — | `data/output_data/tsc/sumo_sga_flx_colight/hz4x4/sga_flx_noalign_regr_s7` |  |
| `tarlp_att_s17` | att_s17/hz4x4/s17 | 200 | DONE | 2026-09-29 | 356.5/2726/257 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarlp_att_s17` |  |
| `tarlp_att_s27` | att_s27/hz4x4/s27 | 200 | DONE | 2026-09-29 | 399.9/2555/428 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarlp_att_s27` |  |
| `tarlp_att_s37` | att_s37/hz4x4/s37 | 200 | DONE | 2026-09-29 | 357.2/2722/261 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarlp_att_s37` |  |
| `tarlp_att_s47` | att_s47/hz4x4/s47 | 200 | DONE | 2026-09-30 | 356.4/2728/255 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarlp_att_s47` |  |
| `tarlp_att_s7` | att_s7/hz4x4/s7 | 200 | DONE | 2026-09-29 | 355.7/2726/257 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarlp_att_s7` |  |
| `tarlp_gat_s17` | gat_s17/hz4x4/s17 | 200 | DONE | 2026-09-29 | 357.4/2728/255 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarlp_gat_s17` |  |
| `tarlp_gat_s27` | gat_s27/hz4x4/s27 | 200 | DONE | 2026-09-29 | 361.7/2730/253 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarlp_gat_s27` |  |
| `tarlp_gat_s37` | gat_s37/hz4x4/s37 | 200 | DONE | 2026-09-29 | 359.4/2727/256 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarlp_gat_s37` |  |
| `tarlp_gat_s47` | gat_s47/hz4x4/s47 | 200 | DONE | 2026-09-30 | 354.4/2735/248 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarlp_gat_s47` |  |
| `tarlp_gat_s7` | gat_s7/hz4x4/s7 | 200 | DONE | 2026-09-29 | 355.4/2740/243 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarlp_gat_s7` |  |
| `tarlp_gatg_s17` | gatg_s17/hz4x4/s17 | 200 | DONE | 2026-09-29 | 366.8/2728/255 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarlp_gatg_s17` |  |
| `tarlp_gatg_s27` | gatg_s27/hz4x4/s27 | 200 | DONE | 2026-09-29 | 359.0/2728/255 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarlp_gatg_s27` |  |
| `tarlp_gatg_s37` | gatg_s37/hz4x4/s37 | 200 | DONE | 2026-09-29 | 373.4/2533/450 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarlp_gatg_s37` |  |
| `tarlp_gatg_s47` | gatg_s47/hz4x4/s47 | 200 | DONE | 2026-09-30 | 357.9/2727/256 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarlp_gatg_s47` |  |
| `tarlp_gatg_s7` | gatg_s7/hz4x4/s7 | 200 | DONE | 2026-09-29 | 362.4/2734/249 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarlp_gatg_s7` |  |
| `tarlp_sen_s17` | sen_s17/hz4x4/s17 | 200 | DONE | 2026-09-29 | 348.0/2739/244 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarlp_sen_s17` |  |
| `tarlp_sen_s27` | sen_s27/hz4x4/s27 | 200 | DONE | 2026-09-29 | 347.3/2740/243 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarlp_sen_s27` |  |
| `tarlp_sen_s37` | sen_s37/hz4x4/s37 | 200 | DONE | 2026-09-30 | 347.4/2737/246 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarlp_sen_s37` |  |
| `tarlp_sen_s47` | sen_s47/hz4x4/s47 | 200 | DONE | 2026-09-30 | 347.1/2737/246 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarlp_sen_s47` |  |
| `tarlp_sen_s7` | sen_s7/hz4x4/s7 | 200 | DONE | 2026-09-29 | 347.7/2742/241 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarlp_sen_s7` |  |

评估 attempt 共 1354 行，按包/来源聚合：

| 评估包/来源 | n DONE/FAILED/ABORTED | 条件集 | eval seeds |
|---|---|---|---|
| `align_trans_l0.1_s17` | 14/0/0 | all,none | 400007 |
| `align_trans_l0.1_s27` | 14/0/0 | all,none | 400007 |
| `align_trans_l0.1_s37` | 14/0/0 | all,none | 400007 |
| `align_trans_l0.1_s47` | 14/0/0 | all,none | 400007 |
| `align_trans_l0.1_s7` | 14/0/0 | all,none | 400007 |
| `align_trans_l1.0_s17` | 14/0/0 | all,none | 400007 |
| `align_trans_l1.0_s7` | 14/0/0 | all,none | 400007 |
| `align_trans_meta_l0.1_s17` | 14/0/0 | all,none | 400007 |
| `align_trans_meta_l0.1_s27` | 14/0/0 | all,none | 400007 |
| `align_trans_meta_l0.1_s37` | 14/0/0 | all,none | 400007 |
| `align_trans_meta_l0.1_s47` | 14/0/0 | all,none | 400007 |
| `align_trans_meta_l0.1_s7` | 14/0/0 | all,none | 400007 |
| `att7_canonical` | 1/0/0 | — | — |
| `att7_empty` | 1/0/0 | — | — |
| `att7_foreign` | 1/0/0 | — | — |
| `att7_relabeled` | 1/0/0 | — | — |
| `att_17` | 1/0/0 | — | — |
| `att_27` | 1/0/0 | — | — |
| `att_37` | 1/0/0 | — | — |
| `att_47` | 1/0/0 | — | — |
| `att_7` | 1/0/0 | — | — |
| `att_s17_revc` | 1/0/0 | — | — |
| `att_s7_can` | 1/0/0 | — | — |
| `att_s7_emp2` | 1/0/0 | — | — |
| `att_s7_rev` | 0/1/0 | — | — |
| `att_s7_rev2` | 1/0/0 | — | — |
| `colight_event_random_200_s17` | 28/0/0 | all,none | 400007 |
| `colight_event_random_200_s27` | 28/0/0 | all,none | 400007 |
| `colight_event_random_200_s37` | 28/0/0 | all,none | 400007 |
| `colight_event_random_200_s47` | 28/0/0 | all,none | 400007 |
| `colight_event_random_200_s7` | 28/0/0 | all,none | 400007 |
| `flx_align_l0.1_event_random_200_s17` | 28/0/0 | all,none | 400007 |
| `flx_align_l0.1_event_random_200_s27` | 28/0/0 | all,none | 400007 |
| `flx_align_l0.1_event_random_200_s37` | 28/0/0 | all,none | 400007 |
| `flx_align_l0.1_event_random_200_s47` | 28/0/0 | all,none | 400007 |
| `flx_align_l0.1_event_random_200_s7` | 28/0/0 | all,none | 400007 |
| `flxcf2_sga_flx_colight_text_event_200_s17_ep100` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s17_ep150` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s17_ep200` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s17_ep50` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s27_ep100` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s27_ep150` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s27_ep200` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s27_ep50` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s37_ep100` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s37_ep150` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s37_ep200` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s37_ep50` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s47_ep100` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s47_ep150` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s47_ep200` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s47_ep50` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s7_ep100` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s7_ep150` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s7_ep200` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_200_s7_ep50` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s17_ep100` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s17_ep150` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s17_ep200` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s17_ep50` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s27_ep100` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s27_ep150` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s27_ep200` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s27_ep50` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s37_ep100` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s37_ep150` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s37_ep200` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s37_ep50` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s47_ep100` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s47_ep150` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s47_ep200` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s47_ep50` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s7_ep100` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s7_ep150` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s7_ep200` | 1/0/0 | — | — |
| `flxcf2_sga_flx_colight_text_event_random_200_s7_ep50` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s17_ep0_correct` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s17_ep0_wrong_location` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s17_ep0_zero_g` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s17_ep30_correct` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s17_ep30_wrong_location` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s17_ep30_zero_g` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s17_ep60_correct` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s17_ep60_wrong_location` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s17_ep60_zero_g` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s27_ep0_correct` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s27_ep0_wrong_location` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s27_ep0_zero_g` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s27_ep30_correct` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s27_ep30_wrong_location` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s27_ep30_zero_g` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s27_ep60_correct` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s27_ep60_wrong_location` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s27_ep60_zero_g` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s7_ep0_correct` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s7_ep0_wrong_location` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s7_ep0_zero_g` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s7_ep30_correct` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s7_ep30_wrong_location` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s7_ep30_zero_g` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s7_ep60_correct` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s7_ep60_wrong_location` | 1/0/0 | — | — |
| `flxcf_sga_flx_diag60_s7_ep60_zero_g` | 1/0/0 | — | — |
| `gatg` | 1/0/0 | — | — |
| `gatg7_canonical` | 1/0/0 | — | — |
| `gatg7_empty` | 1/0/0 | — | — |
| `gatg7_foreign` | 1/0/0 | — | — |
| `gatg7_relabeled` | 1/0/0 | — | — |
| `gatg_37` | 1/0/0 | — | — |
| `gatg_47` | 1/0/0 | — | — |
| `gatg_s17` | 1/0/0 | — | — |
| `gatg_s17_revc` | 1/0/0 | — | — |
| `gatg_s7` | 1/0/0 | — | — |
| `gatg_s7_can` | 1/0/0 | — | — |
| `gatg_s7_emp2` | 1/0/0 | — | — |
| `gatg_s7_rev` | 0/1/0 | — | — |
| `gatg_s7_rev2` | 1/0/0 | — | — |
| `mp_eventref_v1` | 14/0/0 | — | — |
| `mplight_randE_s17` | 14/0/0 | all,none | 400007 |
| `mplight_randE_s27` | 14/0/0 | all,none | 400007 |
| `mplight_randE_s37` | 14/0/0 | all,none | 400007 |
| `mplight_randE_s47` | 14/0/0 | all,none | 400007 |
| `mplight_randE_s7` | 14/0/0 | all,none | 400007 |
| `probe4_align_trans_l0.1_s17` | 1/0/0 | — | — |
| `probe4_align_trans_l0.1_s7` | 1/0/0 | — | — |
| `probe_align_trans_l0.1_s17` | 1/0/0 | — | — |
| `probe_align_trans_l0.1_s27` | 1/0/0 | — | — |
| `probe_align_trans_l0.1_s37` | 1/0/0 | — | — |
| `probe_align_trans_l0.1_s47` | 1/0/0 | — | — |
| `probe_align_trans_l0.1_s7` | 1/0/0 | — | — |
| `probe_align_trans_l1.0_s17` | 1/0/0 | — | — |
| `probe_align_trans_l1.0_s7` | 1/0/0 | — | — |
| `probe_align_trans_meta_l0.1_s17` | 1/0/0 | — | — |
| `probe_align_trans_meta_l0.1_s27` | 1/0/0 | — | — |
| `probe_align_trans_meta_l0.1_s37` | 1/0/0 | — | — |
| `probe_align_trans_meta_l0.1_s7` | 1/0/0 | — | — |
| `probe_meta_s47` | 1/0/0 | — | — |
| `profdiag` | 0/1/0 | — | — |
| `retr_align_l0.1_s17_ep200` | 1/0/0 | — | — |
| `retr_align_l0.1_s7_ep200` | 1/0/0 | — | — |
| `retr_align_l1.0_s17_ep200` | 1/0/0 | — | — |
| `retr_align_l1.0_s7_ep200` | 1/0/0 | — | — |
| `retr_flx_align_l0.1_event_random_200_s17` | 1/0/0 | — | — |
| `retr_flx_align_l0.1_event_random_200_s27` | 1/0/0 | — | — |
| `retr_flx_align_l0.1_event_random_200_s37` | 1/0/0 | — | — |
| `retr_flx_align_l0.1_event_random_200_s47` | 1/0/0 | — | — |
| `retr_flx_align_l0.1_event_random_200_s7` | 1/0/0 | — | — |
| `retr_probe_s7` | 1/0/0 | — | — |
| `retr_probe_s7_ep0` | 1/0/0 | — | — |
| `sga_colight_text_event_random_200_s17` | 28/0/0 | all,none | 400007 |
| `sga_colight_text_event_random_200_s27` | 28/0/0 | all,none | 400007 |
| `sga_colight_text_event_random_200_s37` | 28/0/0 | all,none | 400007 |
| `sga_colight_text_event_random_200_s47` | 28/0/0 | all,none | 400007 |
| `sga_colight_text_event_random_200_s7` | 28/0/0 | all,none | 400007 |
| `sga_flx_colight_text_event_200_s17` | 16/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007 |
| `sga_flx_colight_text_event_200_s27` | 16/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007 |
| `sga_flx_colight_text_event_200_s37` | 16/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007 |
| `sga_flx_colight_text_event_200_s47` | 16/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007 |
| `sga_flx_colight_text_event_200_s7` | 16/0/0 | all,blockage,blockage_closure,blockage_rain,closure,closure_ | 400007 |
| `sga_flx_colight_text_event_random_200_s17` | 28/0/0 | all,none | 400007 |
| `sga_flx_colight_text_event_random_200_s27` | 28/0/0 | all,none | 400007 |
| `sga_flx_colight_text_event_random_200_s37` | 28/0/0 | all,none | 400007 |
| `sga_flx_colight_text_event_random_200_s47` | 28/0/0 | all,none | 400007 |
| `sga_flx_colight_text_event_random_200_s7` | 28/0/0 | all,none | 400007 |
| `tarlp_att_s17` | 14/0/0 | all,none | 400007 |
| `tarlp_att_s27` | 14/0/0 | all,none | 400007 |
| `tarlp_att_s37` | 14/0/0 | all,none | 400007 |
| `tarlp_att_s47` | 14/0/0 | all,none | 400007 |
| `tarlp_att_s7` | 14/0/0 | all,none | 400007 |
| `tarlp_gat_s17` | 14/11/0 | all,none | 400007 |
| `tarlp_gat_s27` | 14/0/0 | all,none | 400007 |
| `tarlp_gat_s37` | 14/0/0 | all,none | 400007 |
| `tarlp_gat_s47` | 14/0/0 | all,none | 400007 |
| `tarlp_gat_s7` | 14/14/0 | all,none | 400007 |
| `tarlp_gatg_s17` | 14/8/0 | all,none | 400007 |
| `tarlp_gatg_s27` | 14/0/0 | all,none | 400007 |
| `tarlp_gatg_s37` | 14/0/0 | all,none | 400007 |
| `tarlp_gatg_s47` | 14/0/0 | all,none | 400007 |
| `tarlp_gatg_s7` | 14/14/0 | all,none | 400007 |
| `tarlp_sen_s17` | 14/0/0 | all,none | 400007 |
| `tarlp_sen_s27` | 14/0/0 | all,none | 400007 |
| `tarlp_sen_s37` | 14/0/0 | all,none | 400007 |
| `tarlp_sen_s47` | 14/0/0 | all,none | 400007 |
| `tarlp_sen_s7` | 14/14/0 | all,none | 400007 |

## ATT-ENTITY-002 SGA/CoLight 文本事件 200ep 矩阵

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

## 半离线 arterial 1×6 干线实验

- 假设：shared_dqn 在 sumoarterial1x6 干线上按 stage0 采集→stage1 控制→stage2 流水线评估。
- plan_id/队列：artifacts/arterial_experiments/stage{0,1,2}
- 设计稿/证据位置：artifacts/arterial_experiments/**/run_queue.json + run_state
- 结论：stage0/stage1 采集与控制 run 501 完成、8 失败（含 stage1_invalid_scene_routing 隔离批）；stage2 部分留档。
- 登记单元 509（train/eval/smoke/batch/calibration = 157/352/0/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `arterial_goal_single_regression_20260729` | arterial_goal_single_regression_20260729/sumohz1x1/s0 | 1 | DONE | 2026-07-29 | 0/0/8 | `data/output_data/tsc/sumo_dqn/sumohz1x1/arterial_goal_single_regression_20260729` |  |
| `arterial_preflight_runtime_20260730` | arterial_preflight_runtime_20260730/sumoarterial1x6_300_06/s0 | 1 | DONE | 2026-07-29 | 0/0/31 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/arterial_preflight_r` |  |
| `arterial_resume_fault_20260730` | arterial_resume_fault_20260730/sumoarterial1x6_300_06/s9299 | 2 | DONE | 2026-07-29 | 0/0/31 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/arterial_resume_faul` |  |
| `arterial_resume_source_20260730` | arterial_resume_source_20260730/sumoarterial1x6_300_06/s9299 | 2 | DONE | 2026-07-29 | 0/0/31 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/arterial_resume_sour` |  |
| `arterial_smoke_final_20260729` | arterial_smoke_final_20260729/sumoarterial1x6_300_06/s0 | 1 | DONE | 2026-07-29 | 0/0/31 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/arterial_smoke_final` |  |
| `arterial_smoke_load_70003_20260729` | arterial_smoke_load_70003_20260729/sumoarterial1x6_700_03/s0 | 1 | DONE | 2026-07-29 | 0/0/33 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/arterial_smoke_load_` |  |
| `arterial_smoke_load_70006_20260729_r2` | arterial_smoke_load_70006_20260729_r2/sumoarterial1x6_700_06/s0 | 1 | DONE | 2026-07-29 | 0/0/24 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/arterial_smoke_load_` |  |
| `arterial_smoke_offline_20260729` | arterial_smoke_offline_20260729/sumoarterial1x6_300_06/s0 | 0 | DONE | 2026-07-29 | — | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/arterial_smoke_offli` |  |
| `arterial_smoke_online_20260729_r5` | arterial_smoke_online_20260729_r5/sumoarterial1x6_300_06/s0 | 1 | DONE | 2026-07-29 | 0/0/31 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/arterial_smoke_onlin` |  |
| `arterial_smoke_semi_20260729` | arterial_smoke_semi_20260729/sumoarterial1x6_300_03/s0 | 1 | DONE | 2026-07-29 | 0/0/31 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/arterial_smoke_semi_` |  |
| `arterial_smoke_switch_20260729_r2` | arterial_smoke_switch_20260729_r2/sumoarterial1x6_300_03/s0 | 1 | DONE | 2026-07-29 | 0/0/31 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/arterial_smoke_switc` |  |
| `arterial_stage1_gate_seed_metadata_20260730` | arterial_stage1_gate_seed_metadata_20260730/sumoarterial1x6_300_06/s9199 | 1 | DONE | 2026-07-29 | 0/0/31 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/arterial_stage1_gate` |  |
| `artifacts__arterial_experiments__quarantine__stage1_invalid_` | stage1_collector_300_03_seed0/sumoarterial1x6_300_06/s0 | 5 | FAILED | 2026-07-29 | — | `artifacts/arterial_experiments/quarantine/stage1_invalid_scene_routing_20260730/` | alias→sumo_shared_dqn__sumoarterial1x6_300_03_; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_shared_dqn__sumoarterial1x6_300_03__stage1_ |
| `artifacts__arterial_experiments__quarantine__stage1_invalid_` | stage1_collector_300_03_seed1/sumoarterial1x6_300_06/s1 | 4 | FAILED | 2026-07-29 | — | `artifacts/arterial_experiments/quarantine/stage1_invalid_scene_routing_20260730/` | alias→sumo_shared_dqn__sumoarterial1x6_300_03_; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_shared_dqn__sumoarterial1x6_300_03__stage1_ |
| `artifacts__arterial_experiments__quarantine__stage1_invalid_` | stage1_collector_300_06_seed0/sumoarterial1x6_300_06/s0 | 5 | FAILED | 2026-07-29 | — | `artifacts/arterial_experiments/quarantine/stage1_invalid_scene_routing_20260730/` | alias→sumo_shared_dqn__sumoarterial1x6_300_06_; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_shared_dqn__sumoarterial1x6_300_06__stage1_ |
| `artifacts__arterial_experiments__quarantine__stage1_invalid_` | stage1_collector_300_06_seed1/sumoarterial1x6_300_06/s1 | 4 | FAILED | 2026-07-29 | — | `artifacts/arterial_experiments/quarantine/stage1_invalid_scene_routing_20260730/` | alias→sumo_shared_dqn__sumoarterial1x6_300_06_; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_shared_dqn__sumoarterial1x6_300_06__stage1_ |
| `artifacts__arterial_experiments__quarantine__stage1_invalid_` | stage1_collector_700_03_seed0/sumoarterial1x6_300_06/s0 | 5 | FAILED | 2026-07-29 | — | `artifacts/arterial_experiments/quarantine/stage1_invalid_scene_routing_20260730/` | alias→sumo_shared_dqn__sumoarterial1x6_700_03_; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_shared_dqn__sumoarterial1x6_700_03__stage1_ |
| `artifacts__arterial_experiments__quarantine__stage1_invalid_` | stage1_collector_700_03_seed1/sumoarterial1x6_300_06/s1 | 4 | FAILED | 2026-07-29 | — | `artifacts/arterial_experiments/quarantine/stage1_invalid_scene_routing_20260730/` | alias→sumo_shared_dqn__sumoarterial1x6_700_03_; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_shared_dqn__sumoarterial1x6_700_03__stage1_ |
| `artifacts__arterial_experiments__quarantine__stage1_invalid_` | stage1_collector_700_06_seed0/sumoarterial1x6_300_06/s0 | 5 | FAILED | 2026-07-29 | — | `artifacts/arterial_experiments/quarantine/stage1_invalid_scene_routing_20260730/` | alias→sumo_shared_dqn__sumoarterial1x6_700_06_; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_shared_dqn__sumoarterial1x6_700_06__stage1_ |
| `artifacts__arterial_experiments__quarantine__stage1_invalid_` | stage1_collector_700_06_seed1/sumoarterial1x6_300_06/s1 | 4 | FAILED | 2026-07-29 | — | `artifacts/arterial_experiments/quarantine/stage1_invalid_scene_routing_20260730/` | alias→sumo_shared_dqn__sumoarterial1x6_700_06_; quarantined; 同名隔离/残留副本，逻辑同一 run→sumo_shared_dqn__sumoarterial1x6_700_06__stage1_ |
| `stage0_budget_300_06_seed9100` | stage0_budget_300_06/sumoarterial1x6_300_06/s9100 | 100 | DONE | 2026-07-29 | 91.2/4352/113 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage0_budget_300_06` |  |
| `stage0_budget_700_06_seed9101` | stage0_budget_700_06/sumoarterial1x6_700_06/s9101 | 100 | DONE | 2026-07-29 | 101.6/8812/190 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_budget_700_06` |  |
| `stage0_w12_700_06_seed9060` | stage0_w12_700_06/sumoarterial1x6_700_06/s9060 | 25 | DONE | 2026-07-29 | 428.1/1970/1729 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w12_700_06_se` |  |
| `stage0_w12_700_06_seed9061` | stage0_w12_700_06/sumoarterial1x6_700_06/s9061 | 25 | DONE | 2026-07-29 | 199.9/3668/1558 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w12_700_06_se` |  |
| `stage0_w12_700_06_seed9062` | stage0_w12_700_06/sumoarterial1x6_700_06/s9062 | 25 | DONE | 2026-07-29 | 461.5/1574/1740 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w12_700_06_se` |  |
| `stage0_w12_700_06_seed9063` | stage0_w12_700_06/sumoarterial1x6_700_06/s9063 | 25 | DONE | 2026-07-29 | 296.8/2713/1430 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w12_700_06_se` |  |
| `stage0_w12_700_06_seed9064` | stage0_w12_700_06/sumoarterial1x6_700_06/s9064 | 25 | DONE | 2026-07-29 | 287.4/4901/1459 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w12_700_06_se` |  |
| `stage0_w12_700_06_seed9065` | stage0_w12_700_06/sumoarterial1x6_700_06/s9065 | 25 | DONE | 2026-07-29 | 347.6/2649/1412 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w12_700_06_se` |  |
| `stage0_w12_700_06_seed9066` | stage0_w12_700_06/sumoarterial1x6_700_06/s9066 | 25 | DONE | 2026-07-29 | 367.2/3288/1453 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w12_700_06_se` |  |
| `stage0_w12_700_06_seed9067` | stage0_w12_700_06/sumoarterial1x6_700_06/s9067 | 25 | DONE | 2026-07-29 | 419.5/1989/1736 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w12_700_06_se` |  |
| `stage0_w12_700_06_seed9068` | stage0_w12_700_06/sumoarterial1x6_700_06/s9068 | 25 | DONE | 2026-07-29 | 199.9/3668/1558 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w12_700_06_se` |  |
| `stage0_w12_700_06_seed9069` | stage0_w12_700_06/sumoarterial1x6_700_06/s9069 | 25 | DONE | 2026-07-29 | 219.4/3541/1649 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w12_700_06_se` |  |
| `stage0_w12_700_06_seed9070` | stage0_w12_700_06/sumoarterial1x6_700_06/s9070 | 25 | DONE | 2026-07-29 | 299.3/2667/1436 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w12_700_06_se` |  |
| `stage0_w12_700_06_seed9071` | stage0_w12_700_06/sumoarterial1x6_700_06/s9071 | 25 | DONE | 2026-07-29 | 445.8/1797/1664 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w12_700_06_se` |  |
| `stage0_w1_700_06_seed9000` | stage0_w1_700_06/sumoarterial1x6_700_06/s9000 | 25 | DONE | 2026-07-29 | 310.0/3743/1564 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w1_700_06_see` |  |
| `stage0_w2_700_06_seed9010` | stage0_w2_700_06/sumoarterial1x6_700_06/s9010 | 25 | DONE | 2026-07-29 | 172.3/5596/1448 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w2_700_06_see` |  |
| `stage0_w2_700_06_seed9011` | stage0_w2_700_06/sumoarterial1x6_700_06/s9011 | 25 | DONE | 2026-07-29 | 297.6/3008/1720 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w2_700_06_see` |  |
| `stage0_w4_700_06_seed9020` | stage0_w4_700_06/sumoarterial1x6_700_06/s9020 | 25 | DONE | 2026-07-29 | 135.3/5737/1455 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w4_700_06_see` |  |
| `stage0_w4_700_06_seed9021` | stage0_w4_700_06/sumoarterial1x6_700_06/s9021 | 25 | DONE | 2026-07-29 | 325.1/2905/1442 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w4_700_06_see` |  |
| `stage0_w4_700_06_seed9022` | stage0_w4_700_06/sumoarterial1x6_700_06/s9022 | 25 | DONE | 2026-07-29 | 287.3/4588/1564 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w4_700_06_see` |  |
| `stage0_w4_700_06_seed9023` | stage0_w4_700_06/sumoarterial1x6_700_06/s9023 | 25 | DONE | 2026-07-29 | 256.9/4308/1293 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w4_700_06_see` |  |
| `stage0_w8_700_06_seed9040` | stage0_w8_700_06/sumoarterial1x6_700_06/s9040 | 25 | DONE | 2026-07-29 | 170.2/5231/1634 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w8_700_06_see` |  |
| `stage0_w8_700_06_seed9041` | stage0_w8_700_06/sumoarterial1x6_700_06/s9041 | 25 | DONE | 2026-07-29 | 243.8/5302/1315 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w8_700_06_see` |  |
| `stage0_w8_700_06_seed9042` | stage0_w8_700_06/sumoarterial1x6_700_06/s9042 | 25 | DONE | 2026-07-29 | 290.5/3815/1266 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w8_700_06_see` |  |
| `stage0_w8_700_06_seed9043` | stage0_w8_700_06/sumoarterial1x6_700_06/s9043 | 25 | DONE | 2026-07-29 | 298.2/3770/1529 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w8_700_06_see` |  |
| `stage0_w8_700_06_seed9044` | stage0_w8_700_06/sumoarterial1x6_700_06/s9044 | 25 | DONE | 2026-07-29 | 354.6/2736/1583 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w8_700_06_see` |  |
| `stage0_w8_700_06_seed9045` | stage0_w8_700_06/sumoarterial1x6_700_06/s9045 | 25 | DONE | 2026-07-29 | 209.1/4013/1597 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w8_700_06_see` |  |
| `stage0_w8_700_06_seed9046` | stage0_w8_700_06/sumoarterial1x6_700_06/s9046 | 25 | DONE | 2026-07-29 | 502.5/2110/1529 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w8_700_06_see` |  |
| `stage0_w8_700_06_seed9047` | stage0_w8_700_06/sumoarterial1x6_700_06/s9047 | 25 | DONE | 2026-07-29 | 192.4/5861/1563 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage0_w8_700_06_see` |  |
| `stage1_collector_300_03_seed2` | stage1_collector_300_03/sumoarterial1x6_300_03/s2 | 400 | DONE | 2026-07-29 | 90.3/4302/147 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage1_collector_300` |  |
| `stage1_collector_300_03_seed3` | stage1_collector_300_03/sumoarterial1x6_300_03/s3 | 400 | DONE | 2026-07-29 | 90.4/4308/141 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage1_collector_300` |  |
| `stage1_collector_300_03_seed4` | stage1_collector_300_03/sumoarterial1x6_300_03/s4 | 400 | DONE | 2026-07-29 | 90.2/4305/144 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage1_collector_300` |  |
| `stage1_collector_300_06_seed2` | stage1_collector_300_06/sumoarterial1x6_300_06/s2 | 400 | DONE | 2026-07-29 | 91.2/4358/107 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage1_collector_300` |  |
| `stage1_collector_300_06_seed3` | stage1_collector_300_06/sumoarterial1x6_300_06/s3 | 400 | DONE | 2026-07-29 | 91.1/4358/107 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage1_collector_300` |  |
| `stage1_collector_300_06_seed4` | stage1_collector_300_06/sumoarterial1x6_300_06/s4 | 400 | DONE | 2026-07-29 | 91.0/4351/114 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage1_collector_300` |  |
| `stage1_collector_700_03_seed2` | stage1_collector_700_03/sumoarterial1x6_700_03/s2 | 400 | DONE | 2026-07-29 | 99.3/9115/284 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage1_collector_700` |  |
| `stage1_collector_700_03_seed3` | stage1_collector_700_03/sumoarterial1x6_700_03/s3 | 400 | DONE | 2026-07-29 | 98.3/9117/282 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage1_collector_700` |  |
| `stage1_collector_700_03_seed4` | stage1_collector_700_03/sumoarterial1x6_700_03/s4 | 400 | DONE | 2026-07-29 | 98.1/9112/287 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage1_collector_700` |  |
| `stage1_collector_700_06_seed2` | stage1_collector_700_06/sumoarterial1x6_700_06/s2 | 400 | DONE | 2026-07-29 | 100.9/8804/198 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage1_collector_700` |  |
| `stage1_collector_700_06_seed3` | stage1_collector_700_06/sumoarterial1x6_700_06/s3 | 400 | DONE | 2026-07-29 | 103.3/8801/201 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage1_collector_700` |  |
| `stage1_collector_700_06_seed4` | stage1_collector_700_06/sumoarterial1x6_700_06/s4 | 400 | DONE | 2026-07-29 | 102.2/8807/195 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage1_collector_700` |  |
| `stage2_causal_semi_r25_order1_seed1000_stage1_300_06` | stage2_causal_semi_r25_order1/sumoarterial1x6_300_06/s1000 | 200 | DONE | 2026-07-30 | 367.5/2587/1155 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r25_order1_seed1000_stage2_300_03` | stage2_causal_semi_r25_order1/sumoarterial1x6_300_03/s1000 | 200 | DONE | 2026-07-30 | 90.8/4305/144 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r25_order1_seed1000_stage3_700_03` | stage2_causal_semi_r25_order1/sumoarterial1x6_700_03/s1000 | 200 | DONE | 2026-07-30 | 98.0/9117/282 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r25_order1_seed1000_stage4_700_06` | stage2_causal_semi_r25_order1/sumoarterial1x6_700_06/s1000 | 200 | DONE | 2026-07-30 | 100.4/8811/191 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r25_order1_seed1001_stage1_300_06` | stage2_causal_semi_r25_order1/sumoarterial1x6_300_06/s1001 | 200 | DONE | 2026-07-30 | 512.0/1210/1290 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r25_order1_seed1001_stage2_300_03` | stage2_causal_semi_r25_order1/sumoarterial1x6_300_03/s1001 | 200 | DONE | 2026-07-30 | 90.7/4304/145 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r25_order1_seed1001_stage3_700_03` | stage2_causal_semi_r25_order1/sumoarterial1x6_700_03/s1001 | 200 | DONE | 2026-07-30 | 98.4/9115/284 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r25_order1_seed1001_stage4_700_06` | stage2_causal_semi_r25_order1/sumoarterial1x6_700_06/s1001 | 200 | DONE | 2026-07-30 | 101.2/8814/188 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r25_order1_seed1002_stage1_300_06` | stage2_causal_semi_r25_order1/sumoarterial1x6_300_06/s1002 | 200 | DONE | 2026-07-30 | 310.3/2466/978 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r25_order1_seed1002_stage2_300_03` | stage2_causal_semi_r25_order1/sumoarterial1x6_300_03/s1002 | 200 | DONE | 2026-07-30 | 90.3/4305/144 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r25_order1_seed1002_stage3_700_03` | stage2_causal_semi_r25_order1/sumoarterial1x6_700_03/s1002 | 200 | DONE | 2026-07-30 | 97.8/9131/268 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r25_order1_seed1002_stage4_700_06` | stage2_causal_semi_r25_order1/sumoarterial1x6_700_06/s1002 | 200 | DONE | 2026-07-30 | 100.4/8807/195 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r50_order1_seed1000_stage1_300_06` | stage2_causal_semi_r50_order1/sumoarterial1x6_300_06/s1000 | 200 | DONE | 2026-07-30 | 367.5/2587/1155 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r50_order1_seed1000_stage2_300_03` | stage2_causal_semi_r50_order1/sumoarterial1x6_300_03/s1000 | 200 | DONE | 2026-07-30 | 90.9/4306/143 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r50_order1_seed1000_stage3_700_03` | stage2_causal_semi_r50_order1/sumoarterial1x6_700_03/s1000 | 200 | DONE | 2026-07-30 | 98.8/9116/283 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r50_order1_seed1000_stage4_700_06` | stage2_causal_semi_r50_order1/sumoarterial1x6_700_06/s1000 | 200 | DONE | 2026-07-30 | 100.2/8807/195 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r50_order1_seed1001_stage1_300_06` | stage2_causal_semi_r50_order1/sumoarterial1x6_300_06/s1001 | 200 | DONE | 2026-07-30 | 512.0/1210/1290 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r50_order1_seed1001_stage2_300_03` | stage2_causal_semi_r50_order1/sumoarterial1x6_300_03/s1001 | 200 | DONE | 2026-07-30 | 90.6/4301/148 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r50_order1_seed1001_stage3_700_03` | stage2_causal_semi_r50_order1/sumoarterial1x6_700_03/s1001 | 200 | DONE | 2026-07-30 | 99.3/9110/289 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r50_order1_seed1001_stage4_700_06` | stage2_causal_semi_r50_order1/sumoarterial1x6_700_06/s1001 | 200 | DONE | 2026-07-30 | 100.0/8809/193 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r50_order1_seed1002_stage1_300_06` | stage2_causal_semi_r50_order1/sumoarterial1x6_300_06/s1002 | 200 | DONE | 2026-07-30 | 310.3/2466/978 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r50_order1_seed1002_stage2_300_03` | stage2_causal_semi_r50_order1/sumoarterial1x6_300_03/s1002 | 200 | DONE | 2026-07-30 | 90.2/4296/153 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r50_order1_seed1002_stage3_700_03` | stage2_causal_semi_r50_order1/sumoarterial1x6_700_03/s1002 | 200 | DONE | 2026-07-30 | 98.3/9114/285 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r50_order1_seed1002_stage4_700_06` | stage2_causal_semi_r50_order1/sumoarterial1x6_700_06/s1002 | 200 | DONE | 2026-07-30 | 100.0/8810/192 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r75_order1_seed1000_stage1_300_06` | stage2_causal_semi_r75_order1/sumoarterial1x6_300_06/s1000 | 200 | DONE | 2026-07-30 | 367.5/2587/1155 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r75_order1_seed1000_stage2_300_03` | stage2_causal_semi_r75_order1/sumoarterial1x6_300_03/s1000 | 200 | DONE | 2026-07-30 | 91.4/4296/153 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r75_order1_seed1000_stage3_700_03` | stage2_causal_semi_r75_order1/sumoarterial1x6_700_03/s1000 | 200 | DONE | 2026-07-30 | 98.9/9118/281 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r75_order1_seed1000_stage4_700_06` | stage2_causal_semi_r75_order1/sumoarterial1x6_700_06/s1000 | 200 | DONE | 2026-07-30 | 101.0/8808/194 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r75_order1_seed1001_stage1_300_06` | stage2_causal_semi_r75_order1/sumoarterial1x6_300_06/s1001 | 200 | DONE | 2026-07-30 | 512.0/1210/1290 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r75_order1_seed1001_stage2_300_03` | stage2_causal_semi_r75_order1/sumoarterial1x6_300_03/s1001 | 200 | DONE | 2026-07-30 | 91.4/4299/150 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r75_order1_seed1001_stage3_700_03` | stage2_causal_semi_r75_order1/sumoarterial1x6_700_03/s1001 | 200 | DONE | 2026-07-30 | 99.0/9105/294 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r75_order1_seed1001_stage4_700_06` | stage2_causal_semi_r75_order1/sumoarterial1x6_700_06/s1001 | 200 | DONE | 2026-07-30 | 100.6/8808/194 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r75_order1_seed1002_stage1_300_06` | stage2_causal_semi_r75_order1/sumoarterial1x6_300_06/s1002 | 200 | DONE | 2026-07-30 | 310.3/2466/978 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r75_order1_seed1002_stage2_300_03` | stage2_causal_semi_r75_order1/sumoarterial1x6_300_03/s1002 | 200 | DONE | 2026-07-30 | 90.1/4306/143 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r75_order1_seed1002_stage3_700_03` | stage2_causal_semi_r75_order1/sumoarterial1x6_700_03/s1002 | 200 | DONE | 2026-07-30 | 99.8/9107/292 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_causal_semi_r` |  |
| `stage2_causal_semi_r75_order1_seed1002_stage4_700_06` | stage2_causal_semi_r75_order1/sumoarterial1x6_700_06/s1002 | 200 | DONE | 2026-07-30 | 100.8/8808/194 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_causal_semi_r` |  |
| `stage2_continue_online_clear_order1_seed1000_stage1_300_06` | stage2_continue_online_clear_order1/sumoarterial1x6_300_06/s1000 | 200 | DONE | 2026-07-30 | 90.8/4356/109 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_continue_onli` |  |
| `stage2_continue_online_clear_order1_seed1000_stage2_300_03` | stage2_continue_online_clear_order1/sumoarterial1x6_300_03/s1000 | 200 | DONE | 2026-07-30 | 90.5/4299/150 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_continue_onli` |  |
| `stage2_continue_online_clear_order1_seed1000_stage3_700_03` | stage2_continue_online_clear_order1/sumoarterial1x6_700_03/s1000 | 200 | DONE | 2026-07-30 | 99.0/9114/285 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_continue_onli` |  |
| `stage2_continue_online_clear_order1_seed1000_stage4_700_06` | stage2_continue_online_clear_order1/sumoarterial1x6_700_06/s1000 | 200 | DONE | 2026-07-30 | 102.2/8805/197 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_continue_onli` |  |
| `stage2_continue_online_clear_order1_seed1001_stage1_300_06` | stage2_continue_online_clear_order1/sumoarterial1x6_300_06/s1001 | 200 | DONE | 2026-07-30 | 90.5/4352/113 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_continue_onli` |  |
| `stage2_continue_online_clear_order1_seed1001_stage2_300_03` | stage2_continue_online_clear_order1/sumoarterial1x6_300_03/s1001 | 200 | DONE | 2026-07-30 | 90.0/4297/152 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_continue_onli` |  |
| `stage2_continue_online_clear_order1_seed1001_stage3_700_03` | stage2_continue_online_clear_order1/sumoarterial1x6_700_03/s1001 | 200 | DONE | 2026-07-30 | 98.0/9114/285 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_continue_onli` |  |
| `stage2_continue_online_clear_order1_seed1001_stage4_700_06` | stage2_continue_online_clear_order1/sumoarterial1x6_700_06/s1001 | 200 | DONE | 2026-07-30 | 99.8/8805/197 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_continue_onli` |  |
| `stage2_continue_online_clear_order1_seed1002_stage1_300_06` | stage2_continue_online_clear_order1/sumoarterial1x6_300_06/s1002 | 200 | DONE | 2026-07-30 | 90.8/4355/110 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_continue_onli` |  |
| `stage2_continue_online_clear_order1_seed1002_stage2_300_03` | stage2_continue_online_clear_order1/sumoarterial1x6_300_03/s1002 | 200 | DONE | 2026-07-30 | 90.7/4302/147 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_continue_onli` |  |
| `stage2_continue_online_clear_order1_seed1002_stage3_700_03` | stage2_continue_online_clear_order1/sumoarterial1x6_700_03/s1002 | 200 | DONE | 2026-07-30 | 99.7/9108/291 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_continue_onli` |  |
| `stage2_continue_online_clear_order1_seed1002_stage4_700_06` | stage2_continue_online_clear_order1/sumoarterial1x6_700_06/s1002 | 200 | DONE | 2026-07-30 | 100.5/8808/194 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_continue_onli` |  |
| `stage2_continue_online_fifo_order1_seed1000_stage1_300_06` | stage2_continue_online_fifo_order1/sumoarterial1x6_300_06/s1000 | 200 | DONE | 2026-07-30 | 90.8/4356/109 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_continue_onli` |  |
| `stage2_continue_online_fifo_order1_seed1000_stage2_300_03` | stage2_continue_online_fifo_order1/sumoarterial1x6_300_03/s1000 | 200 | DONE | 2026-07-30 | 90.3/4308/141 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_continue_onli` |  |
| `stage2_continue_online_fifo_order1_seed1000_stage3_700_03` | stage2_continue_online_fifo_order1/sumoarterial1x6_700_03/s1000 | 200 | DONE | 2026-07-30 | 99.1/9115/284 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_continue_onli` |  |
| `stage2_continue_online_fifo_order1_seed1000_stage4_700_06` | stage2_continue_online_fifo_order1/sumoarterial1x6_700_06/s1000 | 200 | DONE | 2026-07-30 | 100.1/8811/191 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_continue_onli` |  |
| `stage2_continue_online_fifo_order1_seed1001_stage1_300_06` | stage2_continue_online_fifo_order1/sumoarterial1x6_300_06/s1001 | 200 | DONE | 2026-07-30 | 90.5/4352/113 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_continue_onli` |  |
| `stage2_continue_online_fifo_order1_seed1001_stage2_300_03` | stage2_continue_online_fifo_order1/sumoarterial1x6_300_03/s1001 | 200 | DONE | 2026-07-30 | 90.6/4303/146 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_continue_onli` |  |
| `stage2_continue_online_fifo_order1_seed1001_stage3_700_03` | stage2_continue_online_fifo_order1/sumoarterial1x6_700_03/s1001 | 200 | DONE | 2026-07-30 | 98.6/9118/281 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_continue_onli` |  |
| `stage2_continue_online_fifo_order1_seed1001_stage4_700_06` | stage2_continue_online_fifo_order1/sumoarterial1x6_700_06/s1001 | 200 | DONE | 2026-07-30 | 100.2/8805/197 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_continue_onli` |  |
| `stage2_continue_online_fifo_order1_seed1002_stage1_300_06` | stage2_continue_online_fifo_order1/sumoarterial1x6_300_06/s1002 | 200 | DONE | 2026-07-30 | 90.8/4355/110 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_continue_onli` |  |
| `stage2_continue_online_fifo_order1_seed1002_stage2_300_03` | stage2_continue_online_fifo_order1/sumoarterial1x6_300_03/s1002 | 200 | DONE | 2026-07-30 | 90.5/4307/142 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_continue_onli` |  |
| `stage2_continue_online_fifo_order1_seed1002_stage3_700_03` | stage2_continue_online_fifo_order1/sumoarterial1x6_700_03/s1002 | 200 | DONE | 2026-07-30 | 97.7/9106/293 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_continue_onli` |  |
| `stage2_continue_online_fifo_order1_seed1002_stage4_700_06` | stage2_continue_online_fifo_order1/sumoarterial1x6_700_06/s1002 | 200 | DONE | 2026-07-30 | 100.6/8809/193 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_continue_onli` |  |
| `stage2_e2e_smoke_causal_r50_seed19000_stage1_300_06` | stage2_e2e_smoke_causal_r50/sumoarterial1x6_300_06/s19000 | 1 | DONE | 2026-07-30 | 255.9/2050/1241 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_e2e_smoke_cau` |  |
| `stage2_e2e_smoke_causal_r50_seed19000_stage2_300_03` | stage2_e2e_smoke_causal_r50/sumoarterial1x6_300_03/s19000 | 1 | DONE | 2026-07-30 | 250.1/2108/1286 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_e2e_smoke_cau` |  |
| `stage2_e2e_smoke_causal_r50_seed19000_stage3_700_03` | stage2_e2e_smoke_causal_r50/sumoarterial1x6_700_03/s19000 | 1 | DONE | 2026-07-30 | 209.4/3884/1654 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_e2e_smoke_cau` |  |
| `stage2_e2e_smoke_causal_r50_seed19000_stage4_700_06` | stage2_e2e_smoke_causal_r50/sumoarterial1x6_700_06/s19000 | 1 | DONE | 2026-07-30 | 131.7/5736/1456 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_e2e_smoke_cau` |  |
| `stage2_low_epsilon_continue_online_diagnostic_order1_seed100` | stage2_low_epsilon_continue_online_diagnostic_order1/sumoarterial1x6_300_06/s1000 | 200 | DONE | 2026-07-30 | 90.8/4356/109 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_low_epsilon_c` |  |
| `stage2_low_epsilon_continue_online_diagnostic_order1_seed100` | stage2_low_epsilon_continue_online_diagnostic_order1/sumoarterial1x6_300_03/s1000 | 200 | DONE | 2026-07-30 | 90.4/4301/148 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_low_epsilon_c` |  |
| `stage2_low_epsilon_continue_online_diagnostic_order1_seed100` | stage2_low_epsilon_continue_online_diagnostic_order1/sumoarterial1x6_700_03/s1000 | 200 | DONE | 2026-07-30 | 98.4/9113/286 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_low_epsilon_c` |  |
| `stage2_low_epsilon_continue_online_diagnostic_order1_seed100` | stage2_low_epsilon_continue_online_diagnostic_order1/sumoarterial1x6_700_06/s1000 | 200 | DONE | 2026-07-30 | 102.4/8806/196 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_low_epsilon_c` |  |
| `stage2_low_epsilon_continue_online_diagnostic_order1_seed100` | stage2_low_epsilon_continue_online_diagnostic_order1/sumoarterial1x6_300_06/s1001 | 200 | DONE | 2026-07-30 | 90.5/4352/113 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_low_epsilon_c` |  |
| `stage2_low_epsilon_continue_online_diagnostic_order1_seed100` | stage2_low_epsilon_continue_online_diagnostic_order1/sumoarterial1x6_300_03/s1001 | 200 | DONE | 2026-07-30 | 90.2/4301/148 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_low_epsilon_c` |  |
| `stage2_low_epsilon_continue_online_diagnostic_order1_seed100` | stage2_low_epsilon_continue_online_diagnostic_order1/sumoarterial1x6_700_03/s1001 | 200 | DONE | 2026-07-30 | 99.0/9115/284 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_low_epsilon_c` |  |
| `stage2_low_epsilon_continue_online_diagnostic_order1_seed100` | stage2_low_epsilon_continue_online_diagnostic_order1/sumoarterial1x6_700_06/s1001 | 200 | DONE | 2026-07-30 | 100.6/8808/194 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_low_epsilon_c` |  |
| `stage2_low_epsilon_continue_online_diagnostic_order1_seed100` | stage2_low_epsilon_continue_online_diagnostic_order1/sumoarterial1x6_300_06/s1002 | 200 | DONE | 2026-07-30 | 90.8/4355/110 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_low_epsilon_c` |  |
| `stage2_low_epsilon_continue_online_diagnostic_order1_seed100` | stage2_low_epsilon_continue_online_diagnostic_order1/sumoarterial1x6_300_03/s1002 | 200 | DONE | 2026-07-30 | 90.3/4306/143 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_low_epsilon_c` |  |
| `stage2_low_epsilon_continue_online_diagnostic_order1_seed100` | stage2_low_epsilon_continue_online_diagnostic_order1/sumoarterial1x6_700_03/s1002 | 200 | DONE | 2026-07-30 | 98.4/9119/280 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_low_epsilon_c` |  |
| `stage2_low_epsilon_continue_online_diagnostic_order1_seed100` | stage2_low_epsilon_continue_online_diagnostic_order1/sumoarterial1x6_700_06/s1002 | 200 | DONE | 2026-07-30 | 100.7/8806/196 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_low_epsilon_c` |  |
| `stage2_non_causal_full_history_upper_bound_r50_order1_seed10` | stage2_non_causal_full_history_upper_bound_r50_order1/sumoarterial1x6_300_06/s1000 | 200 | DONE | 2026-07-30 | 91.6/4348/117 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_non_causal_fu` |  |
| `stage2_non_causal_full_history_upper_bound_r50_order1_seed10` | stage2_non_causal_full_history_upper_bound_r50_order1/sumoarterial1x6_300_03/s1000 | 200 | DONE | 2026-07-30 | 90.5/4300/149 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_non_causal_fu` |  |
| `stage2_non_causal_full_history_upper_bound_r50_order1_seed10` | stage2_non_causal_full_history_upper_bound_r50_order1/sumoarterial1x6_700_03/s1000 | 200 | DONE | 2026-07-30 | 99.1/9122/277 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_non_causal_fu` |  |
| `stage2_non_causal_full_history_upper_bound_r50_order1_seed10` | stage2_non_causal_full_history_upper_bound_r50_order1/sumoarterial1x6_700_06/s1000 | 200 | DONE | 2026-07-30 | 100.8/8809/193 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_non_causal_fu` |  |
| `stage2_non_causal_full_history_upper_bound_r50_order1_seed10` | stage2_non_causal_full_history_upper_bound_r50_order1/sumoarterial1x6_300_06/s1001 | 200 | DONE | 2026-07-30 | 91.3/4349/116 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_non_causal_fu` |  |
| `stage2_non_causal_full_history_upper_bound_r50_order1_seed10` | stage2_non_causal_full_history_upper_bound_r50_order1/sumoarterial1x6_300_03/s1001 | 200 | DONE | 2026-07-30 | 90.7/4300/149 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_non_causal_fu` |  |
| `stage2_non_causal_full_history_upper_bound_r50_order1_seed10` | stage2_non_causal_full_history_upper_bound_r50_order1/sumoarterial1x6_700_03/s1001 | 200 | DONE | 2026-07-30 | 99.1/9103/296 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_non_causal_fu` |  |
| `stage2_non_causal_full_history_upper_bound_r50_order1_seed10` | stage2_non_causal_full_history_upper_bound_r50_order1/sumoarterial1x6_700_06/s1001 | 200 | DONE | 2026-07-30 | 102.8/8809/193 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_non_causal_fu` |  |
| `stage2_non_causal_full_history_upper_bound_r50_order1_seed10` | stage2_non_causal_full_history_upper_bound_r50_order1/sumoarterial1x6_300_06/s1002 | 200 | DONE | 2026-07-30 | 91.0/4351/114 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage2_non_causal_fu` |  |
| `stage2_non_causal_full_history_upper_bound_r50_order1_seed10` | stage2_non_causal_full_history_upper_bound_r50_order1/sumoarterial1x6_300_03/s1002 | 200 | DONE | 2026-07-30 | 90.5/4296/153 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage2_non_causal_fu` |  |
| `stage2_non_causal_full_history_upper_bound_r50_order1_seed10` | stage2_non_causal_full_history_upper_bound_r50_order1/sumoarterial1x6_700_03/s1002 | 200 | DONE | 2026-07-30 | 97.8/9121/278 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage2_non_causal_fu` |  |
| `stage2_non_causal_full_history_upper_bound_r50_order1_seed10` | stage2_non_causal_full_history_upper_bound_r50_order1/sumoarterial1x6_700_06/s1002 | 200 | DONE | 2026-07-30 | 99.7/8804/198 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage2_non_causal_fu` |  |
| `sumo_shared_dqn__sumoarterial1x6_300_03__stage1_collector_30` | stage1_collector_300_03/sumoarterial1x6_300_03/s0 | 400 | DONE | 2026-07-29 | 90.6/4302/147 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage1_collector_300` |  |
| `sumo_shared_dqn__sumoarterial1x6_300_03__stage1_collector_30` | stage1_collector_300_03/sumoarterial1x6_300_03/s1 | 400 | DONE | 2026-07-29 | 90.2/4298/151 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_03/stage1_collector_300` |  |
| `sumo_shared_dqn__sumoarterial1x6_300_06__stage1_collector_30` | stage1_collector_300_06/sumoarterial1x6_300_06/s0 | 400 | DONE | 2026-07-29 | 90.8/4355/110 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage1_collector_300` |  |
| `sumo_shared_dqn__sumoarterial1x6_300_06__stage1_collector_30` | stage1_collector_300_06/sumoarterial1x6_300_06/s1 | 400 | DONE | 2026-07-29 | 90.5/4353/112 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_300_06/stage1_collector_300` |  |
| `sumo_shared_dqn__sumoarterial1x6_700_03__stage1_collector_70` | stage1_collector_700_03/sumoarterial1x6_700_03/s0 | 400 | DONE | 2026-07-29 | 98.8/9109/290 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage1_collector_700` |  |
| `sumo_shared_dqn__sumoarterial1x6_700_03__stage1_collector_70` | stage1_collector_700_03/sumoarterial1x6_700_03/s1 | 400 | DONE | 2026-07-29 | 98.6/9111/288 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_03/stage1_collector_700` |  |
| `sumo_shared_dqn__sumoarterial1x6_700_06__stage1_collector_70` | stage1_collector_700_06/sumoarterial1x6_700_06/s0 | 400 | DONE | 2026-07-29 | 100.6/8804/198 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage1_collector_700` |  |
| `sumo_shared_dqn__sumoarterial1x6_700_06__stage1_collector_70` | stage1_collector_700_06/sumoarterial1x6_700_06/s1 | 400 | DONE | 2026-07-29 | 100.6/8806/196 | `data/output_data/tsc/sumo_shared_dqn/sumoarterial1x6_700_06/stage1_collector_700` |  |

评估 attempt 共 352 行，按包/来源聚合：

| 评估包/来源 | n DONE/FAILED/ABORTED | 条件集 | eval seeds |
|---|---|---|---|
| `stage_1` | 88/0/0 | — | — |
| `stage_2` | 88/0/0 | — | — |
| `stage_3` | 88/0/0 | — | — |
| `stage_4` | 88/0/0 | — | — |

## TARL 复现线（v1/graph/fixed/mp/smoke）

- 假设：复现 TARL 论文机制并做机制审计（v21 前置）。
- plan_id/队列：reproduction/tarl_tsc
- 设计稿/证据位置：reproduction/tarl_tsc/{scripts,tests,provenance.py}
- 结论：132 完成、11 失败（含审计/契约 smoke）；为 v21 正式批提供机制证据。
- 登记单元 143（train/eval/smoke/batch/calibration = 115/0/28/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `tarl_dqn_pilot5_20260914` | tarl_dqn_pilot5_20260914/hz4x4/s0 | 5 | DONE | 2026-09-14 | 629.2/925/1515 | `data/output_data/tsc/sumo_dqn/hz4x4/tarl_dqn_pilot5_20260914` |  |
| `tarl_dqn_smoke_20260914` | tarl_dqn_smoke_20260914/hz4x4/s0 | — | FAILED | — | — | `data/output_data/tsc/sumo_dqn/hz4x4/tarl_dqn_smoke_20260914` |  |
| `tarl_dqn_smoke_20260914_r2` | tarl_dqn_smoke_20260914_r2/hz4x4/s0 | — | FAILED | — | — | `data/output_data/tsc/sumo_dqn/hz4x4/tarl_dqn_smoke_20260914_r2` |  |
| `tarl_dqn_smoke_20260914_r3` | tarl_dqn_smoke_20260914_r3/hz4x4/s0 | 0 | FAILED | 2026-09-14 | 629.2/925/1515 | `data/output_data/tsc/sumo_dqn/hz4x4/tarl_dqn_smoke_20260914_r3` |  |
| `tarl_dqn_smoke_20260914_r4` | tarl_dqn_smoke_20260914_r4/hz4x4/s0 | 0 | FAILED | 2026-09-14 | 629.2/925/1515 | `data/output_data/tsc/sumo_dqn/hz4x4/tarl_dqn_smoke_20260914_r4` |  |
| `tarl_dqn_smoke_20260914_r5` | tarl_dqn_smoke_20260914_r5/hz4x4/s0 | 0 | FAILED | 2026-09-14 | 629.2/925/1515 | `data/output_data/tsc/sumo_dqn/hz4x4/tarl_dqn_smoke_20260914_r5` |  |
| `tarl_dqn_smoke_20260914_r6` | tarl_dqn_smoke_20260914_r6/hz4x4/s0 | 1 | DONE | 2026-09-14 | 629.2/925/1515 | `data/output_data/tsc/sumo_dqn/hz4x4/tarl_dqn_smoke_20260914_r6` |  |
| `tarl_graph_v1_attention_single_pilot2` | tarl_graph_v1_attention_single_pilot2/hz4x4/s7 | 2 | DONE | 2026-09-15 | 546.8/1006/1328 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_graph_v1_attention_single_pi` |  |
| `tarl_graph_v1_concat_single_pilot2` | tarl_graph_v1_concat_single_pilot2/hz4x4/s7 | 2 | DONE | 2026-09-15 | 561.1/694/1416 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_graph_v1_concat_single_pilot2` |  |
| `tarl_graph_v1_gat_single_pilot2` | tarl_graph_v1_gat_single_pilot2/hz4x4/s7 | 2 | DONE | 2026-09-15 | 630.2/714/1442 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_graph_v1_gat_single_pilot2` |  |
| `tarl_graph_v1_gating_single_pilot2` | tarl_graph_v1_gating_single_pilot2/hz4x4/s7 | 2 | DONE | 2026-09-15 | 561.8/699/1408 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_graph_v1_gating_single_pilot2` |  |
| `tarl_graph_v1_sensor_single_pilot2` | tarl_graph_v1_sensor_single_pilot2/hz4x4/s7 | 2 | DONE | 2026-09-15 | 526.6/1076/1357 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_graph_v1_sensor_single_pilot2` |  |
| `tarl_smoke_20260914` | tarl_smoke_20260914/hz4x4/s0 | — | FAILED | 2026-09-14 | — | `data/output_data/tsc/sumo_fixedtime/hz4x4/tarl_smoke_20260914` |  |
| `tarl_smoke_20260914_r2` | tarl_smoke_20260914_r2/hz4x4/s0 | — | FAILED | 2026-09-14 | — | `data/output_data/tsc/sumo_fixedtime/hz4x4/tarl_smoke_20260914_r2` |  |
| `tarl_smoke_20260914_r3` | tarl_smoke_20260914_r3/hz4x4/s0 | 0 | DONE | 2026-09-14 | — | `data/output_data/tsc/sumo_fixedtime/hz4x4/tarl_smoke_20260914_r3` |  |
| `tarl_v1_text_contract_canonical2` | tarl_v1_text_contract_canonical2/hz4x4/s7 | 2 | DONE | 2026-09-15 | 561.8/699/1408 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_text_contract_canonical2` |  |
| `tarl_v1_text_contract_empty2` | tarl_v1_text_contract_empty2/hz4x4/s7 | 2 | DONE | 2026-09-15 | 553.8/699/1408 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_text_contract_empty2` |  |
| `tarl_v1_text_contract_shifted2` | tarl_v1_text_contract_shifted2/hz4x4/s7 | 2 | DONE | 2026-09-15 | 558.8/699/1408 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_text_contract_shifted2` |  |
| `tarl_v21_attention_protocol_pilot` | tarl_v21_attention_protocol_pilot/hz4x4/s7 | 2 | FAILED | 2026-09-15 | 542.4/979/1356 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v21_attention_protocol_pilot` |  |
| `tarl_v21_attention_protocol_pilot_retry1` | tarl_v21_attention_protocol_pilot_retry1/hz4x4/s7 | 2 | DONE | 2026-09-15 | 542.4/979/1356 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v21_attention_protocol_pilot` |  |
| `tarl_v21_attention_resume_audit2` | tarl_v21_attention_resume_audit2/hz4x4/s7 | 3 | DONE | 2026-09-15 | 520.4/2017/958 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v21_attention_resume_audit2` |  |
| `tarl_v21_gat_protocol_pilot` | tarl_v21_gat_protocol_pilot/hz4x4/s7 | 2 | FAILED | 2026-09-15 | 579.8/693/1443 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v21_gat_protocol_pilot` |  |
| `tarl_v21_gat_protocol_pilot_retry1` | tarl_v21_gat_protocol_pilot_retry1/hz4x4/s7 | 2 | DONE | 2026-09-15 | 579.8/693/1443 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v21_gat_protocol_pilot_retry1` |  |
| `tarl_v21_gating_explicit_seed_resume_audit` | tarl_v21_gating_explicit_seed_resume_audit/hz4x4/s7 | 3 | DONE | 2026-09-15 | 448.4/1193/1081 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v21_gating_explicit_seed_resume` |  |
| `tarl_v21_gating_protocol_pilot` | tarl_v21_gating_protocol_pilot/hz4x4/s7 | 2 | FAILED | 2026-09-15 | 556.6/700/1407 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v21_gating_protocol_pilot` |  |
| `tarl_v21_gating_protocol_pilot_retry1` | tarl_v21_gating_protocol_pilot_retry1/hz4x4/s7 | 2 | DONE | 2026-09-15 | 556.6/700/1407 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v21_gating_protocol_pilot_retry` |  |
| `tarl_v21_gating_resume_audit1` | tarl_v21_gating_resume_audit1/hz4x4/s7 | 2 | DONE | 2026-09-15 | 574.4/1620/1096 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v21_gating_resume_audit1` |  |
| `tarl_v21_gating_resume_audit2` | tarl_v21_gating_resume_audit2/hz4x4/s7 | 3 | DONE | 2026-09-15 | 574.4/1620/1096 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v21_gating_resume_audit2` |  |
| `tarl_fixed_s17_20260914` | tarl_fixed_s17_20260914/hz4x4/s17 | 0 | DONE | 2026-09-14 | — | `data/output_data/tsc/sumo_fixedtime/hz4x4/tarl_fixed_s17_20260914` |  |
| `tarl_fixed_s27_20260914` | tarl_fixed_s27_20260914/hz4x4/s27 | 0 | DONE | 2026-09-14 | — | `data/output_data/tsc/sumo_fixedtime/hz4x4/tarl_fixed_s27_20260914` |  |
| `tarl_fixed_s37_20260914` | tarl_fixed_s37_20260914/hz4x4/s37 | 0 | DONE | 2026-09-14 | — | `data/output_data/tsc/sumo_fixedtime/hz4x4/tarl_fixed_s37_20260914` |  |
| `tarl_fixed_s47_20260914` | tarl_fixed_s47_20260914/hz4x4/s47 | 0 | DONE | 2026-09-14 | — | `data/output_data/tsc/sumo_fixedtime/hz4x4/tarl_fixed_s47_20260914` |  |
| `tarl_fixed_s7_20260914` | tarl_fixed_s7_20260914/hz4x4/s7 | 0 | DONE | 2026-09-14 | — | `data/output_data/tsc/sumo_fixedtime/hz4x4/tarl_fixed_s7_20260914` |  |
| `tarl_maxpressure_20260914` | tarl_maxpressure_20260914/hz4x4/s0 | — | FAILED | 2026-09-14 | — | `data/output_data/tsc/sumo_maxpressure/hz4x4/tarl_maxpressure_20260914` |  |
| `tarl_maxpressure_20260914_r2` | tarl_maxpressure_20260914_r2/hz4x4/s0 | 0 | DONE | 2026-09-14 | — | `data/output_data/tsc/sumo_maxpressure/hz4x4/tarl_maxpressure_20260914_r2` |  |
| `tarl_mp_s17_20260914` | tarl_mp_s17_20260914/hz4x4/s17 | 0 | DONE | 2026-09-14 | — | `data/output_data/tsc/sumo_maxpressure/hz4x4/tarl_mp_s17_20260914` |  |
| `tarl_mp_s27_20260914` | tarl_mp_s27_20260914/hz4x4/s27 | 0 | DONE | 2026-09-14 | — | `data/output_data/tsc/sumo_maxpressure/hz4x4/tarl_mp_s27_20260914` |  |
| `tarl_mp_s37_20260914` | tarl_mp_s37_20260914/hz4x4/s37 | 0 | DONE | 2026-09-14 | — | `data/output_data/tsc/sumo_maxpressure/hz4x4/tarl_mp_s37_20260914` |  |
| `tarl_mp_s47_20260914` | tarl_mp_s47_20260914/hz4x4/s47 | 0 | DONE | 2026-09-14 | — | `data/output_data/tsc/sumo_maxpressure/hz4x4/tarl_mp_s47_20260914` |  |
| `tarl_mp_s7_20260914` | tarl_mp_s7_20260914/hz4x4/s7 | 0 | DONE | 2026-09-14 | — | `data/output_data/tsc/sumo_maxpressure/hz4x4/tarl_mp_s7_20260914` |  |
| `tarl_v1__v0fjq45_gating_multi-event_seed17_empty` | tarl_v1__v0fjq45_gating_multi-event_seed17_empty/hz4x4/s17 | 50 | DONE | 2026-09-15 | 355.9/2708/275 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_multi-event_` |  |
| `tarl_v1__v0fjq45_gating_multi-event_seed17_shifted` | tarl_v1__v0fjq45_gating_multi-event_seed17_shifted/hz4x4/s17 | 50 | DONE | 2026-09-15 | 339.2/2737/246 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_multi-event_` |  |
| `tarl_v1__v0fjq45_gating_multi-event_seed27_empty` | tarl_v1__v0fjq45_gating_multi-event_seed27_empty/hz4x4/s27 | 50 | DONE | 2026-09-15 | 339.0/2741/242 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_multi-event_` |  |
| `tarl_v1__v0fjq45_gating_multi-event_seed27_shifted` | tarl_v1__v0fjq45_gating_multi-event_seed27_shifted/hz4x4/s27 | 50 | DONE | 2026-09-15 | 338.6/2741/242 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_multi-event_` |  |
| `tarl_v1__v0fjq45_gating_multi-event_seed37_empty` | tarl_v1__v0fjq45_gating_multi-event_seed37_empty/hz4x4/s37 | 50 | DONE | 2026-09-15 | 338.8/2741/242 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_multi-event_` |  |
| `tarl_v1__v0fjq45_gating_multi-event_seed37_shifted` | tarl_v1__v0fjq45_gating_multi-event_seed37_shifted/hz4x4/s37 | 50 | DONE | 2026-09-15 | 338.8/2739/244 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_multi-event_` |  |
| `tarl_v1__v0fjq45_gating_multi-event_seed47_empty` | tarl_v1__v0fjq45_gating_multi-event_seed47_empty/hz4x4/s47 | 50 | DONE | 2026-09-15 | 338.3/2740/243 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_multi-event_` |  |
| `tarl_v1__v0fjq45_gating_multi-event_seed47_shifted` | tarl_v1__v0fjq45_gating_multi-event_seed47_shifted/hz4x4/s47 | 50 | DONE | 2026-09-15 | 339.8/2737/246 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_multi-event_` |  |
| `tarl_v1__v0fjq45_gating_multi-event_seed7_empty` | tarl_v1__v0fjq45_gating_multi-event_seed7_empty/hz4x4/s7 | 50 | DONE | 2026-09-15 | 339.5/2741/242 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_multi-event_` |  |
| `tarl_v1__v0fjq45_gating_multi-event_seed7_shifted` | tarl_v1__v0fjq45_gating_multi-event_seed7_shifted/hz4x4/s7 | 50 | DONE | 2026-09-15 | 338.7/2738/245 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_multi-event_` |  |
| `tarl_v1__v0fjq45_gating_single-event_seed17_empty` | tarl_v1__v0fjq45_gating_single-event_seed17_empty/hz4x4/s17 | 50 | DONE | 2026-09-15 | 333.9/2739/244 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_single-event` |  |
| `tarl_v1__v0fjq45_gating_single-event_seed17_shifted` | tarl_v1__v0fjq45_gating_single-event_seed17_shifted/hz4x4/s17 | 50 | DONE | 2026-09-15 | 333.2/2739/244 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_single-event` |  |
| `tarl_v1__v0fjq45_gating_single-event_seed27_empty` | tarl_v1__v0fjq45_gating_single-event_seed27_empty/hz4x4/s27 | 50 | DONE | 2026-09-15 | 333.4/2743/240 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_single-event` |  |
| `tarl_v1__v0fjq45_gating_single-event_seed27_shifted` | tarl_v1__v0fjq45_gating_single-event_seed27_shifted/hz4x4/s27 | 50 | DONE | 2026-09-15 | 333.9/2738/245 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_single-event` |  |
| `tarl_v1__v0fjq45_gating_single-event_seed37_empty` | tarl_v1__v0fjq45_gating_single-event_seed37_empty/hz4x4/s37 | 50 | DONE | 2026-09-15 | 334.5/2737/246 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_single-event` |  |
| `tarl_v1__v0fjq45_gating_single-event_seed37_shifted` | tarl_v1__v0fjq45_gating_single-event_seed37_shifted/hz4x4/s37 | 50 | DONE | 2026-09-15 | 334.6/2740/243 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_single-event` |  |
| `tarl_v1__v0fjq45_gating_single-event_seed47_empty` | tarl_v1__v0fjq45_gating_single-event_seed47_empty/hz4x4/s47 | 50 | DONE | 2026-09-15 | 333.7/2740/243 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_single-event` |  |
| `tarl_v1__v0fjq45_gating_single-event_seed47_shifted` | tarl_v1__v0fjq45_gating_single-event_seed47_shifted/hz4x4/s47 | 50 | DONE | 2026-09-15 | 333.6/2743/240 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_single-event` |  |
| `tarl_v1__v0fjq45_gating_single-event_seed7_empty` | tarl_v1__v0fjq45_gating_single-event_seed7_empty/hz4x4/s7 | 50 | DONE | 2026-09-15 | 333.8/2739/244 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_single-event` |  |
| `tarl_v1__v0fjq45_gating_single-event_seed7_shifted` | tarl_v1__v0fjq45_gating_single-event_seed7_shifted/hz4x4/s7 | 50 | DONE | 2026-09-15 | 333.8/2743/240 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1__v0fjq45_gating_single-event` |  |
| `tarl_v1_attention_interrupted_resume2` | tarl_v1_attention_interrupted_resume2/hz4x4/s7 | 2 | DONE | 2026-09-15 | 546.8/1006/1328 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_attention_interrupted_res` |  |
| `tarl_v1_attention_normal_seed7_validation50` | tarl_v1_attention_normal_seed7_validation50/hz4x4/s7 | 50 | DONE | 2026-09-15 | 332.2/2742/241 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_attention_normal_seed7_va` |  |
| `tarl_v1_gat_normal_seed7_convergence50` | tarl_v1_gat_normal_seed7_convergence50/hz4x4/s7 | 50 | DONE | 2026-09-15 | 332.4/2739/244 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_gat_normal_seed7_convergence50` |  |
| `tarl_v1_gating_normal_seed7_validation50` | tarl_v1_gating_normal_seed7_validation50/hz4x4/s7 | 50 | DONE | 2026-09-15 | 331.7/2737/246 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_gating_normal_seed7_validati` |  |
| `tarl_v1_resume_learned_interrupted4` | tarl_v1_resume_learned_interrupted4/hz4x4/s7 | 4 | DONE | 2026-09-15 | 557.0/1841/1132 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_resume_learned_interrupte` |  |
| `tarl_v1_resume_learned_reference4` | tarl_v1_resume_learned_reference4/hz4x4/s7 | 4 | DONE | 2026-09-15 | 557.0/1841/1132 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_resume_learned_reference4` |  |
| `tarl_v1_sensor_normal_seed17_convergence50` | tarl_v1_sensor_normal_seed17_convergence50/hz4x4/s17 | 50 | DONE | 2026-09-15 | 331.3/2739/244 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_sensor_normal_seed17_converg` |  |
| `tarl_v1_sensor_normal_seed7_convergence50` | tarl_v1_sensor_normal_seed7_convergence50/hz4x4/s7 | 50 | DONE | 2026-09-15 | 332.1/2738/245 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_sensor_normal_seed7_converge` |  |
| `tarl_v1_ykz6h7nw_attention_multi-event_seed17` | tarl_v1_ykz6h7nw_attention_multi-event_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 340.2/2738/245 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_multi-` |  |
| `tarl_v1_ykz6h7nw_attention_multi-event_seed27` | tarl_v1_ykz6h7nw_attention_multi-event_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 339.3/2733/250 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_multi-` |  |
| `tarl_v1_ykz6h7nw_attention_multi-event_seed37` | tarl_v1_ykz6h7nw_attention_multi-event_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 341.1/2736/247 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_multi-` |  |
| `tarl_v1_ykz6h7nw_attention_multi-event_seed47` | tarl_v1_ykz6h7nw_attention_multi-event_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 339.8/2739/244 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_multi-` |  |
| `tarl_v1_ykz6h7nw_attention_multi-event_seed7` | tarl_v1_ykz6h7nw_attention_multi-event_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 339.9/2740/243 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_multi-` |  |
| `tarl_v1_ykz6h7nw_attention_normal_seed17` | tarl_v1_ykz6h7nw_attention_normal_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 332.4/2737/246 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_normal` |  |
| `tarl_v1_ykz6h7nw_attention_normal_seed27` | tarl_v1_ykz6h7nw_attention_normal_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 333.1/2740/243 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_normal` |  |
| `tarl_v1_ykz6h7nw_attention_normal_seed37` | tarl_v1_ykz6h7nw_attention_normal_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 332.9/2739/244 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_normal` |  |
| `tarl_v1_ykz6h7nw_attention_normal_seed47` | tarl_v1_ykz6h7nw_attention_normal_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 332.6/2741/242 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_normal` |  |
| `tarl_v1_ykz6h7nw_attention_normal_seed7` | tarl_v1_ykz6h7nw_attention_normal_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 332.2/2742/241 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_normal` |  |
| `tarl_v1_ykz6h7nw_attention_single-event_seed17` | tarl_v1_ykz6h7nw_attention_single-event_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 334.1/2738/245 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_single` |  |
| `tarl_v1_ykz6h7nw_attention_single-event_seed27` | tarl_v1_ykz6h7nw_attention_single-event_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 333.8/2740/243 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_single` |  |
| `tarl_v1_ykz6h7nw_attention_single-event_seed37` | tarl_v1_ykz6h7nw_attention_single-event_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 334.8/2735/248 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_single` |  |
| `tarl_v1_ykz6h7nw_attention_single-event_seed47` | tarl_v1_ykz6h7nw_attention_single-event_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 334.1/2739/244 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_single` |  |
| `tarl_v1_ykz6h7nw_attention_single-event_seed7` | tarl_v1_ykz6h7nw_attention_single-event_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 333.2/2740/243 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/tarl_v1_ykz6h7nw_attention_single` |  |
| `tarl_v1_ykz6h7nw_concat_multi-event_seed17` | tarl_v1_ykz6h7nw_concat_multi-event_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 339.1/2739/244 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_multi-event_` |  |
| `tarl_v1_ykz6h7nw_concat_multi-event_seed27` | tarl_v1_ykz6h7nw_concat_multi-event_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 338.4/2738/245 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_multi-event_` |  |
| `tarl_v1_ykz6h7nw_concat_multi-event_seed37` | tarl_v1_ykz6h7nw_concat_multi-event_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 338.3/2739/244 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_multi-event_` |  |
| `tarl_v1_ykz6h7nw_concat_multi-event_seed47` | tarl_v1_ykz6h7nw_concat_multi-event_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 337.5/2740/243 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_multi-event_` |  |
| `tarl_v1_ykz6h7nw_concat_multi-event_seed7` | tarl_v1_ykz6h7nw_concat_multi-event_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 340.7/2739/244 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_multi-event_` |  |
| `tarl_v1_ykz6h7nw_concat_normal_seed17` | tarl_v1_ykz6h7nw_concat_normal_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 331.8/2741/242 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_normal_seed1` |  |
| `tarl_v1_ykz6h7nw_concat_normal_seed27` | tarl_v1_ykz6h7nw_concat_normal_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 331.9/2739/244 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_normal_seed2` |  |
| `tarl_v1_ykz6h7nw_concat_normal_seed37` | tarl_v1_ykz6h7nw_concat_normal_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 333.4/2740/243 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_normal_seed3` |  |
| `tarl_v1_ykz6h7nw_concat_normal_seed47` | tarl_v1_ykz6h7nw_concat_normal_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 332.6/2737/246 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_normal_seed4` |  |
| `tarl_v1_ykz6h7nw_concat_normal_seed7` | tarl_v1_ykz6h7nw_concat_normal_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 334.6/2740/243 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_normal_seed7` |  |
| `tarl_v1_ykz6h7nw_concat_single-event_seed17` | tarl_v1_ykz6h7nw_concat_single-event_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 334.6/2742/241 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_single-event` |  |
| `tarl_v1_ykz6h7nw_concat_single-event_seed27` | tarl_v1_ykz6h7nw_concat_single-event_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 335.0/2738/245 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_single-event` |  |
| `tarl_v1_ykz6h7nw_concat_single-event_seed37` | tarl_v1_ykz6h7nw_concat_single-event_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 333.8/2737/246 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_single-event` |  |
| `tarl_v1_ykz6h7nw_concat_single-event_seed47` | tarl_v1_ykz6h7nw_concat_single-event_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 333.4/2740/243 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_single-event` |  |
| `tarl_v1_ykz6h7nw_concat_single-event_seed7` | tarl_v1_ykz6h7nw_concat_single-event_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 334.7/2734/249 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/tarl_v1_ykz6h7nw_concat_single-event` |  |
| `tarl_v1_ykz6h7nw_gat_multi-event_seed17` | tarl_v1_ykz6h7nw_gat_multi-event_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 338.2/2739/244 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_multi-event_seed17` |  |
| `tarl_v1_ykz6h7nw_gat_multi-event_seed27` | tarl_v1_ykz6h7nw_gat_multi-event_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 337.3/2742/241 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_multi-event_seed27` |  |
| `tarl_v1_ykz6h7nw_gat_multi-event_seed37` | tarl_v1_ykz6h7nw_gat_multi-event_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 337.9/2741/242 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_multi-event_seed37` |  |
| `tarl_v1_ykz6h7nw_gat_multi-event_seed47` | tarl_v1_ykz6h7nw_gat_multi-event_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 338.1/2742/241 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_multi-event_seed47` |  |
| `tarl_v1_ykz6h7nw_gat_multi-event_seed7` | tarl_v1_ykz6h7nw_gat_multi-event_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 337.5/2741/242 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_multi-event_seed7` |  |
| `tarl_v1_ykz6h7nw_gat_normal_seed17` | tarl_v1_ykz6h7nw_gat_normal_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 331.8/2740/243 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_normal_seed17` |  |
| `tarl_v1_ykz6h7nw_gat_normal_seed27` | tarl_v1_ykz6h7nw_gat_normal_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 331.8/2739/244 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_normal_seed27` |  |
| `tarl_v1_ykz6h7nw_gat_normal_seed37` | tarl_v1_ykz6h7nw_gat_normal_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 331.5/2741/242 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_normal_seed37` |  |
| `tarl_v1_ykz6h7nw_gat_normal_seed47` | tarl_v1_ykz6h7nw_gat_normal_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 331.1/2740/243 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_normal_seed47` |  |
| `tarl_v1_ykz6h7nw_gat_normal_seed7` | tarl_v1_ykz6h7nw_gat_normal_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 332.4/2739/244 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_normal_seed7` |  |
| `tarl_v1_ykz6h7nw_gat_single-event_seed17` | tarl_v1_ykz6h7nw_gat_single-event_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 333.3/2739/244 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_single-event_seed1` |  |
| `tarl_v1_ykz6h7nw_gat_single-event_seed27` | tarl_v1_ykz6h7nw_gat_single-event_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 333.1/2740/243 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_single-event_seed2` |  |
| `tarl_v1_ykz6h7nw_gat_single-event_seed37` | tarl_v1_ykz6h7nw_gat_single-event_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 333.8/2742/241 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_single-event_seed3` |  |
| `tarl_v1_ykz6h7nw_gat_single-event_seed47` | tarl_v1_ykz6h7nw_gat_single-event_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 333.4/2740/243 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_single-event_seed4` |  |
| `tarl_v1_ykz6h7nw_gat_single-event_seed7` | tarl_v1_ykz6h7nw_gat_single-event_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 334.0/2736/247 | `data/output_data/tsc/sumo_tarl_gat/hz4x4/tarl_v1_ykz6h7nw_gat_single-event_seed7` |  |
| `tarl_v1_ykz6h7nw_gating_multi-event_seed17` | tarl_v1_ykz6h7nw_gating_multi-event_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 338.7/2736/247 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_multi-event_` |  |
| `tarl_v1_ykz6h7nw_gating_multi-event_seed27` | tarl_v1_ykz6h7nw_gating_multi-event_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 339.1/2741/242 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_multi-event_` |  |
| `tarl_v1_ykz6h7nw_gating_multi-event_seed37` | tarl_v1_ykz6h7nw_gating_multi-event_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 339.0/2738/245 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_multi-event_` |  |
| `tarl_v1_ykz6h7nw_gating_multi-event_seed47` | tarl_v1_ykz6h7nw_gating_multi-event_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 338.5/2739/244 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_multi-event_` |  |
| `tarl_v1_ykz6h7nw_gating_multi-event_seed7` | tarl_v1_ykz6h7nw_gating_multi-event_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 338.1/2742/241 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_multi-event_` |  |
| `tarl_v1_ykz6h7nw_gating_normal_seed17` | tarl_v1_ykz6h7nw_gating_normal_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 331.7/2738/245 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_normal_seed1` |  |
| `tarl_v1_ykz6h7nw_gating_normal_seed27` | tarl_v1_ykz6h7nw_gating_normal_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 331.9/2740/243 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_normal_seed2` |  |
| `tarl_v1_ykz6h7nw_gating_normal_seed37` | tarl_v1_ykz6h7nw_gating_normal_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 332.7/2742/241 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_normal_seed3` |  |
| `tarl_v1_ykz6h7nw_gating_normal_seed47` | tarl_v1_ykz6h7nw_gating_normal_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 332.4/2742/241 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_normal_seed4` |  |
| `tarl_v1_ykz6h7nw_gating_normal_seed7` | tarl_v1_ykz6h7nw_gating_normal_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 331.7/2737/246 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_normal_seed7` |  |
| `tarl_v1_ykz6h7nw_gating_single-event_seed17` | tarl_v1_ykz6h7nw_gating_single-event_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 333.7/2738/245 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_single-event` |  |
| `tarl_v1_ykz6h7nw_gating_single-event_seed27` | tarl_v1_ykz6h7nw_gating_single-event_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 334.2/2739/244 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_single-event` |  |
| `tarl_v1_ykz6h7nw_gating_single-event_seed37` | tarl_v1_ykz6h7nw_gating_single-event_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 333.1/2738/245 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_single-event` |  |
| `tarl_v1_ykz6h7nw_gating_single-event_seed47` | tarl_v1_ykz6h7nw_gating_single-event_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 333.3/2741/242 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_single-event` |  |
| `tarl_v1_ykz6h7nw_gating_single-event_seed7` | tarl_v1_ykz6h7nw_gating_single-event_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 334.0/2737/246 | `data/output_data/tsc/sumo_tarl_gating/hz4x4/tarl_v1_ykz6h7nw_gating_single-event` |  |
| `tarl_v1_ykz6h7nw_sensor_multi-event_seed17` | tarl_v1_ykz6h7nw_sensor_multi-event_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 337.6/2735/248 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_multi-event_` |  |
| `tarl_v1_ykz6h7nw_sensor_multi-event_seed27` | tarl_v1_ykz6h7nw_sensor_multi-event_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 337.5/2740/243 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_multi-event_` |  |
| `tarl_v1_ykz6h7nw_sensor_multi-event_seed37` | tarl_v1_ykz6h7nw_sensor_multi-event_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 337.9/2740/243 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_multi-event_` |  |
| `tarl_v1_ykz6h7nw_sensor_multi-event_seed47` | tarl_v1_ykz6h7nw_sensor_multi-event_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 337.5/2737/246 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_multi-event_` |  |
| `tarl_v1_ykz6h7nw_sensor_multi-event_seed7` | tarl_v1_ykz6h7nw_sensor_multi-event_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 337.4/2741/242 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_multi-event_` |  |
| `tarl_v1_ykz6h7nw_sensor_normal_seed17` | tarl_v1_ykz6h7nw_sensor_normal_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 331.3/2739/244 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_normal_seed1` |  |
| `tarl_v1_ykz6h7nw_sensor_normal_seed27` | tarl_v1_ykz6h7nw_sensor_normal_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 331.5/2739/244 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_normal_seed2` |  |
| `tarl_v1_ykz6h7nw_sensor_normal_seed37` | tarl_v1_ykz6h7nw_sensor_normal_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 331.0/2743/240 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_normal_seed3` |  |
| `tarl_v1_ykz6h7nw_sensor_normal_seed47` | tarl_v1_ykz6h7nw_sensor_normal_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 331.4/2743/240 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_normal_seed4` |  |
| `tarl_v1_ykz6h7nw_sensor_normal_seed7` | tarl_v1_ykz6h7nw_sensor_normal_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 332.1/2738/245 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_normal_seed7` |  |
| `tarl_v1_ykz6h7nw_sensor_single-event_seed17` | tarl_v1_ykz6h7nw_sensor_single-event_seed17/hz4x4/s17 | 50 | DONE | 2026-09-15 | 332.4/2737/246 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_single-event` |  |
| `tarl_v1_ykz6h7nw_sensor_single-event_seed27` | tarl_v1_ykz6h7nw_sensor_single-event_seed27/hz4x4/s27 | 50 | DONE | 2026-09-15 | 332.7/2739/244 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_single-event` |  |
| `tarl_v1_ykz6h7nw_sensor_single-event_seed37` | tarl_v1_ykz6h7nw_sensor_single-event_seed37/hz4x4/s37 | 50 | DONE | 2026-09-15 | 333.1/2739/244 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_single-event` |  |
| `tarl_v1_ykz6h7nw_sensor_single-event_seed47` | tarl_v1_ykz6h7nw_sensor_single-event_seed47/hz4x4/s47 | 50 | DONE | 2026-09-15 | 333.1/2739/244 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_single-event` |  |
| `tarl_v1_ykz6h7nw_sensor_single-event_seed7` | tarl_v1_ykz6h7nw_sensor_single-event_seed7/hz4x4/s7 | 50 | DONE | 2026-09-15 | 332.9/2739/244 | `data/output_data/tsc/sumo_tarl_sensor/hz4x4/tarl_v1_ykz6h7nw_sensor_single-event` |  |

## TARL v21 正式批

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

## Plan5-B100 跨算法锚点/校准（Phase0）

- 假设：DDQN/CTXDDQN 在 S1-S4 场景 100ep 锚点 + 校准，建立跨算法比较底座。
- plan_id/队列：plan5_b100 logical_run_id=P5-{ANCHOR,CAL}-<ALGO>-<场景>-SD<seed>
- 设计稿/证据位置：data/output_data/cross_algorithm/plan5_b100/{runs,manifest,audit}
- 结论：Phase0 基础设施（fixedtime 参照/probe/resume 等价/并发门禁/环境/provenance）与 DDQN/CTXDDQN 锚点、校准全部完成；Phase0 不构成正式批次结论。
- 登记单元 68（train/eval/smoke/batch/calibration = 40/0/0/10/18）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `P5-PHASE0/environment` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/environment` | Phase0 基础设施/门禁证据批次: plan5 micromamba 环境冻结 |
| `P5-PHASE0/fixedtime` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/fixedtime` | Phase0 基础设施/门禁证据批次: FixedTime 参照（S1-S4×repeat×2） |
| `P5-PHASE0/probe_evaluation_runs` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/probe/evaluation_runs` | Phase0 基础设施/门禁证据批次: 探针评估执行 |
| `P5-PHASE0/probe_plan5_fixed_probe_v1` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/probe/plan5_fixed_probe_v1` | Phase0 基础设施/门禁证据批次: plan5_fixed_probe_v1 冻结探针 |
| `P5-PHASE0/provenance` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/provenance` | Phase0 基础设施/门禁证据批次: 源码 provenance/bundle |
| `P5-PHASE0/resource_gate` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/resource_gate` | Phase0 基础设施/门禁证据批次: 1/4/8 并发门禁 |
| `P5-PHASE0/resource_gate_attempt_1_failed` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/resource_gate_attempt_1_failed` | Phase0 基础设施/门禁证据批次: 并发门禁失败 attempt |
| `P5-PHASE0/resource_gate_attempt_2` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/resource_gate_attempt_2` | Phase0 基础设施/门禁证据批次: 并发门禁 attempt2 |
| `P5-PHASE0/resource_gate_attempt_3` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/resource_gate_attempt_3` | Phase0 基础设施/门禁证据批次: 并发门禁 attempt3 |
| `P5-PHASE0/resume_equivalence` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/resume_equivalence` | Phase0 基础设施/门禁证据批次: resume 等价性验证 |
| `P5-CAL-PPO-S2-LR1E4-EC001-SD100` | PPO_CAL/sumohz1x1/s100 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR1E4-EC001-SD100` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR1E4-EC001-SD101` | PPO_CAL/sumohz1x1/s101 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR1E4-EC001-SD101` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR1E4-EC001-SD102` | PPO_CAL/sumohz1x1/s102 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR1E4-EC001-SD102` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR1E4-EC01-SD100` | PPO_CAL/sumohz1x1/s100 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR1E4-EC01-SD100` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR1E4-EC01-SD101` | PPO_CAL/sumohz1x1/s101 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR1E4-EC01-SD101` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR1E4-EC01-SD102` | PPO_CAL/sumohz1x1/s102 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR1E4-EC01-SD102` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR25E5-EC001-SD100` | PPO_CAL/sumohz1x1/s100 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR25E5-EC001-SD10` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR25E5-EC001-SD101` | PPO_CAL/sumohz1x1/s101 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR25E5-EC001-SD10` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR25E5-EC001-SD102` | PPO_CAL/sumohz1x1/s102 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR25E5-EC001-SD10` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR25E5-EC01-SD100` | PPO_CAL/sumohz1x1/s100 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR25E5-EC01-SD100` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR25E5-EC01-SD101` | PPO_CAL/sumohz1x1/s101 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR25E5-EC01-SD101` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR25E5-EC01-SD102` | PPO_CAL/sumohz1x1/s102 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR25E5-EC01-SD102` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR5E4-EC001-SD100` | PPO_CAL/sumohz1x1/s100 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR5E4-EC001-SD100` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR5E4-EC001-SD101` | PPO_CAL/sumohz1x1/s101 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR5E4-EC001-SD101` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR5E4-EC001-SD102` | PPO_CAL/sumohz1x1/s102 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR5E4-EC001-SD102` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR5E4-EC01-SD100` | PPO_CAL/sumohz1x1/s100 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR5E4-EC01-SD100` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR5E4-EC01-SD101` | PPO_CAL/sumohz1x1/s101 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR5E4-EC01-SD101` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR5E4-EC01-SD102` | PPO_CAL/sumohz1x1/s102 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR5E4-EC01-SD102` | attempts=1; type=CAL |
| `P5-ANCHOR-CTXDDQN-S1-SD0` | CTXDDQN_ANCHOR/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S1-SD0` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S1-SD1` | CTXDDQN_ANCHOR/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S1-SD1` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S1-SD2` | CTXDDQN_ANCHOR/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S1-SD2` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S1-SD3` | CTXDDQN_ANCHOR/sumohz1x1_config2/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S1-SD3` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S1-SD4` | CTXDDQN_ANCHOR/sumohz1x1_config2/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S1-SD4` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S2-SD0` | CTXDDQN_ANCHOR/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S2-SD0` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S2-SD1` | CTXDDQN_ANCHOR/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S2-SD1` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S2-SD2` | CTXDDQN_ANCHOR/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S2-SD2` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S2-SD3` | CTXDDQN_ANCHOR/sumohz1x1/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S2-SD3` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S2-SD4` | CTXDDQN_ANCHOR/sumohz1x1/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S2-SD4` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S3-SD0` | CTXDDQN_ANCHOR/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S3-SD0` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S3-SD1` | CTXDDQN_ANCHOR/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S3-SD1` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S3-SD2` | CTXDDQN_ANCHOR/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S3-SD2` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S3-SD3` | CTXDDQN_ANCHOR/sumohz1x1_config4/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S3-SD3` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S3-SD4` | CTXDDQN_ANCHOR/sumohz1x1_config4/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S3-SD4` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S4-SD0` | CTXDDQN_ANCHOR/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S4-SD0` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S4-SD1` | CTXDDQN_ANCHOR/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S4-SD1` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S4-SD2` | CTXDDQN_ANCHOR/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S4-SD2` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S4-SD3` | CTXDDQN_ANCHOR/sumohz1x1_config3/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S4-SD3` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S4-SD4` | CTXDDQN_ANCHOR/sumohz1x1_config3/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S4-SD4` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-DDQN-S1-SD0` | DDQN_ANCHOR/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S1-SD0` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S1-SD1` | DDQN_ANCHOR/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S1-SD1` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S1-SD2` | DDQN_ANCHOR/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S1-SD2` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S1-SD3` | DDQN_ANCHOR/sumohz1x1_config2/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S1-SD3` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S1-SD4` | DDQN_ANCHOR/sumohz1x1_config2/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S1-SD4` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S2-SD0` | DDQN_ANCHOR/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S2-SD0` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S2-SD1` | DDQN_ANCHOR/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S2-SD1` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S2-SD2` | DDQN_ANCHOR/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S2-SD2` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S2-SD3` | DDQN_ANCHOR/sumohz1x1/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S2-SD3` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S2-SD4` | DDQN_ANCHOR/sumohz1x1/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S2-SD4` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S3-SD0` | DDQN_ANCHOR/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S3-SD0` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S3-SD1` | DDQN_ANCHOR/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S3-SD1` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S3-SD2` | DDQN_ANCHOR/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S3-SD2` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S3-SD3` | DDQN_ANCHOR/sumohz1x1_config4/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S3-SD3` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S3-SD4` | DDQN_ANCHOR/sumohz1x1_config4/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S3-SD4` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S4-SD0` | DDQN_ANCHOR/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S4-SD0` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S4-SD1` | DDQN_ANCHOR/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S4-SD1` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S4-SD2` | DDQN_ANCHOR/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S4-SD2` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S4-SD3` | DDQN_ANCHOR/sumohz1x1_config3/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S4-SD3` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S4-SD4` | DDQN_ANCHOR/sumohz1x1_config3/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S4-SD4` | attempts=3; type=ANCHOR |

## Plan1 半离线 DQN 基线（hz1x1 系列）

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

## Milestone0 实验基础设施烟测

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

## 论文基础设施验证批次

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

## 杂项一次性探针

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

## 附：追溯口径说明

- `eval` 行 = 冻结评估 attempt（评估包内 `attempts/` 或 tsc 输出树下独立 eval 目录），`behavior_source` 回指训练 run/包。
- `partial_*`/`quarantine`/`packages_INVALID_no_ckpt_load` 桶中的 attempt 按隔离语义登记为 ABORTED/FAILED，不与 DONE 混计。
- 残留 `运行中` 状态且核查无存活进程（2026-10-07）者登记 ABORTED，notes 记已写训练条数。
- §4.4 seed 贯通自检：对全部 DONE 训练行按 (campaign,arm) 分组比对终值三指标，未发现同配置不同 seed 终值全同，无 flag 行。
- 相同 run 的隔离/重复派发副本以 `alias_of` 指向 canonical run，统计时去重。
