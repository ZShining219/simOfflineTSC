# ATT-ENTITY-003 TARL 对齐/传感矩阵（flx/tarlp/stage2-5）

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

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
