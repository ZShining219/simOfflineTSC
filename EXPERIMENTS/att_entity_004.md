# ATT-ENTITY-004 结构化事件输入 + 语义辅助（注意力线当前主线）

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：事件信息以结构化 z_task 槽位 + 语义辅助头进入 TARL 注意力，应同时提升场景区分与决策优化（H: 结构化+aux > 纯文本注入）。
- plan_id/队列：att004_wave1/fix/struct/formal/ablation/holdout/e200 波次队列
- 设计稿/证据位置：artifacts/att_entity_004/{run_queue_att004_*.json, run_state_*/}；configs/tsc/att_entity_004/generated/*.yml；artifacts/att_entity_004/{ABLATION_ARMS_DESIGN,DIVIDE_CONQUER_DESIGN,SGA_REPAIR_DESIGN}.md
- 结论：fixG/fixR/fixD/crossq/ablA/ablation/a2s/a3s/holdout 波次均完成并留档；
- **fixG 收官测评收割（2026-10-07，eval 实为 10-05 已跑完、此前未汇总判读）**：fe_fixG_* 网格 5 seed×{ep50/100/150/200}×5 条件×3 eval seed 共 300 run 全存在。判读：①优化层——fixG vs a3s 同 seed/同 eval 配对 canonical TT ep50 −24.9s、ep100 −8.9s（收敛更快且更优）；②表征层——wrong_type−canonical 中位 +1.34s 非零，类型语义仍在影响行为；③**效用层未翻正**——canonical−empty 中位 +0.58s（均值 +2.27s 被 s? 单点 +28.7s 拉高），文本通道相对 empty 仍无净收益，独家信息设计需求不变。evidence：hz4x4/fe_fixG_*。stg2 波次登记未执行、2026-10-07 裁决转 SCOPE_DISCARD，stg1/ctl 部分臂 DROPPED（SCOPE_DISCARD）；SGA-GB 梯度线经 pilot 判定为负结果后停止（SGA_GB_PILOT_RESULT.md）；三指标+条件矩阵证据在 data/output_data/analysis/att_entity_004/ 与 baseline_cmp.md。
- 登记单元 2039（train/eval/smoke/batch/calibration = 86/1893/34/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `ev2p_fixG_s7_a0_can` | fixG_s7/hz4x4/s7 | 20 | DONE | 2026-10-07 | — | `data/output_data/analysis/att_entity_004/eval_v2/ev2p_fixG_s7_a0_can` | w34_2 zero-train probe: alpha_scale=0 (CPU-only, canonical condition) |
| `ev2p_fixG_s7_a0p25_can` | fixG_s7/hz4x4/s7 | 20 | DONE | 2026-10-07 | — | `data/output_data/analysis/att_entity_004/eval_v2/ev2p_fixG_s7_a0p25_can` | w34_2 zero-train probe: alpha_scale=0.25 (CPU-only, canonical condition) |
| `ev2p_fixG_s7_a0p5_can` | fixG_s7/hz4x4/s7 | 20 | DONE | 2026-10-07 | — | `data/output_data/analysis/att_entity_004/eval_v2/ev2p_fixG_s7_a0p5_can` | w34_2 zero-train probe: alpha_scale=0.5 (CPU-only, canonical condition) |
| `ev2p_fixG_s7_a1p5_can` | fixG_s7/hz4x4/s7 | 20 | DONE | 2026-10-07 | — | `data/output_data/analysis/att_entity_004/eval_v2/ev2p_fixG_s7_a1p5_can` | w34_2 zero-train probe: alpha_scale=1.5 (CPU-only, canonical condition) |
| `ev2p_fixG_s7_a2_can` | fixG_s7/hz4x4/s7 | 20 | DONE | 2026-10-07 | — | `data/output_data/analysis/att_entity_004/eval_v2/ev2p_fixG_s7_a2_can` | w34_2 zero-train probe: alpha_scale=2 (CPU-only, canonical condition) |
| `ev2p_fixG_s7_km10_can` | fixG_s7/hz4x4/s7 | 20 | DONE | 2026-10-07 | — | `data/output_data/analysis/att_entity_004/eval_v2/ev2p_fixG_s7_km10_can` | w34_2 zero-train probe: inject_shift=-10 (CPU-only, canonical condition) |
| `ev2p_fixG_s7_km20_can` | fixG_s7/hz4x4/s7 | 20 | DONE | 2026-10-07 | — | `data/output_data/analysis/att_entity_004/eval_v2/ev2p_fixG_s7_km20_can` | w34_2 zero-train probe: inject_shift=-20 (CPU-only, canonical condition) |
| `ev2p_fixG_s7_km5_can` | fixG_s7/hz4x4/s7 | 20 | DONE | 2026-10-07 | — | `data/output_data/analysis/att_entity_004/eval_v2/ev2p_fixG_s7_km5_can` | w34_2 zero-train probe: inject_shift=-5 (CPU-only, canonical condition) |
| `ev2p_fixG_s7_kp10_can` | fixG_s7/hz4x4/s7 | 20 | DONE | 2026-10-07 | — | `data/output_data/analysis/att_entity_004/eval_v2/ev2p_fixG_s7_kp10_can` | w34_2 zero-train probe: inject_shift=10 (CPU-only, canonical condition) |
| `ev2p_fixG_s7_kp20_can` | fixG_s7/hz4x4/s7 | 20 | DONE | 2026-10-07 | — | `data/output_data/analysis/att_entity_004/eval_v2/ev2p_fixG_s7_kp20_can` | w34_2 zero-train probe: inject_shift=20 (CPU-only, canonical condition) |
| `ev2p_fixG_s7_kp5_can` | fixG_s7/hz4x4/s7 | 20 | DONE | 2026-10-07 | — | `data/output_data/analysis/att_entity_004/eval_v2/ev2p_fixG_s7_kp5_can` | w34_2 zero-train probe: inject_shift=5 (CPU-only, canonical condition) |
| `lag_blockthrot_fixG_s7` | fixG_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_blockthrot_fixG_s7` | REGISTERED+RUNNING: lag_utility probe — lane_blockage 节流 oracle（堵塞边在下游节点减服务，acti |
| `lag_corridorL120_clbind_s7` | closure_bind_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_corridorL120_clbind_s7` | REGISTERED: lag_utility probe — corridor-throttle oracle，lead=120（窗含预告段预置）, eval |
| `lag_corridorL120_fixG_s7` | fixG_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_corridorL120_fixG_s7` | REGISTERED: lag_utility probe — corridor-throttle oracle，lead=120（窗含预告段预置）, eval |
| `lag_corridor_clbind_s7` | closure_bind_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_corridor_clbind_s7` | REGISTERED: lag_utility probe — corridor-throttle oracle（馈入+上游最小走廊相位），lead=0, ev |
| `lag_corridor_fixG_s7` | fixG_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_corridor_fixG_s7` | REGISTERED: lag_utility probe — corridor-throttle oracle（馈入+上游最小走廊相位），lead=0, ev |
| `lag_lead120_clbind_s7` | closure_bind_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_lead120_clbind_s7` | REGISTERED: lag_utility probe — TARL_ANNOUNCE_LEAD_S=120, canonical, eval_v2 40e |
| `lag_lead120_fixG_s7` | fixG_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_lead120_fixG_s7` | REGISTERED: lag_utility probe — TARL_ANNOUNCE_LEAD_S=120 (report early 120s, cle |
| `lag_lead300_clbind_s7` | closure_bind_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_lead300_clbind_s7` | REGISTERED: lag_utility probe — TARL_ANNOUNCE_LEAD_S=300, canonical, eval_v2 40e |
| `lag_lead300_fixG_s7` | fixG_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_lead300_fixG_s7` | REGISTERED: lag_utility probe — TARL_ANNOUNCE_LEAD_S=300, canonical, eval_v2 40e |
| `lag_oracle_clbind_s7` | closure_bind_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_oracle_clbind_s7` | REGISTERED: lag_utility probe — oracle dead-lane mask on closure_bind arm, eval_ |
| `lag_oracle_fixG_s7` | fixG_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_oracle_fixG_s7` | REGISTERED: lag_utility probe — oracle dead-lane mask (closure feeder in-lane ze |
| `lag_recboost_clbind_s7` | closure_bind_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_recboost_clbind_s7` | REGISTERED+RUNNING: lag_utility probe — corridor recovery-boost oracle（恢复窗 end.. |
| `lag_recboost_fixG_s7` | fixG_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_recboost_fixG_s7` | REGISTERED+RUNNING: lag_utility probe — corridor recovery-boost oracle（恢复窗 end.. |
| `lag_rr_can_fixG_s7` | fixG_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_rr_can_fixG_s7` | REGISTERED+DONE: lag_utility — TARL_REROUTE_P=0.5 下 canonical eval_v2 基线（reroute |
| `lag_rr_corridor_fixG_s7` | fixG_s7/hz4x4/s7 | 40 | DONE | 2026-10-09 | — | `data/output_data/analysis/att_entity_004/lag_utility/lag_rr_corridor_fixG_s7` | REGISTERED+DONE: lag_utility — TARL_REROUTE_P=0.5 下 corridor-throttle oracle（判 r |
| `att004_abl_smoke_concat` | abl_smoke_concat/hz4x4/s7 | 2 | DONE | 2026-10-03 | 204.2/617/1440 | `data/output_data/tsc/sumo_tarl_concat/hz4x4/att004_abl_smoke_concat` |  |
| `att004_abl_smoke_selfattn` | abl_smoke_selfattn/hz4x4/s7 | 2 | DONE | 2026-10-03 | 361.8/562/1538 | `data/output_data/tsc/sumo_tarl_selfattn/hz4x4/att004_abl_smoke_selfattn` |  |
| `att004_atsmoke_ctrl` | atsmoke_ctrl/hz4x4/s7 | 1 | DONE | 2026-10-05 | 661.8/2331/652 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_atsmoke_ctrl` |  |
| `att004_bert_regress` | bert_regress/hz4x4/s7 | 4 | DONE | 2026-10-01 | 160.6/22/220 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_bert_regress` |  |
| `att004_closure_bind_smoke` | closure_bind_s7/hz4x4/s7 | 2 | DONE | 2026-10-07 | 587.1/—/— | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_closure_bind_smoke` | closure_bind 2ep smoke before formal 200ep run |
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
| `att004_closure_bind_s7` | closure_bind_s7/hz4x4/s7 | 200 | DONE | 2026-10-07 | — | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_closure_bind_s7` | 200ep DONE; last20 TT 345.5; 16 unique fallback binds logged |
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
| `att004_paircurr_s7` | paircurr_s7/hz4x4/s7 | 200 | DONE | 2026-10-09 | — | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_paircurr_s7` | M1 配对课程训练臂：100 对（同 schedule 同 sumo_seed 仅 canonical/empty 不同），fixG 同超参；验证配对能否把 c |
| `att004_sgagb_s17` | sgagb/hz4x4/s17 | 100 | DONE | 2026-10-01 | 360.5/2725/258 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_sgagb_s17` |  |
| `att004_sgagb_s7` | sgagb/hz4x4/s7 | 100 | DONE | 2026-10-01 | 362.0/2728/255 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_sgagb_s7` |  |
| `att004_stg1_s27` | stg1/hz4x4/s27 | 111 | ABORTED | 2026-10-01 | 356.7/2716/267 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_stg1_s27` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 111 条 |
| `att004_stg1_s7` | stg1/hz4x4/s7 | 109 | ABORTED | 2026-10-01 | 356.2/2720/263 | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_stg1_s7` | run_status 残留'运行中'；无存活进程(2026-10-07 核查)；已写训练记录 109 条 |
| `att004_surge_pair_s7` | surge_pair_s7/hz4x4/s7 | 200 | RUNNING | 2026-10-09 | — | `data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_surge_pair_s7` | REGISTERED: pairaux surge 臂——配对课程(同 schedule/同 seed can/emp)×surge 计划，ztask_voca |
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

评估 attempt 共 1893 行，按包/来源聚合：

| 评估包/来源 | n DONE/FAILED/ABORTED | 条件集 | eval seeds |
|---|---|---|---|
| `a2s` | 49/0/0 | — | — |
| `a3s` | 122/0/0 | — | — |
| `a3s_s17` | 47/0/0 | canonical,empty | — |
| `a3s_s7` | 48/0/0 | canonical,empty,wrong_type | — |
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
| `att004_paircurr_s7/checkpoints/resumable/episode_0200.pt` | 3/0/0 | canonical,empty,foreign,wrong_type | — |
| `closure_bind_s7` | 3/0/0 | canonical,empty,wrong_type | — |
| `colight` | 72/0/0 | — | — |
| `colight_s7_e200` | 1/0/0 | physical_only | — |
| `crossq_s17` | 60/0/0 | — | — |
| `crossq_s27` | 60/0/0 | — | — |
| `crossq_s37` | 60/0/0 | — | — |
| `crossq_s47` | 60/0/0 | — | — |
| `crossq_s7` | 60/0/0 | — | — |
| `fixD_s17` | 45/0/0 | — | — |
| `fixD_s7` | 45/0/0 | — | — |
| `fixG_s17` | 62/18/0 | canonical,empty | — |
| `fixG_s27` | 62/20/0 | canonical,empty | — |
| `fixG_s37` | 62/19/0 | canonical,empty | — |
| `fixG_s47` | 62/35/0 | canonical,empty | — |
| `fixG_s7` | 63/18/0 | canonical,empty,wrong_type | — |
| `fixR_s17` | 47/0/0 | canonical,empty | — |
| `fixR_s7` | 47/0/0 | canonical,empty | — |
| `fixedtime` | 37/0/0 | physical_only | — |
| `ft_ev` | 3/0/0 | — | — |
| `ft_noev` | 6/0/0 | — | — |
| `ft_noevB` | 6/0/0 | — | — |
| `ho_t3_s17` | 15/0/0 | — | — |
| `ho_t3_s7` | 15/0/0 | — | — |
| `ho_t4_s17` | 15/0/0 | — | — |
| `ho_t4_s7` | 15/0/0 | — | — |
| `maxpressure` | 37/0/0 | physical_only | — |
| `mplight` | 73/0/0 | physical_only | — |
| `s17_can` | 2/0/0 | — | — |
| `s17_emp` | 1/0/0 | — | — |
| `smoke_colight` | 1/0/0 | — | — |
| `smoke_fixed` | 1/0/0 | — | — |
| `smoke_mplight` | 1/0/0 | — | — |

<!-- EXTRA-BEGIN -->
## eval_v2 批次（w34_1，2026-10-07，commit 3a04dad feat/w34-eval-v2-closure）

- 协议：`tools/run_tarl_struct_cf.py --eval_protocol eval_v2` + `trainer.sumo_seed_per_episode: true`；
  plan 集 e 以 `--seed 9000+e` 起 SUMO（event_plan_log 记 sumo_seed），单 run 产 40 个可配对集。
  eval_v1（run 级固定种子）口径并存，配置族 `ev2_{a3s,fixG,fixR}.yml`。
- 队列：`artifacts/att004_eval_v2_w34/run_queue_w34_ev2.json`（20 run，parallel 8，tmux `w34_ev2`）。
  预算裁决：canonical+empty 全 9 臂种（18 run）+ wrong_type 限 fixG_s7/a3s_s7（2 run）；
  wrong_type 其余臂种以 canonical run 内嵌 cf 变体（逐决策 action_changed/q_l1）补证。
- ckpt：fixG s7/17/27/37/47 ep200；a3s s7/17 ep100；fixR s7/17 ep100。
- 分析：`tools/att004_paired_util.py` canonical−empty 逐集配对 + 事件窗分层 →
  `data/output_data/analysis/att_entity_004/eval_v2_paired.md`。
- closure_bind 臂：`att004_closure_bind_200.yml`（fixG 族 + model.closure_binding=nearest_upstream，
  覆写见 docs/att004_closure_bind_plan.md），s7/200ep 队列 `run_queue_w34_closure.json`，tmux `w34_closure`。

## w34_3 批次（2026-10-08，commit 4f0f23c+ feat/w34-eval-v2-closure）

- **T2 常规基线 eval_v2 同集对照**：`ev2b_{fixedtime,maxpressure,colight,mplight}.yml` 40 集
  （plan 0-39 + sumo_seed=9000+e，经 `tools/att004_baseline_eval.py` 驱动），
  队列 `run_queue_w34_baselines.json`。结果：fixedtime TT 446.5 / maxpressure 341.6 /
  colight@ep200 342.1 / **mplight@ep200 337.8** vs fixG_can 344.1 ——
  mplight **40/40 全胜** fixG（Δ−6.28s，最强基线），maxpressure 40/40（Δ−2.49s），
  colight 36/40（Δ−2.00s）；非语义基线差距未闭合，对最强基线残差 ~6.3s TT。
  注意：`train_model` 必须为真以过 `task.py` 的 `train()` 门（学习由工具中性化），
  `update_target_rate` 需非零（除零 bug 已踩过）。
- **T1 closure_bind 收官（DONE）**：训练 200ep 完成（近 20 集 TT 345.5），
  `ev2c_closure_bind_s7_{can,emp,wt}` ×40 集全 SUCCEEDED
  （分析 `eval_v2_closure_bind{,_wt}/`）。判读：
  ① gap 4 集窗内 Δqueue 从旧臂 ≈0 → **+1.04**——fallback 送达实证，
  但送达后仍付窗内代价（覆盖补全≠覆盖收益）；
  ② 全集收窄成立——ΔTT +1.141→+0.437（−62%），窗内 Δqueue +6.53→+2.04
  （−69%），窗外 ≈0，no_event 集 ≈0；残余 +0.44s 仍正；
  ③ 审计复跑 4 gap upstream 全可解析，三条件在线 log（各 1292 行）
  与离线预测逐位一致，eval 集覆盖 42/46→**46/46**；
  ④ can−wt −0.56s TT / −3.77 窗内——语义正确性边界保留。
  canonical 绝对水平 343.39 优于 fixG_can 0.70s；闭环口径未过线
  （仍逊 empty、逊 maxpressure ~2s）。
- **T3 preview 臂设计注记**：REPORT_w34_3.md §T3（announce_lead_time + announced
  状态族，仅设计不实现；对照合法性已论证）。

## w34_4 批次（2026-10-09，feat/w34-eval-v2-closure）

- **lag_utility 探针线收官**（判读 `analysis/att_entity_004/lag_utility/LAG_VERDICT.md`）：
  hz4x4 + 相位级动作下 closure/blockage 双向皆无可兑现 headroom——
  corridor-throttle +1.40/+1.48s、lead120 更坏、recovery-boost +1.77/+1.55s、
  blockage-throttle +1.19s、deadlane 遮蔽 ≈0、reroute(TARL_REROUTE_P) 下
  节流仍 +1.32s。机理：困死需求与活需求共享进口道与相位，方向性服务不可分离。
- **pairaux 线**：plan 行级 `sumo_seed/text_condition` 覆盖 + `hz4x4_pair_v1.yml`
  配对生成器（同 schedule/同 seed 仅文本有无）+ `model.reward_aux`
  （事件语义辅助奖励，仅注 replay 不动评估，window active|recovery|both）。
  训练臂 `att004_paircurr_s7` 200ep RUNNING。
- **demand_surge 事件原型**（commit 5d31b9a）：第四事件族——origin/destination
  边 + rate_vph 注入，`findRoute` 绑定时一次算路、active 窗按
  3600/rate 周期确定性 `vehicle.add`（失败丢弃计数，offered-rate 语义）；
  文本报告 + 严格 grounding（origin 边，destination 暂不入文）；
  z_task type 码 4（surge 臂需 `model.ztask_vocabs: [3,5,4,5,6]` 拓宽
  生命周期+类型嵌入与语义辅助头，默认保持旧 ckpt 可加载）。冒烟：
  `dev/lag_utility/probe_surge_inject.py` libsumo 直跑验证
  （600vph/300s → inserted=50/failed=0，active/cleared 报告渲染+grounding 通过）；
  `att004_surge_smoke.yml` 3 集训练+eval 全绿。**设计动机**：唯一
  pre-symptom 可兑现的事件族——"预告→提前多服务"在相位粒度下方向可分
  （与 closure 的"避免"需求相反）。未决：announced/active 状态位、
  destination 编码、surge 配对评估计划。
- **surge 表征深化（同日 r2）**：报告生命周期扩三态——surge 专属
  `announced` 状态在 `TARL_ANNOUNCE_LEAD_S` 窗内发未来时态文本
  （"is expected... starting at B seconds"），begin 处转 active、end 处
  转 cleared，非 surge 族逐位不变；文本新增 "bound for junction J"
  （目的边 to_junction），grounding 增 `bound` junction 角色 + rate_vph
  提取；`scene_context`/`to_gnode` 把 announced 视为 live 报告。
  **z_task field0 从二值变生命周期码**（0 无 / 1 announced / 2 active），
  编码器 present 门改 `(>0)` 位判定，surge 行同时落在 origin 馈入节点与
  bound 目的节点（走廊两端可见，不泄露特权路由），severity←rate/3600；
  `model.ztask_vocabs` 全元组覆盖（surge 臂 [3,5,4,5,6]，默认保持旧
  ckpt 可加载）。配对计划 `hz4x4_surge_pair_v1.yml`（100 surge 对 +
  20% normal，逐对 sumo_seed=9000+p）生成并冒烟 4ep 全绿
  （libsumo 探针三态 + trainer 配对/生命周期端到端）。
<!-- EXTRA-END -->