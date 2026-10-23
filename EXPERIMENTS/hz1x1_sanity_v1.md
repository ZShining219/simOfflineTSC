# hz1x1 口径体检（对照 v3 协议）

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：在已验证干净的 hz1x1 原始真实包上用与 v3 逐位一致的协议训练 4 臂，验证训练+eval 记录口径是否正常——若形态健康则确认 xiasha1 异常为数据专属，若异常则管线有更深缺陷。
- plan_id/队列：hz1x1_sanity_v1
- 设计稿/证据位置：docs/experiments/2026-10-23_hz1x1_sanity.md；4 臂(d3qn/frap/sac/presslight)×4 原始包(bc-tyc/qc-yn/kn-hz/sb-sx)×3 seeds=48 run；200ep+每ep贪心评估+pseudo_step，协议同 v3。
- 结论：进行中（2026-10-23 队列派发）。
- 登记单元 48（train/eval/smoke/batch/calibration = 48/0/0/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `hz1sn_d3qn_bctyc_s0` | d3qn/hz1x1_bctyc/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/hz1sn_d3qn_bctyc_s0` | hz1x1 口径体检：原始真实包 sumohz1x1；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_d3qn_bctyc_s1` | d3qn/hz1x1_bctyc/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/hz1sn_d3qn_bctyc_s1` | hz1x1 口径体检：原始真实包 sumohz1x1；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_d3qn_bctyc_s2` | d3qn/hz1x1_bctyc/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/hz1sn_d3qn_bctyc_s2` | hz1x1 口径体检：原始真实包 sumohz1x1；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_d3qn_knhz_s0` | d3qn/hz1x1_knhz/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/hz1sn_d3qn_knhz_s0` | hz1x1 口径体检：原始真实包 sumohz1x1_config3；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_d3qn_knhz_s1` | d3qn/hz1x1_knhz/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/hz1sn_d3qn_knhz_s1` | hz1x1 口径体检：原始真实包 sumohz1x1_config3；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_d3qn_knhz_s2` | d3qn/hz1x1_knhz/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/hz1sn_d3qn_knhz_s2` | hz1x1 口径体检：原始真实包 sumohz1x1_config3；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_d3qn_qcyn_s0` | d3qn/hz1x1_qcyn/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/hz1sn_d3qn_qcyn_s0` | hz1x1 口径体检：原始真实包 sumohz1x1_config2；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_d3qn_qcyn_s1` | d3qn/hz1x1_qcyn/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/hz1sn_d3qn_qcyn_s1` | hz1x1 口径体检：原始真实包 sumohz1x1_config2；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_d3qn_qcyn_s2` | d3qn/hz1x1_qcyn/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/hz1sn_d3qn_qcyn_s2` | hz1x1 口径体检：原始真实包 sumohz1x1_config2；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_d3qn_sbsx_s0` | d3qn/hz1x1_sbsx/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/hz1sn_d3qn_sbsx_s0` | hz1x1 口径体检：原始真实包 sumohz1x1_config4；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_d3qn_sbsx_s1` | d3qn/hz1x1_sbsx/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/hz1sn_d3qn_sbsx_s1` | hz1x1 口径体检：原始真实包 sumohz1x1_config4；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_d3qn_sbsx_s2` | d3qn/hz1x1_sbsx/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/hz1sn_d3qn_sbsx_s2` | hz1x1 口径体检：原始真实包 sumohz1x1_config4；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_frap_bctyc_s0` | frap/hz1x1_bctyc/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/hz1sn_frap_bctyc_s0` | hz1x1 口径体检：原始真实包 sumohz1x1；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_frap_bctyc_s1` | frap/hz1x1_bctyc/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/hz1sn_frap_bctyc_s1` | hz1x1 口径体检：原始真实包 sumohz1x1；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_frap_bctyc_s2` | frap/hz1x1_bctyc/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/hz1sn_frap_bctyc_s2` | hz1x1 口径体检：原始真实包 sumohz1x1；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_frap_knhz_s0` | frap/hz1x1_knhz/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/hz1sn_frap_knhz_s0` | hz1x1 口径体检：原始真实包 sumohz1x1_config3；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_frap_knhz_s1` | frap/hz1x1_knhz/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/hz1sn_frap_knhz_s1` | hz1x1 口径体检：原始真实包 sumohz1x1_config3；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_frap_knhz_s2` | frap/hz1x1_knhz/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/hz1sn_frap_knhz_s2` | hz1x1 口径体检：原始真实包 sumohz1x1_config3；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_frap_qcyn_s0` | frap/hz1x1_qcyn/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/hz1sn_frap_qcyn_s0` | hz1x1 口径体检：原始真实包 sumohz1x1_config2；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_frap_qcyn_s1` | frap/hz1x1_qcyn/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/hz1sn_frap_qcyn_s1` | hz1x1 口径体检：原始真实包 sumohz1x1_config2；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_frap_qcyn_s2` | frap/hz1x1_qcyn/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/hz1sn_frap_qcyn_s2` | hz1x1 口径体检：原始真实包 sumohz1x1_config2；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_frap_sbsx_s0` | frap/hz1x1_sbsx/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/hz1sn_frap_sbsx_s0` | hz1x1 口径体检：原始真实包 sumohz1x1_config4；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_frap_sbsx_s1` | frap/hz1x1_sbsx/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/hz1sn_frap_sbsx_s1` | hz1x1 口径体检：原始真实包 sumohz1x1_config4；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_frap_sbsx_s2` | frap/hz1x1_sbsx/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/hz1sn_frap_sbsx_s2` | hz1x1 口径体检：原始真实包 sumohz1x1_config4；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_presslight_bctyc_s0` | presslight/hz1x1_bctyc/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_presslight/sumohz1x1/hz1sn_presslight_bctyc_s0` | hz1x1 口径体检：原始真实包 sumohz1x1；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_presslight_bctyc_s1` | presslight/hz1x1_bctyc/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_presslight/sumohz1x1/hz1sn_presslight_bctyc_s1` | hz1x1 口径体检：原始真实包 sumohz1x1；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_presslight_bctyc_s2` | presslight/hz1x1_bctyc/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_presslight/sumohz1x1/hz1sn_presslight_bctyc_s2` | hz1x1 口径体检：原始真实包 sumohz1x1；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_presslight_knhz_s0` | presslight/hz1x1_knhz/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_presslight/sumohz1x1_config3/hz1sn_presslight_knhz_s0` | hz1x1 口径体检：原始真实包 sumohz1x1_config3；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_presslight_knhz_s1` | presslight/hz1x1_knhz/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_presslight/sumohz1x1_config3/hz1sn_presslight_knhz_s1` | hz1x1 口径体检：原始真实包 sumohz1x1_config3；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_presslight_knhz_s2` | presslight/hz1x1_knhz/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_presslight/sumohz1x1_config3/hz1sn_presslight_knhz_s2` | hz1x1 口径体检：原始真实包 sumohz1x1_config3；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_presslight_qcyn_s0` | presslight/hz1x1_qcyn/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_presslight/sumohz1x1_config2/hz1sn_presslight_qcyn_s0` | hz1x1 口径体检：原始真实包 sumohz1x1_config2；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_presslight_qcyn_s1` | presslight/hz1x1_qcyn/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_presslight/sumohz1x1_config2/hz1sn_presslight_qcyn_s1` | hz1x1 口径体检：原始真实包 sumohz1x1_config2；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_presslight_qcyn_s2` | presslight/hz1x1_qcyn/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_presslight/sumohz1x1_config2/hz1sn_presslight_qcyn_s2` | hz1x1 口径体检：原始真实包 sumohz1x1_config2；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_presslight_sbsx_s0` | presslight/hz1x1_sbsx/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_presslight/sumohz1x1_config4/hz1sn_presslight_sbsx_s0` | hz1x1 口径体检：原始真实包 sumohz1x1_config4；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_presslight_sbsx_s1` | presslight/hz1x1_sbsx/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_presslight/sumohz1x1_config4/hz1sn_presslight_sbsx_s1` | hz1x1 口径体检：原始真实包 sumohz1x1_config4；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_presslight_sbsx_s2` | presslight/hz1x1_sbsx/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_presslight/sumohz1x1_config4/hz1sn_presslight_sbsx_s2` | hz1x1 口径体检：原始真实包 sumohz1x1_config4；协议与 xiasha1_online_v3 逐位一致；condition=full |
| `hz1sn_sac_bctyc_s0` | sac/hz1x1_bctyc/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_sac/sumohz1x1/hz1sn_sac_bctyc_s0` | hz1x1 口径体检：原始真实包 sumohz1x1；协议与 xiasha1_online_v3 逐位一致；condition=native_explorati |
| `hz1sn_sac_bctyc_s1` | sac/hz1x1_bctyc/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_sac/sumohz1x1/hz1sn_sac_bctyc_s1` | hz1x1 口径体检：原始真实包 sumohz1x1；协议与 xiasha1_online_v3 逐位一致；condition=native_explorati |
| `hz1sn_sac_bctyc_s2` | sac/hz1x1_bctyc/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_sac/sumohz1x1/hz1sn_sac_bctyc_s2` | hz1x1 口径体检：原始真实包 sumohz1x1；协议与 xiasha1_online_v3 逐位一致；condition=native_explorati |
| `hz1sn_sac_knhz_s0` | sac/hz1x1_knhz/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_sac/sumohz1x1_config3/hz1sn_sac_knhz_s0` | hz1x1 口径体检：原始真实包 sumohz1x1_config3；协议与 xiasha1_online_v3 逐位一致；condition=native_e |
| `hz1sn_sac_knhz_s1` | sac/hz1x1_knhz/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_sac/sumohz1x1_config3/hz1sn_sac_knhz_s1` | hz1x1 口径体检：原始真实包 sumohz1x1_config3；协议与 xiasha1_online_v3 逐位一致；condition=native_e |
| `hz1sn_sac_knhz_s2` | sac/hz1x1_knhz/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_sac/sumohz1x1_config3/hz1sn_sac_knhz_s2` | hz1x1 口径体检：原始真实包 sumohz1x1_config3；协议与 xiasha1_online_v3 逐位一致；condition=native_e |
| `hz1sn_sac_qcyn_s0` | sac/hz1x1_qcyn/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_sac/sumohz1x1_config2/hz1sn_sac_qcyn_s0` | hz1x1 口径体检：原始真实包 sumohz1x1_config2；协议与 xiasha1_online_v3 逐位一致；condition=native_e |
| `hz1sn_sac_qcyn_s1` | sac/hz1x1_qcyn/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_sac/sumohz1x1_config2/hz1sn_sac_qcyn_s1` | hz1x1 口径体检：原始真实包 sumohz1x1_config2；协议与 xiasha1_online_v3 逐位一致；condition=native_e |
| `hz1sn_sac_qcyn_s2` | sac/hz1x1_qcyn/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_sac/sumohz1x1_config2/hz1sn_sac_qcyn_s2` | hz1x1 口径体检：原始真实包 sumohz1x1_config2；协议与 xiasha1_online_v3 逐位一致；condition=native_e |
| `hz1sn_sac_sbsx_s0` | sac/hz1x1_sbsx/s0 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_sac/sumohz1x1_config4/hz1sn_sac_sbsx_s0` | hz1x1 口径体检：原始真实包 sumohz1x1_config4；协议与 xiasha1_online_v3 逐位一致；condition=native_e |
| `hz1sn_sac_sbsx_s1` | sac/hz1x1_sbsx/s1 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_sac/sumohz1x1_config4/hz1sn_sac_sbsx_s1` | hz1x1 口径体检：原始真实包 sumohz1x1_config4；协议与 xiasha1_online_v3 逐位一致；condition=native_e |
| `hz1sn_sac_sbsx_s2` | sac/hz1x1_sbsx/s2 | 200 | REGISTERED | — | — | `data/output_data/tsc/sumo_sac/sumohz1x1_config4/hz1sn_sac_sbsx_s2` | hz1x1 口径体检：原始真实包 sumohz1x1_config4；协议与 xiasha1_online_v3 逐位一致；condition=native_e |
