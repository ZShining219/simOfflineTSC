# scen_cov_v1 场景覆盖率：半离线 vs 纯在线（pilot）

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：同一训练预算 / 同一场景顺序 / 同一 seeds 下，半离线（offline_ratio=0.25 仅 st2–4 生效，全程有效混合比 ≈18.75%）控制器的场景覆盖率高于纯在线；即 ΔCoverage_L1 ≥ +0.25 且逐 cell 方向 ≥3/4 一致。
- plan_id/队列：scen_cov_v1（v2 冻结队列 sha256 0b0aef99…，256 tasks = 16 train + 240 eval）
- 设计稿/证据位置：artifacts/scen_cov_v1/{run_queue.json,run_state,coverage_report.json,coverage_cells.csv}；docs/scenario_taxonomy.md（δ=0.05/η=0.10，v1.1 只批 L2）；docs/experiments/2026-11-02_scen_cov_v1.md；configs/events/plans/hz1x1_single_v1.yml（L3 事件层只定义不跑）
- 结论：pilot 档（25 ep/stage，train seeds n=2）：L1 覆盖 1.00（裕度 +0.078）vs 在线 0.25（−0.293），Δcov=+0.75、方向 4/4，主假设成立；L2 外推覆盖 0.375 vs 0.500 不支持"半离线外推更强"，但裕度 −0.013 vs −0.240；反例 sparse_ultra 半离线双指标均不过（比 MP 差 11.2%），须与 L1 同段引用。
- 登记单元 256（train/eval/smoke/batch/calibration = 16/240/0/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `scv1_ddqn_clear_o2_sd0_st1` | ddqn_clear/sumohz1x1/s0 | 25 | DONE | 2026-11-02 | 255.4/1054/226 | `data/output_data/tsc/sumo_dqn/sumohz1x1/scv1_ddqn_clear_o2_sd0_st1` | REGISTERED: scen_cov_v1 online train stage=1 offline_ratio=0.0 ／ scen_cov_script |
| `scv1_ddqn_clear_o2_sd0_st2` | ddqn_clear/sumohz1x1_config4/s0 | 25 | DONE | 2026-11-02 | 79.3/1620/51 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/scv1_ddqn_clear_o2_sd0_st2` | REGISTERED: scen_cov_v1 online train stage=2 offline_ratio=0.25 ／ scen_cov_scrip |
| `scv1_ddqn_clear_o2_sd0_st3` | ddqn_clear/sumohz1x1_config2/s0 | 25 | DONE | 2026-11-02 | 74.6/1383/28 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/scv1_ddqn_clear_o2_sd0_st3` | REGISTERED: scen_cov_v1 online train stage=3 offline_ratio=0.25 ／ scen_cov_scrip |
| `scv1_ddqn_clear_o2_sd0_st4` | ddqn_clear/sumohz1x1_config3/s0 | 25 | DONE | 2026-11-02 | 68.5/727/16 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/scv1_ddqn_clear_o2_sd0_st4` | REGISTERED: scen_cov_v1 online train stage=4 offline_ratio=0.25 ／ scen_cov_scrip |
| `scv1_ddqn_clear_o2_sd1_st1` | ddqn_clear/sumohz1x1/s1 | 25 | DONE | 2026-11-02 | 222.9/1208/246 | `data/output_data/tsc/sumo_dqn/sumohz1x1/scv1_ddqn_clear_o2_sd1_st1` | REGISTERED: scen_cov_v1 online train stage=1 offline_ratio=0.0 ／ scen_cov_script |
| `scv1_ddqn_clear_o2_sd1_st2` | ddqn_clear/sumohz1x1_config4/s1 | 25 | DONE | 2026-11-02 | 84.0/1615/56 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/scv1_ddqn_clear_o2_sd1_st2` | REGISTERED: scen_cov_v1 online train stage=2 offline_ratio=0.25 ／ scen_cov_scrip |
| `scv1_ddqn_clear_o2_sd1_st3` | ddqn_clear/sumohz1x1_config2/s1 | 25 | DONE | 2026-11-02 | 75.0/1380/31 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/scv1_ddqn_clear_o2_sd1_st3` | REGISTERED: scen_cov_v1 online train stage=3 offline_ratio=0.25 ／ scen_cov_scrip |
| `scv1_ddqn_clear_o2_sd1_st4` | ddqn_clear/sumohz1x1_config3/s1 | 25 | DONE | 2026-11-02 | 69.0/727/16 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/scv1_ddqn_clear_o2_sd1_st4` | REGISTERED: scen_cov_v1 online train stage=4 offline_ratio=0.25 ／ scen_cov_scrip |
| `scv1_soddqn_o2_sd0_st1` | soddqn/sumohz1x1/s0 | 25 | DONE | 2026-11-02 | 255.4/1054/226 | `data/output_data/tsc/sumo_dqn/sumohz1x1/scv1_soddqn_o2_sd0_st1` | REGISTERED: scen_cov_v1 semi_offline train stage=1 offline_ratio=0.0 ／ scen_cov_ |
| `scv1_soddqn_o2_sd0_st2` | soddqn/sumohz1x1_config4/s0 | 25 | DONE | 2026-11-02 | 85.5/1613/58 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/scv1_soddqn_o2_sd0_st2` | REGISTERED: scen_cov_v1 semi_offline train stage=2 offline_ratio=0.25 ／ scen_cov |
| `scv1_soddqn_o2_sd0_st3` | soddqn/sumohz1x1_config2/s0 | 25 | DONE | 2026-11-02 | 77.1/1380/31 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/scv1_soddqn_o2_sd0_st3` | REGISTERED: scen_cov_v1 semi_offline train stage=3 offline_ratio=0.25 ／ scen_cov |
| `scv1_soddqn_o2_sd0_st4` | soddqn/sumohz1x1_config3/s0 | 25 | DONE | 2026-11-02 | 73.3/725/18 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/scv1_soddqn_o2_sd0_st4` | REGISTERED: scen_cov_v1 semi_offline train stage=4 offline_ratio=0.25 ／ scen_cov |
| `scv1_soddqn_o2_sd1_st1` | soddqn/sumohz1x1/s1 | 25 | DONE | 2026-11-02 | 222.9/1208/246 | `data/output_data/tsc/sumo_dqn/sumohz1x1/scv1_soddqn_o2_sd1_st1` | REGISTERED: scen_cov_v1 semi_offline train stage=1 offline_ratio=0.0 ／ scen_cov_ |
| `scv1_soddqn_o2_sd1_st2` | soddqn/sumohz1x1_config4/s1 | 25 | DONE | 2026-11-02 | 84.4/1619/52 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/scv1_soddqn_o2_sd1_st2` | REGISTERED: scen_cov_v1 semi_offline train stage=2 offline_ratio=0.25 ／ scen_cov |
| `scv1_soddqn_o2_sd1_st3` | soddqn/sumohz1x1_config2/s1 | 25 | DONE | 2026-11-02 | 74.9/1379/32 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/scv1_soddqn_o2_sd1_st3` | REGISTERED: scen_cov_v1 semi_offline train stage=3 offline_ratio=0.25 ／ scen_cov |
| `scv1_soddqn_o2_sd1_st4` | soddqn/sumohz1x1_config3/s1 | 25 | DONE | 2026-11-02 | 72.0/725/18 | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/scv1_soddqn_o2_sd1_st4` | REGISTERED: scen_cov_v1 semi_offline train stage=4 offline_ratio=0.25 ／ scen_cov |

评估 attempt 共 240 行，按包/来源聚合：

| 评估包/来源 | n DONE/FAILED/ABORTED | 条件集 | eval seeds |
|---|---|---|---|
| `maxpressure` | 24/0/0 | baseline | 101,102,103 |
| `scv1_ddqn_clear_o2_sd0_st1` | 3/0/0 | online | 101,102,103 |
| `scv1_ddqn_clear_o2_sd0_st2` | 6/0/0 | online | 101,102,103 |
| `scv1_ddqn_clear_o2_sd0_st3` | 9/0/0 | online | 101,102,103 |
| `scv1_ddqn_clear_o2_sd0_st4` | 36/0/0 | online | 101,102,103 |
| `scv1_ddqn_clear_o2_sd1_st1` | 3/0/0 | online | 101,102,103 |
| `scv1_ddqn_clear_o2_sd1_st2` | 6/0/0 | online | 101,102,103 |
| `scv1_ddqn_clear_o2_sd1_st3` | 9/0/0 | online | 101,102,103 |
| `scv1_ddqn_clear_o2_sd1_st4` | 36/0/0 | online | 101,102,103 |
| `scv1_soddqn_o2_sd0_st1` | 3/0/0 | semi_offline | 101,102,103 |
| `scv1_soddqn_o2_sd0_st2` | 6/0/0 | semi_offline | 101,102,103 |
| `scv1_soddqn_o2_sd0_st3` | 9/0/0 | semi_offline | 101,102,103 |
| `scv1_soddqn_o2_sd0_st4` | 36/0/0 | semi_offline | 101,102,103 |
| `scv1_soddqn_o2_sd1_st1` | 3/0/0 | semi_offline | 101,102,103 |
| `scv1_soddqn_o2_sd1_st2` | 6/0/0 | semi_offline | 101,102,103 |
| `scv1_soddqn_o2_sd1_st3` | 9/0/0 | semi_offline | 101,102,103 |
| `scv1_soddqn_o2_sd1_st4` | 36/0/0 | semi_offline | 101,102,103 |

<!-- EXTRA-BEGIN -->
## 人工补充（P5 执行与口径）

> 状态：P5 已收官（attempt2 256/256 SUCCEEDED、0 FAILED；台账 256 行全 DONE，结果已回填见下）。P6 结论另存本机 `agenthub/lab/ledger-notes/2026-11-03-P6-结论.md`（本机路径，不在仓内、不入 git，非仓内链接）

- 场景与口径：[docs/scenario_taxonomy.md](../docs/scenario_taxonomy.md)（δ=0.05、η=0.10 冻结；**v1.1 已批，只批 L2**：`unfinished` 加绝对地板 `max(1.10·MP, MP+2)`，L1 原式不动，见卡 §3.3）
- 口径变更记录：v1.0 冻结于 2026-11-02 设计时；v1.1（2026-11-03 00:10 课题主管批，只批 L2）因 L2 空载格阈值退化提出（数据评测复核 2.1），**派发前 0 run，不触发重打基线**
- 实验卡：[docs/experiments/2026-11-02_scen_cov_v1.md](../docs/experiments/2026-11-02_scen_cov_v1.md)
- 场景集（冻结）：L1 = s1 / c4 / c2 / c3（order o2，训练见过）；L2 = sparse_ultra / dense_extreme / lt_extreme / asym_wonly（训练未见，exploratory）；L3 事件层只定义不跑
- 臂：`ddqn_clear`（在线，offline_ratio=0，切场景清 replay）/ `soddqn`（半离线，offline_ratio=0.25、causal history、scene_intersection_balanced）；train seeds 0/1（确认性复跑加 2）；eval sumo seeds 101–103
- 预算：单链 ≈26 min（4 阶段 × 25 ep）；4~6 链；无 GPU（CPU 口径）；硬顶 600 train ep / 324 eval attempts
- 停止条件：卡片 §3.5 八条（S1 冒烟 / S2 链中断 / S3 饱和 / S4 全灭 / S5 分辨率 / S6 预算顶 / S7 外部守护进程 / S8 负结果）
- 执行：attempt2 `16:38:37Z→17:18:08Z`，`256/256 SUCCEEDED`、`FAILED 0`、`rc=0`、零重启零熔断；`comparable=true`、`missing=0`；全 CPU，4 卡空闲。台账 256 行全 DONE（ledger-only `3663698`）
- 结果（优化层，pilot 档 25 ep/stage，train seeds n=2）：**L1 半离线 Coverage 1.00 / Margin +0.078 vs 在线 0.25 / −0.293，Δcov=+0.75、Δmargin=+0.371、方向 4/4**；逐格看在线只在末段 c3 覆盖，s1/c4/c2 的 ATT 分别劣于 maxpressure 34.8%/53.9%/106.1%。**S5 `flag=false` 表示分辨率足够（|Δ|=0.75 ≫ 臂内 seed 极差 0），非负面。**
- 结果（L2 外推，混合信号须双报）：覆盖 0.375 vs 0.500（半离线少 1 格/两 seed 合计 3/8 格），但 margin −0.013 vs −0.240（半离线更近阈值，Δmargin=+0.227）。`guard_effective_cells=[]`：口径 v1.1 护栏本轮零翻转，对数字无影响。
- 后续：A 100 ep/stage 正式档（≈3.2 CPU-h、wall 55–75 min，成本表见实验卡 §5.3）／B L3 事件层 P5b／C order o3 复现。今晚不开跑，等用户选。
<!-- EXTRA-END -->