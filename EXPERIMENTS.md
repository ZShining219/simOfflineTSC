# EXPERIMENTS — 实验台账（索引）

> 追溯登记 2026-10-07（治理分支 gov/experiment-governance）。
> 事实源 = `ledger/runs.jsonl`（每 run 一行机读）；本文件 = 索引层，明细按 campaign 分文件 `EXPERIMENTS/<campaign>.md`——**按需取读，勿通读**。
> 口径：`run_kind=train|eval|smoke|batch|calibration`；eval 行为单次冻结评估 attempt，`behavior_source` 指回训练 run 包名；`alias_of` 非空者不作独立样本统计。
> 机器一律代号 34/73；路径均为仓内相对路径。

<!-- PANEL-BEGIN -->
## 当前面板（2026-10-23，campaign 状态刷新）

- **当前有效设计栈**：`agent/scene_attention.py` + `agent/tarl.py`——per-node z_task 结构化槽位 + SemanticAuxHead 辅助监督 + 反事实探针四层判定；配置族 `configs/tsc/att_entity_004/`；参考臂 `a3s`，最新修复线 `fixG`（收官测评已收割：vs a3s 收敛更快更优 ep100 −8.9s，但效用层未翻正 canonical−empty 中位 +0.58s）。
- **纯离线基线（MPLight 修订）**：`hz1x1_offline_v1` 使用同一控制拓扑、四种车流上的四类 DONE 源（d3qn/FRAP/SAC/MPLight）；d3qn reward 在离线视图乘 `1/12`，其余倍率为 1；27 池 162 行已于 2026-10-23 收口为 DONE（batch_dqn/cql_dqn 各 81），旧三控制器方案 144 行 SCOPE_DISCARD。E5 是四臂×四车流各一来源的平衡稀疏混合池。xiasha1_offline_v1/v2 全部作废。
- **代码位置**：`codex/milestone0-experiment-infrastructure` 分支（收编点 c8be158..32d3e18，已与主干合并 70f3dbf）；main=治理层+半离线基线。
- **代号地图**（历史代际，明细在各 campaign 节）：
  - `tarl_reproduction`/`tarl_v21_formal` = TARL 论文复现基线（完结，仅对照用）
  - `att_entity_002` = SGA/concat/MPLight 文本注入矩阵（完结：H2 支持、H3 否）
  - `att_entity_003` = TARL 对齐/传感矩阵 flx（完结）
  - `att_entity_004` = **当前主线**（z_task+aux+反事实）；`sga_gb` 梯度门线 pilot 负结果已停
  - `ha_sodqn` = **半离线主线**（hz1x1，Plan1→HOA→DHOA/OWP 历史采样矩阵，E4=100 身份正式批已收官）
  - `arterial_1x6`/`plan5_b100` = 半离线线（技术点3，独立于注意力线）
- **最新结论**：z_task 语义保真达标、held-out 证实文本携场景身份（t34_devin）；"canonical≈empty"修正为"事件窗内小幅净负"（t34_cc，配对 n=59）；road_closure 覆盖缺口待裁决修复。
- **接管测试**：t34_devin / t34_cc 均 DONE（test/t34_* 分支）。
- **证据完整性**：台账 15735 行，evidence_lost=19（18 条为历史 FAILED/ABORTED 的产物丢失；另 1 条为 `att004_surge_pair_s7` 僵尸清理时核定无产物。2026-11-06 巡检核定；`scen_cov_v1` 256/256 在盘（证据根见团队侧 decisions evidence-04）。
- **推定写入（非实测终态，勿当实测结果）**：`att004_surge_pair_s7`（`ABORTED`；依据＝无存活进程 / 无 tmux / run_dir 与 artifacts 均无产物）；`signal_control_v1` 4 条（`SCOPE_DISCARD`；依据＝主树 / dsh-* / tmp 全根 find 不存在、队列文件不在盘）。依据已写入各条 `notes`，2026-11-06 治理核定。
- **证据核查（2026-11-12，机器 73，`python -m lab audit` 可复现）**：上条 evidence_lost=19 只统计了已标记行。本机与各工作树中实际找不到 `run_dir` 的有 10,449 行（多数可能在机器 34，待在 34 上运行 `lab audit` 区分"在 34 上"与"真丢失"）；所有历史 run 的代码版本均非运行时记录（manifest 的 baseline_commit 是写死常量）；1,426 行台账 `config_sha` 与 run 目录不符；`xiasha1_online_v3`、`hz1x1_sanity_v1` 的台账指标混入了训练回合（最优臂不变，部分中间排名改变）。**已统一口径**：台账指标标准 = 最佳 checkpoint 评估，946 行已按此复算（旧值留在 `metrics_legacy`），其余行以 `metric_source` 标注口径，索引中非标准数字标 ⟨旧口径⟩/⟨评估⟩；本面板及各节手写结论中的数字仍为旧口径，引用前核对。引用任何结果前先 `python -m lab show <run_id>`。详见 `docs/plan_refactor_202611.md`。
- **⚠ 注意力线代码缺失（2026-11-12）**：`tarl_*` 智能体与 scene_attention 文本编码器的实现 `reproduction/tarl_tsc/` 从未完整入库，已于 `8bae003` 删除；att_entity_003/004 当前无法复现。很可能仍存于机器 34，恢复步骤见 `docs/plan_refactor_202611.md` §1.1。
<!-- PANEL-END -->

## 0. 总览

登记 run 总数：**15715**（train 3273、eval attempt 12110、smoke 140、calibration 18、batch 46）

| campaign | DONE | FAILED | ABORTED | SUPERSEDED | REGISTERED | SCOPE_DISCARD | LEGACY_UNCLEAR | 合计 |
|---|---|---|---|---|---|---|---|---|
| att_entity_004 | 1871 | 126 | 31 | 0 | 0 | 11 | 0 | 2039 |
| att_entity_003 | 1372 | 67 | 0 | 0 | 0 | 0 | 0 | 1439 |
| att_entity_002 | 2206 | 26 | 59 | 0 | 0 | 0 | 0 | 2291 |
| arterial_1x6 | 501 | 8 | 0 | 0 | 0 | 0 | 0 | 509 |
| tarl_reproduction | 132 | 11 | 0 | 0 | 0 | 0 | 0 | 143 |
| tarl_v21_formal | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 15 |
| plan5_b100 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 68 |
| plan1_dqn | 27 | 0 | 4 | 4 | 0 | 0 | 0 | 35 |
| ha_sodqn | 172 | 0 | 2 | 0 | 0 | 0 | 0 | 174 |
| milestone0_infra | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |
| paper_infra_validation | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 19 |
| misc_probes | 1 | 5 | 0 | 0 | 0 | 1 | 0 | 7 |
| xiasha1_online_v1 | 24 | 0 | 12 | 0 | 0 | 252 | 0 | 288 |
| xiasha1_explore_v1 | 6 | 0 | 6 | 0 | 0 | 0 | 0 | 12 |
| xiasha1_online_v2 | 288 | 0 | 0 | 0 | 0 | 0 | 0 | 288 |
| xiasha1_cross_eval_v1 | 384 | 0 | 0 | 0 | 0 | 0 | 0 | 384 |
| xiasha1_online_v3 | 288 | 0 | 0 | 0 | 0 | 0 | 0 | 288 |
| xiasha1_v3_cross_eval_v1 | 384 | 0 | 0 | 0 | 0 | 0 | 0 | 384 |
| hz1x1_sanity_v1 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 60 |
| xiasha1_offline_v1 | 0 | 0 | 0 | 0 | 0 | 126 | 0 | 126 |
| xiasha1_offline_v2 | 0 | 0 | 0 | 0 | 0 | 126 | 0 | 126 |
| hz1x1_offline_v1 | 162 | 0 | 0 | 0 | 0 | 144 | 0 | 306 |
| scen_cov_v1 | 256 | 0 | 0 | 0 | 0 | 0 | 0 | 256 |
| cl_hz1x1 | 5368 | 0 | 0 | 0 | 0 | 216 | 0 | 5584 |
| hitl_live_ab | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| hitl_live_baseline | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| hitl_live_gate | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |
| hitl_live_pilot5 | 75 | 0 | 0 | 0 | 0 | 0 | 0 | 75 |
| hitl_live_pseudo | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 11 |
| hitl_live_r2 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |
| hitl_live_r3 | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 150 |
| hitl_live_r4 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 20 |
| hitl_warmloop | 26 | 0 | 0 | 0 | 2 | 0 | 0 | 28 |
| seq8_anchor | 292 | 0 | 0 | 0 | 0 | 0 | 0 | 292 |
| seq8_demand | 24 | 0 | 0 | 0 | 0 | 18 | 0 | 42 |
| signal_control_v1 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 4 |
| takeover_probe_sig15 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 24 |
| xiasha1_compound_probe_v1 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 22 |
| xiasha1_compound_train_v1 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |
| xiasha1_v2_bestckpt_ts_v1 | 96 | 0 | 0 | 0 | 0 | 0 | 0 | 96 |
| **合计** | 14454 | 243 | 114 | 4 | 2 | 898 | 0 | **15715** |

状态语义见 GOVERNANCE.md §2。LEGACY_UNCLEAR=0 表示全部可识别 run 均通过队列/状态文件/目录归属证据还原出 campaign；个别 run 的臂内身份仍以命名约定为据，已在 notes 标注处保持保守。

## 1. campaign 索引

| campaign | runs | D/F/A/SD | 一句话结论 | 明细 |
|---|---|---|---|---|
| `att_entity_004` | 2039 | 1871/126/31/11 | 当前主线：保真达标；canonical≈empty 修正为窗内小幅净负；closure 缺口待修 | [明细](EXPERIMENTS/att_entity_004.md) |
| `att_entity_003` | 1439 | 1372/67/0/0 | TARL 对齐/传感矩阵完成（flx）；文本反事实探针完成 | [明细](EXPERIMENTS/att_entity_003.md) |
| `att_entity_002` | 2291 | 2206/26/59/0 | SGA/concat 文本注入矩阵：H2 支持、H3 未支持 | [明细](EXPERIMENTS/att_entity_002.md) |
| `arterial_1x6` | 509 | 501/8/0/0 | 半离线干线 stage0/1/2 流水线完成 | [明细](EXPERIMENTS/arterial_1x6.md) |
| `tarl_reproduction` | 143 | 132/11/0/0 | TARL 论文复现基线（完结，仅对照） | [明细](EXPERIMENTS/tarl_reproduction.md) |
| `tarl_v21_formal` | 15 | 15/0/0/0 | TARL v2.1 正式评估（完结） | [明细](EXPERIMENTS/tarl_v21_formal.md) |
| `plan5_b100` | 68 | 68/0/0/0 | Plan5 b100 半离线批 | [明细](EXPERIMENTS/plan5_b100.md) |
| `plan1_dqn` | 35 | 27/0/4/0 | Plan1 在线 DQN 基线 | [明细](EXPERIMENTS/plan1_dqn.md) |
| `ha_sodqn` | 174 | 172/0/2/0 | 半离线主线：E4 正式批完成；历史利用降遗忘但有适应代价 | [明细](EXPERIMENTS/ha_sodqn.md) |
| `milestone0_infra` | 12 | 12/0/0/0 | 基础设施验证 | [明细](EXPERIMENTS/milestone0_infra.md) |
| `paper_infra_validation` | 19 | 19/0/0/0 | paper 基建验证 | [明细](EXPERIMENTS/paper_infra_validation.md) |
| `misc_probes` | 7 | 1/5/0/1 | 零散探针 | [明细](EXPERIMENTS/misc_probes.md) |
| `xiasha1_online_v1` | 288 | 24/0/12/252 | xiasha1 纯在线 off-policy 基线（已收口：24 DONE/12 ABORTED/252 SCOPE_DISCARD） | [明细](EXPERIMENTS/xiasha1_online_v1.md) |
| `xiasha1_explore_v1` | 12 | 6/0/6/0 | xiasha1 充分探索验证（已暂停：slow 6 DONE，full 6 ABORTED；转 v2） | [明细](EXPERIMENTS/xiasha1_explore_v1.md) |
| `xiasha1_online_v2` | 288 | 288/0/0/0 | 已完成 288/288（几何截断 errata，指标以 v3 为准）：DQN 家族仍早收敛；PressLight 最差，FRAP/MPLight 服务更好但 TT 更高 | [明细](EXPERIMENTS/xiasha1_online_v2.md) |
| `xiasha1_cross_eval_v1` | 384 | 384/0/0/0 | 已完成 384/384（几何截断 errata）：FRAP/MPLight 相对泛化最好；DQN 家族绝对跨源 TT 最好；PressLight 最差 | [明细](EXPERIMENTS/xiasha1_cross_eval_v1.md) |
| `xiasha1_online_v3` | 288 | 288/0/0/0 | 已完成 288/288（几何修复正式口径）：ddqn/d3qn/sac TT 领先；uninserted≈1（v2 为 42–151，截断伪影消除） | [明细](EXPERIMENTS/xiasha1_online_v3.md) |
| `xiasha1_v3_cross_eval_v1` | 384 | 384/0/0/0 | 已完成 384/384：SAC/PressLight 泛化最稳（gap 9.0/15.4%），DQN 家族最差（27.5–49.0%），v1 排序翻转 | [明细](EXPERIMENTS/xiasha1_v3_cross_eval_v1.md) |
| `hz1x1_sanity_v1` | 60 | 60/0/0/0 | 已完成 60/60（5 臂）：体检签名全过（thr≈需求、uninserted≈0、量级正常）——记录管线无罪，xiasha1 异常为数据专属 | [明细](EXPERIMENTS/hz1x1_sanity_v1.md) |
| `xiasha1_offline_v1` | 126 | 0/0/0/126 | 作废：xiasha 数据不再用于 offline；126 SCOPE_DISCARD；相关历史源仅留档 | [明细](EXPERIMENTS/xiasha1_offline_v1.md) |
| `xiasha1_offline_v2` | 126 | 0/0/0/126 | 作废：xiasha 源全量弃用，126 SCOPE_DISCARD，队列 dropped；转 hz1x1_offline_v1 | [明细](EXPERIMENTS/xiasha1_offline_v2.md) |
| `hz1x1_offline_v1` | 306 | 162/0/0/144 | MPLight 修订离线基线已收官：162/162 DONE（batch_dqn/cql_dqn 各 81，末次 eval TT 均值 82.5/80.8s），旧矩阵 144 SCOPE_DISCARD；d3qn÷12 对齐 FRAP/SAC/MPLight；末端关闭 bootstrap | [明细](EXPERIMENTS/hz1x1_offline_v1.md) |
| `scen_cov_v1` | 256 | 256/0/0/0 | 半离线 L1 覆盖 1.00 vs 在线 0.25（pilot）；L2 不支持外推，sparse_ultra 反例须同段 | [明细](EXPERIMENTS/scen_cov_v1.md) |
| `cl_hz1x1` | 5584 | 5368/0/0/216 | hz1x1 持续学习：off-policy 机制性遗忘三骨干证实；+policy 锚 λπ=1 把 worst 退化压至 ≤1.0s 近零塑性代价；ρ 剂量响应显示锚定可省回放预算且 ρ=0.25 为甜点 | [明细](EXPERIMENTS/cl_hz1x1.md) |
| `hitl_live_ab` | 6 | 6/0/0/0 | — | [明细](EXPERIMENTS/hitl_live_ab.md) |
| `hitl_live_baseline` | 2 | 2/0/0/0 | — | [明细](EXPERIMENTS/hitl_live_baseline.md) |
| `hitl_live_gate` | 30 | 30/0/0/0 | — | [明细](EXPERIMENTS/hitl_live_gate.md) |
| `hitl_live_pilot5` | 75 | 75/0/0/0 | — | [明细](EXPERIMENTS/hitl_live_pilot5.md) |
| `hitl_live_pseudo` | 11 | 11/0/0/0 | — | [明细](EXPERIMENTS/hitl_live_pseudo.md) |
| `hitl_live_r2` | 30 | 30/0/0/0 | — | [明细](EXPERIMENTS/hitl_live_r2.md) |
| `hitl_live_r3` | 150 | 150/0/0/0 | — | [明细](EXPERIMENTS/hitl_live_r3.md) |
| `hitl_live_r4` | 20 | 20/0/0/0 | — | [明细](EXPERIMENTS/hitl_live_r4.md) |
| `hitl_warmloop` | 28 | 26/0/0/0 | — | [明细](EXPERIMENTS/hitl_warmloop.md) |
| `seq8_anchor` | 292 | 292/0/0/0 | — | [明细](EXPERIMENTS/seq8_anchor.md) |
| `seq8_demand` | 42 | 24/0/0/18 | — | [明细](EXPERIMENTS/seq8_demand.md) |
| `signal_control_v1` | 4 | 0/0/0/4 | — | [明细](EXPERIMENTS/signal_control_v1.md) |
| `takeover_probe_sig15` | 24 | 24/0/0/0 | — | [明细](EXPERIMENTS/takeover_probe_sig15.md) |
| `xiasha1_compound_probe_v1` | 22 | 22/0/0/0 | — | [明细](EXPERIMENTS/xiasha1_compound_probe_v1.md) |
| `xiasha1_compound_train_v1` | 30 | 30/0/0/0 | — | [明细](EXPERIMENTS/xiasha1_compound_train_v1.md) |
| `xiasha1_v2_bestckpt_ts_v1` | 96 | 96/0/0/0 | — | [明细](EXPERIMENTS/xiasha1_v2_bestckpt_ts_v1.md) |

## 附：追溯口径说明

- `eval` 行 = 冻结评估 attempt（评估包内 `attempts/` 或 tsc 输出树下独立 eval 目录），`behavior_source` 回指训练 run/包。
- `partial_*`/`quarantine`/`packages_INVALID_no_ckpt_load` 桶中的 attempt 按隔离语义登记为 ABORTED/FAILED，不与 DONE 混计。
- 残留 `运行中` 状态且核查无存活进程（2026-10-07）者登记 ABORTED，notes 记已写训练条数。
- §4.4 seed 贯通自检：对全部 DONE 训练行按 (campaign,arm) 分组比对终值三指标，未发现同配置不同 seed 终值全同，无 flag 行。
- 相同 run 的隔离/重复派发副本以 `alias_of` 指向 canonical run，统计时去重。
