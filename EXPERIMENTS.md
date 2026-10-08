# EXPERIMENTS — 实验台账（索引）

> 追溯登记 2026-10-07（治理分支 gov/experiment-governance）。
> 事实源 = `ledger/runs.jsonl`（每 run 一行机读）；本文件 = 索引层，明细按 campaign 分文件 `EXPERIMENTS/<campaign>.md`——**按需取读，勿通读**。
> 口径：`run_kind=train|eval|smoke|batch|calibration`；eval 行为单次冻结评估 attempt，`behavior_source` 指回训练 run 包名；`alias_of` 非空者不作独立样本统计。
> 机器一律代号 34/73；路径均为仓内相对路径。

<!-- PANEL-BEGIN -->
## 当前面板（2026-10-07，campaign 收官时刷新）

- **当前有效设计栈**：`agent/scene_attention.py` + `agent/tarl.py`——per-node z_task 结构化槽位 + SemanticAuxHead 辅助监督 + 反事实探针四层判定；配置族 `configs/tsc/att_entity_004/`；参考臂 `a3s`，最新修复线 `fixG`（收官测评已收割：vs a3s 收敛更快更优 ep100 −8.9s，但效用层未翻正 canonical−empty 中位 +0.58s）。
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

## 1. campaign 索引

| campaign | runs | D/F/A/SD | 一句话结论 | 明细 |
|---|---|---|---|---|
| `att_entity_004` | 1978 | 1812/126/30/10 | 当前主线：保真达标；canonical≈empty 修正为窗内小幅净负；closure 缺口待修 | [明细](EXPERIMENTS/att_entity_004.md) |
| `att_entity_003` | 1439 | 1372/67/0/0 | TARL 对齐/传感矩阵完成（flx）；文本反事实探针完成 | [明细](EXPERIMENTS/att_entity_003.md) |
| `att_entity_002` | 2291 | 2206/26/59/0 | SGA/concat 文本注入矩阵：H2 支持、H3 未支持 | [明细](EXPERIMENTS/att_entity_002.md) |
| `arterial_1x6` | 509 | 501/8/0/0 | 半离线干线 stage0/1/2 流水线完成 | [明细](EXPERIMENTS/arterial_1x6.md) |
| `tarl_reproduction` | 143 | 132/11/0/0 | TARL 论文复现基线（完结，仅对照） | [明细](EXPERIMENTS/tarl_reproduction.md) |
| `tarl_v21_formal` | 15 | 15/0/0/0 | TARL v2.1 正式评估（完结） | [明细](EXPERIMENTS/tarl_v21_formal.md) |
| `plan5_b100` | 68 | 68/0/0/0 | Plan5 b100 半离线批 | [明细](EXPERIMENTS/plan5_b100.md) |
| `plan1_dqn` | 35 | 27/0/4/0 | Plan1 在线 DQN 基线 | [明细](EXPERIMENTS/plan1_dqn.md) |
| `milestone0_infra` | 12 | 12/0/0/0 | 基础设施验证 | [明细](EXPERIMENTS/milestone0_infra.md) |
| `paper_infra_validation` | 19 | 19/0/0/0 | paper 基建验证 | [明细](EXPERIMENTS/paper_infra_validation.md) |
| `misc_probes` | 7 | 1/5/0/1 | 零散探针 | [明细](EXPERIMENTS/misc_probes.md) |

## 附：追溯口径说明

- `eval` 行 = 冻结评估 attempt（评估包内 `attempts/` 或 tsc 输出树下独立 eval 目录），`behavior_source` 回指训练 run/包。
- `partial_*`/`quarantine`/`packages_INVALID_no_ckpt_load` 桶中的 attempt 按隔离语义登记为 ABORTED/FAILED，不与 DONE 混计。
- 残留 `运行中` 状态且核查无存活进程（2026-10-07）者登记 ABORTED，notes 记已写训练条数。
- §4.4 seed 贯通自检：对全部 DONE 训练行按 (campaign,arm) 分组比对终值三指标，未发现同配置不同 seed 终值全同，无 flag 行。
- 相同 run 的隔离/重复派发副本以 `alias_of` 指向 canonical run，统计时去重。
