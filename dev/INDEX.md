# dev/INDEX — 一次性脚本与散装驱动登记

> 追溯登记 2026-10-07（gov/experiment-governance 分支）。登记范围：仓根、`scripts/`、
> `tools/`、`dev/` 下的启动/驱动/一次性脚本。**本轮盘点未发现可判定为"孤儿"的脚本，
> 未做任何移动**；存疑项见文末。后续一次性脚本一律放 `dev/<topic>/` 并在此登记。

## 仓根入口（非一次性，保留原位）

| 路径 | 用途 | 关联 | 判定 |
|---|---|---|---|
| `run.py` | 实验主入口/参数定义 | 全仓 | 正式入口 |
| `offline_run.py` | 离线数据训练入口 | 半离线线 | 正式入口 |
| `sequential_run.py` | sequential 模块入口 | sequential/ | 模块入口 |
| `arterial_run.py` | arterial 线入口 | arterial/ | 模块入口 |
| `environment.py` | 环境探测辅助 | — | 基础设施 |
| `scripts/compare_run_configs.py` | M0 run 配置比对（调 utils/run_config_compare） | milestone0_infra | 工具件，保留 |

## tools/ — 一次性启动/驱动脚本（campaign 绑定，原位保留）

以下为"一次性矩阵派发/批次驱动"性质脚本。它们绑定了已完成 campaign 的派发方式，
是证据链的一部分（对应 `logs/*.log`、`artifacts/*/run_state_*`），**不移**；
今后同用途新脚本不得再写进 tools/，须放 `dev/<topic>/`。

| 路径 | 用途 | 关联 campaign |
|---|---|---|
| `tools/dense_eval_matrix.sh` | dense 条件评估矩阵派发 | att_entity_004 |
| `tools/mid15_eval_matrix.sh` | mid15 评估矩阵派发 | att_entity_004 |
| `tools/mid20_eval_matrix.sh` | mid20 评估矩阵派发 | att_entity_004 |
| `tools/single_eval_matrix.sh` | single 条件评估矩阵 | att_entity_004 |
| `tools/sga_cf_probe_matrix.sh` / `sga_cf_probe_matrix2.sh` | sga 反事实探针矩阵 | att_entity_002 |
| `tools/flx_diag_probe_matrix.sh` / `flx_stage2_probe_matrix.sh` | flx 诊断/stage2 探针矩阵 | att_entity_003 |
| `tools/att004_baseline_matrix.sh` | att004 基线评估矩阵 | att_entity_004 |
| `tools/att004_statelib_matrix.sh` | att004 state-lib 矩阵 | att_entity_004 |

## tools/ — campaign 专用分析/构建脚本（原位保留）

`att004_*`（12 件：cond_eval、baseline_eval、build_queue、focus_null/score、holdout_plans、
probe3_sup、probes、single_cmp_layered、status、statelib_matrix.sh）、`flx_*`/`sga_*`/
`analyze_*`/`build_*`/`stage5_*`/`aggregate_text_utility`/`allocation_accuracy_*`/
`plan1_*`/`audit_arterial_*`/`build_arterial_*`/`capture_arterial_*`/`run_arterial_*`/
`run_tarl_*`/`run_align_retrieval`/`run_sga_counterfactual` 等——均为对应 campaign
的一次性但**证据关联**分析/构建件，归属见 `EXPERIMENTS.md` 各节；不移。

## tools/ — 可复用工具（长期件，原位保留）

`run_experiment_queue.py`（队列 runner）、`run_paper_baseline.py`、`training_dashboard.py`、
`feishu_wiki.py`、`resource_efficiency_audit.py`、`resource_evidence_dashboard.py`、
`ha_continuous_resource_audit.py`、`render_resource_efficiency_cei.py`、
`resource_metric_audit.py`、`traffic_visualization.py`、`sumo_gui_comparison.py`、
`sumo_html_comparison.py`、`export_compact_experiment_tables.py`、`experiment_plotting/`、
`traffic_flow_profile/`、`xiasha_sumo/`、`convert_mplight_ckpt.py`、`mb_gate_precheck.py`、
`watch_stage5_eval.py`、`stage_eval_aggregate.py`、`run_tarl_grad_diag.py` 等；
登记/说明以 `tools/README.md` 为准。

## artifacts/ 内部驱动（证据附属，原位保留）

`artifacts/att_entity_004/run_state_*/run_*.sh`、`artifacts/sga_concat_colight_200_v1/
{wave1_watcher,flip_epsilon_after_stragglers}.sh` 等——评估/看门狗脚本已随状态目录
成为 campaign 证据本体，不移出 artifacts/。

## dev/ 现状

- `dev/govern_1007/`：本治理任务的授权暂存位（当前为空，临时产物放治理工作区 scratch）。
- `dev/_legacy/20261007/`：见 `dev/_legacy/20261007/README.md`（本轮无移入件）。

## 存疑/待裁决

1. `tools/*_matrix.sh` 8 件矩阵驱动：倾向归档 `dev/_legacy/`（治理 §6.1 已禁止该模式
   新写），但它们与 `logs/*.log`、run_state 证据链绑定，本轮保守不移；请裁决。
2. `logs/*.log`（根目录运行时日志）、`final_result/*.log`：产物类散件，是否归
   `dev/_legacy/` 或并入对应 run 目录，待裁决（不删）。
3. `interpretation/`（空目录）、`dev/govern_1007/`（空）：保留原位待裁决。
