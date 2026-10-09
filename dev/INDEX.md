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

- `dev/ledger/`：台账构建器（inventory.py / build_ledger.py / gen_experiments.py +
  README），2026-10-07 裁决收仓长期维护；源自治理工作区 govern_1007/scratch/。
- `dev/govern_1007/`：本治理任务的授权暂存位（当前为空）。
- `dev/_legacy/20261007/`：见 `dev/_legacy/20261007/README.md`。
- `dev/takeover_probe/`：控制器接管与调度器探针族（探索件）。
  `probe.py` 为接管分叉探针：复用 tools/sumo_live_broadcast 控制器适配器 +
  world 构建；前缀到指定决策步后切换接管控制器，逐决策记录车道级排队，
  分叉点状态指纹校验确定性；支持 `--inject` 突发车流注入
  （注入实现已收编 `tools/sumo_live_broadcast/injects.py`，与转播服务共用）。
  `dispatch_state.py`（车道/全局/事件文本特征化+播报延迟）、
  `dispatch_model.py`（Scene-Guided Attention / SelfAttention /
  Concat 三变体 DispatchRouter——对应任务书三方对比：场景引导注意力 /
  自注意力Transformer / 无注意力基线；attn 变体可选
  `--residual-pool mms` 残差增强：mean+max+std 仅统计物理特征）、
  `router_runtime.py`（窗口化接管运行循环）、`train_router.py`（SMDP Double-DQN
  训练，支持目标条件化 reward + 配对课程 + aux 对齐衰减 + 多事件干扰注入）、
  `eval_router.py`（冻结评测 A/B/C/D+文本消融，`--extra NAME=CKPT` 扩臂）、
  `scan_pairs.py`（反事实胜者矩阵扫描：固定臂在不同
  事件参数下的最优翻转）、`eval_objective.py`（目标条件化配对评测：
  同物理事件仅声明目标不同；单事件格为 global/focus 成员对，
  `edgeA+edgeB` 多事件格为 focus@A/focus@B 绑定翻转对）、
  `make_figs.py`（机制类扩展出图：配对 regret、消融阶梯、绑定一致性、
  训练曲线、排队轨迹、注意力热力图，以及 held-out 逐时间对比曲线
  （queue/throughput/事件车道/控制器时间线 fig8–fig11），PNG 单出）、
  `collect_attn_trace.py`（冻结 checkpoint 注意力权重轨迹采集，
  供 fig7 热力图；windows 日志含 `attn_weights`+`lane_names`）、
  `probe_repr.py`（表征层探针：同 rollout 下三冻结编码器配对取 z，
  测任务书主指标——事件影响预测 / 拥堵等级分类 / 压缩保持 /
  参数量与延迟，留一 episode CV）、
  `channel_ablate.py`（信道归因：基于 z_dump 重建 snap，
  event_flag 与文本信道独立扰动，测绑定/Δz/决策翻转；
  免 SUMO 免重训）、`fork_probe.py`（诊断性分叉：同 seed 重放 +
  指定窗口强制控制器，测关键窗臂选的反事实代价；
  `run_router_episode` 加 `force_choices` 钩子）。
  `collect_scene_dump.py`（场景蒸馏数据集采集：A 自然匹配 /
  B 失配播报 / C 纯播报 / D 无播报四相，全 8 边覆盖，
  每窗存 lanes/text/global + 场景标签）、`scene_distill.py`
  （表征层蒸馏基准：各编码臂同输入同 z 维同场景目标训练产 z，
  场景保真探针 + 冻结 z 线性探针 + episode 级切分 +
  未见播报边 held-out 泛化，与调度性能解耦的第一级评测；
  编码臂含 sga 复合模块 / attn / selfattn / concat / flat-mlp）。
  训练课程支持
  `--periods/--durs/--onsets` 网格覆盖（重度偏向校准用），
  评测支持 `--events` 追加未见事件格
  为场景引导注意力调度
  Pilot 配套件。产物在
  `output_data/dev_takeover_probe/`（不入库）；
  转播侧 `DispatchController` 适配器在 `tools/sumo_live_broadcast/controllers.py`。
- `dev/hitl_live/`：HITL 拓展实验件（设计依据 `docs/plan_hitl_live.md`）。
  `package.py` 为会话打包器：把转播录制会话（`--record` 产出的
  transitions/proposals/interventions JSONL）打包成 Plan 1 兼容
  `trajectory/` 分片（复用 `EpisodeTrajectoryWriter`）+ Plan 2 风格
  `source_shards.jsonl` + `datasets/<network>/<stage>/manifest.json`，
  语义哈希复用 `dataset._scene_semantics` 同口径计算；产物可被
  `OfflineTrajectoryDataset`/`validate_offline_dataset` 直接消费
  （自检内嵌于打包流程）。变长 episode、HITL 溯源进
  `behavior_mode`/`hitl/labels.jsonl` 副产物，不伪造正式 run 元数据。
  用法：`python dev/hitl_live/package.py --session <session_dir>
  [--session <session_dir>...] --out <archive_dir> --network <name>
  [--seed N] [--strict]`（多会话合并：episode 全局重编号、global_step
  连续化、归属记 labels/source_shards；另发 `expert_pool_manifest.json`
  供 sequential 专家腿消费）。产物 `output_data/hitl_archives/`（不入库）。
- `dev/hitl_live/paired_probe.py`：三臂（student/hitl/expert）同 seed 配对
  运行时探针——第 0 层"专家资格体检"工具。
- `dev/hitl_live/make_figs.py`：probe 佐证图（排队轨迹+gate 标记/终端三指标/
  事件窗放大/成本-效果）。
- `dev/hitl_live/build_pilot_plan.py`：C 通道训练臂对照 plan 生成器——
  同一 ep0 checkpoint/seed 出 HITL_C0/HITL_EXP/HITL_CTRL/HITL_DEMO 四臂
  （差异仅 `expert_pool_manifest`/`expert_ratio`）；产物用
  `sequential_run.py run-child` 逐臂执行（探索性 pilot，不入台账）。
- `dev/hitl_live/curve_figs.py`：训练期佐证图——自主冻结 eval 学习曲线
  （vs 决策数）、AULC、对 C0 逐点 delta、专家腿消耗量（sampling_windows
  账本）。

## 已裁决归档（2026-10-07 Z 裁）

1. `tools/*_matrix*.sh` 10 件矩阵驱动 → 主树 `dev/_legacy/20261007/matrix_drivers/`
   （§6.1 已禁止该模式新写；原路径留此记录供证据链对照：原 `tools/`、配套运行日志
   原 `logs/*.log`）。
2. 根目录散件 `logs/`、`final_result/`、`interpretation/`（空目录）→ 主树
   `dev/_legacy/20261007/loose_1007/`（保持目录原名，不并入 run 目录以免改产物原始性）。
3. 台账 6 个 REGISTERED → SCOPE_DISCARD：`att004__a2_stg2_s{7,17,27,37,47}`
   （a2s 臂 stg2 已挂起）与 `inspect_mplight`（基线已被 baseline_eval 覆盖）。
   附注已写入 runs.jsonl 各行 notes，EXPERIMENTS.md 已重生成。
