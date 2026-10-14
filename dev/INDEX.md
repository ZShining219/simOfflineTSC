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
- `dev/w34/probes/`：w34_2 零训练探针分析件——`p2_binding_audit.py`（离线
  绑定审计：bound/fallback_bind/gap/co_active 分类 + 配对差分层）、
  `p4_stratify.py`（臂/种子/事件类/窗长分层 + 稀疏窗伪影检验）。消费
  eval_v2_paired 产物，不改原始 run 输出；随 feat/w34-eval-v2-closure 入库。
- `dev/lag_utility/`：滞后窗口（pre-symptom）效用分析件——`phase_map.py`
  （离线导出 hz4x4 相位→lanelink 映射 + 进口道特征序 → `hz4x4_phase_map.json`）、
  `lag_analysis.py`（事件→症状滞后分布 / bound-feeder-down 三侧发散 /
  首发散时刻 / 死需求份额 / foreign 特异性，消费 eval_v2 dump+probe）、
  `run_oracle_deadlane.py`（closure 激活期馈入节点死车道观测遮蔽 oracle，
  判决文本通道可兑现上界）、`probe_junction_block.py`（libsumo 微探针：
  封闭边需求车辆的滞留位置/车道/箱内阻塞核验）、
  `run_oracle_corridor.py`（corridor oracle：closure/ blockage 窗内在目标
  节点强制选"走廊方向服务最少/最多"相位，判决双向 headroom；
  `--kind` 选事件族、`--mode min|max`、`--window active|recovery`、
  `--lead` 可把窗延伸到预告段测预置收益）、
  `probe_surge_inject.py`（libsumo 微探针：不走 trainer 直绑
  SumoEventRuntime，验证 demand_surge 的路由绑定/确定性注入/announced→
  active→cleared 三态报告渲染/严格 grounding 全链路）。探索期实现，
  产物落 `data/output_data/analysis/att_entity_004/lag_utility/`。
- `dev/pairaux/`：配对课程 + 事件语义辅助奖励线（73 侧思路本地化）——
  `build_pair_plan.py` 把源 plan 的每条 episode 复制成
  `{canonical, empty}` 对（同 schedule/同 SUMO seed，仅文本有无不同），
  产出 `configs/events/plans/hz4x4_pair_v1.yml`（200 行=100 对）；
  surge 线源 plan 由 `tools/build_random_event_plan.py --kinds demand_surge`
  生成（连通 OD 对经 sumolib getShortestPath 验证），配对版
  `hz4x4_surge_pair_v1.yml`（200 行=100 对，eval=三走廊 surge schedule
  `configs/events/hz4x4_surge_eval_v1.yml`）；
  配套机制在 `world/sumo_events/plans.py`（行级 `sumo_seed`/
  `text_condition` 覆盖）与 `reproduction/tarl_tsc/parent_adapter.py`
  （`_effective_text_condition` 逐集生效 + `model.reward_aux` 辅助奖励，
  仅注入 replay、不动评估 metric，window 支持 active|recovery|both）。
  训练臂配置 `att004_paircurr_{smoke,200}.yml` /
  `att004_paircurr_aux_smoke.yml`。
- `dev/govern_1007/`：本治理任务的授权暂存位（当前为空）。
- `dev/scene_semantics/`：轨迹场景语义标注探索件（P0 窗口恢复/统计 → P2 事实描述+冻结编码 →
  P3 三层匹配验证），探索期实现；执行依据 `docs/plan_trajectory_scene_semantics.md`，
  产物落 `data/output_data/scene_semantics/`。
- `dev/cl_hz1x1/`：hz1x1 持续学习（sequential stage 链 + 冻结下三角评估矩阵）
  队列生成件——`build_queue.py` 产出 `artifacts/cl_hz1x1/run_queue_cl1.json`
  （1476 任务：ddqn/dueling_ddqn/frap×{clear,fifo}+ppo×{clear} 四阶段链 ×
  2 序 × 3 种子 + 单任务基线 + 逐 cell 冻结评估）与台账 REGISTERED 行；
  依赖 `run.py` 的 `model.stage_checkpoint`/`replay_policy`/`epsilon_mode` 路径。
  姊妹件 `build_queue_so.py` 产出 `run_queue_so1.json`（612 任务：半离线臂
  soddqn/soduel/sofrap × 2 序 × 3 种子，offline_ratio=0.25 因果历史，
  在线 buffer 随场景清空、历史包按 stage 归档 `history_archive/`）；
  基线复用 cl1（offline_ratio=0 同码路）。
  锚定件 `build_queue_soa.py`（argv: smoke|screen|main）产出
  `run_queue_soa_<phase>.json`——与 so1 同协议但 stage>1 额外加载
  `teacher_paths`（各可见场景的 stage 末冻结 ckpt）并在离线样本上加
  输出锚定损失 `λ_π·KL(softmax(Q_teacher/T)‖softmax(Q_θ/T))`
  +可选 `λ_q·Huber`（mask 已执行动作维）；锚定实现位于
  `agent/dqn.py::_anchor_loss` 与 `agent/frap.py::_anchor_loss`，
  诊断经 `train_diagnostics` 字段（metrics schema v5）落 records.jsonl。
  收割/绘图件——
  `harvest_matrix.py` 产出 `matrix_report{,_so}.json`（下三角矩阵+最深退化汇总），
  `harvest_soa.py` 产出 `matrix_report_soa_<prefix>.json`（同构口径，适配
  soa 命名前缀；argv: `<run_id前缀> [输出文件名]`），
  `plot_so_compare.py` 产出 `fig_so_{retention_trajectory,worst_degradation}.png`
  （clear/fifo/半离线三条件对照，逐种子细线+均值粗线+基线虚线），
  `plot_soa_compare.py` 产出 `fig_soa_{retention_trajectory,worst_degradation}.png`
  （o2 三条件对照：replay / +policy锚 / +policy+value锚），
  `plot_soa_panels.py` 产出 `fig_soa_{cellmap,bars,diag}.png`
  （逐 cell ΔATT 热力三角、o2+o3 双序 worst 柱状、锚定/TD 诊断曲线），
  `plot_soa_rho.py` 产出 `fig_soa_rho.png`
  （ρ∈{0.125,0.25,0.5} × {replay,+P} 剂量-响应：worst 退化 + 对角线塑性）。
  诊断件 `drift_diag.py`——历史策略/价值漂移诊断（冻结推理，不改 ckpt）：
  `pool` 用基准 ckpt 在 held-out sumo 种子上 greedy rollout 建场景公共
  参考状态池（~1500 态/场景，c4/c2 合并两臂基准并留 src 标签）；
  `measure` 对链上 ep{0,50,100} ckpt 测 Q-L1/去中心化 Q-L1/贪心翻转率
  （累计+相邻）/Action Gap，含 archive 补充数据集与跨臂决策差异行；
  `mkqueue` 生成中途 ckpt 冻结 TT 补评队列（治理路径 runner）；
  `harvest`/`report` 汇总 TT 并出漂移-进度曲线与漂移-TT 对应图。
  产物 `artifacts/cl_hz1x1/{drift_pool/,drift_*_rows.json,drift_*_tt.json,
  drift_figs/,run_queue_drift_*.json,run_state_drift_*/}`。
  诊断件 `grad_diag.py`——纯离线梯度协调探针：从 resumable checkpoint
  重建模型+优化器，以 so1 归档 transition 为新旧数据探针，测 batch 梯度
  余弦/符号冲突/范数比与单步真实更新前后旧场景 TD 误差变化；
  产出 `artifacts/cl_hz1x1/grad_diag.json`（5 边界 × so/clear 双 ckpt）。
  诊断件 `fqe_probe.py`——离线策略价值评估（OPE）可行性探针（纯离线，
  不改 ckpt、不跑新仿真）：6 个候选策略/种子 = s1 基准 st1ep100 +
  clear/so st4 ep{0,50,100}；真值取既有 ev*/drift_* 冻结 eval 格的
  reward_sum/travel_time（eval 种子 101–103）；以 so 臂 st1
  `history_archive` 归档 transition 为独立历史轨迹（shard 末条标记
  terminal），半梯度 FQE（γ=0.95，DQNNet 同构）逐策略估计 V；
  覆盖变体 all/early(ep1–50)/late(ep51–100)/n4k/explor（仅 off-greedy
  转移）探覆盖-误差关系；输出 Spearman(V̂, 真回报) 与
  good/bad 策略分离判定；附 FQE 随机种子方差探针（信号 vs 噪声）。
  产物 `artifacts/cl_hz1x1/{fqe_s1.json,fqe_s1_noise.json,mc_s1.jsonl}`。
  审计：Bellman target/γ/reward/done 口径均无误；失败根因为
  覆盖不足——各候选策略所选动作在档案状态上的支持率仅
  10–50%，且崩溃 clear 策略在每种覆盖变体下支持率最低
  （late 变体 base 69% vs clear_ep50 27%）。
  口径核验（--mc 子进程单回合 rollout，`mc_s1.jsonl` 63 行）：
  逐决策步 reward 的 J_γ=Σ0.95^t·r_t 与官方 reward_sum 对拍
  max|Δ|=0.01；J_γ 真值排序改变（sd0 base 退至第 5、clear_ep50
  差距缩至 ~3×）但 FQE SEP 仍 4/15——口径对齐不能挽救排序。
  结论：单点 FQE 排序不可靠（SEP 4/15、ΔV̂≈种子噪声），
  支持走不确定性约束的历史策略保护路线。
- `dev/_legacy/20261007/`：见 `dev/_legacy/20261007/README.md`。

## 已裁决归档（2026-10-07 Z 裁）

1. `tools/*_matrix*.sh` 10 件矩阵驱动 → 主树 `dev/_legacy/20261007/matrix_drivers/`
   （§6.1 已禁止该模式新写；原路径留此记录供证据链对照：原 `tools/`、配套运行日志
   原 `logs/*.log`）。
2. 根目录散件 `logs/`、`final_result/`、`interpretation/`（空目录）→ 主树
   `dev/_legacy/20261007/loose_1007/`（保持目录原名，不并入 run 目录以免改产物原始性）。
3. 台账 6 个 REGISTERED → SCOPE_DISCARD：`att004__a2_stg2_s{7,17,27,37,47}`
   （a2s 臂 stg2 已挂起）与 `inspect_mplight`（基线已被 baseline_eval 覆盖）。
   附注已写入 runs.jsonl 各行 notes，EXPERIMENTS.md 已重生成。
