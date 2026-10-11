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
- `dev/takeover_probe/`：场景引导注意力信号控制探针族（探索件）。
  **当前主线（2026-10 定型）**：单层场景条件信号控制器——
  `SceneJointPolicy`（signal_model.py canonical 标记）：dqn_s1
  底座图内保留（ckpt 初始化 + `--base-lr` 微调）+ sga 编码器
  （车道自注意力+播报交叉注意力+边身份绑定）+ 门控有界残差
  Q = q_base + dose·spread·gate(bind)·tanh(δ(z))。
  训练链路：① adapter 教师（scene_mp.py SceneGuidedDQN：冻结
  底座+注意力启发式偏置）跑事件局 → 赢家过滤 → ② 学生加权
  CE 蒸馏（`--distill-*`，sig9_joint_bc，-2.7% 胜 dqn_s1 与
  MP，且超教师）→ ③ 原装 reward TD 微调（`--orig-reward` =
  -12×入口车道排队均值，与 dqn_s1 收敛目标同义；
  `--init-from` 热启动学生权重；sig10_joint_rl）。
  入口：`train_signal.py`/`eval_signal.py`；
  `signal_runtime.py` 为 10s 决策粒度 episode 驱动器。
  产物 `output_data/dev_takeover_probe/`（不入库）：sig5_*
  编码器消融 / sig6* from-scratch 信号 agent（学不会驾驶，
  已弃） / sig7_res_* 冻结底座学习残差（+0.7%，被 joint
  取代） / sig8_joint_sga 错 reward 联合微调（+9%，灾难性
  遗忘，反例） / sig9_joint_bc 蒸馏学生（当前最优） /
  sig10_joint_rl 原装目标 RL 续训。
  **已否决归档**：双层 DispatchRouter 架构（上层选控制器、
  下层选相位）移 `_router/`（含 train/eval/runtime 及表征
  探针，见目录 README）；DispatchRouter 类仅在
  dispatch_model.py 留档供归档件引用。注意与 codex 分支
  scene_attention+tarl（EXPERIMENTS.md 的"注意力线"）是
  不同项目不同路网，勿混。
  **仍活用的共享件**：`probe.py`（build_world /
  build_controllers / DEFAULT_DQN_S1）、`dispatch_state.py`
  （FeatureBuilder：车道 token+播报 text_vec+global）、
  `dispatch_model.py`（编码器族，见文件头 NOTE）、
  `scene_mp.py`（adapter 教师；文件名只提 MP 但含
  SceneGuidedDQN）、`plot_scene_effect.py`（介入前后曲线）、
  `tools/sumo_live_broadcast/controllers.py`（与直播系统
  共享的 DQN 推理适配——改动影响两条线）。
  历史陷阱（已修勿复踩）：eval 每臂重解析 injects（
  apply_injects 会改写 dict）+ 每臂重绑控制器（world.reset()
  重建 intersections）；scenedqn t_min=0 对齐原生 DQN 纪律。
- `dev/takeover_probe/dispatch_specialist.py`：注意力 key 派发壳
  （探索件）。`KeyDispatchAgent`：encoder 绑定报告→(kind,edge)
  key≥bind_tau → 注册表查专项，否则基座 act()；规则派发非学习
  策略（避开被否 SMDP router 的双层决策）；oracle 模式读
  `fb.event_fields` 作鉴别上限；audit 记录逐 tick 派发轨迹。
- `dev/takeover_probe/dispatch_eval.py`：派发门评测（探索件）。
  surge 网格 ×{dispatch_attn, dispatch_oracle, dispatch_blind,
  dispatch_mp(无文本规则专项), joint_always, FT/MP/dqn_s1}。
  dispatch_gate_v1 结果（n=16）：blind≡dqn_s1 逐位回退、
  attn≡oracle 鉴别无损、dispatch_mp 全场最优
  (ev 15207 vs MP 15392；queue_int −5% vs MP)。
- `dev/takeover_probe/dispatch_fig.py`：派发门佐证图生成（探索件）。
  单事件轨迹 episode 出四联图：车道×车道自注意力（交通响应层，
  sa_sal）+ text->lane 交叉注意力热力图（报告臂 vs 盲化臂同标尺）
  + 全边 mass 束/接管曲线（fig_attn.png）；gate 聚合出 base vs
  Scene-Guided Attention 归一化优化柱状（fig_bars.png）。产物在
  `output_data/dev_takeover_probe/dispatch_figs_v1/`。
- `dev/takeover_probe/nl_scene.py`：自然语言播报→场景字段结构化
  前端（探索件）。`nl_render` 把播报字段渲染成多模板 NL 句式
  （模拟异构上报），`NLSceneParser`（hash n-gram 线性分类器 +
  边名字面提取）按"模板留出"训练评估；`NLDispatchAgent` 在 act
  内完成 NL→字段→text_vec 重建，下游注意力绑定/规则派发零改动。
  演示 `output_data/dev_takeover_probe/nl_scene_v1/nl_demo.json`：
  留出话术 acc≈0.71（硬负例/未见过恢复句式仍误）、NL 臂接管
  72 tick 与结构化臂逐位一致，命中未见过恢复句式的变体仅丢尾段
  接管（60 tick）——优雅降级而非误派发。
- `dev/hitl_live/`：HITL 拓展实验件（设计依据 `docs/plan_hitl_live.md`）。
  采集端 `tools/sumo_live_broadcast` 已有真人专家席：`--expert human`
  + gate 命中冻结仿真弹横幅（相位按钮+倒计时），`--hitl-human-timeout`
  超时沉默仍计费，答案落 hitl.expert_id='human'（详见设计文档专家席行）。
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
  （vs 决策数，多 seed mean±std 带）、AULC、对 C0 同 seed 配对 delta、
  专家腿消耗量（sampling_windows 账本）；eval 快照 digest 去重的
  前导/内段别名按"同快照同估值"语义补齐（与 paired_bootstrap 同口径）。
- `dev/hitl_live/dump_lane_topology.py`：场景车道拓扑转储（in_lanes 顺序、
  车道长度→容量、相位→服务车道映射），B 通道特征/教师分的 ctx 来源。
- `dev/hitl_live/segments.py`：B 通道片段切割 + 确定性教师分
  （h=-0.5Q-0.5U，逐车道 q_l/c_l + 相位服务史重建 u_l）+ spec §6.2
  奖励模型特征 [obs|one_hot(a)|obs'|q/c|u/T_u]。
- `dev/hitl_live/build_pairs.py`：档案→片段→偏好对数据集
  （片段均值特征 + H 标签 + 平局保留）。
- `dev/hitl_live/train_reward.py`：偏好奖励模型训练（128→128→1，BCE，
  episode 级留出，早停；产 reward_model.pt + normalization.json +
  reward_manifest.json）。
- `dev/hitl_live/calibrate_reward.py`：B 通道归一化统计冻结（校准集上
  h/f_hat 的 μ/σ，产 reward_calibration.json 供训练器 shaping）。
- `dev/hitl_live/paired_bootstrap.py`：同 seed 配对 bootstrap（10000
  重采样；AULC_delta/final_delta + 95% CI，spec §11.4 口径）。
- `sequential/reward_model.py`：偏好奖励网络 PreferenceRewardNet
  （128→128→1，温度 0.25，BCE 含平局 y=0.5）。
- `sequential/reward_shaping.py`：B 通道奖励塑形——r=r0+λ·z（B1 规则
  教师分 / B2 冻结模型分；z=clip((f-μ)/σ,-3,3)；崩溃回退 r0；仅作用
  在线腿）。塑形器在首个转移处惰性构建（SUMO 连接在 reset 前未载路
  网）；轨迹 npz 保留环境真值 r_base，塑形审计（z/h/f_hat/r_base 均
  值、回退计数）经 ReplayMetadata.reward_aux 汇总进每集
  replay_diagnostics JSON。经 plan child `trainer_overrides` 接入
  （reward_mode/reward_lambda/reward_calibration_path/
  reward_model_path）。
- `sequential/demo_ac.py`：A 通道 DemoLight 机制——actor/critic 各为
  DQNNet(d→20→20→P)；预训练 = actor Gumbel-NLL（独立均匀流）+
  critic PER（α0.6/β0.4→1，n-step≤10 后缀 + margin 0.8 + L2 1e-5，
  目标网每 50 步同步）；在线逐决策更新（critic 0.5TD + actor 优势
  -logπ·A）；在线起点重置优化器+目标网拷贝 critic（spec 边界）。
  plan child `policy='demo_ac'` + `demo_pretrain` 块接入；快照复用
  online_only 通道（actor logits argmax = argmax π，evaluator 无分叉）。
- `dev/hitl_live/build_demos.py`：示范集构建——打包档案→扁平
  (obs,aE,r,obs',done,episode_id) npz，episode 序保持（n-step 后缀
  按 episode_id 分组推导）。
- `sequential/adjudication.py`：B′ 采样时裁定层——判官面板
  （terms 带参治理词表 / reward_model 冻结 RM / qgap 收敛 DQN 快照，
  qgap 硬门槛 episode≥min_episode 且 gradient_updates>0，ep0 档拒载），
  `z=clip(Σw_j·z_j/Σ|w_j|,±clip)` 权重归一合成，单判官崩溃降级不
  回退 r0；manifest digest 入 checkpoint（resume 强校验防换判官）。
  转移只存 r_base+裁定特征（payload.aux / reward_aux），三腿在
  sample 后同口径 r_eff=r_base+λz 进 TD loss；腿特征经
  `leg_features_from_view` 按 (run_id,episode_id) 分组回放
  ServiceGapTracker 重建。经 plan child `trainer_overrides` 的
  `adjudication` spec + `adjudication_ctx[_path]` 接入；与旧
  reward_mode b1/b2 互斥。
- `dev/hitl_live/calibrate_adjudication.py`：B′ 判官面板逐判官归一化
  校准——在 expert_pool_manifest 列表上逐转移算各判官 raw 分，回填
  spec 的 mu/sigma/calibration_count 产冻结 manifest
  （`output_data/hitl_probe/adjudication/manifest_v1_calibrated.json`）。
  已跑：hitl_ep50+student_ep50+expert_ep50 三池 731 转移校准一版。
- `dev/hitl_live/build_pilot_plan.py` 增 `--bprime-arms`/`--bpc-arms`
  + `--adjudication-manifest`/`--adjudication-ctx`：HITL_BP（纯在线
  统一裁定）/ HITL_BPC（裁定+专家腿，旧 B 不能表达的组合）两臂。

## 已裁决归档（2026-10-07 Z 裁）

1. `tools/*_matrix*.sh` 10 件矩阵驱动 → 主树 `dev/_legacy/20261007/matrix_drivers/`
   （§6.1 已禁止该模式新写；原路径留此记录供证据链对照：原 `tools/`、配套运行日志
   原 `logs/*.log`）。
2. 根目录散件 `logs/`、`final_result/`、`interpretation/`（空目录）→ 主树
   `dev/_legacy/20261007/loose_1007/`（保持目录原名，不并入 run 目录以免改产物原始性）。
3. 台账 6 个 REGISTERED → SCOPE_DISCARD：`att004__a2_stg2_s{7,17,27,37,47}`
   （a2s 臂 stg2 已挂起）与 `inspect_mplight`（基线已被 baseline_eval 覆盖）。
   附注已写入 runs.jsonl 各行 notes，EXPERIMENTS.md 已重生成。
