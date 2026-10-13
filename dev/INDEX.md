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
  **内部意图层（sig12 起定型，sig15 走廊版）**：外部接口仍是
  现象播报（kind/edge/severity，无指令）；`dispatch_state.
  intent_vec/intent_watch` 把它编译为两套按边向量——θ_reward
  责任集（事件边+feeder+discharge，`--intent-coord-w` 分摊权重，
  `--scene-reward-w` 以 Σ(q·θ̂)/Σθ̂ 加权排队罚进 reward）与
  θ_watch 观察集（+pred-of-feeder 截流来路边，aux 加权 CE 教
  L2 绑定 w 铺开）。拓扑由 `FeatureBuilder._coord_topology`
  从 SUMO lane.getLinks 实时解析（1×1 走廊网 pred/succ +
  2-hop 来路）。训练链路：① adapter 教师（scene_mp.py SceneGuidedDQN：冻结
  底座+注意力启发式偏置）跑事件局 → 赢家过滤 → ② 学生加权
  CE 蒸馏（`--distill-*`，sig9_joint_bc，-2.7% 胜 dqn_s1 与
  MP，且超教师）→ ③ 原装 reward TD 微调（`--orig-reward` =
  -12×入口车道排队均值，与 dqn_s1 收敛目标同义；
  `--init-from` 热启动学生权重；sig10_joint_rl）。
  ④ 意图效用纯在线续训（sig11 无意图对照 → sig12 单边意图，
  文本转承重；sig13/14 走廊意图负结果：空走廊无回报 →
  sig15 corridor-load 场景：主事件+静默 feeder surge+throttle
  discharge，`--p-corridor`，文本臂在走廊指标上分化）。
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
- `dev/takeover_probe/attn_text_probe.py`：E5 响应契约探针
  （探索件）。同状态换文本（真/谎/无）逐 tick 量化：逐边绑定
  质量、w-θ_watch 余弦对齐（`align_true/lie/none`、谎报对谎
  走廊 `align_lie_on_liewatch`）、w-L1 文本敏感度、残差幅度、
  动作翻转率；`--corridor` 复现训练场景。产物
  `output_data/dev_takeover_probe/attn_probe_c2/`：sga15 谎报
  对齐谎报走廊 0.42>真 0.35（注意力跟报告所指）；sga16(aux=0)
  对齐全平、w-L1 0.03——θ_watch 监督是可审计契约非性能件。
- `dev/takeover_probe/joint_attn_fig.py`：反事实注意力热力图
  （探索件）。单状态三文本臂六面板热力，corridor 模式按
  θ_watch 分层标记（事件边/协调腿/截流来路）。产物
  `attn_probe_fig_c/fig_joint_attn.png`。
- `dev/takeover_probe/evidence_fig.py`：量化证据四联图（探索件）。
  汇总已收割 results.json/探针 JSON 出 E1 走廊治理 / E5 契约
  / E7 跨路网收益% / E2 机制选型四联柱状，不再跑仿真。产物
  `evidence_fig/evidence.png`。
- `dev/takeover_probe/report_effect_fig.py`：单事件一句话证据图
  （探索件）。同 episode 三文本臂：上=注意力压在当前播报所指
  走廊的比例（逐 tick 解析播报边，谎报臂画谎走廊），下=事件边
  排队曲线。注意 rec['text'] 记的是真实播报句，override 只改
  模型输入；lane_queue 以车道名为键（勿用下标直接查）。产物
  `evidence_fig/report_effect.png`。
- `dev/takeover_probe/corridor_delta_fig.py`：一句话柱状图（探索件）。
  从 results.json 直接出"报告所指走廊排队 vs 无文本臂"的下降
  比例柱状（走廊=事件边+协调腿合计），fixedtime 超框加截断
  标记。产物 `evidence_fig/corridor_delta.png`。
- `dev/takeover_probe/sig15_table.py`：SIGMA 式汇总表（探索件）。
  跨 eval 目录拼 mean±std 主表 + paired Wilcoxon + 延迟/强度/
  跨场景小表；读 `results.json`['arms'] 的逐 case 记录。
- `dev/takeover_probe/sig15_panels.py`：三联证据图（探索件）。
  A=走廊收益配对%柱(带SEM+Wilcoxon星标) B=延迟鲁棒曲线
  C=强度剂量响应；数据源 sig15_eval_main8/delay*/sev。
  产物 `evidence_fig/sig15_panels.png`。
- `dev/takeover_probe/taskbook_figs.py`：任务书佐证图组（探索件）。
  一图一问题六联：F1 感知-响应契约（attn_probe_c2 反事实
  探针：真/谎/无文本对齐 + sga16 无监督对照）/ F2 感知-浓缩
  表征（probe_repr_id 四编码器三指标）/ F3 决策-主效应
  （逐case配对连线+分指标增益分布+参照口径归因）/ F4 决策-
  设计消融（意图粒度阶梯 + 非joint编码器参照臂）/ F5 决策-
  鲁棒边界（delay 全档 + distract + sev 分档的配对增益）/
  F6 决策-迁移（sig15_eval_{S1,S3,S4} 主模型零样本迁移，勿与
  sig15_S*_eval 场景内重训混淆）。决策层统一用"对盲化孪生
  逐case配对增益%"呈现，绝对值柱图压不出 ~9% 差。配对按
  (event,第k次)键对齐。Wilcoxon 复用 sig15_table scipy-free
  实现；中文标签需 NotoSansCJK 字体。
  产物 `taskbook_figs/f{1..6}_*.png`。
- `dev/takeover_probe/sga_arch_fig.py`：现行架构图（探索件）。
  外部输入仅 SUMO 观测 + 现象播报；播报穿 SGA 虚线边界分
  两条通路：① text_vec（推理：query + fuse3 语义通道）
  ② 意图编译（训练：θ_reward 塑形 / θ_watch aux 监督）。
  编码器内为递进主干（lane_proj→车道自注意力→身份绑定
  交叉注意力）+ 并行旁支（mms 残差、text_proj）→ fuse3
  浓缩 z；副产物 w 只作门控/审计。底座 dqn_s1 图内微调 +
  门控有界残差 → 相位；底条蒸馏训练链路（含 sig6/7/8
  否决理由）。产物 `taskbook_figs/f0_architecture.png`。
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
- `hitl/`：HITL 拓展实验件（设计依据 `docs/plan_hitl_live.md`）。
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
  用法：`python hitl/pipeline/package.py --session <session_dir>
  [--session <session_dir>...] --out <archive_dir> --network <name>
  [--seed N] [--strict]`（多会话合并：episode 全局重编号、global_step
  连续化、归属记 labels/source_shards；另发 `expert_pool_manifest.json`
  供 sequential 专家腿消费）。产物 `output_data/hitl_archives/`（不入库）。
- `hitl/probe/paired_probe.py`：三臂（student/hitl/expert）同 seed 配对
  运行时探针——第 0 层"专家资格体检"工具。
- `hitl/figs/make_figs.py`：probe 佐证图（排队轨迹+gate 标记/终端三指标/
  事件窗放大/成本-效果）。
- `hitl/plans/build_pilot_plan.py`：C 通道训练臂对照 plan 生成器——
  同一 ep0 checkpoint/seed 出 HITL_C0/HITL_EXP/HITL_CTRL/HITL_DEMO 四臂
  （差异仅 `expert_pool_manifest`/`expert_ratio`）；产物用
  `sequential_run.py run-child` 逐臂执行（探索性 pilot，不入台账）。
- `hitl/figs/curve_figs.py`：训练期佐证图——自主冻结 eval 学习曲线
  （vs 决策数，多 seed mean±std 带）、AULC、对 C0 同 seed 配对 delta、
  专家腿消耗量（sampling_windows 账本）；eval 快照 digest 去重的
  前导/内段别名按"同快照同估值"语义补齐（与 paired_bootstrap 同口径）。
- `hitl/plans/build_r2_plan.py`：r2 增补臂 plan 生成（探索件）。
  克隆 pilot5 已验证 child 为模板（不重新推导 initial_state/digest），
  生成 6 新臂×5seed=30 run：BPT/BPM/BPQ 逐判官消融（单判官 manifest），
  EXPB/EXPL 大纠偏池（2190 转移 18 会话）r=0.25/0.05，
  APIQB 大示范池（1462 转移）双预训。产 `pilot5/plan_r2.json`。
  注意：HITL_EXPERT 的 run_id 历史名 HITL_EXP-SD*，克隆时须按
  -SD 后缀重建而非条件串替换；run-child 的 --attempt-dir 须用
  runs/<id>/attempt_001 两层结构（决定 eval 落 pilot5/_shared_evaluation）。
- `hitl/plans/build_r3_plan.py`：r3 多场景复现 plan 生成（探索件）。
  克隆 pilot5/r2 已验证 child 为模板，把每通道代表臂
  （C0 基线/APIQB 双预训/B2 塑形/EXPB 纠偏池）×5seed 铺到三个
  杭州真实路口配置（sumohz1x1_config2/3/4 = qc-yn/kn-hz/sb-sx，
  1417/743/1671 veh，论文式"同拓扑换流量分布"多场景）。B2 为
  零样本 RM 迁移（bc-tyc 训练直接部署）。initial_state/digest 复用
  `initial_state_catalog.json` 预登记项；注意 seed0 存在
  `_r2` 重复 run 目录且 ep0 内容不同，须按 catalog sha 匹配选
  checkpoint，不能取 glob 首个。产 `pilot5/plan_r3.json`（60 children）。
- `hitl/plans/dispatch_r3.sh`：r3 run-child 批量派发器（探索件）。
  attempt_dir 用 `multiscene/<tag>/runs/<id>`——runtime 取
  dirname×2 为 launch_root，故 eval 落 `multiscene/_shared_evaluation`
  （三场景共享，run_id 含 -TAG- 区分，统计时按 tag 过滤）。
  需 env: SUMO_HOME=<colight>/lib/python3.10/site-packages/sumo +
  CUDA_VISIBLE_DEVICES=""。幂等：completed.json 存在则跳过。
- `hitl/figs/deck_figs.py`：HITL 汇报组图（探索件）。
  按"一图一事、对比锚定 C0"组织：fig0 系统集成消融
  （主场景 FULL(=示范+偏好双通道集成；C 腿在 demo_ac 路径
  未接线，经 HITL_AB 臂比特级复核确认) vs 单通道臂 vs C0，论文"完整版vs
  简化版"位）；fig3 全场景系统曲线总图（4 场景行×3 指标列，
  只画 C0 vs FULL 两条线=系统口径，5seed±SEM；fig1/旧fig3
  单通道版已弃）；fig2 冻结 RM 偏好预测准确率
  （in-domain 94.6% vs 零样本 qc 78.7%/kn 83.3%/sb 80.8%，
  2400 对/场景，规则教师标签）——对应论文表4.3 的
  跨控制器鲁棒性叙事位。pairs_<tag>/topology_<cfg>.json 为
  其配套产物（build_pairs.py + dump_lane_topology.py 同流程）。
  产物 `pilot5/figs_deck/`。
- `hitl/figs/hitl_arch_fig.py`：HITL 模块架构图（探索件）。
  三层单系统口径：在线会话层（人只在专家席：gate 触发问人→
  接管/同意→recorder 审计）→数据资产层（一份录制分层为
  demo/片段+偏好/纠偏转移三种人源数据）→训练层（A 双预训/
  B 奖励塑形/C 回放腿同作用于 HASODQNAgent）；底条列证据链
  六环，虚线箭头表"新学生回会话层"闭环。中文需 NotoSansCJK。
  产物 `pilot5/figs_deck/hitl_arch.png`。
- `hitl/figs/gate_modes_table.py`：gate 触发策略消融表
  （探索件）。五模式各 10 seed 配对采集（gate_modes/<mode>/，
  hitl 臂，同 inject/冷却；实际查询量对齐 ~4-8/ep）：queries、takeover、
  takeover_yield、事件窗内触发占比、终端三指标。
  配套改动：tools/sumo_live_broadcast/gate.py 新增
  random/margin/window 三模式（check 扩 margin/now 可选参，
  向后兼容；margin=学生 Q top1−top2 近期分位自适应触发，
  window=事件窗 oracle，窗由 paired_probe 从 inject 自动推导）；
  engine.py 两处 check 调用补传学生 margin（SnapshotController
  inference_q_values）与仿真秒。产 `gate_modes_table.json`。
- `hitl/plans/build_gate_plan.py`：gate 消融训练 plan（探索件）。
  克隆 plan_r2 的 HITL_EXPB 模板（大池 r=0.25），五臂各换一个
  gate 模式采集的纠偏池（pools/hitl_gate_<mode>，seed0-9×10 会话
  合并，~1210 转移/池）；臂 HITL_GATE_{OD,RAND,MARGIN,PERIOD,WIN}。
  产 `pilot5/plan_gate.json`（25 run，campaign=hitl_live_gate，
  统计在 `pilot5/stats_gate/`，评测已归并 pilot5/_shared_evaluation）。
- `hitl/probe/rm_env_corr.py`：RM 打分 vs 环境奖励相关性
  （探索件，对齐顾第三章 PCC 验证层）。冻结偏好 RM 对三源池
  （专家/学生/hitl）片段打分 vs 同段平均环境奖励：Pearson
  0.34~0.53（专家集最强）、Spearman 0.20~0.37——编码了交通
  语义排序但不等价重构真奖励。产 `reward_model_v1/rm_env_corr.json`。
  配套终局基线采集：`output_data/hitl_probe/baseline_{fixedtime,
  sotl}`（各 10 seed live，ft tt=175.6/sotl 103.6/mp 83.4）+
  `hitl_probe/_meta/hitl_cost_table.json`（问人成本与数据资产表）。
- `hitl/figs/multiscene_table.py`：r3 多场景汇总（探索件）。
  live 三臂（student/hitl/expert×10seed）终端指标表 + 逐场景
  paired_bootstrap（臂 vs 场景本地 C0，复用 collect/paired_deltas/
  bootstrap，run_id 按 -TAG- 拆分）。产 `multiscene/summary_r3.json`。
- `hitl/figs/hitl_score_fig.py`：三通道全臂森林图（探索件）。
  合并 pilot5 各 stats_*/paired_bootstrap.json（baseline=C0），
  4 指标 AULC delta + 95% CI + 显著性星标，按 A/B/B'/C 通道
  着色；含文档未载的 EXPC(r=.125)/EXPD(衰减)/EXPF(gateq 池)
  及 r2 增补 6 臂（BPT/BPM/BPQ/EXPB/EXPL/APIQB，共 21 臂）。
  产物 `output_data/hitl_probe/pilot5/figs_ALL/hitl_score.png`。
- `hitl/pipeline/dump_lane_topology.py`：场景车道拓扑转储（in_lanes 顺序、
  车道长度→容量、相位→服务车道映射），B 通道特征/教师分的 ctx 来源。
- `hitl/pipeline/segments.py`：B 通道片段切割 + 确定性教师分
  （h=-0.5Q-0.5U，逐车道 q_l/c_l + 相位服务史重建 u_l）+ spec §6.2
  奖励模型特征 [obs|one_hot(a)|obs'|q/c|u/T_u]。
- `hitl/pipeline/build_pairs.py`：档案→片段→偏好对数据集
  （片段均值特征 + H 标签 + 平局保留）。
- `hitl/pipeline/train_reward.py`：偏好奖励模型训练（128→128→1，BCE，
  episode 级留出，早停；产 reward_model.pt + normalization.json +
  reward_manifest.json）。
- `hitl/probe/calibrate_reward.py`：B 通道归一化统计冻结（校准集上
  h/f_hat 的 μ/σ，产 reward_calibration.json 供训练器 shaping）。
- `hitl/probe/paired_bootstrap.py`：同 seed 配对 bootstrap（10000
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
- `hitl/pipeline/build_demos.py`：示范集构建——打包档案→扁平
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
- `hitl/probe/calibrate_adjudication.py`：B′ 判官面板逐判官归一化
  校准——在 expert_pool_manifest 列表上逐转移算各判官 raw 分，回填
  spec 的 mu/sigma/calibration_count 产冻结 manifest
  （`output_data/hitl_probe/adjudication/manifest_v1_calibrated.json`）。
  已跑：hitl_ep50+student_ep50+expert_ep50 三池 731 转移校准一版。
- `hitl/plans/build_pilot_plan.py` 增 `--bprime-arms`/`--bpc-arms`
  + `--adjudication-manifest`/`--adjudication-ctx`：HITL_BP（纯在线
  统一裁定）/ HITL_BPC（裁定+专家腿，旧 B 不能表达的组合）两臂。
- `hitl/audit/action_ranking_audit.py`：动作排序误排审计
  （纯事后分析，不跑仿真）——读 `output_data/allocation_accuracy_formal_cf/`
  的 Q 预测 + 8 动作反事实推演 + oracle 标签，按 (method,scene,state) 计算
  Q 序 vs 实际回报序的 Spearman/Kendall/成对反序率/选中动作真实名次，以及
  chosen−oracle 的归一化 regret 与 Δqueue AUC/Δreal_delay/Δthroughput 原始
  交通损失；含方法内汇总与 R25/R50 对 CONT 的同状态配对 bootstrap。
  产物写 `output_data/action_ranking_audit/`。
- `hitl/audit/counterfactual_v2.py`：v2 严格反事实推演器
  （fresh world + loadState + 固定 --seed → 同后续实现；逐决策记录原 DQN
  reward、γ=0.95 折现回报与交通指标）。模式：`continuations`（indep/cont
  双接管 90s）、`multiseed`（高 gap 对 × 多未来 seed）、`boundary`（在
  30/60/90s 决策点记录 obs+两套 Q 向量，供 bootstrap 修正）、`episodes`
  （推演至 episode 终止 3600s 求全回报近似）。产物
  `output_data/action_ranking_audit/v2_rollouts/`。
- `hitl/audit/analyze_v2.py`：v2 四任务分析——训练奖励口径
  回报排序、indep-vs-cont 配对敏感性、多种子稳定性、场景/负载/相位/时段
  分层。产物 `v2_*.csv` + `v2_action_ranking_report.md`。
- `hitl/audit/analyze_bootstrap.py`：高差异状态三口径对比
  （R1 60s 截断回报 / R2 边界 Q bootstrap 修正 / R3 全 episode 回报），
  含与已存分支的确定性逐位校验。产物 `v2_bootstrap_*.csv` +
  `v2_bootstrap_eval_report.md`。
- `hitl/audit/mine_preference_pairs.py`：从 multiseed 反事实
  分支挖多种子确认的动作偏好对（a+ 在 ≥4/5 种子下 60s 折现回报更优），
  产物 `preference_pairs.json`（112 对）。
- `hitl/audit/rank_aux_train.py`：L = L_TD + λ·L_rank 辅助
  微调——TD 与 agent/dqn.py 同式（DQNNet/目标网/γ=0.95/RMSprop/MSE），
  rank 项为 logistic softplus(-(Q+−Q-))；含 before/after 场景排序指标与
  固定批 TD 漂移度量。--arm {A,B,C,D}：A=1×TD、B=2×TD（等量计算对照）、
  C=TD+rank、D=TD+反事实回报回归。试点产物 `rank_aux_pilot/` +
  `rank_aux_pilot_report.md`；四臂对照 `rank_aux_abcd/` + `rank_aux_abcd_report.md`。
  注意：import agent 前必须按 counterfactual_v2 方式设 SUMO_HOME/SUMO_BINARY
  环境变量，否则包 __init__ 内某处会静默退出。
- `hitl/audit/rank_aux_eval.py`：贪心策略整 episode SUMO 交通
  评估（throughput/avg_travel_time/unfinished + queue_auc/real_delay），
  fresh world + --seed 多种子。
- `hitl/audit/divergence_probe.py`：双 checkpoint 闭环分歧探针——
  driver 驱动、shadow 同状态影子评估，argmax 分歧即存 saveState 快照 +
  逐决策状态特征 JSONL。产物 `divergence_bc/`。
- `hitl/audit/divergence_cf.py`：分歧快照反事实核验——
  强制 a_driver/a_shadow 两分支 × {driver 自身, indep-seed0} 接管 × 多
  未来 seed，复用 counterfactual_v2.run_branch。产物 `divergence_bc/cf*/`。
- `hitl/audit/analyze_divergence.py`：分歧率画像 +
  分歧动作反事实差值统计（C 方向归一化：Δ>0 即 C 的动作为优）。

## 已裁决归档（2026-10-07 Z 裁）

1. `tools/*_matrix*.sh` 10 件矩阵驱动 → 主树 `dev/_legacy/20261007/matrix_drivers/`
   （§6.1 已禁止该模式新写；原路径留此记录供证据链对照：原 `tools/`、配套运行日志
   原 `logs/*.log`）。
2. 根目录散件 `logs/`、`final_result/`、`interpretation/`（空目录）→ 主树
   `dev/_legacy/20261007/loose_1007/`（保持目录原名，不并入 run 目录以免改产物原始性）。
3. 台账 6 个 REGISTERED → SCOPE_DISCARD：`att004__a2_stg2_s{7,17,27,37,47}`
   （a2s 臂 stg2 已挂起）与 `inspect_mplight`（基线已被 baseline_eval 覆盖）。
   附注已写入 runs.jsonl 各行 notes，EXPERIMENTS.md 已重生成。
