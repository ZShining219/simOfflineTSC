# scene_attention — 场景引导注意力调度（本仓主线）

> 包形状约定见 GOVERNANCE §3.1。本包为**方法包**，升格自 `dev/takeover_probe`
> （2026-10-13，git rename 可追溯）。`dev/takeover_probe/` 原地留指路 stub。

## 问什么

自然语言播报（事件/意图）能否作为**条件信号**提升信号控制——单决策层
场景条件控制器，而非双层路由（G3 已否决）。当前设计栈：
`SceneJointPolicy` = dqn_s1 底座 + sga 注意力模块 + 有界残差；
训练链路 = adapter 教师蒸馏（sig9_joint_bc）→ 原装 reward RL 微调
（sig10_joint_rl）→ 意图编译层（sig15）。

## 入口面表

| 面 | 命令 | 说明 |
|---|---|---|
| 门面 | `python -m scene_attention` | 打印本表 |
| 训练 | `python -m scene_attention.train --args` | BC→RL 两阶段训练器 |
| 评测 | `python -m scene_attention.eval --args` | 事件场景冻结评测 |
| 探针 | `python -m scene_attention.probe.takeover / .attn_text / .dispatch_eval` | 接管 fork 探针 / 文本门探针 / 调度门评测 |
| 出图 | `scene_attention/figs/*.py` | 产物件（非入口），直接 python 跑 |
| 归档 | `scene_attention/_legacy/router/` | G3 router 否决代际快照（附死因 README） |

## 文件归属

- `train.py`/`eval.py`：训练/评测面；
- `env.py`：事件 episode 运行时（`run_signal_episode`）；
- `signal_model.py`：`SceneJointPolicy`（canonical）/`SceneResidualPolicy`（其父类，底座冻结残差）/`SceneSignalAgent`（sga/attn ckpt 装载面+蒸馏编码器供体）；
- `dispatch_model.py`：`SceneAttention`/`CompositeSceneEncoder`（sga，canonical 编码器）/`DispatchRouter`（已否决研究架构，仅留作 `hitl.live.controllers` 调度工具与归档 ckpt 装载）/`collate`；
- 已移除（git 历史可查）：`selfattn`/`concat` 对照编码器与 `selfattn_st`/`concat_st`/`mlp_st` state-only 消融臂——仅服务于已冻结的消融证据；
- `dispatch_specialist.py`：`KeyDispatchAgent`/`wrap_controller`（意图分发特化）；
- `dispatch_state.py`：`FeatureBuilder`/`fb_kwargs_from_meta`（状态构造，含 `_coord_topology`）；
- `scene_mp.py`：G4 Guided 代际变体（`SceneGuidedMP/DQN`、`bound_dqn_agent`）；
- `nl_scene.py`：NL 播报→结构化字段管线；
- `events.py`：sig 事件常量**唯一源**（从 router 时代抽出，见 `_legacy/router/README`）。

## 机制谱系与产物映射

代际链（输出均落 `data/output_data/dev_takeover_probe/`，~168 目录）：

| 代际 | 机制 | 产物前缀 | 件数/时段 | 状态 |
|---|---|---|---|---|
| G3 Router 双层 | `DispatchRouter` SMDP 选控制器 | `router_*`/`eval_obj_*`/`eval_p*`/`probe_repr*`/`channel_ablate`/`scan_pairs`/`stage1_physics` | ~45 / 10-08~09 | **否决**：上层退化为控制器选择；代码封存 `_legacy/router/` |
| 编码器消融矩阵 | sga vs selfattn vs concat vs *_st | `sig_*`/`sig2-5_*`/`eval_sig1/2/3/5_*` | ~20 / 10-09~10 | 证据冻结；消融臂代码已移除（git 可查） |
| 信号 agent 代际 | sig6 from-scratch→sig7 冻结残差→sig8 错 reward joint | `sig6*`/`sig7_res_sga`/`sig8_joint_sga` | ~6 / 10-10 | 取代：见 `signal_model.py` 头注 |
| **sig9-16 Joint+意图层** | `SceneJointPolicy` + intent_vec + corridor | `sig9_joint_bc`/`sig10_joint_rl`/`sig11*`/`sig12_joint_intent`/`sig13,14_corridor`/`sig15_*`/`sig16_noaux` | **~38 / 10-10~12** | **canonical**：台账 `takeover_probe_sig15` 24 行全 DONE |
| 意图层外围 | KeyDispatch 派发 + NL 前端 | `dispatch_gate_v1`/`spec_gate_congest_v1`/`sig15_dispatch_eval`/`nl_scene_v1` | ~5 / 10-11 | 维护中（`sig15_dispatch_eval` 属主线） |
| 参照系/图件 | fixedtime/mp/dqn_s1 参照 + fork 探针 + 出图 | `s2_ref_*`/`s5_ref_*`/`s2_fork*`/`figs`/`plots`/`*_fig` | ~23 / 10-08~12 | 对照证据 |

主线读法：`sig9_joint_bc`（adapter 教师蒸馏）→ `sig10_joint_rl`（原装 reward TD）→ `sig11`（无意图对照）→ `sig12_joint_intent`（单边意图）→ `sig13/14_corridor`（走廊负结果）→ `sig15_corrload + S1/S3/S4`（corridor-load 主链，多场景泛化，14 个评测变体）→ `sig16_noaux`（无辅助对照）。产物目录名即 run_id；ledger 只覆盖 sig15 起（治理 10-07 落地，此前靠目录自述）。

## 工单与产物

- 事件常量：`scene_attention/events.py`（采样空间单源；`configs/scene_attention/` 工单覆盖为下一步）；
- 产物桶沿用 `data/output_data/dev_takeover_probe/`（证据路径冻结，不改名）；
- campaign 明细：`EXPERIMENTS/takeover_probe_sig15.md`、台账 `line=takeover_probe`。

## 边界

- 只许被 `hitl.live.controllers` 以 adapter 方式引用（DispatchRouter/FeatureBuilder 懒加载）；
- 依赖下层契约（`world/`、`hitl.live.injects` 播报 spec 解析）；
- G3 router 双层架构已否决，`_legacy/` 内件不作架构参照。
