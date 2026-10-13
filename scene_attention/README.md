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

## 工单与产物

- 事件常量：`scene_attention/events.py`（采样空间单源；`configs/scene_attention/` 工单覆盖为下一步）；
- 产物桶沿用 `output_data/dev_takeover_probe/`（证据路径冻结，不改名）；
- campaign 明细：`EXPERIMENTS/takeover_probe_sig15.md`、台账 `line=takeover_probe`。

## 边界

- 只许被 `hitl.live.controllers` 以 adapter 方式引用（DispatchRouter/FeatureBuilder 懒加载）；
- 依赖下层契约（`world/`、`hitl.live.injects` 播报 spec 解析）；
- G3 router 双层架构已否决，`_legacy/` 内件不作架构参照。
