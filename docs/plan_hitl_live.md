# 人在回路（HITL）× 半离线拓展 —— 设计与执行参考

> 状态：设计定稿（2026-10-09，Z 已确认方向与角色）。
> 定位：阶段性执行参考；实验登记走台账，本文件不作为实验事实源。
> 上游依据：`/projects/子任务1.docx`（子任务1.2 人在回路）、`/projects/hitl_traffic_spec_v0_4`（SPEC v0.4 + HITL_FULL_DESIGN）。

## 0. 一句话

把"专家席"接进 `tools/sumo_live_broadcast`：虚拟专家（MaxPressure）先坐进去按任务书语义产生干预/示范数据，数据打包成与 Plan 系档案同构的只读档案喂给半离线训练线（`sequential/`），用同 seed 对照实验证明人在回路有效；真人未来可同席位顶替，接口零改造。

## 1. 角色锁定（Z 裁决 2026-10-09）

| 角色 | 取值 | 说明 |
|---|---|---|
| 学生（被纠偏对象） | `snap_sumohz1x1`（S2，p1_formal_dqn_sumohz1x1_config2 ep400） | 训练对象=控制器级 Q 网络 |
| 专家 | `maxpressure`（LegacyAgentController） | hitl 仓认证基准 mp_n_m0≈45.07 |
| dsp 调度线 | **不掺和** | dsp 是调度层（选谁开）；HITL 是训练层（控制器怎么变好）。HITL 产物未来可经 `DispatchRouter.add_controller` 注册为 dsp 新臂 |
| 场景 | S2 = sumohz1x1（单路口）起步 | 与 spec 单路口口径、快照输入维度一致 |

## 2. 任务书对照结论

- §1.2 三类 HITL 方法（演示模仿/奖励塑造/交互修正）→ 本线 A/B/C 三通道。
- "（状态，专家动作）序列"→ recorder 逐决策区间录制。
- "专家方案库（场景特征+预期目标）"→ 人工/专家档案 + inject `objective` 声明字段。
- "关键节点监听+被动触发/主动探询/周期审查"→ gate 三 mode。
- "专家数据入混合回放池、多池存储、**更高权重**、按与示范差距引导采样"→ 半离线新增独立"专家池"第三条腿，独立配比参数。
- spec 口径分歧声明：spec §0"零真人"指虚拟专家替身；本设计专家接口同构，真人可换坐。
- 暂后置：多专家冲突融合、周期审查视图、90%/10% 验收（任务书自定后置项）。

## 3. 链路

```
转播会话（S2；演示 1×，生产可加速/无头复用同组件）
  学生 snap_sumohz1x1 控盘
    gate 命中（排队/服务间隔触发，预算 ⌊0.1×360⌋=36/ep，冷却 3 决策）
    → 专家席答题（虚拟：当场答；真人：请求横幅等点击/超时沉默仍计费）
    → arbiter：学生建议 → 专家建议 → 执行动作
    → recorder 记账
  A 通道：MP 控盘 episode → （状态, 专家动作） 序列
  C 通道：被纠偏区间的执行转移（只学 accepted_action）
        ↓
  打包 = 与 dataset/offline_trajectory_dataset 同 schema 的只读档案 + manifest + hash
        ↓
  sequential 新臂：原配方 + 专家池第三条腿（独立配比，不进 offline_ratio 语义）
        ↓
  新快照（如 dqn_s2_hitl）→ SnapshotController 契约 → 转播菜单同场景对比演示
```

零回归规则：**全部 HITL 能力 opt-in；不配参数 = 不存在**。存量臂/池/台账/转播服务语义不动。

## 4. 模块设计

新增 `tools/sumo_live_broadcast/` 内（工具层能力，非实验逻辑）：

| 模块 | 职责 |
|---|---|
| `recorder.py` | 会话级录制：`transitions.jsonl` + `proposals.jsonl` + `interventions.jsonl` + `manifest.json`，落在 `runtime_dir/hitl/<session_id>/` |
| `gate.py`（二期） | 触发器 periodic/on_demand + 预算/冷却/eligible；虚拟专家当场答，真人走 awaiting 横幅 |
| 专家席适配 | 复用 controllers.py 统一面（bind/decide）；virtual=控制器实例，human=UI 应答；同接口 |

实验侧（打包/对照/出图）归 `dev/hitl_live/`，登记 `dev/INDEX.md`。

### recorder 契约（已实现侧的核心）

- 观测/奖励复用学生骨架同款生成器（`LaneVehicleGenerator(lane_count, in_only=True, average=None)` 得 state；`IntersectionPhaseGenerator(cur_phase)` 得 phase；`lane_waiting_count` 负值 ×12 得每秒 reward），与 `sequential` 训练/推理路径逐位一致
- 每受控 rank 一条转移：决策边界 T 采 `(state, phase)`，区间 [T, T+Δt) 内每秒累 reward，下一决策边界关闭为 `(next_state, next_phase)`——严格保序保证 episode 内 state 链不断（数据集校验硬性要求）
- 审计字段：`student_id/student_action/expert_id/expert_action/accepted_action/queried/intervened/manual_mid_interval/source/answer_latency_s`
- episode 边界 = reset/会话结束；终止一律 `truncated=True`（转播无自然终止）
- JSONL 落盘；NPZ 分片 + manifest 由 dev 侧打包器后做

## 5. 量化验证体系（四层，便宜→贵）

- **第 0 层体检**（不花训练钱）：MP 专家在转播里控盘跑 episode，三指标 vs 学生快照 → 专家是否真强；人工/专家转移覆盖格互补率；纠偏后排队回落对比
- **第 1 层核心对照**：同快照同 seed 同预算——基线 / +示范 / +纠偏 / +两者 / **等量机器数据对照臂**（排除"只是数据多"）；学习曲线 + AULC + 终点配对 + 95% bootstrap 区间，三指标（travel_time/throughput/unfinished）强制同报
- **第 2 层事件窗专项**：同 inject 调度下各臂排队轨迹（事件前/中/后分段），峰值/恢复时间——主张"HITL 价值集中在突发段"
- **第 3 层过程账本**：请求数/预算、接管率、同动作率、超时率；成本-效果散点；assisted vs autonomous 分开报（spec 硬性）
- **第 4 层转播演示**：基线快照 vs HITL 快照同场景切换演示 + 干预时间线；`sumo_html_comparison` 双跑回放页

执行顺序：recorder → 第 0 层体检 → 过线后跑第 1 层对照；2/3/4 层为第 1 层数据的切片展示，不另花训练钱。

## 6. 未决项与边界

- gate 命中是否冻结仿真等答：虚拟专家不冻结；真人模式冻结（语义贴 spec"一次请求一个区间"），超时沉默=held 继续仍计 query
- B 偏好标注（RM/PBRS）二期；可先走"示范相似度稠密奖励"轻量替代
- 干预比例随学习阶段动态调整（任务书"初期高频→中期预算→成熟监督"）——gate 阶段感知参数化，对齐 hitl 仓 gap 语义
- 多路口（S5）：gate 按受控 rank 逐路口请求或合并请求，实现时定
- 干预预算：B=⌊0.1×N_nominal⌋；S2 N=3600/10=360 → 36
