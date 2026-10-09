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

| 模块 | 状态 | 职责 |
|---|---|---|
| `recorder.py` | 已落地 | 会话级录制：`transitions.jsonl` + `proposals.jsonl` + `interventions.jsonl` + `manifest.json`，落在 `--record DIR/<session_id>/` |
| `gate.py` | 已落地 | 触发器 periodic/on_demand + 预算/冷却/eligible；预算默认 spec 公式 B=⌊0.1×N⌋ |
| 专家席 | 已落地（virtual+human） | `--expert NAME` 虚拟专家当场答；`--expert human` 真人席：gate 命中→`awaiting='expert_request'` 横幅冻结仿真（每路口列排队/触发原因/学生建议+相位按钮+倒计时），`--hitl-human-timeout`（默认30s）到期沉默仍计费（查询在 gate.check 时已消耗）；答案经 `resolve choices={rank:phase}` 作为专家动作执行，hitl.expert_id='human'；仲裁序 学生提案 < 专家答 < 人工 pending |
| `sequential/expert_pool.py` + HA 第三腿 | 已落地 | `ExpertPoolArchive` 读打包档案的 `expert_pool_manifest.json` → `VisibleArchiveView` → RAND 采样器；`HASODQNAgent` 加 `expert_pool_manifest/expert_ratio/expert_capacity` 三参，`loss = Σ(count_i/64)·loss_i`（online/offline/expert 三腿），账本记 `expert_transition_ids/expert_episode_ids/expert_stability`；checkpoint `ha_sodqn.expert` 块前后兼容（旧档无此块要求 expert_ratio=0）；plan 侧 child 键 `expert_pool_manifest/expert_ratio/expert_capacity`（runtime `.get()` 兼容旧 plan）。不配参数 = 逐位等价现状 |

CLI：`--record DIR` 开录制；`--expert maxpressure --hitl-gate periodic|on_demand [--hitl-period N | --hitl-queue-threshold X | --hitl-queue-growth X] [--hitl-budget B] [--hitl-cooldown C]` 开虚拟专家干预。全部 opt-in，不配 = 不存在。

实验侧（打包/对照/出图）归 `dev/hitl_live/`，登记 `dev/INDEX.md`。

训练臂 pilot 管线（已落地）：`build_pilot_plan.py` 从同一 ep0 checkpoint + 同 seed
生成 HITL_C0 / HITL_EXP(纠偏池) / HITL_CTRL(等量机器数据) / HITL_DEMO(纯示范池)
四臂 plan；逐臂 `sequential_run.py run-child`（注意 `CUDA_VISIBLE_DEVICES=""`——
checkpoint 的 rng_state_digest 按 CPU-only 语义钉死，GPU 可见时 digest 不匹配）；
`curve_figs.py` 从 `_shared_evaluation` + `replay_diagnostics` 抽自主冻结 eval
学习曲线/AULC/对基线 delta/专家腿消耗账本。`ReplayDiagnostics` 已补
`expert_samples_by_episode`/`expert_sample_count_by_scene`/`loss_expert` 字段。

**C 通道 5-seed 配对结果（2026-10-09，pilot 档）**：20 run（4 臂×5 seed，
ep0/seed 配对，20ep/臂，120 决策/ep，ratio 25%）全部完成；
`paired_bootstrap.py`（10000 重采样，spec §11.4）对 C0 的 AULC 差：

| 指标 | HITL_EXP 纠偏池 | HITL_CTRL 等量机器 | HITL_DEMO 纯示范 |
|---|---|---|---|
| travel_time | **+11.80** CI[+2.0,+26.1] P=0.004 | **+11.47** CI[+0.5,+22.8] P=0.017 | +1.25 ns |
| throughput | **-20.69** CI[-36.4,-5.0] P=0.992 | -3.13 ns | +0.38 ns |
| unfinished | +3.45 CI[-1.2,+8.1] P=0.076 | **+4.88** CI[+1.4,+8.8] P=0.000 | +1.61 ns |
| rrt | **+19.23** CI[+3.4,+34.1] P=0.007 | +0.36 ns | -0.05 ns |

诚实判读（负结果，推翻单 seed pilot 的方向）：
1. 在 20ep/近饱和场景/25% 配比/243 条小池的配置下，**掺静态池整体损害
   学习**——EXP 四指标一致劣化且统计显著；等量机器池同样劣化 → 此配
   置下"更多静态数据"本身即伤，未析出专家专属收益；
2. 机理推因：243 条纠偏转移 ~142× 过采样，危机状态占比失真（接管只
   发生在 gate 触发的拥堵边界），对早期值学习是 off-distribution 冲击；
3. 与单 seed（SD0）pilot 结论相反，正是多 seed 配对要排掉的噪声——
   SD0 上 EXP 终点 80.4<C0 81.3 的表象不具代表性；
4. 下一步建议（未做）：扩池（多会话数千条）+ 降配比（5-10%）+ 延长
   训练；或改 C1 训练期在线询专家语义（池随在线分布更新）。
注意：评测按 checkpoint digest 去重，ep1/ep2 权重仍等于 ep0（更新未
启动）时被去重归属首个请求者；`paired_bootstrap.py` 已按"同快照同评
估值"语义补齐这些前导缺口（非缺失数据，是确定性别名）。

B 通道管线（已落地 pilot 档，spec §6.2/§6.3/§9.1）：
`segments.py` 片段切割+确定性教师分（h=-0.5Q-0.5U，q_l/c_l 逐车道 +
相位服务史重建 u_l）→ `build_pairs.py` 偏好对（平局保留，episode 内
两两配对）→ `train_reward.py` 128→128→1 奖励网络（温度 0.25 BCE，
episode 级留出+早停）→ `calibrate_reward.py` 冻结 μ/σ → 训练器
`reward_shaping.py`：r=r0+λ·z（B1 规则分/B2 模型分；仅作用在线腿，
aux 存 r_base/h/z；崩溃回退 r0）。child `trainer_overrides` 携带
reward_mode 等键（runtime 并入 trainer_config）；B×C 互斥校验在
`_validate_ha_config`。B0≡C0 同配置复用 C0 run 结果，不重复计。

**B 通道 5-seed 配对结果（2026-10-09，pilot 档）**：10 run（B1/B2×5 seed，
与 C 同 ep0/seed/预算；λ=0.1）全部完成；对 C0 的 AULC 差：

| 指标 | HITL_B1 规则教师分 | HITL_B2 冻结奖励模型 |
|---|---|---|
| travel_time | -1.56 CI[-3.3,+0.2] P=0.947 | +1.90 CI[-2.7,+7.9] ns |
| throughput | +16.90 CI[-6.9,+42.7] ns | **+31.11** CI[+8.5,+53.7] P(<0)=0.004 |
| unfinished | +1.30 ns | +0.04 ns |
| rrt | -16.59 CI[-39.5,+3.0] P=0.944 | **-27.32** CI[-48.1,-6.6] P=0.996 |

诚实判读（首个正向信号，仍 pilot 级）：
1. **B2 是全线首个统计显著的正向臂**——spec 主口径 RRT 显著改善
   （CI 全负），throughput 显著更高；travel_time/unfinished 中性；
   与 C 通道"静态池伤学习"方向相反：塑形改的是奖励信号而非数据
   分布，不引入 off-distribution 冲击；
2. B1 方向一致但更弱（RRT/tt 的 P≈0.94，CI 触零）——确定性教师分
   与学到的 f̂ 同号但分辨力更低；奖励网络的平滑/泛化带来了增量；
3. 塑形审计（replay_diagnostics `reward_aux`）：B1 ep10
   h_mean=-0.98/z∈[-1.5,+2.5]、B2 f̂_mean=-2.49/z 截 ±3；fallback=0，
   λ·z 相对 r_base≈-93 的量级约 0.3%——收益来自信号形状而非幅值；
4. 边界：20ep 近饱和场景 + 单路口 + 小偏好集（720 对）；B2 终点
   指标 CI 仍跨零（final RRT d=-23.5 CI[-77,+11]）——正向结论目
   前只在 AULC（学习速度）层面显著，终点优势未证。

**评估器修正（2026-10-09）**：`world.close()` 原插在 metric 读取之前
导致 `FatalTraCIError: network not yet constructed`（eval 全灭）；
已改为全部指标先取（live engine）→ close 刷 tripinfo → RRT 解析。
RRT 口径：(1/N_due)Σ[min(arrival,T)−planned_depart]，未到达计 T。

A 通道管线（pilot 档，spec §8）：`sequential/demo_ac.py` 的
`DemoACAgent`——actor/critic 共用 DQNNet(d→20→20→P) 躯干，同一 ep0
checkpoint 逐位初始化；`demo_pretrain` 块控制 A0/Aπ/AQ/AπQ 开关；
示范集 `build_demos.py` 从打包档案产扁平 npz（episode 序保持）。
偏差声明：示范来源为 MP 控盘池（expert_ep50，244 转移）而非 spec 的
5×SOTL×360=1800；SOTL 专家本体未移植；合法动作 mask 恒等（S2 全相
位合法）。

### recorder 契约（已落地）

- 观测/奖励复用学生骨架同款生成器（`LaneVehicleGenerator(lane_count, in_only=True, average=None)` 得 state；`IntersectionPhaseGenerator(cur_phase)` 得 phase；`lane_waiting_count` 负值 ×12 得每秒 reward），与 `sequential` 训练/推理路径逐位一致
- 每受控 rank 一条转移：决策边界 T 采 `(state, phase)`，区间 [T, T+Δt) 内每秒累 reward，下一决策边界关闭为 `(next_state, next_phase)`——严格保序保证 episode 内 state 链不断（数据集校验硬性要求）
- `hitl` 审计块：`student_id/student_action/queried/query_reason/expert_id/expert_action/accepted_action/intervened`；区间级 `mid_interval_override`、`reward_steps/reward_error`；区间关闭时采 `metrics`（queue/approximate_delay/real_delay/throughput/waiting_time，对齐 Plan 1 分片辅助字段）
- episode 边界 = reset/会话结束；悬空区间以 `truncated=True` 封存（转播无自然终止）
- JSONL 落盘；NPZ 分片 + manifest 由 dev 侧打包器后做

### packager 契约（已落地，`dev/hitl_live/package.py`）

- 输入：录制会话目录（`manifest.json` + `transitions.jsonl` + 可选 proposals/interventions）；`source_config` 从会话 manifest 取，亦可 `--config` 覆盖
- 输出 `<out>/`：`trajectory/`（Plan 1 同构分片，复用 `EpisodeTrajectoryWriter` 保证逐位同格式）+ `source_shards.jsonl` + `datasets/<network>/<stage>/manifest.json` + `hitl/` 审计副产物（session_manifest/proposals/interventions 副本 + `labels.jsonl` 逐转移归属表 + index）+ `manifest.json` 根清单 + `packaging_report.json`
- 直接可消费：`OfflineTrajectoryDataset(<stage>/manifest.json, seed)` 与 `validate_offline_dataset` 内嵌自检；`observations = state + one_hot(phase)`，obs_dim 与学生网络输入一致（S2=16）
- 语义哈希五元组复用 `dataset._scene_semantics` 计算（`config/simulator_resolved.cfg` 落绝对路径 dir）；`green_action_count` 与录制 action_dim 不符即报错
- 变长 episode：不伪装等长，`expected_decisions_per_episode` 记最大单集数；逐集校验由打包器自带校验器完成（`trajectory/validation.json`，`validator: hitl_packager`）
- 归属标签：转移 `behavior_mode` ∈ {hitl_autonomous, hitl_expert_agreed, hitl_expert_took_over, hitl_operator}；逐转移明细（queried/双方动作/执行动作/reward_steps）在 `hitl/labels.jsonl`，键对齐 `transition_unique_key=(run_id, shard_sha256, episode_id, decision_step)` 供混合池加权
- 隔离语义：非法/断链/缺字段 episode 打包前剔除并记 `packaging_report.json`（state/phase 链断、action/phase 越界——含黄灯程序索引、reward 采样失败、rank 集漂移、未正常 truncated 收尾）；`--strict` 遇隔离即中止。labels.jsonl 保留全部被剔除行及原因
- 不伪造正式 run 元数据：档案 `archive_kind='hitl_live_recording'`，`run_id=session_id`；不进 `prepare_plan2_datasets` 的正式矩阵路径

冒烟验证（真实 SUMO，S2 fixedtime 学生 + MP 专家 periodic(3)）：76 仿真秒 → 8 转移 → 打包 valid，obs_dim=16，ds6 MP 接管正确落 `hitl_expert_took_over`；`dataset_load/statistics` 正常。

## 5. 量化验证体系（四层，便宜→贵）

- **第 0 层体检**（不花训练钱）：MP 专家在转播里控盘跑 episode，三指标 vs 学生快照 → 专家是否真强；人工/专家转移覆盖格互补率；纠偏后排队回落对比
- **第 1 层核心对照**：同快照同 seed 同预算——基线 / +示范 / +纠偏 / +两者 / **等量机器数据对照臂**（排除"只是数据多"）；学习曲线 + AULC + 终点配对 + 95% bootstrap 区间，三指标（travel_time/throughput/unfinished）强制同报
- **第 2 层事件窗专项**：同 inject 调度下各臂排队轨迹（事件前/中/后分段），峰值/恢复时间——主张"HITL 价值集中在突发段"
- **第 3 层过程账本**：请求数/预算、接管率、同动作率、超时率；成本-效果散点；assisted vs autonomous 分开报（spec 硬性）
- **第 4 层转播演示**：基线快照 vs HITL 快照同场景切换演示 + 干预时间线；`sumo_html_comparison` 双跑回放页

执行顺序：recorder → 第 0 层体检 → 过线后跑第 1 层对照；2/3/4 层为第 1 层数据的切片展示，不另花训练钱。

## 6. 未决项与边界

- gate 命中是否冻结仿真等答：虚拟专家不冻结；真人模式冻结（语义贴 spec"一次请求一个区间"），超时沉默=held 继续仍计 query
- B 偏好标注（RM/PBRS）pilot 档已落地（见 §4）；B2-U 在线更新奖励
  版本切换、多专家冲突标注仍为未做项
- 干预比例随学习阶段动态调整（任务书"初期高频→中期预算→成熟监督"）——gate 阶段感知参数化，对齐 hitl 仓 gap 语义
- 多路口（S5）：gate 按受控 rank 逐路口请求或合并请求，实现时定
- 干预预算：B=⌊0.1×N_nominal⌋ 公式已按 spec 落地（N=T/Δt；
  T=1200/Δt=10 → B=12）；probe 端由 horizon 推导，CLI 开放会话保
  留显式 `--hitl-budget` 默认并在 manifest 记解析值
