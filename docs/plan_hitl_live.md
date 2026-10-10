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

**B′ 通道：采样时统一裁定层（2026-10-10 落地，spec v0.5 扩展）**

设计动因：B1/B2 在 `remember()` 入池前塑形，只作用在线腿——离线
腿/专家腿保留原生 r0，两腿奖励口径混杂，B×C 因此被硬互斥。B′ 把
评分挪到消费侧：**转移只存 `r_base` + 裁定特征 `{q_ratio, gap_norm}`
（payload.aux/metadata.reward_aux），塑形值永不落盘**；`sample`
抽出后三腿（online/offline/expert）过同一判官面板出同一口径
`r_eff = r_base + λ·z` 再进 TD loss。

`sequential/adjudication.py` 判官面板：
- `terms`——带参治理词条 `raw=Σw_k·φ_k`（现词表：
  queue_per_capacity/queue_max/unserved_max/unserved_mean，词表即
  配置可扩），B1 公式即 `{queue_per_capacity:-0.5, unserved_max:-0.5}`
  特例；
- `reward_model`——冻结 PreferenceRewardNet（B2 同件）；
- `qgap`——**收敛后 DQN 快照**（`Q(s,a_taken)−max_a Q` 认可度分）；
  硬资格门槛 `episode>=min_episode(默认20) 且 gradient_updates>0`，
  ep0/未训档一律拒载（`allow_unconverged` 仅测试逃生口，旗记入
  manifest）。判官网络初始化包 `fork_rng`——不污染
  `rng_state_digest`（rebind 审计约束）。
- 合成：`z=clip(Σw_j·z_j/Σ|w_j|, ±clip)`，逐判官先 `(raw−μ_j)/σ_j`
  归一再加权和——异构判官共 z 量纲；单判官崩溃该判官 z=0 且权重
  归一剔除，面板降级不回退 r0。
- 投票聚合为后续扩展（现 detail 已暴露逐判官 z/raw 可接投票层）。

目标漂移防护（Z 关切的"动态奖励"问题的工程答案）：
- **run 内冻结**：判官面板 spec = 版本化 manifest，digest 进
  checkpoint `adjudication` 块；`load_full_state_dict` 对
  manifest_digest 强校验——换判官=新 stage，静默 resume 直接拒
  （4 项单测锁定）；
- **幅值预算**：λ=0.1、z 截 ±3 → 塑形上限 ±0.3，对 r_base≈-80/93
  量级 ~0.3%，判官再错也晃不动目标；
- **真值兜底**：eval/指标只看环境 r_base 世界，塑形是脚手架。

腿侧特征重建：`leg_features_from_view` 按 `(run_id, episode_id)`
分组 + `decision_step` 排序，逐 episode 回放 `ServiceGapTracker`
重建 u_l/q_ratio——与在线 `_adj_features` 同一机制；懒构建+按
view 缓存，resume 后确定性重放。在线 `(stage_index, local_episode)`
为 episode 键防跨 stage 污染。

冒烟验证（sumohz1x1，2ep，三判官面板 terms+RM+qgap(ep400 档)）：
- `HITL_BP-SD0` 在线腿：120/120 update 窗带 adjudication 块，
  `missing_features=0`、`judge_fallbacks=0`，三判官 z 分布可分
  （terms −0.94 / RM −1.83 / qgap −2.09）；
- `HITL_BPC-SD0`（expert_ratio=0.25，hitl_ep50 池）：**expert 腿
  全部转移吃到同一 manifest 的裁定**（missing=0），expert
  z_mean=−0.06 vs online z_mean=−1.62——判官语义正确（纠偏数据
  被评得更接近专家期望），这正是 B+C 在旧方案里无法表达的组合；
- `tests/test_adjudication.py` 23 例锁不变量：同转移两腿同 r_eff、
  缺特征→z=0、判官崩溃只降权、digest 稳定/敏感、qgap 资格门槛、
  (run_id,episode_id) 分组防串、checkpoint manifest 绑定。

plan 侧：`build_pilot_plan.py --bprime-arms/--bpc-arms` +
`--adjudication-manifest spec.json --adjudication-ctx topo.json`
生成 HITL_BP / HITL_BPC 臂（BPC 走 expert_pool_* 既有键）。
未做：判官质量离线评估（分歧矩阵/预测效度）、qgap μ/σ 正式校准、
B′/BPC 的 5-seed 配对证据、投票聚合、词条权重寻优。

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

**A 通道 5-seed 配对结果（2026-10-09，pilot 档）**：20 run（A0/Aπ/AQ/
AπQ×5 seed，同 ep0/seed/预算；pretrain 2000 步）全部完成。A 内部基线
为 A0（DemoAC 不预训），外部参照为 C0（普通 DQN）；对 C0 的 AULC 差：

| 指标 | A0 不预训 | Aπ 仅 actor | AQ 仅 critic | AπQ 双预训 |
|---|---|---|---|---|
| travel_time | +18.55 CI[+5.9,+37.5] 差 | +70.88 CI[+15.5,+144.1] 差 | **-4.12** CI[-6.0,-2.0] P=1.000 | **-23.25** CI[-27.3,-18.8] P=1.000 |
| throughput | -59.70 CI[-120.3,+0.1] | +97.87 ns | -49.45 CI[-81.7,-17.5] 差 | **+252.91** CI[+213.3,+290.7] P=0.000 |
| unfinished | +1.30 ns | **-48.48** CI[-71.0,-21.3] P=1.000 | +1.58 ns | **-86.25** CI[-90.7,-82.2] P=1.000 |
| rrt | +55.40 CI[+3.3,+109.6] 差 | -75.97 CI[-178.2,+47.8] ns | +50.83 CI[+22.3,+81.7] 差 | **-212.63** CI[-245.4,-177.6] P=1.000 |

诚实判读（全线最强臂 + 一个明确负臂，pilot 级）：
1. **AπQ 是全部 10 臂中的绝对胜者**：四指标对 C0 全部统计显著改
   善，AULC tt=81.8（C0=105.0，-22%），ep20 冻结 eval 79.4s 低于
   MP 专家单独控盘（83.7s）——"好老师教差学生"在双预训通道成立；
   曲线形态上 ep1 起即 ~80s 平台（示范预训直接给学生教师级策略），
   收益是初始策略质量而非学习速度；
2. **Aπ 单 actor 预训是显著负臂**（tt AULC 175.9，ep20 发散到
   222.4）：actor 模仿 MP 动作分布但 critic 未对齐→在线 AC 更新
   把策略拉崩；说明 DemoLight 机制里 actor/critic 预训不可拆分；
3. AQ 仅 critic 预训小幅改善 tt（-4.12 显著）但 throughput/RRT
   显著变差——critic 单独热启动引入系统性偏差；
4. A0（DemoAC 不预训）劣于普通 DQN——per-decision AC 更新在冷启
   动下比 DQN 批量更新更不稳；A 通道的价值完全来自示范预训；
5. AπQ 终点优势经 final_bootstrap 复核同样全显著：tt -7.6
   CI[-19.5,-1.0]、rrt -81.1 CI[-137.6,-46.8]、throughput +106.0
   CI[+58.4,+182.6]、unfinished -78.2 CI[-99.2,-58.4]——收益不止
   学习速度，收敛终点也更优；
6. 边界：示范池仅 244 条（spec 口径 1800）、MP 而非 SOTL、单路口
   近饱和场景；示范数据与评测同场景，泛化未证。

**三通道合并结论（pilot 级，5 seed 配对）**：
- AπQ（示范预训）> B2（奖励塑形）> C0（基线）> C 静态池各臂；
- 正向通道 = 改初始策略（AπQ）与改奖励信号（B2）；往回放池掺静
  态数据（C 全部形态）在该配置下均伤学习；
- 与"已收敛后优化空间小"的定位一致：HITL 价值体现在学习/启动
  阶段——AπQ 把起点直接抬到教师水平，B2 在学习期持续给正确方向。

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
