# xiasha1_offline：异构历史经验的纯离线学习刻画实验（E0–E5）

> 状态：**已作废**（2026-10-23 用户裁定 xiasha1 数据全部退出本轮离线训练；
> `xiasha1_offline_v1/v2` 的 252 行均为 `SCOPE_DISCARD`，v2 队列已标 dropped；
> 在线源数据留作历史证据，当前基线转由 `hz1x1_offline_v1` 承载）
> 发起日期：2026-10-22 ｜ 负责人/Agent：Z / Devin
> 最近更新：2026-10-23

## 关联

- 当前有效 campaign：无；离线基线转为 `hz1x1_offline_v1`
- 历史 campaign：`xiasha1_offline_v1`（126 行 SCOPE_DISCARD，产物
  已清理，见 §4.4）
- 历史队列：`artifacts/xiasha1_offline_v2/run_queue.json`（已标 dropped，
  不得派发；artifacts 为 gitignore 仓外资产）
- 数据集：`data/output_data/offline_datasets/xiasha1/v2_hetero/`
  （只读索引根，27 源 run / 5400 shards / 1.944M transitions；
  21 个 spec manifest 于 `datasets/<spec>/manifest.json`，
  不复制源分片；`source_root_id=xiasha1_online_v3_root`）
- 数据源：`xiasha1_online_v3` 的 DONE run 轨迹
  `trajectory/episodes/episode_*.npz`（schema A，72k transitions/run；
  v1 曾用 `xiasha1_online_v2` 同构轨迹，因截断几何作废——`xi2on_*`
  源 run 保留作历史证据，不再用于训练）
- 评估协议：与 v3 训练期评估同口径（贪心 episode、SUMO
  `fixed_default`、legacy `pseudo_step`）
- 产物目录：`data/output_data/offline_tsc/sumo_{batch_dqn,cql_dqn}/xiasha1_asym_ns/xi2off_*`

## 1. 改动意图

在线 TSC 训练持续产生历史交互经验，来自不同控制策略、不同需求场景
和变化的探索行为，天然异构且质量不均。本实验是"未经质量筛选的异构
历史经验离线利用"研究的**阶段①刻画实验**：构建异构性递增的固定经验
池，系统考察现有离线算法（`batch_dqn` / `cql_dqn`）的学习表现与失效
边界，为阶段②"不依赖人工质量分层的数据价值自主利用"提供 motivation
证据与对照基线。

设计原则（用户确认）：

- **固定池、从头均匀采样**，池内不做课程/阶段投喂；异构性只由池构成
  引入；
- **全窗口入池**：每个源 run 200 ep 全部收录，不按质量/阶段切片；
  `epsilon`/`episode_id`/`run_id` 等质量元数据仅做事后归因，不进
  池组成规则；
- 阶段投喂（E6 族）与自主价值加权（阶段②）均不在本轮范围。

与既有线的边界：本线是**纯离线**（训练零仿真交互，仅评估节点用
SUMO），区别于 ha_sodqn/cl_hz1x1 半离线线（在线交互+历史归档混合）
与 Plan 2 线（sumohz1x1 四场景正式协议）。本 campaign 复用 Plan 2 的
数据集格式与离线 trainer，扩展其构建器以支持 xiasha 数据源与跨控制器
混合。

## 2. 改动模块

- `data/dataset/offline_trajectory_dataset.py`：构建器参数化扩展
  （向后兼容，Plan 2 默认行为逐位不变）：
  - `expected_networks`/`expected_behavior_seeds`/`expected_episodes`/
    `stages` 透出至调用方；
  - `_validate_source_run` 增加 `require_plan1_formal` 与
    `allowed_agents` 开关（xiasha 源 run 前缀 `xi2on_*`、agent ∈
    {dqn,sac,frap,mplight}；d3qn 等 DQN 变体的 manifest agent 为
    `dqn`，臂身份由 run-list `arm` 列承载）；
  - run-list 支持可选 `arm` 列，唯一键 (arm, network,
    behavior_training_seed)；manifest 增 `source_arms`/`source_cells`
    审计字段；索引条目带 `arm`；
  - 跨臂混合时 `compare_runs` 可放宽为语义等价检查（语义哈希、
    action_dim、决策步数、state 宽度一致），源 run 配置差异允许
    （`relaxed_compatibility`）；
  - `dataset_specs` 显式 spec 模式替代 Plan 2 stage 矩阵；新增
    `dataset_kind: multi_scene`（多源场景、含评估场景）与
    `source_cells`（per-cell (network,arm) 精确选择，供 E5 臂随
    场景变化的稀疏混合）；采样策略多场景强制
    `scene_balanced_batch`；
  - 跨源 `state_width` 一致性门禁（`seen_state_widths`）。
- `sequential/offline.py`：`prepare-plan2` 透出
  `--expected-networks/--expected-behavior-seeds/--allowed-agents/
  --no-plan1-formal/--relaxed-compatibility/--dataset-specs`；
  `train` 新增 `--experiment-config`（YAML overlay 经
  `merge_dicts` 并入 config，归档 `experiment_overlay.yml`，
  机制同 `run.py`）。
- `configs/offline_tsc/xiasha1_v1.yml`（新增 overlay）：
  `trainer.evaluation_updates` = 每 1800 updates 一点共 81 节点
  {0,1800,…,144000}；`base.yml` 不动。
- `dev/xiasha1_offline/{build_datasets,build_queue}.py` 与
  `dev/ledger/register_xiasha1_offline_v1.py`（登 `dev/INDEX.md`）。
- 台账登记件与本文档。

## 3. 实验设计

### 3.1 记号规范

- 场景字母（第一字母）：A=`asym_ns`，B=`dense_extreme`，
  C=`fluct_burst`，D=`asym_nonly`（沿用 cross_eval_v1 四包）；
- 控制器字母（第二字母）：a=`d3qn`，b=`sac`，c=`frap`，d=`mplight`；
- **cell = 单个 v2 run**（200 ep × 360 决策 = 72,000 transitions）；
- 池 = cell 集合；每个混合池按"所有 cell 同取 seed s"构造 3 个平行
  复本（s ∈ {0,1,2}）；
- `presslight` 不可混（18 维观测含出口道 + `-pressure` 奖励，与 DQN
  系 16 维 `-queue` 语义不同），全矩阵排除。

### 3.2 实验矩阵

每实验单元 = 3 池复本 × {batch_dqn, cql_dqn} × 3 离线 seeds = 18 run。

| 实验 | 池构成 | transitions | eff. epochs¹ | 评估场景 | 问题 |
|---|---|---:|---:|---|---|
| E0 | Aa | 72k | 128 | A | 单 run 全探索数据能否被离线学习（对齐基线） |
| E1 | Aa(s)+Aa(s′) | 144k | 64 | A | 同臂异 seed 轨迹的增益（复本对 {0,1}/{1,2}/{0,2}） |
| E2 | Aa+Ab | 144k | 64 | A | 换控制器是稀释还是互补 |
| E3 | Aa+Ab+Ac+Ad | 288k | 32 | A | 多控制器全混合 |
| E4 | Aa+Ba+Ca | 216k | 42.7 | A（池内） | 多场景同控制器、含目标场景 |
| E4-LOO | Ba+Ca+Da | 216k | 42.7 | A（池外） | 跨需求泛化，对标 cross_eval_v1 |
| E5 | Aa+Ab+Bc+Dd | 288k | 32 | A | 双轴稀疏、覆盖不均的"真实积累"形态 |

¹ effective epochs = total_updates × batch_size / 池 transitions，
跨池比较的协变量，不拉平（本身即实验变量）。

合计 **7 单元 × 18 = 126 正式 run**。

### 3.3 统一协议

- 训练：144,000 updates、batch 64、RMSprop(lr 0.001, α0.9)、
  γ 0.95、target_update_interval 10、grad_clip 5、全 Q 向量 MSE、
  plain-DQN target（CQL 仅加 α=1.0 正则项）；
- 离线 seeds {1000,1001,1002}（派生 model=s、sampler=s+10000、
  evaluation=s+20000）；
- 评估：**81 节点**（每 1800 updates 一点，含 0 与 144000），每节点
  1 个贪心 episode、SUMO `fixed_default`、evaluation_seed 隔离、
  300 s 超时 + 1 重试；`evaluation/summary.json` 按 min-TT 选 best；
- 信号语义 legacy `pseudo_step`；训练期零环境交互；
- 采样策略：单场景池 `uniform_transition`，多场景池
  `scene_balanced_batch`；
- 派发：`tools/run_experiment_queue.py` + run_queue.json，CPU、
  `--parallel 8–12`。

### 3.4 指标与判定

- 过程：81 点 EVALUATION 曲线（TT/throughput/unfinished/queue/
  waiting_time/action_distribution）+ TRAIN 损失曲线（每 100 updates）；
- 收敛：滑动均值连续 N 点落 final-段 ±5% 内（沿用 v2 conv5 口径），
  报首个稳定 update；
- 对齐判据（E0/E1 主问）：离线 best/final 三指标落在同场景 v2
  online final-20 均值的 ±5% 带内 → 判定"可对齐"；同时报行为数据
  自身质量（npz 内 queue/delay/throughput）作参照带；
- 混合效应（E2–E5）：相对 E1/E0 同场景基线的 best/final Δ 三指标，
  分臂/分场景归因用 transition provenance 事后切片；
- E4-LOO 对照 cross_eval_v1 在线跨包迁移（DQN 家族 TT 退化 ~4–5%）；
- 全部单元强制同报 travel_time / throughput / unfinished_vehicles。

## 4. 实验结果

### 4.1 工程就绪与冒烟（2026-10-22）

- 数据集构建：`v1_hetero` 一次构建通过——27 源 run 全量
  validation.json/config 归档/分片 sha256 校验，5400 shards、
  1,944,000 transitions 入索引；21 个 spec manifest 计数正确
  （e0=72k、e1/e2=144k、e4/e4loo=216k、e3/e5=288k）。
- 冒烟 run：`xi1offsmoke_e0_p0_batch_dqn_o1000`
  （`--total-updates 3600`，百分比 6 评估节点）全链路通过：
  数据集加载 → 16 维 DQNNet → 更新 → 评估节点 → checkpoint →
  `run_status=已完成`（exit_code 0）；`metrics/offline_records.jsonl`
  37 TRAIN + 5 EVALUATION + 1 FINAL_EVALUATION 记录齐全，
  `evaluation/summary.json` 按 min-TT 选出 best=update_003600；
  `config/experiment_overlay.yml` 已归档；evaluation_seed=21000
  （=1000+20000 派生口径生效）。
- 冒烟学习信号：E0 单 run 池 3600 updates 内 TT 128.4→13.5 s、
  queue 12.8→3.4、throughput 347→1842——离线 learner 在单 run
  混合质量数据上确实学习（非正式结果，仅链路验证）。
- overlay 合并核验：`xiasha1_v1.yml` 经 `merge_dicts` 后
  evaluation_updates=81 节点、间距 1800、覆盖 0 与 144000。
- 台账：126 行 REGISTERED（code_commit=0d8e17e，condition=
  dataset_spec 名，behavior_source=v1_hetero/<spec>）；
  队列 `artifacts/xiasha1_offline_v1/run_queue.json` 126 任务就绪，
  **待确认后派发**。

### 4.2 派发与关闭（2026-10-22）

- 11:32 队列经 `tools/run_experiment_queue.py --parallel 12` 派发，
  同步起 TB 投影器（`dev/xiasha1_offline/project_tb.py`，6008）；
- ~11:40 Z 裁定**源数据存在问题**，全量停止：36 SUCCEEDED /
  12 RUNNING 被 kill / 78 PENDING 未拉起；126 台账行转
  SCOPE_DISCARD（notes 记录关停时状态）；已完成产物留存
  `data/output_data/offline_tsc/sumo_{batch,cql}_dqn/xiasha1_asym_ns/`
  仅作过程证据，不作结论依据；
- 工程资产（构建器扩展、CLI 参数、overlay、dev 三件、本记录）
  与数据本身无关，修复数据后可原样复用重跑。

### 4.3 v3 数据对齐重建（2026-10-22）

- 根因确认：v2 及全部历史 xiasha 实验在**截断几何**上运行
  （进口道 106.4m + departPos≈80，见 `2026-10-22_xiasha1_online_v3.md`）；
  v3（`xi3re_*`，288/288 DONE）修复几何（280m 道 + 去 departPos）。
- 对齐核验：v3 源 run 与 v2 结构逐位一致（200ep×360、
  `fixed_default`、schema 1、action_dim 4、manifest agent 映射相同）；
  4 个目标包 net.xml 语义哈希仍全等（多场景门禁通过）；4 臂 shard
  均 12 车道观测、queue 奖励。
- 重建：`v2_hetero`（27 源 run/5400 shards/1.944M transitions，
  `source_root_id=xiasha1_online_v3_root`）构建通过；
  `artifacts/xiasha1_offline_v2/run_queue.json` 126 任务
  （前缀 `xi2off_*`）；台账 126 行 REGISTERED。
- v3 冒烟 `xi2offsmoke_e0_p0_batch_dqn_o1000`（3600 updates）链路
  通过；观测到 v3 几何下的真实动力学：随机 init → 锁死
  （queue 224/TT 1155），~360 updates 即塌缩至"单相位不放行"
  固定点（action 2 占比 100%、sw=0、TT 108/thr 1534）并维持至
  update 1800，后段转向 action 3 主导更劣区域——loss 82→~15 后
  振荡。判读：v2 截断几何掩盖了坏策略代价（cap ~3/lane），v3 下
  离线 DQN 的外推误差/退化策略问题被真实暴露——这正是 E0 与
  CQL 对照臂要刻画的量；小预算冒烟不下学习结论。

### 4.4 v1 离线产物清理（2026-10-23）

- Z 裁：清理 v1 离线训练产物，v2 在线源数据本身不动——
  - 已删：`data/output_data/offline_tsc/sumo_{batch,cql}_dqn/
    xiasha1_asym_ns/xi1off*` 49 个 run 目录（~5.6 GB，含冒烟
    `xi1offsmoke_*`）、数据集索引 `v1_hetero`、派发 artifacts
    `artifacts/xiasha1_offline_v1/`（~161 MB，含队列/run_state/
    TB 投影）；
  - 保留：`xi2on_*` v2 在线源 run（290 目录，历史证据）、
    `v2_hetero` 数据集、`artifacts/xiasha1_offline_v2/` 队列、
    v3 冒烟产物 `xi2offsmoke_*`；
- 台账：v1 126 行 notes 追加清理记录（SCOPE_DISCARD 状态不变）；
  `gen_experiments.py` 补登两 campaign META/SHORT/order，
  EXPERIMENTS 索引与明细重生成；
- 工程资产（构建器扩展、CLI 参数、overlay、dev 三件、登记件）
  与数据无关，继续服务 v2 campaign。

## 4.5 xiasha 数据全量退出离线范围（2026-10-23）

- 用户裁定：xiasha 的 v1/v2 轨迹均不用于本轮离线训练；v2 的 126 条
  REGISTERED 转 `SCOPE_DISCARD`，队列标记 dropped，数据资产留档；
- 后续基线使用 `hz1x1_sanity_v1` 的 d3qn/FRAP/SAC/MPLight 源，详见
  `docs/experiments/2026-10-23_hz1x1_offline_mplight.md`。

## 5. 结论与后续

- v1 campaign 作废且离线训练产物已清理（2026-10-23，§4.4）；随后
  用户裁定 xiasha 全部数据退出 offline 范围，v1/v2 的 252 行均为
  `SCOPE_DISCARD`，v2 队列 dropped，数据不派发；在线源 run 与离线源索引
  只保留为历史证据；
- 当前基线为 `hz1x1_offline_v1` MPLight 修订：四控制器、同构路口不同车流，
  27 池/162 REGISTERED，准备未派发，见
  `EXPERIMENTS/hz1x1_offline_v1.md`；
- 阶段②（自主价值利用）与 E6 阶段投喂族仍待数据就绪后另行立项。

## 勘误（2026-10-22，几何截断缺陷）

本文档 §4.1 及 v1 campaign（`xi1off_*`）的全部指标采集自**截断几何**
的 xiasha1 路网（§4.3 起的 v2 campaign 指标为修复后几何）：进口车道
仅 106.4m，
且路由 `departPos≈80`（车辆自距停车线 ~22-27m 处、以 10m/s 凭空插入——
源自 `tools/xiasha_sumo/converter.py` 相机视野重建遗留语义，经
`dev/demand_packs/build_xiasha_demands.py` 传播至全部 14 包）。

影响口径：
- 每车道可观测排队物理上限 ~3 辆，queue 均值/分位、waiting_time 长尾、
  uninserted 计数、travel_time 绝对值均失真；
- 结论中"臂间相对比较"仅在同几何内成立，**不可与修复后（280m、车道起点
  插入）的 v3 指标直接比较**；
- 台账 `ledger/runs.jsonl` 中全部 xiasha1* campaign 行已加
  `errata=geometry_truncated_v1v2`；正式对比口径以 `xiasha1_online_v3`
  为准。
