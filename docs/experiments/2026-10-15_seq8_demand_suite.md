# SEQ8 — 语义化需求包扩展 + 8 段持续/半离线顺序学习 pilot

> 状态：进行中
> 发起日期：2026-10-15　｜　负责人/Agent：Devin
> 最近更新：2026-10-15

## 关联

- campaign：`EXPERIMENTS/seq8_demand.md`（新建）
- run_id / 队列文件 / bench profile：`python -m sequential build-seq8-plan`（schema_version=2 开放协议）→ `sequential launch`；plan 落 `data/output_data/sequential/seq8_*_formal_20261015/`
- code_commit / config_sha：见各 plan 的 `git_commit`/`config_sha256`
- 产物目录：`data/output_data/sequential/seq8_{norep,rep,drift}_formal_20261015/`

## 1. 改动意图

把持续学习/半离线证据从冻结的 4 场景协议扩展到**语义化需求维度**：在同一
hz1x1 路网（直行+左转拓扑）上生成 15 个语义标注的需求包（稀疏/高流量/
左转结构/直行主导/方向不对称/时变波动），并设计三种 8 段序列形态检验
「学习恢复≠控制恢复」「继续学习改写历史表现」「退化幅度非固定」的普遍性：

- **SEQ8-NOREP**：8 个不重复需求，相邻切换主轴轮换（无回访基线）；
- **SEQ8-REP**：4 锚点包块重复 `A B C D → A B C D`，回访间隔统一=4，检验
  场景回访时控制能力的恢复/改写；
- **SEQ8-DRIFT**：总量恒定≈基线，每步只动一个语义旋钮（左转占比
  5→15→35→55% → 空间集中 → 臂间失衡）的临近相似序列。

半离线语义对齐用户定义：stage k 的离线腿只用**本 run 自己 stage<k 产生
的转移**（不依赖外部 Plan-1/2 档案），即新 `archive_mode: SELF`。Q 锚定
沿用 CQ/CQA 质量选择机制作用于自产档案。

## 2. 改动模块

- 新增 `dev/demand_packs/build_hz1x1_demands.py`：NHPP 需求包生成器
  （md5 稳定种子），产出 raw 目录 + `configs/sim/sumohz1x1_<pack>.cfg` +
  `pack_spec.json` 语义自述 + 集合清单 `data/raw_data/hangzhou_1x1_demand_packs.json`。
- 新增 `sequential/self_archive.py`：SELF 自产档案——per-stage 蓄水池
  （默认 2500/stage，界定 checkpoint 体积）、stage<k 因果可见视图、
  `HistoricalBatch` 转换、DHOA/RAND/COV/CQ/CQA 采样适配。
- 新增 `sequential/seq8_manifest.py` + `configs/sequential/seq8_*.yml`：
  schema_version=2 开放协议（8 段/允许回访重复/任意已注册 cfg 网络名），
  冻结 4 场景协议不动。
- `sequential/ha_agent.py`：`archive_mode: SELF` 接线（begin_stage 视图、
  remember() 归档、checkpoint 往返）、重复网络限制收窄到仅 P1C/P1F。
- `sequential/config.py`：`simulator_config_path` 放开到任意已存在
  `configs/sim/<name>.cfg`（未知名仍 fail-fast）。
- `sequential/runtime.py` + `cli.py`：SELF 不挂外部 archive manifest；
  `build-seq8-plan` verb。

已知边界：hz1x1 拓扑无右转连接（每臂直行+左转），「高右转」物理不可
表达，以 lt_extreme/asym/st_dom 补全转向维度。

## 3. 实验设计

- 场景/路网：hz1x1 同一路网，15 个需求包；三序列各 1 order。
- 臂设置：`CONT-FIFO`（在线基线）/`SELF-DHOA-R50`（自产档案动态均匀采样）
  /`SELF-CQ-R50`（自产档案 + Q 质量锚定工作池），offline_ratio=0.5。
- seed/episode：seeds [0..4]，stage_episodes=[100]×8（smoke=[5]×8），
  steps=3600、action_interval=10、b100 其余超参沿用 plan34_b100。
- 父 checkpoint：复用 engineering_20260726 initial_state_catalog 的 ep0
  初始态（per-seed 跨网络 bitwise 等价已审计），元数据重绑定到各 order
  首网并记 `initial_reuse_note`。
- 指标口径：stage×scene 冻结评估矩阵（下三角因果口径），三指标
  travel_time/throughput/unfinished_vehicles 同报；SELF 可见性/OWP
  manifest 记 producer/seen_per_stage/选中量。

## 4. 实验结果

（收割后回填）

## 5. 结论与后续

（待定）
