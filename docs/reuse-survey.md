# 复用调研（reuse-survey）

> 维护人：工程开发 ｜ 首次建立：2026-11-02（campaign `2026-11-02-cleanup-and-scenario-coverage` P1.5）
> 规则：**未登记进本文件的库不许出现在 import 里；无 license 的代码一律不引。**
> 数据来源：GitHub REST API 实测（2026-11-02 夜），license / 最近 push / star 均为当日快照；仓内路径与版本为 `colight` 环境实测。

## 0. 结论（先看这段）

1. **本阶段不需要新增任何第三方依赖。** P4/P5 要的「场景划分 + 覆盖率」，仓内已有 14 个需求包、3 类事件、22 个冻结事件计划、bench 判定协议，加上已装的 scikit-learn / scipy 就够算。0 安装。
2. 候选共登记 18 条：**已装并可直用 6 / 仅参考 2 / 不采用 10**。采用方式以「已有依赖」为主，没有引入新的复制代码。
3. 两个治理风险（不属我单方面能改的，见 §5）：
   - `DaRL-LibSignal/LibSignal` 上游**没有 LICENSE**，而本仓基座来自它；
   - `THIRD_PARTY_NOTICES.md` 把 d3rlpy 写成 Apache-2.0，**实际 v2.0.4 LICENSE 是 MIT**（已实测文件内容）。
4. 三个环境坑：`d3rlpy` 未安装（Plan 2 纯离线冻结线不可跑，半离线 cl_hz1x1 不依赖它）；`gym` 实装 0.26.2 而 `requirements.txt` 锁 0.21.0；`SUMO_HOME` 未设置（调用 SUMO 自带脚本要绝对路径）。

## A. 第一步：仓内已有能力（优先复用，不写新代码）

| 能力 | 路径 | 核实状态 | 在本次 campaign 里的用途 |
|---|---|---|---|
| 14 个 xiasha1 需求包（sparse_low / dense_high / dense_extreme / rt_high / lt_high / st_dom / asym_ns / asym_we / asym_nonly / fluct_sin / fluct_burst / fluct_ramp / compound_a / compound_b） | `data/raw_data/xiasha1_demand_packs.json` + 14 个 `xiasha1_*_1h/` 目录 | 已用于 `xiasha1_online_v1` 正式 campaign | 场景的「需求分布」维度**现成**，不用自己造 |
| 事件模块 3 类：`lane_blockage` / `road_closure` / `global_rain` | `world/sumo_events/`（README、runtime、schema、plans、trex_blockage 等） | hz4x4 上已工程验证并有对比证据 | 场景的「事件」维度**现成** |
| 22 个冻结事件计划（含训练/评估 held-out 划分） | `configs/events/plans/*.yml` | 已用于 att004 阶段①②；`att004_t4_eval_v1.yml` 是「训练未见布局」先例 | 场景划分与 held-out 评估直接可套 |
| 确定性随机事件计划生成器（同 seed 字节级一致、方法间共享同一 plan 保持配对） | `tools/build_random_event_plan.py` | 已有 `tests/test_event_plans.py` | 场景采样 + 配对实验的现成工具 |
| 基准协议与「阶段①分类关卡」判定 | `bench/BENCH.md`、`bench/profiles/att_hz4x4.yml` | 现行 | 覆盖率「达标」判定口径可参考 |
| 台账三指标（travel_time / throughput / unfinished_vehicles） | `ledger/runs.jsonl`（15479 行，用 jq 过滤） | 事实源 | 覆盖率指标的计算基数 |
| 半离线实现 cl_hz1x1（逐 stage 进程 + history_archive 全量归档 + 教师锚定） | README「研究线 1」+ `run.py` + `dev/cl_hz1x1/` | 现行激活口径 | P5 的实验臂 |
| 传统基线 + RL 基线（FixedTime/SOTL/MaxPressure/DQN/FRAP/CoLight/PressLight/MPLight/SAC…） | `agent/`、`configs/tsc/`、`common/registry.py` | 已实现并注册 | 对照组与「哪个场景熟、哪个不熟」的臂 |
| 需求包构建脚本先例 | `dev/demand_packs/build_xiasha_notrain_queue.py` | 已用 | 若 P4 要新增需求包，照此写 |
| SUMO 自带场景生成工具 | `site-packages/sumo/tools/{randomTrips,routeSampler,osmGet}.py`；`envs/colight/bin/{duarouter,od2trips,activitygen,netconvert}` | sumolib/traci 1.25.0 已装，二进制在 | 真要造新需求/路径时用；**不复制 SUMO 源码** |

## B. 第二步：已装依赖（能直接用，别重装）

| 包 | 实装版本 | 与本 campaign 的关系 |
|---|---|---|
| scikit-learn | 1.7.2 | 场景簇聚类（KMeans / Agglomerative）+ silhouette，覆盖率分母 K 的来源；**直接可用** |
| scipy | 1.15.3 | 覆盖率熵、bootstrap / 置换检验（半离线 vs 在线按 seed 配对）；**直接可用** |
| pandas / numpy | 2.3.3 / 1.26.4 | 指标汇总 |
| gym | 0.26.2（`requirements.txt` 锁 0.21.0，**已漂移**） | 现有 world 的 API 语义以代码为准；P2 整理不许动它 |
| torch / torch-geometric | 1.13.1+cu116 / 2.7.0 | 现有 agent、CoLight 图后端 |
| sumolib / traci | 1.25.0 | SUMO 接口（libsumo/traci 两条） |
| CityFlow | 0.1 | 第二仿真器对照 |
| pfrl | 已装（无 `__version__` 属性） | PPO/PFRL 臂 |
| d3rlpy | **未安装** | Plan 2 纯离线冻结线要跑才需要；半离线 cl_hz1x1 **不依赖**它 → P5 用不到就不装 |

## C. 第三步：GitHub 公开项目候选（逐条登记）

| 候选 | 链接 | license | 最近更新 | 能直接用吗 | 采用/不采用理由 |
|---|---|---|---|---|---|
| SUMO | https://github.com/eclipse-sumo/sumo | EPL-2.0 | 2026-11-02 | 已装 1.25.0，能用 | **采用（已有依赖）**：traci/libsumo 直接用；EPL 属弱 copyleft，**不复制其源码**，只当依赖 |
| SUMO 工具脚本（randomTrips / routeSampler / duarouter / od2trips / activitygen） | 随 SUMO 发行 | EPL-2.0 | 同上 | 能用（需设 `SUMO_HOME` 或用绝对路径，当前未设） | **采用（已有）**：新场景需求生成；先看仓内 `dev/demand_packs/` 是否有现成包 |
| CityFlow | https://github.com/cityflow-project/CityFlow | Apache-2.0 | 2025-08-19 | 已装 0.1 | **采用（已有依赖）**：跨仿真器对照 |
| LibSignal（本仓基座） | https://github.com/DaRL-LibSignal/LibSignal | **NONE（无 LICENSE）** | 2025-09-19 | **不可用** | **不采用**：无 license 不得复制代码；本仓既存基座属历史遗留，新增代码一律不得再从上游取文件（见 §5 待裁决） |
| pytorch_geometric | https://github.com/pyg-team/pytorch_geometric | MIT | 2026-10-21 | 已装 2.7.0 | **采用（已有依赖）**：CoLight 图后端 |
| d3rlpy | https://github.com/takuseno/d3rlpy | MIT（v2.0.4 LICENSE 实测为 MIT，非 Apache-2.0） | 2025-09-10 | pinned 2.0.4 可用（需安装） | **采用（既有）**：Plan 2 冻结线上游；**不升级**——v2.8.x 是重写 API，升级=算法语义变更，要走 GOVERNANCE §5.5 |
| scikit-learn | https://github.com/scikit-learn/scikit-learn | BSD-3-Clause | 2026-11-02 | 已装 1.7.2 | **采用（已有依赖）**：场景聚类 + 覆盖率分母 |
| scipy | https://github.com/scipy/scipy | BSD-3-Clause | 2026-11-02 | 已装 1.15.3 | **采用（已有依赖）**：熵 / 配对显著性检验 |
| tianshou | https://github.com/thu-ml/tianshou | MIT | 2026-04-03 | 能装（py3.10 支持） | **仅参考**：offline 算法（CQL/IQL）实现读法参考；不引入——与 `trainer/` 重复且训练语义不同 |
| coverage.py | https://github.com/coveragepy/coveragepy | Apache-2.0 | 2026-11-02 | 能装（dev-only） | **暂不采用**：P3 关心「关键入口没坏」，全量测试已 13 分 39 秒，再加覆盖率翻倍；收益对不起时间 |
| osmnx | https://github.com/gboeing/osmnx | MIT | 2026-07-31 | 能装 | **不采用**：路网固定（xiasha1 / hz4x4），抓 OSM 无收益，还拉 geopandas/pyproj 重依赖 |
| sumo-rl | https://github.com/LucasAlegre/sumo-rl | MIT | 2026-03-08 | 能装但冲突 | **不采用**：与 `world/` 重复封装，且是新一代 gym API 的 wrapper；引入=同进程两套环境语义 |
| flow | https://github.com/flow-project/flow | MIT | 2024-07-27 | 不推荐 | **不采用**：场景生成/RL 框架与仓内重叠，维护慢、依赖旧，py3.10 上要额外验证 |
| stable-baselines3 | https://github.com/DLR-RM/stable-baselines3 | MIT | 2026-10-26 | 能装 | **不采用**：会替换现有 DQN/PPO 实现并改动基准口径，属语义变更 |
| SALib | https://github.com/SALib/SALib | MIT | 2026-10-30 | 能装 | **不采用**：场景维度敏感性用 numpy 手算方差/置换就够，引库成本 > 收益 |
| hypothesis | https://github.com/HypothesisWorks/hypothesis | 待核实（GitHub 自动识别为 NOASSERTION，疑 MPL-2.0） | 2026-11-01 | 待定 | **不采用**：license 未能机器确认，按「无明确 license 不引」处理；且测试面收益低 |
| Gymnasium | https://github.com/Farama-Foundation/Gymnasium | MIT | 2026-10-29 | 能装但冲突 | **不采用**：现有 world 走 gym 0.21 语义；迁移是独立提案，不在 P2/P4 里夹带 |
| DataLight（旧） | https://github.com/LiangZhang1996/DataLight-old | GPL-3.0 | 2024-05-02 | 理论能跑 | **不采用**：GPL-3.0 传染风险 + 2024 停更 |
| ADAC-traffic | https://github.com/qcri/ADAC-traffic | MIT | 2022-07-11 | 能跑但旧 | **不采用**：2022 停更、非主线，无直接可搬模块 |

## D. 「看起来能用、实际不行」的坑

1. **Gymnasium / sumo-rl 系**：不是「装个包就能用」。仓内 `world/` 是按 gym 0.21 风格写的，同进程混两套 API 会直接在 reset/step 语义上翻车，而且会动到既有 run 的可比性。
2. **OSMnx / flow 的重依赖**：能装，但会拉 geopandas/shapely 等，和我们锁定的 numpy 1.26 / torch 1.13 组合要重新验证；没有人手维护就是技术债。
3. **升级 d3rlpy 到 2.8.x**：v2 API 重写，2.0.4 的 pin 是有原因的，升级会改 Plan 2 评测口径。
4. **LibSignal 上游无 license**：「抄一段代码进来」这条路法律上不通，只能读思路自己写——本仓已经这么做了，别再回头取文件。
5. **SUMO 两套安装**：`sumolib/traci` 是 python 包，`sumo/duarouter/...` 是二进制，`SUMO_HOME` 没设时 `randomTrips.py` 之类会报找不到 sumo。用绝对路径，不改全局环境。

## E. 本阶段实际采用的复用动作（P4/P5 可直接落地）

- **0 个新依赖、0 行复制代码。**
- 场景划分建议（给方案设计做输入，定稿权在 P4）：需求包 14 × 事件类别 3 × 时段切分 2 = 84 格；覆盖率 = 达标格数 / 84，或按簇加权有效覆盖熵（scipy）。
- 若 P4 要新需求包：用 SUMO 工具脚本生成，登记 `dev/demand_packs/`，产物进 `data/raw_data/`。
- 覆盖率计算的统计口径（配对检验、seed 聚合）由数据评测定，我按定稿实现，不自己发明口径。

## F. 待裁决 / 已上报

1. **LibSignal 上游无 LICENSE**：本仓是 public 仓且基座来自它。属于治理问题，需 Chief 决定（联系上游 / README 加说明 / 不动）；我不自行处理。
2. **`THIRD_PARTY_NOTICES.md` 的 d3rlpy 许可证写错**（Apache-2.0 → 实为 MIT）。建议 P2 由治理管家修正；我不动根文件，避免与 `dsh/steward` 分支撞车。
3. **d3rlpy 未安装**：若 P5 排期里出现 Plan 2 纯离线臂，需要先批准安装（`pip install --no-deps -r requirements-offline.txt`），不属我单方面决定。
