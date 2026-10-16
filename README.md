# simOfflineTSC

面向交通信号控制（Traffic Signal Control，TSC）的跨仿真器实验框架，基于 [LibSignal](https://github.com/DaRL-LibSignal/LibSignal) 整理，提供与 OpenAI Gym 风格兼容的交通环境、统一训练流程，以及传统控制和强化学习基线。

本仓库在统一框架上持续开展多条研究线，当前包括场景引导注意力（campaign `att_entity_*`）与**半离线持续学习**（当前激活口径为 `cl_hz1x1` 线：自产历史归档 + 教师锚定，见下文「研究线」节；更早的 Plan 1–4 / HA-SODQN 线已冻结保留）等；各研究线的 run 级事实与结论统一登记在仓库台账中。

## 仓库地图

- **实验进度与结论**：`EXPERIMENTS.md` 顶部「当前面板」；逐 run 机读事实源 `ledger/runs.jsonl`，各 campaign 明细在 `EXPERIMENTS/`。动态实验状态一律以台账为准，本文件不复述。
- **实验纪律**：`GOVERNANCE.md`（登记→冒烟→派发→收割→回填全流程）；正式验收口径见 `bench/`。
- **Agent 上手**：`AGENTS.md`；一次性脚本登记见 `dev/INDEX.md`。
- **注意力线代码**：`scene_attention/` 方法包 + `agent/` 注册臂（codex 分支工作已并入 main；`_legacy/router/` 为已否决架构存档）。
- **专项手册**：`docs/`（Plan 2 支持说明、半离线复现指南、cl_hz1x1 Q 锚定设计稿等，见下文各节指针）。

## 功能概览

- 支持 CityFlow 与 SUMO 交通仿真器（libsumo / traci 接口）。
- 传统控制：FixedTime、SOTL、MaxPressure；强化学习：DQN、FRAP、CoLight、PressLight、MPLight、MADDPG、MAGD、PPO 等。
- 统一包形状入口：`python run.py`（Online/传统）+ `python -m <pkg>[.<face>]`（sequential 半离线、scene_attention 注意力、hitl 通道；见「实验入口边界」）。
- 配套工具：路网与交通流格式转换、SUMO HTML 回放对比、浏览器实时转播（见「工具与可视化」）。

## 实验入口边界（统一包形状，GOVERNANCE §3.1）

拉起语法统一为 **`python -m <pkg>[.<face>] [args]`**；`python -m <pkg>` 打印该包面表。

| 工作 | 面 | 配置 | 产物 |
| --- | --- | --- | --- |
| Online/传统 TSC（fixedtime/dqn/colight 等） | `python run.py` | `configs/tsc/` | `data/output_data/tsc/` |
| Plan 2 纯 Offline DQN | `python -m sequential.offline train` | `configs/offline_tsc/` | `data/output_data/offline_tsc/` |
| Plan 3/4 Sequential / HA-SODQN / HITL 通道训练（❄️ 冻结，仅复跑底稿） | `python -m sequential <verb>`（manifest 驱动） | `configs/sequential/` | `data/output_data/{sequential,ha_sodqn,hitl_probe}/` |
| 半离线持续学习正式批（✅ 当前唯一激活链路 cl_hz1x1） | `python tools/run_experiment_queue.py`（队列 manifest 驱动，生成件 `dev/cl_hz1x1/`） | `dev/cl_hz1x1/` overlays + `configs/tsc/` | `data/output_data/tsc/` + `artifacts/cl_hz1x1/` |
| 场景引导注意力训练/评测 | `python -m scene_attention.train` / `.eval` | `scene_attention/events.py` 常量源 | `data/output_data/dev_takeover_probe/` |
| HITL 实时服务 | `python -m hitl.serve` | `--scene/--dispatch/--gate/--record` | `data/output_data/hitl_probe/sessions/` |
| HITL 数据面/plan/探针/审计 | `python -m hitl.pipeline <stage>` / `hitl.plans build --round <r>` / `hitl.probe.<x>` / `hitl.audit.<x>` | 见 `hitl/README.md` | `data/output_data/hitl_probe/` |

兼容 shim（过渡，将移除）：`python sequential_run.py`→`-m sequential`，`python offline_run.py`→`-m sequential.offline`。

调用规则：

- 各面必须使用各自命令，不能用相似参数名替代实验语义：不要通过 `run.py --agent batch_dqn` 启动 Offline，也不要通过 `-m sequential.offline` 启动 Online/传统实验。
- `-m sequential.offline` 训练期间不会向数据集追加 transition，也不会修改源 Plan 1 NPZ。
- HA-SODQN 不能通过修改 YAML 直接切换为 Double DQN 或 PPO；跨算法验证需要先增加算法 identity、训练 target/loss、checkpoint 和 evaluator 适配。
- 正式 run 只走两条路径：队列 runner（`tools/run_experiment_queue.py`+`run_queue_*.json`）或 `bench/` profile；一次性件归 `dev/<topic>/` 并登 `dev/INDEX.md`。
- 出图/表格脚本（`<pkg>/figs/` 等）为产物件不是入口。

详细手册：Offline 数据门禁与恢复见 [Plan 2 支持说明](docs/plan2_offline_support.md)；半离线科学设计、结果边界与迁移步骤见 [半离线实验总结与跨算法复现实用指南](docs/semi_offline_cross_algorithm_reproduction.md)；HITL 通道与边界见 [hitl/README.md](hitl/README.md)；注意力线见 [scene_attention/README.md](scene_attention/README.md)。

## 研究线：半离线持续学习

**语义定义**：多场景顺序训练下，stage k 的每个训练 batch 由「当前在线回放」
+「本 run 自己 stage 1..k−1 产生的历史转移」（因果可见、均匀随机采样、
`offline_ratio` 混入比例）两部分构成；可选在离线样本上叠加**教师锚定
loss**——各历史场景 stage 末的冻结网络给出参考 Q，约束 `λ_π·KL(softmax
(Q_teacher/T)‖softmax(Q_θ/T))` 使非执行动作的 Q 面不被共享权重漂移改写。

**当前唯一激活实现 = `cl_hz1x1` 线**（campaign `cl_hz1x1`，5584 run 已收口）：
`agent/dqn.py`+`agent/frap.py` 的 `collect_history`/`_anchor_loss` +
`arterial/replay.py` 的 `HistoricalPool`/`mixed_batch`（`global_uniform` 或
`scene_intersection_balanced`）+ `tools/run_experiment_queue.py` 队列编排
（stage=进程边界，`history_paths`/`teacher_paths`/`depends_on` 逐段交接），
队列生成件与收割件在 `dev/cl_hz1x1/`。结论摘要见 EXPERIMENTS.md 面板；
设计稿 `docs/plan_qanchor_clhz1x1.md`。

**已冻结保留**：Plan 1–4 / HA-SODQN 链（`sequential/` 方法包，manifest 驱动
的单进程 stage 循环 + 外部 Plan-2 档案 P1C/P1F 可见性 + DHOA/RAND/CQ/CQA
采样器）。证据链完整、可审计、可复跑底稿，但**不再发起新实验**——与
cl_hz1x1 并存会造成「两套半离线口径」歧义，故冻结。

**已否决支线（负证据保留，勿再启用）**：
- **CQ/CQA 数据层选池**（按转移质量分锚选固定工作池 OWP）——seq8 pilot
  中实测在自产档案口径下不优于均匀重放且稀疏场景退化更重，2026-10-16
  裁决作废；ha_sodqn 历史批中的 CQ/CQA 结果仅作对比留档；
- **sequential `archive_mode: SELF`**（in-process stage<k 蓄水池档案）——
  与 cl_hz1x1 全量落盘归档语义重复实现，同日裁决移除代码；
- **seq8 pilot**（需求语义多样性 8 段序列）——campaign 关闭，18 完成 run
  留作资产；15 个需求包与序列定义保留为可复用资产。

半离线正式场景为四个 SUMO 杭州单路口（16 维 observation：8 维进入车道 count + 8 维当前相位 one-hot；8 个绿灯动作）：

| 代称 | network | 数据目录 |
| --- | --- | --- |
| S1 | `sumohz1x1_config2` | `data/raw_data/hangzhou_1x1_qc-yn_18041608_1h/` |
| S2 | `sumohz1x1` | `data/raw_data/hangzhou_1x1_bc-tyc_18041610_1h/` |
| S3 | `sumohz1x1_config4` | `data/raw_data/hangzhou_1x1_sb-sx_18041607_1h/` |
| S4 | `sumohz1x1_config3` | `data/raw_data/hangzhou_1x1_kn-hz_18041608_1h/` |

正式场景顺序：O1 = S4→S1→S3→S2，O2 = S2→S3→S1→S4，O3 = S1→S4→S2→S3，O4 = S3→S2→S4→S1。

## 安装

- 建议使用独立 Conda 环境（本仓库开发环境名为 `colight`），Python 3.9。
- `python -m pip install -r requirements.txt` 安装核心依赖。
- PyTorch（原项目用 1.11.0）、CityFlow、SUMO 按需单独安装：
  - PyTorch 安装命令见 [官方安装页](https://pytorch.org/get-started/locally/)；注意旧版 Gym 和图神经网络扩展的兼容性。
  - CityFlow 源码安装见 [官方文档](https://cityflow.readthedocs.io/en/latest/install.html)，验证 `python -c "import cityflow"`。
  - SUMO 见 [官方安装文档](https://sumo.dlr.de/docs/Installing/index.html)，需设置 `SUMO_HOME` 与 `PYTHONPATH`，验证 `python -c "import sumolib, traci"`。
- Plan 2 另需 `python -m pip install --no-deps -r requirements-offline.txt`（固定 `d3rlpy==2.0.4`，避免扰动现有 Torch/Gym/NumPy）。第三方来源与适配边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- 部分智能体有额外依赖，如 PPO/PFRL 需要 `pfrl`，CoLight 部分实现需要 PyTorch Geometric。

seed 语义按入口区分：`run.py --seed` 是 Online 流程的 `training_seed`；`offline_run.py train --seed` 是 Plan 2 的 base seed，并默认派生 `model_init_seed=base`、`dataset_sampler_seed=base+10000`、`evaluation_seed=base+20000`（可覆盖，分别写入 metadata/checkpoint）。SUMO 默认采用固定默认随机实现（`sumo_seed_mode=fixed_default`），仅显式传 `--sumo-seed` 才生效。

## 快速开始

每个入口一条最小命令；完整参数手册见对应文档。

```bash
# Online/传统（参数详见 python run.py --help 与 configs/tsc/*.yml）
python run.py --agent maxpressure --world sumo --network sumo1x1 \
  --interface libsumo --prefix sumo-baseline --seed 0

# Plan 2 纯 Offline（prepare-plan2 / validate-dataset / train 三步，
# manifest schema v2，详见 Plan 2 支持说明）
python offline_run.py train --agent batch_dqn --network sumohz1x1 \
  --dataset-manifest /path/to/manifest.json --prefix p2_seed1000 --seed 1000

# Sequential / HA-SODQN（❄️已冻结，manifest 驱动；新半离线实验用
# cl_hz1x1 队列路径，见上表）
python sequential_run.py --help
```

正式 Sequential/HA 实验需先构建并 validate manifest，launch 会产生大规模 SUMO 训练并要求显式授权；执行顺序、资产清单与门禁见 [半离线复现指南](docs/semi_offline_cross_algorithm_reproduction.md)。

## 输出目录

各入口输出彼此隔离，均不写入其他实验的目录：

```text
data/output_data/tsc/                    # Online/传统
data/output_data/offline_datasets/plan2/ # Plan 2 只读数据索引
data/output_data/offline_tsc/            # Plan 2 训练输出
data/output_data/analysis/               # 分析汇总
data/output_data/sequential/             # Sequential DQN
data/output_data/ha_sodqn/               # HA-SODQN engineering / 正式运行 / 分析
```

`data/output_data/`、根目录 `data/output_data/`、`analysis/`、`tmp/` 以及 checkpoint/日志/图像类文件均已在 `.gitignore` 中忽略，可再生成的大型产物不提交 Git。

## 项目结构

| 路径 | 说明 |
| --- | --- |
| run.py | Online/传统实验入口（offline_run.py/sequential_run.py 为过渡 shim） |
| environment.py | 环境探测辅助 |
| agent/ | 传统控制及强化学习智能体 |
| world/ | CityFlow、SUMO 等仿真器适配层 |
| trainer/、task/ | 训练评估流程与任务编排 |
| configs/{tsc,offline_tsc,sequential,sim,evaluation} | 智能体、离线、顺序/半离线、仿真器及评估配置 |
| data/ | 数据层：`raw_data/`（路网输入，只读）、`output_data/`（run 产物证据桶，gitignore）、`dataset/`（数据集接口代码） |
| common/、generator/ | 注册器与配置加载、特征生成器 |
| sequential/ | 方法包（❄️已冻结）：持续学习+HA-SODQN 半离线（manifest 驱动；README 方法卡）。新半离线实验走 cl_hz1x1 |
| scene_attention/ | 方法包：场景引导注意力主线（`python -m scene_attention`；README 方法卡） |
| hitl/ | 通道包：人在回路实时+数据面（`python -m hitl`；README 通道卡与边界） |
| GOVERNANCE.md、EXPERIMENTS.md、EXPERIMENTS/、ledger/ | 实验治理规范、台账索引与各 campaign 明细、机读台账 |
| bench/ | 正式验收口径（BENCH.md + profiles） |
| dev/ | 一次性脚本唯一合法堆放处 + INDEX.md 登记 |
| docs/ | 计划文档 + **实验记录**（`docs/experiments/`，每实验一份固定格式，规则见 `docs/README.md`）+ 里程碑归档 `docs/done/` |
| tools/、tests/ | 分析与可视化工具（见下节）、最小回归测试 |

## 工具与可视化

- **`tools/xiasha_sumo/`**：xiasha1*1 语义事件到 SUMO 逐车/flow 需求和信号路网的转换。最小示例 `python -m tools.xiasha_sumo --all`，参数与信号规则见 [tools/xiasha_sumo/README.md](tools/xiasha_sumo/README.md)。
- **`tools/traffic_flow_profile/`**：SUMO 单场景车流需求评估、标准度量计算与固定子图报告。
- **`common/converter.py`**：CityFlow 与 SUMO 路网/交通流互转；`cd common && python converter.py --typ s2c`（默认处理 cologne3，转换其他数据需改 parse_args 默认路径）。
- **`tools/sumo_html_comparison.py`**：将已记录的 SUMO 决策经 libsumo 按原控制间隔重放，生成自包含 HTML 多控制器对比页面（不改源记录/路网/配置）。支持多方法对比与追加冻结记录；最小示例：

  ```bash
  python tools/sumo_html_comparison.py --scene S2 --fixedtime-only \
    --evaluation-seed 10000 --start 0 --end 3600 --sample-every 1 \
    --output /tmp/s2_fixedtime_html
  ```

  辅助实现 `tools/sumo_gui_comparison.py`（SUMO-GUI/TraCI 截图或 GIF 回放）；页面产物输出到 `/tmp` 等产物目录，不提交 Git。
- **`hitl/live/`（原 `tools/sumo_live_broadcast/`）**：浏览器实时转播服务（libsumo 实时步进，SSE 收帧 + HTTP POST 控制）。支持运行时切换 FixedTime/MaxPressure/SOTL/DQN 快照/人工接管、相位卡片点击接管、变速与决策点等待确认；多路口场景（如 S5=sumohz4x4）下控制器逐路口决策，相位面板可通过下拉或点击地图路口切换。地图渲染可插拔：默认 deck.gl WebGL 渲染（`static/vendor/deck.min.js`，MIT），`?renderer=canvas` 可切回无依赖 Canvas 实现，两者实现同一契约（`static/js/renderer.js`）。最小示例 `python -m hitl.serve --scene S2 --port 8010`；常驻托管脚本 `hitl/live/supervise.sh`。回归测试 `tests/test_sumo_live_broadcast.py`。

  转播层另注册了场景引导调度控制器 `dsp` / `dspconcat` / `dspself`（`hitl/live/controllers.py` 的 `DispatchController`）：内部按 120 s 窗口在候选控制器间选接管臂，调度轨迹通过事件流透出。

## 三条研究线

本仓在 LibSignal 基座（`agent/`+`world/`+`trainer/`+`run.py`）之上并行推进三条研究线，均为 Route A 包形状（`python -m <pkg>` 打面表，方法卡见各自 README）：

### 1. 半离线 / 持续学习（cl_hz1x1 当前线 + `sequential/` 已冻结）

在线交互 + 过去阶段自产历史数据的混合训练线。**当前实现**：`cl_hz1x1` 队列路径（run.py 逐 stage 进程 + `history_archive` 全量归档 + 教师锚定 loss；生成/收割件 `dev/cl_hz1x1/`，产物 `data/output_data/tsc/` + `artifacts/cl_hz1x1/`）。**冻结保留**：Plan 2 纯离线（`python -m sequential.offline`）、manifest 驱动 HA-SODQN（`python -m sequential`，已冻结）、HITL 采集数据回流训练。产物 `data/output_data/{offline_tsc,offline_datasets,sequential,ha_sodqn}/`（冻结层）。

### 2. 人在回路（`hitl/`）

实时播报→人判→回注控制的通道线（非 train/eval 型包）：`hitl.serve` 实时转播服务（libsumo 步进+SSE+接管控制）、`hitl.pipeline` 会话打包/特征生成、`hitl.plans` 轮次 plan 生成、`hitl.probe`/`hitl.audit` 探针与审计。训练态归 sequential，本包无 train 面。产物 `data/output_data/hitl_probe/`。

### 3. 场景引导注意力调度（`scene_attention/`）

事件播报文本编码为场景 query，引导对车道状态 token 的交叉注意力，产出浓缩场景表征注入信号控制。当前主线为单层场景条件控制器 `SceneJointPolicy`（dqn_s1 底座 + sga 注意力 + 门控有界残差），训练链路 adapter 教师蒸馏→原装 reward RL 微调→意图编译层；历史对照臂（selfattn/concat/state-only 消融）已从维护面移除（git 可查），SMDP Router 双层架构已否决（归档 `_legacy/router/`）。机制谱系与产物映射见其 README。产物 `data/output_data/dev_takeover_probe/`（沿用旧桶名）。

历史注记：控制器注册表与场景编码器解耦的 `DispatchRouter` 双层架构已被否决（上层动作空间退化为控制器选择），实现留存于 `scene_attention/_legacy/router/`（附死因 README）；编码器活组件（`CompositeSceneEncoder` 等）在 `scene_attention/dispatch_model.py`。

## 在其他服务器复现

Git 只同步代码、配置与场景；trajectory、checkpoint、dataset index 与正式结果需另行迁移。完整资产清单、路径重定位、哈希验证与门禁顺序见 [复现指南·服务器迁移](docs/semi_offline_cross_algorithm_reproduction.md#8-新服务器迁移清单)。

## 开发约定

- 新实验优先通过 configs/ 管理参数，避免硬编码；四类入口不混用。
- **每次实验（含相关改动）在 `docs/experiments/` 建档**：一份固定格式文档回答改动意图/改动模块/实验设计/实验结果，模板 `docs/experiments/TEMPLATE.md`，规则见 `docs/README.md` 与 `GOVERNANCE.md` §4.5。
- 修改智能体或训练流程后，至少运行一个受影响仿真器的最小代表性实验，并记录仿真器、agent、网络、随机种子和完整命令；正式实验另记 Order/Archive/seed/manifest hash 与恢复链。
- 不提交 `data/output_data/`、checkpoint、缓存等可再生成产物；大规模数据用外部存储并在文档中给出校验信息。
- 实验相关的一次性脚本放 `dev/<topic>/` 并在 `dev/INDEX.md` 登记。

## 项目来源

本仓库代码基于 [DaRL-LibSignal/LibSignal](https://github.com/DaRL-LibSignal/LibSignal) 整理。原项目提供跨仿真器交通信号控制环境、多种传统与强化学习基线，以及 SUMO/CityFlow 数据转换能力。

相关 sim-to-real 工作：

- [UGAT：Uncertainty-aware Grounded Action Transformation](https://github.com/darl-libsignal/ugat)
- [PromptGAT：Prompt to Transfer](https://github.com/DaRL-LibSignal/PromptGAT)

使用代码、数据集或第三方仿真器时，请同时遵守对应上游项目及数据文件附带的许可证和引用要求。
