# simOfflineTSC

面向交通信号控制（Traffic Signal Control，TSC）的跨仿真器实验框架，基于 [LibSignal](https://github.com/DaRL-LibSignal/LibSignal) 整理，提供与 OpenAI Gym 风格兼容的交通环境、统一训练流程，以及传统控制和强化学习基线。

本仓库在统一框架上持续开展多条研究线，当前包括场景引导注意力（campaign `att_entity_*`）与半离线强化学习（Plan 1–4 / HA-SODQN）等；各研究线的 run 级事实与结论统一登记在仓库台账中。

## 仓库地图

- **实验进度与结论**：`EXPERIMENTS.md` 顶部「当前面板」；逐 run 机读事实源 `ledger/runs.jsonl`，各 campaign 明细在 `EXPERIMENTS/`。动态实验状态一律以台账为准，本文件不复述。
- **实验纪律**：`GOVERNANCE.md`（登记→冒烟→派发→收割→回填全流程）；正式验收口径见 `bench/`。
- **Agent 上手**：`AGENTS.md`；一次性脚本登记见 `dev/INDEX.md`。
- **注意力线代码**：`agent/scene_attention.py` + `agent/tarl.py` 等当前位于 `codex/milestone0-experiment-infrastructure` 分支。
- **专项手册**：`docs/`（Plan 2 支持说明、半离线复现指南等，见下文各节指针）。

## 功能概览

- 支持 CityFlow 与 SUMO 交通仿真器（libsumo / traci 接口）。
- 传统控制：FixedTime、SOTL、MaxPressure；强化学习：DQN、FRAP、CoLight、PressLight、MPLight、MADDPG、MAGD、PPO 等。
- 四类互不混用的实验入口：`run.py`（Online/传统）、`offline_run.py`（纯 Offline）、`sequential_run.py`（Sequential / HA-SODQN 半离线）。
- 配套工具：路网与交通流格式转换、SUMO HTML 回放对比、浏览器实时转播（见「工具与可视化」）。

## 实验入口边界

| 项目 | Online/传统 TSC | Plan 2 纯 Offline DQN | Plan 3/4 Sequential DQN | HA-SODQN 半离线 |
| --- | --- | --- | --- | --- |
| 正确入口 | `run.py` | `offline_run.py` | `sequential_run.py` | `sequential_run.py` 的 HA 子命令 |
| 主要配置 | `configs/tsc/` | `configs/offline_tsc/` | `configs/sequential/plan34*.yml` | `configs/sequential/ha_sodqn_b100.yml` |
| 当前支持 | 传统控制、DQN、PPO 等既有 agent | `batch_dqn`、`cql_dqn` | Independent DQN | Independent DQN + 静态历史混合采样 |
| 训练数据 | 仿真中在线生成 | 冻结的 Plan 1 trajectory | 四场景顺序在线交互 | 顺序在线交互 + Plan 1 静态档案 |
| 主要输出 | `data/output_data/tsc/` | `data/output_data/offline_tsc/` | `data/output_data/sequential/` | `data/output_data/ha_sodqn/` |
| 额外依赖 | 核心依赖和对应 agent/仿真器 | 另加 `requirements-offline.txt` | PyTorch、SUMO | PyTorch、SUMO 和已验收历史资产 |

调用规则：

- 四类入口必须使用各自命令，不能用相似参数名替代实验语义：不要通过 `run.py --agent batch_dqn` 启动 Offline，也不要通过 `offline_run.py` 启动 Online/传统实验。
- `offline_run.py` 训练期间不会向数据集追加 transition，也不会修改源 Plan 1 NPZ。
- HA-SODQN 不能通过修改 YAML 直接切换为 Double DQN 或 PPO；跨算法验证需要先增加算法 identity、训练 target/loss、checkpoint 和 evaluator 适配。

详细手册：Offline 数据门禁与恢复见 [Plan 2 支持说明](docs/plan2_offline_support.md)；半离线科学设计、结果边界与迁移步骤见 [半离线实验总结与跨算法复现实用指南](docs/semi_offline_cross_algorithm_reproduction.md)。

## 研究线：半离线实验链（Plan 1–4 / HA-SODQN）

该研究线由四个阶段构成：Plan 1 Online DQN 采集（建立单场景在线基线、trajectory 与可恢复初始化）→ Plan 2 纯 Offline 学习 → Plan 3/4 Sequential 顺序训练（比较场景切换时的 replay 策略）→ HA-SODQN 半离线历史利用。各阶段证据状态与结论见 `EXPERIMENTS.md` 面板及 `EXPERIMENTS/` 明细。

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

# Sequential / HA-SODQN（manifest 驱动，不接受 --agent/--network）
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

`data/output_data/`、根目录 `output_data/`、`analysis/`、`tmp/` 以及 checkpoint/日志/图像类文件均已在 `.gitignore` 中忽略，可再生成的大型产物不提交 Git。

## 项目结构

| 路径 | 说明 |
| --- | --- |
| run.py / offline_run.py / sequential_run.py | 三类实验入口（见「实验入口边界」） |
| environment.py | 环境探测辅助 |
| agent/ | 传统控制及强化学习智能体 |
| world/ | CityFlow、SUMO 等仿真器适配层 |
| trainer/、task/ | 训练评估流程与任务编排 |
| configs/{tsc,offline_tsc,sequential,sim,evaluation} | 智能体、离线、顺序/半离线、仿真器及评估配置 |
| data/raw_data/ | 路网、交通流和信号方案数据 |
| common/、generator/、dataset/ | 注册器与配置加载、特征生成器、数据集接口 |
| sequential/ | 顺序训练 agent、runtime、历史档案、恢复、验证和分析实现 |
| GOVERNANCE.md、EXPERIMENTS.md、EXPERIMENTS/、ledger/ | 实验治理规范、台账索引与各 campaign 明细、机读台账 |
| bench/ | 正式验收口径（BENCH.md + profiles） |
| dev/ | 一次性脚本唯一合法堆放处 + INDEX.md 登记 |
| docs/ | Plan、goal 及专项支持说明等阶段性参考文件 |
| scripts/ | 辅助脚本（如 run 配置比对） |
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
- **`tools/sumo_live_broadcast/`**：浏览器实时转播服务（libsumo 实时步进，SSE 收帧 + HTTP POST 控制）。支持运行时切换 FixedTime/MaxPressure/SOTL/DQN 快照/人工接管、相位卡片点击接管、变速与决策点等待确认；多路口场景（如 S5=sumohz4x4）下控制器逐路口决策，相位面板可通过下拉或点击地图路口切换。最小示例 `python -m tools.sumo_live_broadcast --scene S2 --port 8010`；常驻托管脚本 `supervise.sh` 遵循仓库 `*.sh` 约定不入 Git。回归测试 `tests/test_sumo_live_broadcast.py`。

## 在其他服务器复现

Git 只同步代码、配置与场景；trajectory、checkpoint、dataset index 与正式结果需另行迁移。完整资产清单、路径重定位、哈希验证与门禁顺序见 [复现指南·服务器迁移](docs/semi_offline_cross_algorithm_reproduction.md#8-新服务器迁移清单)。

## 开发约定

- 新实验优先通过 configs/ 管理参数，避免硬编码；四类入口不混用。
- 修改智能体或训练流程后，至少运行一个受影响仿真器的最小代表性实验，并记录仿真器、agent、网络、随机种子和完整命令；正式实验另记 Order/Archive/seed/manifest hash 与恢复链。
- 不提交 `data/output_data/`、checkpoint、缓存等可再生成产物；大规模数据用外部存储并在文档中给出校验信息。
- 实验相关的一次性脚本放 `dev/<topic>/` 并在 `dev/INDEX.md` 登记。

## 项目来源

本仓库代码基于 [DaRL-LibSignal/LibSignal](https://github.com/DaRL-LibSignal/LibSignal) 整理。原项目提供跨仿真器交通信号控制环境、多种传统与强化学习基线，以及 SUMO/CityFlow 数据转换能力。

相关 sim-to-real 工作：

- [UGAT：Uncertainty-aware Grounded Action Transformation](https://github.com/darl-libsignal/ugat)
- [PromptGAT：Prompt to Transfer](https://github.com/DaRL-LibSignal/PromptGAT)

使用代码、数据集或第三方仿真器时，请同时遵守对应上游项目及数据文件附带的许可证和引用要求。
