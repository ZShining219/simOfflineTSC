# simOfflineTSC

面向交通信号控制（Traffic Signal Control，TSC）的跨仿真器实验框架。本项目基于 LibSignal 整理，提供与 OpenAI Gym 风格兼容的交通环境、统一训练流程，以及传统控制和强化学习基线，可用于单路口与多路口场景实验。

## 主要功能

- 支持 CityFlow 与 SUMO 交通仿真器。
- 原有 Online/传统控制实验通过 `run.py` 组织训练、测试、日志与模型保存。
- 支持固定时制、SOTL、MaxPressure 等传统控制方法。
- 支持 DQN、FRAP、CoLight、PressLight、MPLight、MADDPG、MAGD、PPO 等强化学习方法。
- 包含合成路网、杭州、纽约、科隆等多种实验数据与配置。
- 提供 SUMO 与 CityFlow 路网、交通流格式转换工具。
- 提供与 Online 入口隔离的 Plan 2 纯 Offline Batch-DQN/CQL-DQN 训练、数据校验、断点恢复和结果汇总能力。

## 原项目能力与 Offline 扩展边界

本仓库保留原有 LibSignal Online/传统控制流程，并在其旁路新增 Plan 2 纯 Offline DQN。Offline 是独立扩展，不会替换或改变原有实验入口。两种训练模式不能混用：

| 项目 | 原项目 Online/传统 TSC | Plan 2 纯 Offline DQN 扩展 |
| --- | --- | --- |
| 正确入口 | `run.py` | `offline_run.py` |
| 配置目录 | `configs/tsc/` | `configs/offline_tsc/` |
| 支持算法 | 原有 DQN、传统控制及其他既有 agent | `batch_dqn`、`cql_dqn` |
| 训练数据 | 运行中与仿真器交互，并使用 Online replay | 只读使用已验收的 Plan 1 正式 trajectory |
| 仿真器用途 | 参与训练和评估 | 只在固定节点参与评估，不参与训练数据生成 |
| 训练输出 | `data/output_data/tsc/` | `data/output_data/offline_tsc/` |
| 额外依赖 | `requirements.txt` 及对应 agent/仿真器依赖 | 另加 `requirements-offline.txt` |

调用规则：

- 运行原有 Online DQN、FixedTime、MaxPressure 或其他原有 agent 时，只使用 `run.py`。
- 运行 Plan 2 Batch-DQN/CQL-DQN 时，只使用 `offline_run.py train`。
- 不要通过 `run.py --agent batch_dqn` 或 `run.py --agent cql_dqn` 启动 Offline。
- 不要通过 `offline_run.py` 启动原有 Online DQN 或传统控制实验。
- `offline_run.py` 训练期间不会向数据集追加 transition，也不会修改源 Plan 1 NPZ。

完整的 Offline 数据门禁、算法语义、恢复和汇总说明见 [docs/plan2_offline_support.md](docs/plan2_offline_support.md)。

## 项目结构

| 路径 | 说明 |
| --- | --- |
| run.py | 实验主入口和命令行参数定义 |
| offline_run.py | Plan 2 数据准备、校验和纯离线训练入口 |
| agent/ | 传统控制及强化学习智能体 |
| world/ | CityFlow、SUMO 等仿真器适配层 |
| trainer/ | 训练与评估流程 |
| task/ | 交通信号控制任务编排 |
| configs/tsc/ | 智能体和训练参数 |
| configs/offline_tsc/ | Plan 2 Offline DQN 参数；不影响 Online 配置 |
| configs/sim/ | 仿真器及路网配置 |
| data/raw_data/ | 路网、交通流和信号方案数据 |
| common/ | 注册器、配置加载、指标及格式转换工具 |
| generator/ | 状态、相位和车辆特征生成器 |
| dataset/ | 数据集接口 |
| docs/ | Plan、goal 及其他阶段性执行参考文件 |
| tools/traffic_flow_profile/ | SUMO 单场景车流需求评估、标准度量计算与固定子图报告工具 |

### docs 目录

docs/ 用于保存与用户讨论后形成的 plan、goal，以及其他非永久但需要在任务执行期间持续参考的文件。这些文件用于明确阶段目标、执行步骤、决策和进度；内容失效后应及时更新或清理，避免过期信息影响后续工作。

## 环境要求

原始项目基于以下环境开发：

- Linux（推荐）
- Python 3.9
- PyTorch 1.11.0
- CityFlow 或 SUMO

仓库中的 requirements.txt 仅包含核心 Python 依赖。PyTorch、仿真器及部分智能体所需的扩展库需要根据实际实验单独安装。

Plan 2 额外固定使用 `d3rlpy==2.0.4`。为避免改动当前 Torch/Gym/NumPy，进入 `colight` 环境后使用 `python -m pip install --no-deps -r requirements-offline.txt`。第三方来源与适配边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 安装

### 1. 克隆仓库

~~~bash
git clone https://github.com/ZShining219/simOfflineTSC.git
cd simOfflineTSC
python -m pip install -r requirements.txt
~~~

建议使用独立的 Python 3.9 虚拟环境。

### 2. 安装 PyTorch

请根据本机 CUDA 或 CPU 环境，从 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择合适的安装命令。原项目使用 PyTorch 1.11.0；若使用其他版本，请留意旧版 Gym 和图神经网络扩展的兼容性。

### 3. 配置 CityFlow

CityFlow 的编译和安装方式请参考 [CityFlow 官方文档](https://cityflow.readthedocs.io/en/latest/install.html)。从源码安装的基本流程如下：

~~~bash
git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow
python -m pip install .
python -c "import cityflow; print(cityflow.Engine)"
~~~

### 4. 配置 SUMO

请参考 [SUMO 官方安装文档](https://sumo.dlr.de/docs/Installing/index.html) 安装 SUMO，并设置环境变量：

~~~bash
export SUMO_HOME=/path/to/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
python -c "import sumolib, traci; print('SUMO Python API 可用')"
~~~

使用 SUMO 时，run.py 的 --interface 参数可以选择 libsumo（默认、速度较快）或 traci。

seed 的含义由入口明确区分：`run.py --seed` 是原有 Online 流程的 `training_seed`；`offline_run.py train --seed` 是 Plan 2 的 `offline_training_seed`。Offline 数据来源中的 Online seeds 另行记录为 `behavior_training_seeds`，不能与 Offline 重复 seed 混用。两种入口都不新增或管理 SUMO seed，也不向 SUMO 启动命令传递 seed；运行证据固定记录 `sumo_seed_mode=fixed_default`。因此，这些训练 seed 都不能解释为不同的 SUMO 微观交通随机实现。

### 5. 可选智能体依赖

部分智能体需要额外依赖。例如，PPO/PFRL 实现需要安装 pfrl：

~~~bash
python -m pip install pfrl
~~~

CoLight 的部分实现可能需要 PyTorch Geometric 及其扩展。请按 PyTorch 与 CUDA 版本参考 [PyTorch Geometric 安装文档](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。

## 原项目 Online/传统实验快速开始

原有 Online 和传统控制实验通过 `run.py` 启动：

~~~bash
python run.py
~~~

默认参数使用 CityFlow、DQN、cityflow1x1 网络和在线生成的数据集。也可以显式指定实验参数：

~~~bash
python run.py \
  --task tsc \
  --agent dqn \
  --world cityflow \
  --network cityflow1x1 \
  --dataset onfly \
  --prefix baseline \
  --seed 0 \
  --ngpu -1
~~~

SUMO 示例：

~~~bash
python run.py \
  --agent maxpressure \
  --world sumo \
  --network sumo1x1 \
  --interface libsumo \
  --prefix sumo-baseline \
  --seed 0
~~~

### `run.py` 常用参数（仅限原有 Online/传统流程）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| --task | tsc | 任务类型 |
| --agent | dqn | 控制器或智能体名称 |
| --world | cityflow | 仿真器，可选 cityflow、sumo |
| --network | cityflow1x1 | configs/sim/ 中的网络配置名称 |
| --dataset | onfly | 数据集类型 |
| --thread_num | 4 | CityFlow 仿真线程数 |
| --ngpu | -1 | GPU 编号，-1 表示不指定 GPU |
| --prefix | test | 本次实验输出标识 |
| --seed | 空 | 随机种子 |
| --interface | libsumo | SUMO 接口，可选 libsumo、traci |
| --delay_type | apx | 延误计算方式，可选近似值 apx 或真实值 real |

训练轮数、批量大小、学习率、测试频率和日志路径等参数位于 configs/tsc/*.yml；仿真路网、交通流文件和仿真参数位于 configs/sim/*.cfg。

## Plan 2 Offline 扩展快速开始

Plan 2 分三步执行：先从明确的 Plan 1 正式运行白名单生成只读数据索引，再校验 manifest，最后训练：

```bash
python offline_run.py prepare-plan2 --run-list plan1_formal_runs.csv --dataset-id plan2_formal_v1
python offline_run.py validate-dataset --manifest /path/to/manifest.json
python offline_run.py train --agent batch_dqn --network sumohz1x1 \
  --dataset-manifest /path/to/manifest.json --prefix p2_batch_seed1000 --seed 1000
```

`--seed` 在该入口中明确表示 `offline_training_seed`，正式取值为 1000～1004；源轨迹中的 0～4 始终记录为 `behavior_training_seeds`。更多命令、resume 和汇总格式见 [Plan 2 支持说明](docs/plan2_offline_support.md)。

## 输出结果与目录隔离

原有 Online/传统实验输出保存在：

~~~text
data/output_data/tsc/<world>_<agent>/<network>/<prefix>/
~~~

Plan 2 生成的只读数据索引保存在：

~~~text
data/output_data/offline_datasets/plan2/<dataset_id>/
~~~

Plan 2 Offline 训练输出保存在：

~~~text
data/output_data/offline_tsc/sumo_<batch_dqn|cql_dqn>/<network>/<prefix>/
~~~

Plan 2 汇总输出保存在：

~~~text
data/output_data/analysis/plan2/<analysis_id>/
~~~

这些目录彼此隔离。Offline 数据索引只引用 Plan 1 trajectory，不复制或修改源 NPZ；Offline checkpoint、指标和日志也不会写入原有 Online 运行目录。`data/output_data/` 已在 `.gitignore` 中忽略，避免提交大型实验产物。

## 格式转换

common/converter.py 提供 CityFlow 与 SUMO 路网、交通流格式转换逻辑。当前源码快照在 parse_args() 中启用了 SUMO 转 CityFlow 的参数，默认处理 cologne3 数据：

~~~bash
cd common
python converter.py --typ s2c
~~~

转换其他数据时，需要修改 common/converter.py 中 parse_args() 的默认输入与输出路径。CityFlow 转 SUMO 的参数模板也保留在该函数中，但默认处于注释状态；使用 --typ c2s 前应先启用并配置 or_cityflownet、sumonet、or_cityflowtraffic 和 sumotraffic 参数。

转换工具会依赖 SUMO Python API，并可能生成中间文件。正式实验前请检查输出路网、信号相位与交通流是否一致，建议先在数据副本上操作。

## 开发与复现实验建议

- 新实验优先通过 configs/ 管理参数，避免在代码中硬编码。
- 修改智能体或训练流程后，至少运行一个受影响仿真器的最小代表性实验。
- 记录仿真器、智能体、网络、随机种子和完整启动命令。
- 不要提交 data/output_data/、模型检查点、缓存或其他可再生成的大文件。
- 大规模数据建议使用 Git LFS 或外部数据存储，并在文档中提供下载和校验信息。

## 项目来源

本仓库代码基于 [DaRL-LibSignal/LibSignal](https://github.com/DaRL-LibSignal/LibSignal) 整理。原项目提供跨仿真器交通信号控制环境、多种传统与强化学习基线，以及 SUMO/CityFlow 数据转换能力。

相关 sim-to-real 工作：

- [UGAT：Uncertainty-aware Grounded Action Transformation](https://github.com/darl-libsignal/ugat)
- [PromptGAT：Prompt to Transfer](https://github.com/DaRL-LibSignal/PromptGAT)

使用代码、数据集或第三方仿真器时，请同时遵守对应上游项目及数据文件附带的许可证和引用要求。
