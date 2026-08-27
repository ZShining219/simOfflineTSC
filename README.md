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
- 提供基于 Plan 1 静态历史档案与顺序在线交互的 HA-SODQN 半离线实验链路；当前正式实现为 Independent DQN，跨算法支持边界和服务器迁移步骤见 [半离线实验总结与跨算法复现实用指南](docs/semi_offline_cross_algorithm_reproduction.md)。

## 实验模式与入口边界

本仓库在原有 LibSignal Online/传统控制流程旁路增加了纯 Offline、Sequential 和 HA-SODQN 半离线实验。四类入口拥有不同的数据来源、训练语义和输出结构，不能混用：

| 项目 | Online/传统 TSC | Plan 2 纯 Offline DQN | Plan 3/4 Sequential DQN | HA-SODQN 半离线 |
| --- | --- | --- | --- | --- |
| 正确入口 | `run.py` | `offline_run.py` | `sequential_run.py` | `sequential_run.py` 的 HA 子命令 |
| 主要配置 | `configs/tsc/` | `configs/offline_tsc/` | `configs/sequential/plan34*.yml` | `configs/sequential/ha_sodqn_b100.yml` |
| 当前支持 | 传统控制、DQN、PPO 等既有 agent | `batch_dqn`、`cql_dqn` | Independent DQN | Independent DQN + 静态历史混合采样 |
| 训练数据 | 仿真中在线生成 | 冻结的 Plan 1 trajectory | 四场景顺序在线交互 | 顺序在线交互 + Plan 1 静态档案 |
| 仿真器用途 | 训练和评估 | 仅固定节点评估 | 每个 stage 训练和冻结评估 | 每个 stage 训练和冻结评估 |
| 主要输出 | `data/output_data/tsc/` | `data/output_data/offline_tsc/` | `data/output_data/sequential/` | `data/output_data/ha_sodqn/` |
| 额外依赖 | 核心依赖和对应 agent/仿真器 | 另加 `requirements-offline.txt` | PyTorch、SUMO | PyTorch、SUMO 和已验收历史资产 |

调用规则：

- 运行原有 Online DQN、FixedTime、MaxPressure 或其他原有 agent 时，只使用 `run.py`。
- 运行 Plan 2 Batch-DQN/CQL-DQN 时，只使用 `offline_run.py train`。
- 运行顺序 DQN、CS-HR 或 HA-SODQN 时，只使用 `sequential_run.py` 对应的 manifest、launch、status、audit 和 analyze 子命令。
- 不要通过 `run.py --agent batch_dqn` 或 `run.py --agent cql_dqn` 启动 Offline。
- 不要通过 `offline_run.py` 启动原有 Online DQN 或传统控制实验。
- `offline_run.py` 训练期间不会向数据集追加 transition，也不会修改源 Plan 1 NPZ。
- 当前 HA-SODQN 不能通过修改 YAML 直接切换为 Double DQN 或 PPO；跨算法验证需要先增加算法 identity、训练 target/loss、checkpoint 和 evaluator 适配。

完整的 Offline 数据门禁、算法语义、恢复和汇总说明见 [Plan 2 支持说明](docs/plan2_offline_support.md)；半离线科学设计、结果边界、跨算法方案和服务器迁移步骤见 [半离线实验总结与跨算法复现实用指南](docs/semi_offline_cross_algorithm_reproduction.md)。

## 当前科学实验体系

项目围绕四个 SUMO 杭州单路口场景形成了由 Online 数据采集、纯 Offline 学习、顺序训练到半离线历史利用的实验链：

| 阶段 | 研究问题 | 当前证据状态 |
| --- | --- | --- |
| Plan 1 Online DQN | 建立单场景在线基线、trajectory 和可恢复初始化 | 四场景 × 5 seeds，共 20 个正式运行已验收 |
| Plan 2 Offline DQN | 固定数据下比较 Batch-DQN、CQL-DQN、数据阶段和留一场景迁移 | Plan 2A/2B/2C 共 160 个正式身份已验收，训练环境交互为 0 |
| Plan 3/4 Sequential DQN | 比较场景切换时 clear、FIFO 和 matched-wait replay 策略 | b100 的 60 个正式运行已完成；b400 历史批次不能据局部结果视为完整验收 |
| HA-SODQN | 检验 Plan 1 静态历史能否缓解顺序训练中的遗忘 | E0～E4 已完成；E4 为 4 orders × 5 seeds × 5 conditions，共 100 个身份 |

HA-SODQN 的 E4 配对结果显示：历史利用在配对均值上明显降低旧场景 average/worst forgetting 并提高 retention，同时带来小幅当前场景 normalized travel-time AULC 代价。P1C 只允许使用已完成场景的历史，是主因果设置；P1F 从 T1 即可访问当前和未来场景，只能作为非因果 full-history reference。现有结果只证明 Independent DQN 协议下的表现，不能直接外推到 Double DQN、PPO 或其他 TSC agent。

### 半离线正式场景

| 代称 | network | 数据目录 |
| --- | --- | --- |
| S1 | `sumohz1x1_config2` | `data/raw_data/hangzhou_1x1_qc-yn_18041608_1h/` |
| S2 | `sumohz1x1` | `data/raw_data/hangzhou_1x1_bc-tyc_18041610_1h/` |
| S3 | `sumohz1x1_config4` | `data/raw_data/hangzhou_1x1_sb-sx_18041607_1h/` |
| S4 | `sumohz1x1_config3` | `data/raw_data/hangzhou_1x1_kn-hz_18041608_1h/` |

四个场景均为单个受控路口，使用 16 维 observation（8 维进入车道 count + 8 维当前相位 one-hot）和 8 个绿灯动作。正式顺序为：

| Order | 场景顺序 |
| --- | --- |
| O1 | S4 → S1 → S3 → S2 |
| O2 | S2 → S3 → S1 → S4 |
| O3 | S1 → S4 → S2 → S3 |
| O4 | S3 → S2 → S4 → S1 |

## 项目结构

| 路径 | 说明 |
| --- | --- |
| run.py | 实验主入口和命令行参数定义 |
| offline_run.py | Plan 2 数据准备、校验和纯离线训练入口 |
| sequential_run.py | Sequential DQN、CS-HR 和 HA-SODQN manifest/运行/审计入口 |
| agent/ | 传统控制及强化学习智能体 |
| world/ | CityFlow、SUMO 等仿真器适配层 |
| trainer/ | 训练与评估流程 |
| task/ | 交通信号控制任务编排 |
| configs/tsc/ | 智能体和训练参数 |
| configs/offline_tsc/ | Plan 2 Offline DQN 参数；不影响 Online 配置 |
| configs/sequential/ | 顺序训练、CS-HR 和 HA-SODQN 冻结实验配置 |
| configs/sim/ | 仿真器及路网配置 |
| data/raw_data/ | 路网、交通流和信号方案数据 |
| common/ | 注册器、配置加载、指标及格式转换工具 |
| generator/ | 状态、相位和车辆特征生成器 |
| dataset/ | 数据集接口 |
| sequential/ | 顺序训练 agent、runtime、历史档案、OWP、恢复、验证和分析实现 |
| docs/ | Plan、goal 及其他阶段性执行参考文件 |
| tools/traffic_flow_profile/ | SUMO 单场景车流需求评估、标准度量计算与固定子图报告工具 |
| tools/xiasha_sumo/ | xiasha1*1 语义事件到 SUMO 逐车、flow 和信号路网的转换工具 |
| tools/sumo_html_comparison.py | 将已记录的 SUMO 决策回放为可交互的单 HTML 多控制器对比页面 |
| tools/sumo_gui_comparison.py | 基于 SUMO-GUI/TraCI 的截图或 GIF 回放工具 |
| tests/test_sumo_html_comparison.py | SUMO HTML 回放模块的最小回归测试 |

Xiasha SUMO 转换最小示例（产物位于 `data/raw_data/xiasha1*1/`）：

```bash
/home/dev/miniforge3/envs/colight/bin/python -m tools.xiasha_sumo --all
```

逐车需求使用 `xiasha1_sumo_vehicles.rou.xml`，聚合需求使用名称不同的 `xiasha1_sumo_flows.rou.xml`；两者及信号路网、附加文件和 `.sumocfg` 均由工具生成。详细参数与信号规则见 [tools/xiasha_sumo/README.md](tools/xiasha_sumo/README.md)。

### SUMO HTML 回放与对比模块

`tools/sumo_html_comparison.py` 用于审查已有 SUMO 决策记录，不进行训练，也不修改源记录、路网或仿真配置。工具通过无界面 SUMO/libsumo 按原始控制间隔重放动作，将路网、信号灯、车辆位置、速度、停车车辆、占有率和累计 throughput 等状态嵌入一个自包含 HTML 文件中；浏览器端无需再次连接 SUMO-GUI 即可回放。

页面支持：

- 同一 S1--S4 场景下的多控制器同步回放；包含四组方法时，顶部窗口固定按 `online DQN`、`fixedtime`、`DHOA`、`CONT DQN` 顺序显示。
- 单路口聚焦、路口细节覆盖层、车辆正常行驶与堵塞/停止的颜色区分，以及时间轴、播放、逐步推进、缩放和拖动。
- 将新的冻结验证记录追加到已有页面；追加模式只回放新增方法，已有方法的状态数组保持不变。省略显式 evaluation seed 时可使用正式记录对应的 `fixed_default` SUMO 随机实现。

仅回放 FixedTime 的最小示例：

```bash
python tools/sumo_html_comparison.py \
  --scene S2 \
  --fixedtime-only \
  --evaluation-seed 10000 \
  --start 0 --end 3600 --sample-every 1 \
  --output /tmp/s2_fixedtime_html
```

多方法对比时，为每个方法提供对应的 `decisions.jsonl`：

```bash
python tools/sumo_html_comparison.py \
  --scene S2 \
  --online-dqn /path/to/online_dqn/decisions.jsonl \
  --fixedtime /path/to/fixedtime/decisions.jsonl \
  --hadhoa /path/to/hadhoa/decisions.jsonl \
  --evaluation-seed 10000 \
  --start 0 --end 3600 --sample-every 1 \
  --output /tmp/s2_comparison_html
```

在已有页面上追加冻结 CONT 记录：

```bash
python tools/sumo_html_comparison.py \
  --append-to /tmp/s2_comparison_html \
  --append-method cont_o2_final \
  --append-input /path/to/cont_o2/decisions.jsonl \
  --output /tmp/s2_cont_o2_html_fixed_default
```

如需通过本机地址查看大页面，可在输出目录启动临时静态服务：

```bash
python -m http.server 8765 --bind 127.0.0.1 \
  --directory /tmp/s2_cont_o2_html_fixed_default
```

然后访问 `http://127.0.0.1:8765/sumo_html_comparison.html`，结束服务时在终端按 `Ctrl+C`。HTML、`comparison_states.json` 和 `manifest.json` 是可再生成的实验产物，建议输出到 `/tmp` 或其他实验产物目录，不提交到 Git。

该模块的源码 `tools/sumo_html_comparison.py`、辅助回放实现 `tools/sumo_gui_comparison.py` 和测试 `tests/test_sumo_html_comparison.py` 已纳入 Git 跟踪。当前工作区的修改仍需通过正常的 `git diff` 检查后再提交；页面生成物不作为模块源码版本管理对象。

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

seed 的含义由入口明确区分：`run.py --seed` 是原有 Online 流程的 `training_seed`；`offline_run.py train --seed` 是 Plan 2 的 base/offline training seed。Offline 默认显式派生 `model_init_seed=base`、`dataset_sampler_seed=base+10000`、`evaluation_seed=base+20000`，并分别写入 metadata/checkpoint；也可通过对应 CLI 参数覆盖。源轨迹中的 Online seeds 始终另记为 `behavior_training_seeds`。SUMO 默认仍采用固定默认随机实现并记录 `sumo_seed=null`、`sumo_seed_mode=fixed_default`；只有显式传入 `--sumo-seed` 才会加入 SUMO 命令。

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
python offline_run.py prepare-plan2 --run-list plan1_formal_runs.csv \
  --dataset-id plan2_formal_v2 --source-root /data/plan1
python offline_run.py validate-dataset --manifest /path/to/manifest.json \
  --source-root /data/plan1
python offline_run.py train --agent batch_dqn --network sumohz1x1 \
  --dataset-manifest /path/to/manifest.json --prefix p2_batch_seed1000 --seed 1000
```

当前 manifest schema 为 v2，保存 source root ID、相对路径、原绝对路径、SHA-256、builder/feature/trajectory schema 和场景语义哈希；旧 v1 manifest 会被明确拒绝，必须用新 dataset ID 重建。`--source-root` 可在数据移动或 mount point 变化后重定位。更多命令、resume、I/O 基准和汇总格式见 [Plan 2 支持说明](docs/plan2_offline_support.md)。

## Sequential 与半离线实验快速开始

`sequential_run.py` 使用 manifest 管理顺序实验，不接受 `run.py` 风格的 `--agent/--network` 直接启动方式。先查看当前 checkout 支持的子命令：

```bash
python sequential_run.py --help
```

### Sequential DQN

已有 Plan 1 正式运行目录和白名单时，可先导入对应 parent checkpoint 并生成 parent catalog，再构建和验证顺序实验计划：

```bash
python sequential_run.py import-parents \
  --whitelist data/output_data/analysis/plan1/p1_formal_20_trajectory_whitelist_20260722.csv \
  --output-dir data/output_data/sequential/plan34_engineering \
  --config configs/sequential/plan34_b100.yml

python sequential_run.py build-plan \
  --parent-catalog data/output_data/sequential/plan34_engineering/parent_catalog.json \
  --config configs/sequential/plan34_b100.yml \
  --output data/output_data/sequential/plan34_engineering/formal_manifest.json

python sequential_run.py validate \
  --plan data/output_data/sequential/plan34_engineering/formal_manifest.json
```

正式 launch 会产生大规模 SUMO 训练并要求显式授权。执行前应先使用 pilot、检查磁盘和并发预算，并确认 manifest 冻结的 Git commit 与当前 checkout 一致。

### HA-SODQN

HA-SODQN 还依赖 Plan 1 episode-0 checkpoint、Plan 1 trajectory、Plan 2 Q1 schema-v2 数据集、behavior-seed audit、archive root manifest 和 initial-state catalog。资产就绪后先构建 smoke：

```bash
python sequential_run.py build-ha-plan \
  --stage smoke \
  --config configs/sequential/ha_sodqn_b100.yml \
  --output data/output_data/ha_sodqn/engineering/smoke_manifest.json

python sequential_run.py validate-ha-plan \
  --plan data/output_data/ha_sodqn/engineering/smoke_manifest.json

python sequential_run.py launch \
  --manifest data/output_data/ha_sodqn/engineering/smoke_manifest.json \
  --output-root data/output_data/ha_sodqn/smoke \
  --max-child 4

python sequential_run.py audit-ha \
  --plan data/output_data/ha_sodqn/engineering/smoke_manifest.json \
  --output-root data/output_data/ha_sodqn/smoke \
  --output data/output_data/ha_sodqn/engineering/smoke_audit.json
```

冻结配置默认包含 P1C/P1F、DHOA/RAND/COV/CQ/CQA、R25/R50/R75、O1～O4 和 seeds 0～4。不要在缺少历史资产、未通过 smoke/完整单运行门禁或 checkout 与 manifest commit 不一致时启动正式矩阵。完整资产重建顺序和跨算法建议见 [半离线复现指南](docs/semi_offline_cross_algorithm_reproduction.md)。

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

Sequential DQN 输出保存在：

~~~text
data/output_data/sequential/<experiment_id>/
~~~

HA-SODQN 的工程 manifest、正式运行和分析通常分别保存在：

~~~text
data/output_data/ha_sodqn/engineering_<date>/
data/output_data/ha_sodqn/<experiment_id>/
data/output_data/ha_sodqn/analysis_<commit>/<stage>/
~~~

这些目录彼此隔离。Offline 数据索引和 HA archive 只引用 Plan 1 trajectory，不复制或修改源 NPZ；Offline/Sequential/HA checkpoint、指标和日志也不会写入原有 Online 运行目录。`data/output_data/`、根目录历史 `output_data/`、`analysis/` 和 `tmp/` 均已在 `.gitignore` 中忽略，避免提交大型实验产物。

## 在其他服务器复现

Git 可以统一代码、配置和四个正式 SUMO 场景，但不会同步 trajectory、checkpoint、dataset index 或正式实验结果。推荐流程为：

1. 在目标服务器 clone 当前实验分支并记录 commit SHA；
2. 安装并记录 Python、PyTorch、SUMO、CUDA/driver 版本；
3. 通过 `rsync`、对象存储或共享文件系统迁移 Plan 1 source runs，不把大型资产提交到 Git；
4. 若 checkout 绝对路径变化，重建目标服务器白名单、Plan 2 schema-v2 index、initial-state catalog、behavior audit 和 archive manifest；
5. 先运行 CLI/config 检查和 smoke，再运行完整 100×4 单 run 门禁；
6. 所有正式 manifest 记录 Git commit、配置 SHA、数据 SHA、Order、seed 和 attempt/resume lineage。

仅执行 `git pull` 可以得到公共代码和场景设施，但不足以直接运行 HA-SODQN。所需资产清单、路径重定位命令、哈希验证和跨算法正式矩阵见 [服务器迁移章节](docs/semi_offline_cross_algorithm_reproduction.md#8-新服务器迁移清单)。

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
- Online、Offline、Sequential 和 HA-SODQN 必须使用各自入口，不能用相似参数名替代实验语义。
- 修改智能体或训练流程后，至少运行一个受影响仿真器的最小代表性实验。
- 记录仿真器、智能体、网络、随机种子和完整启动命令。
- 顺序和半离线正式实验还应记录 Order、Archive、Method、Ratio、manifest/config/data hash 和恢复链。
- 不要提交 `data/output_data/`、`output_data/`、`analysis/`、`tmp/`、模型检查点、缓存或其他可再生成的大文件。
- 大规模数据建议使用 Git LFS 或外部数据存储，并在文档中提供下载和校验信息。

## 项目来源

本仓库代码基于 [DaRL-LibSignal/LibSignal](https://github.com/DaRL-LibSignal/LibSignal) 整理。原项目提供跨仿真器交通信号控制环境、多种传统与强化学习基线，以及 SUMO/CityFlow 数据转换能力。

相关 sim-to-real 工作：

- [UGAT：Uncertainty-aware Grounded Action Transformation](https://github.com/darl-libsignal/ugat)
- [PromptGAT：Prompt to Transfer](https://github.com/DaRL-LibSignal/PromptGAT)

使用代码、数据集或第三方仿真器时，请同时遵守对应上游项目及数据文件附带的许可证和引用要求。
