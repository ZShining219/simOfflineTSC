# 项目工具入口与维护说明

本文维护近期新增的论文奖励训练入口、训练看板与飞书 Wiki 工具。所有命令从仓库根目录执行，项目运行使用 `colight` 环境。项目总览与 FRAP/CoLight 依赖、原奖励训练接入验证见 [项目 README](../README.md#5-可选智能体依赖)。

## 入口选择

| 用途 | 入口 | 说明 |
| --- | --- | --- |
| 原有在线训练与传统控制 | `python run.py` | 原模型身份与奖励；既有恢复能力仍走此入口 |
| 三模型专属奖励及监控 | `python -m tools.run_paper_baseline` | 仅 `paper_frap/paper_presslight/paper_colight`，当前仅 SUMO |
| 查看训练曲线、重建导出 | `python -m tools.training_dashboard serve/render` | 读取奖励训练入口生成的日志；render 写回该运行的监控导出文件 |
| 事件机制验证与五条件对比 | `python -m world.sumo_events` / `python -m world.sumo_events.compare` | [事件模块说明](../world/sumo_events/README.md)，不是强化学习训练入口 |
| 飞书知识库操作 | `python tools/feishu_wiki.py` | 独立工具，不在训练期间自动同步 |

已有工具继续使用各自说明：[Xiasha 转换](xiasha_sumo/README.md)、[实验绘图](experiment_plotting/README.md)、[SUMO HTML 回放](../README.md#sumo-html-回放与对比模块)。本页不替代其使用契约。

## 模型专属奖励与训练看板

2026-09-20 新增显式入口 `python -m tools.run_paper_baseline`。它复用现有 Runner、训练循环、评估隔离与标准检查点，通过注册器选择 `paper_frap`、`paper_presslight`、`paper_colight`。旧 `frap/presslight/colight` 及其奖励不变；新入口是选定作者代码奖励的 SUMO 适配版，不代表算法与原论文已经完整对齐。历史配置保留旧 World 行为；新实验通过 `paper_p0.yml` 显式启用修正后的信号控制，见 [P0 工程门禁](#p0-工程门禁)。短运行只验证工程链路。

### 奖励定义与实验身份

奖励以独立 YAML、不可变实例及哈希绑定，避免不同模型互相调用：

| 新模型 | 专属配置 | 写入 replay 的每秒奖励 | 统计范围 |
| --- | --- | --- | --- |
| `paper_frap` | [frap.yml](../configs/rewards/frap.yml) | `-0.25 × Q_in / 20` | 参与相位竞争的直行/左转车道；hz4x4 为八条，不包含常开放右转 |
| `paper_presslight` | [presslight.yml](../configs/rewards/presslight.yml) | `-0.25 × abs(Q_in − Q_out) / 20` | 所有进口、出口车道的当前排队总数；abs 在求和差之后 |
| `paper_colight` | [colight.yml](../configs/rewards/colight.yml) | `-0.25 × Q_in / 20` | 每个路口全部进口车道；局部奖励向量不改成全网统一奖励 |

这里 `Q` 是完整车道内当前速度 `<0.1 m/s` 的普通车辆数，不读取事件真值、未来信息、200 米观测裁剪或历史等待缓存；人工障碍车排除。FRAP 的共用右转车道限制只约束 FRAP，不阻止另外两种模型使用自己的奖励。每秒先求各自统计量和符号/缩放，现有 trainer 再按动作区间求一次均值；TD 阶段不再重复除以 20。

以上 `-0.25` 和 `/20` 来自明确记录的作者代码分支，不是为了统一三种方法而任意设定。FRAP 参照作者 `anon_env.py` 分支；PressLight 参照 `runexp.py` 的 `mode=3` 与 `anon_env.py` 的排队压力绝对值；CoLight 参照其排队奖励与 `CoLight_agent.py` 的 TD 缩放。各 YAML 保留源仓库、固定提交和适配说明。论文文字、作者其他分支、当前 SUMO 统计可能不同，因此不混称为唯一的“原论文奖励”。

实现分工为 [common/paper_rewards.py](../common/paper_rewards.py) 的配置/公式校验、[world/paper_rewards.py](../world/paper_rewards.py) 的 SUMO 统计、[agent/paper_baselines.py](../agent/paper_baselines.py) 的模型专属奖励和 [trainer/paper_trainer.py](../trainer/paper_trainer.py) 的显式接入。模型错用其他模型配置、未知字段、错误统计范围直接报错；初始化后不再动态读取全局配置。运行生成 `reward_contract.json`，记录原始权重、归一化、车道列表、源码哈希和网络身份；标准检查点保存相同契约哈希，拒绝加载不同奖励/旧无契约检查点。改变系数会改变哈希，必须作为新实验身份处理，不能覆盖旧结果。

### 实现分工与扩展边界

| 文件 | 职责与设计原因 |
| --- | --- |
| [run_paper_baseline.py](run_paper_baseline.py) | 显式导入新 agent、注册 PaperTrainer，复用 run.py 的参数和 Runner；注册替换只发生在该启动进程内 |
| [common/paper_rewards.py](../common/paper_rewards.py) | 校验模型所有权、字段和代数，构造不可变 RewardProfile，避免运行中全局配置串用 |
| [world/paper_rewards.py](../world/paper_rewards.py) | SUMO 当前速度/车道统计；缓存键包含路口实例、仿真时刻和速度阈值，避免 reset 后复用上一回合快照 |
| [agent/paper_baselines.py](../agent/paper_baselines.py) | 继承原模型，仅替换 get_reward；FRAP/PressLight 为局部策略，CoLight 保留每路口奖励向量 |
| [trainer/paper_trainer.py](../trainer/paper_trainer.py) | 安装固定事件日程、写奖励契约、绑定检查点、镜像回合记录并采集决策/优化日志 |
| [utils/training_monitor.py](../utils/training_monitor.py) | TensorBoard、JSONL/CSV 与静态图导出；保留训练/评估类型及奖励聚合口径 |
| [training_dashboard.py](training_dashboard.py) | 单独启动本机 TensorBoard 或重建导出，不负责训练调度 |

当前数据路径为：SUMO 当前普通车辆 → 模型专属车道统计 → 奖励符号和缩放 → 动作区间平均 → replay/TD；训练记录另送入监控。事件报告只用于本入口的活动报告计数，**这些 paper_* 策略尚未使用文本或实体关联矩阵作为输入**。

`profile_hash` 标识奖励配置；`reward_contract.json` 中的 `hash` 还绑定路网、实际车道、动作区间和所列源码哈希。检查点按后者拒绝错配。这是奖励接入契约，不是完整实验冻结清单：并未覆盖全部 trainer、监控、依赖及事件日程文件。正式实验仍需保存完整配置、日程、种子与环境身份。

配置职责：`configs/tsc/paper_*.yml` 选择模型及专属奖励文件，`configs/rewards/*.yml` 保存公式和作者代码出处，实验 overlay 调整训练预算、SUMO 种子及 `trainer.event_schedule`。历史配置 [paper_monitor_smoke.yml](../configs/tsc/paper_monitor_smoke.yml) **包含事件**；复验历史正常短运行可在同一入口使用不含事件的 [hz4x4_paper_smoke.yml](../configs/tsc/hz4x4_paper_smoke.yml)，例如：

```bash
python -m tools.run_paper_baseline -w sumo -a paper_frap -n hz4x4 \
  --seed 7 --interface libsumo --prefix paper_frap_normal_smoke_local \
  --experiment-config configs/tsc/hz4x4_paper_smoke.yml
```

上述两个历史短配置都显式设定 `command.sumo_seed: 7`；命令中的 `--seed 7` 设置训练种子。更换种子时需分别设定。固定日程在训练和评估 reset 后重复，尚无独立的随机训练/留出评估事件分布。正常条件也安装空日程，以保持 SUMO 选项一致。

新增模型奖励时需显式增加所有权/统计约束、新注册名和 YAML，再验证实际 replay 奖励与检查点错配拒绝。修改物理或信号执行行为时，先验证真实相位轨迹和事件边界，再重建对应结果身份；不要只更新公式或图表便沿用旧验证结论。

### 看板选择与启动

看板采用 GitHub 开源项目 **[TensorBoard](https://github.com/tensorflow/tensorboard)**（Apache-2.0），本地保存数据，不需要云账号或 TensorFlow 训练框架。调研同时比较了 [Aim](https://github.com/aimhubio/aim) 和 [MLflow](https://github.com/mlflow/mlflow)：本阶段优先解决实时曲线、持久日志和图片导出，选用 TensorBoard；暂不增加模型注册与集中实验追踪服务。

以下命令复验历史奖励/看板接入；新事件研究使用下文 P0 配置。在当前 `colight` 环境安装、启动和训练：

```bash
python -m pip install -r requirements-dashboard.txt

# 保持这个进程运行；看板与训练进程独立。
python -m tools.training_dashboard serve --logdir data/output_data/tsc --port 6006

# 在另一个终端运行，prefix 必须未被使用；可将 paper_frap 换成另外两个模型。
python -m tools.run_paper_baseline -w sumo -a paper_frap -n hz4x4 \
  --seed 7 --interface libsumo --prefix paper_frap_reward_smoke \
  --experiment-config configs/tsc/paper_monitor_smoke.yml
```

浏览器打开 `http://127.0.0.1:6006`。如果训练在远端机器，使用 `ssh -L 6006:127.0.0.1:6006 用户@训练机器` 后在本机打开同一地址。默认只绑定本机地址。训练看板的 Scalars 页查看逐决策奖励、当前停止车辆、活动报告数量，以及每次更新的 TD loss；episode 下按 TRAIN/EVALUATION/FINAL_EVALUATION 分开显示奖励、行程耗时、队列、吞吐量、epsilon、经验池和耗时。Text 页包含运行配置、奖励契约与运行状态。不同奖励版本用不同 tag，跨模型效果应比较交通指标，不能比较 native reward 的绝对大小。TensorBoard 平滑只用于显示，原始数据始终保存。

### 产物、指标与中断处理

每次运行自动产生：

```text
<run>/reward_contract.json          # 奖励归属、公式、作用车道、哈希及限制
<run>/tensorboard/events.*         # 实时日志，停止训练后仍可由看板读取
<run>/monitor/episodes.csv         # 回合记录（含训练/评估类型）
<run>/monitor/decisions.csv         # 训练期间逐决策奖励与事件/交通变化
<run>/monitor/updates.csv           # 每次优化后的 loss 与累计梯度更新数
<run>/monitor/*.jsonl              # 完整原始记录，用于恢复导出
<run>/monitor/episode_curves.png    # 同时生成 SVG
<run>/monitor/decision_curves.png   # 同时生成 SVG
<run>/monitor/status.json          # running/completed/interrupted/failed
```

CSV 每条记录刷新；每回合结束以及正常完成、Ctrl+C、SIGTERM 时自动导出图。手动停止在旧运行状态协议中记录为失败/退出码 130，在监控状态中明确标记 interrupted，不冒充完成。停止看板进程不停止训练。对于 SIGKILL/断电无法执行退出清理的情况，保留已完成记录；status.json 可能仍显示 running，不能代替进程存活检查。以下命令只重建导出，不修复运行状态或恢复训练，可重建 CSV 与图（忽略末尾未写完的 JSONL 行）：

```bash
python -m tools.training_dashboard render --run <run目录>
```

旧单次 smoke 运行保留固定日程；新清单入口支持逐回合计划、独立验证及恢复，见下文“批量实验的统一入口”。旧版无计划运行的监控时间线不自动迁移。固定 `event_schedule` 下，正常对照省略此项即安装空日程。回合队列沿用原有观测指标并明确标记为 legacy metric；逐决策 stopped_vehicles 在所有受控路口进口/出口外部车道的去重集合上统计即时停止车辆（完整车道长度，不包含路口内部连接车道），两者不能混作同一口径。

奖励聚合也明确区分：旧结构化日志的 `reward_mean` 实际是“各路口奖励之和，再按决策次数平均”；监控保留该字段并增加真正的 `reward_per_node_mean`。TensorBoard 分别标为 `network_sum_mean` 和 `node_mean`，静态回合图使用后者，与逐决策的路口平均奖励保持同一尺度。它们都不改变写入 replay 的局部奖励。

### 历史奖励与监控验证

2026-09-20 的已有记录：SUMO 1.27.1/libsumo、hz4x4、训练与仿真种子均为 7，三种模型各训练 2 回合、每回合 300 秒、10 秒决策、batch_size=4。以下通过数量来自此前验证，本次文档维护未重新运行训练。

针对性验证见 [test_paper_rewards_monitor.py](../tests/test_paper_rewards_monitor.py)：奖励符号/归一化、模型所有权、不可变配置、检查点错配拒绝、真实 SUMO 三模型训练、TensorBoard 数据读取、CSV/PNG/SVG 导出、回合/决策奖励聚合一致性和真实停止信号，12 项通过。新奖励工程证据在 `data/output_data/sumo_events/reward_monitor_validation_20260920_v4/`，不属于正式算法效果结果。

在仓库根目录的 `colight` 环境复验：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -s tests/test_paper_rewards_monitor.py
```

默认使用 pytest 临时目录。若指定 `--basetemp`，必须使用新的专用目录，pytest 会清理该目录。缺少可选依赖时部分测试会 skip；skip 不构成训练验证通过。

## P0 工程门禁

本轮仅执行工程验证；P1 pilot/事件强度标定和 P2 正式实验未启动。P0 新增 [paper_p0.yml](../configs/tsc/paper_p0.yml)，使用相同 hz4x4、固定日程、两回合各 300 秒、训练/SUMO 种子 7，并显式选择如下信号控制：

```yaml
trainer:
  signal_control:
    version: sumo-green-yellow-v1
    yellow_seconds: 5
```

[world/sumo_signal_control.py](../world/sumo_signal_control.py) 在 reset 后给当前 World 的路口安装控制逻辑，保留历史 `world_sumo.py`、FixedTimeAgent 和事件/规则映射 v1 源码。旧逻辑的绿灯计时没有在切换后重置，且生成的“黄灯”把退出绿灯的连接直接置红。新逻辑让退出绿灯的连接实际显示 `y`，保持完整 5 秒再放行目标相位，禁止 SUMO 自主推进相位；持续请求同相位不会反复进入黄灯。FixedTime 的 `t_fixed=20` 在新协议下表示 20 秒实际绿灯，切换另加 5 秒黄灯。

策略的 `current_phase` 表示已接受的目标绿灯编号，黄灯期间也保持在合法动作空间；实际灯色由 SUMO 接口和 `signal_is_yellow` 表示。黄灯目标锁定至清空结束，不被期间的新请求缩短或替换。P0 采用 10 秒决策、1 秒仿真步，所有决策边界均检查逻辑相位与实际绿灯一致。非法动作直接拒绝，不通过取模修正。验证范围为 hz4x4 同构八相位、CPU；未做任意异构路网或 GPU 验证。

当前 `TSCEnv` 将固定时长交通任务作为继续型任务处理：`done` 始终为 false，三模型在 horizon 的最后一个真实下一状态上继续 bootstrap；没有把下一回合 reset 状态接到上一条 transition。该边界是本轮保留的项目适配约定，不宣称与每篇原论文完全一致。

PressLight 核查了 16 个独立策略、每路口 24 个进/出车道计数及 8 维相位 one-hot。观测仍沿用 World 的 200 米相关统计，专属奖励则是完整进/出车道的当前停止车辆总数之差取绝对值，再作既定缩放；两种范围不能混称。FRAP 相位竞争映射和 CoLight 图/样本隔离验证继续保留。

修正后的运行额外生成 `experiment_contract.json`，归档实际日程内容及哈希、训练和 SUMO 种子、路网/车流/SUMO 配置、关键源码、模型/训练配置、Python 与关键依赖版本。固定日程明确记录 `events: null` 和 `fixed_schedule_no_rng`，不能把 SUMO 种子称为事件种子。标准检查点保存该契约哈希，拒绝加载不同日程、种子、信号协议或所列资产的检查点。这描述的是固定日程 P0 契约；逐回合计划和独立评估现由下文的清单入口接入，不能混用两种运行身份。

在仓库根目录、`colight` 环境执行新配置（prefix 必须未使用）：

```bash
python -m tools.run_paper_baseline -w sumo -a paper_presslight -n hz4x4 \
  --seed 7 --interface libsumo --prefix paper_presslight_p0_local \
  --experiment-config configs/tsc/paper_p0.yml
```

更换 `-a` 可验证另外两个 paper_* 模型。正常对照复制同一 overlay 并只将 `trainer.event_schedule` 设为 `null`，不要切回未启用 signal_control 的历史 smoke 配置。

复验 P0 及相关回归：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/test_sumo_phase_control.py tests/test_paper_p0.py \
  tests/test_paper_agents_sumo.py tests/test_paper_rewards_monitor.py \
  tests/test_sumo_events.py tests/test_sumo_event_comparison.py \
  tests/test_text_grounding.py tests/test_text_grounding_sumo.py \
  reproduction/tarl_tsc/tests/test_runtime_v21.py \
  reproduction/tarl_tsc/tests/test_semantic_v21.py \
  reproduction/tarl_tsc/tests/test_sampling_v21.py
```

最终回归 **112 passed，0 failed，0 skipped**（244.27 秒，13 条依赖弃用警告）。证据保存在 `data/output_data/paper_robustness/p0_20260920/`：测试日志/XML、各案例 `readiness.json`、`validation.json`、`p0_validation.json`、`reproducibility.json` 和运行配置/检查点。完整执行命令及最终检查结果见该目录的 `summary.json`。初次调试失败日志保留，不计入最终通过结果。新克隆需自行复验，生成的产物不纳入 Git。

P0 检查精确灯色/时长、所有相位对、两种 SUMO 接口、reset 重放、三模型正常/事件训练、梯度与参数更新、评估隔离、检查点恢复/错配拒绝、普通车辆守恒，以及同种子训练重复的模型/优化器/replay/奖励/loss 一致性。完整 3600 秒物理重放使用种子 7，种子 17 另做 300 秒重放，各重复两次。种子 7 的 2983 辆计划需求中，2933 辆已出发、50 辆仍待插入；已出发车辆为 2476 辆到达加 457 辆在途。守恒通过不等于全部需求已出发或拥堵已恢复，P1 必须保留待插入需求和未完成车辆统计。它不校准正式事件强度，不证明策略收敛或鲁棒收益；既有弱局部阻塞、降雨限速代理、无强制绕行及实际障碍放置容差仍需在 P1 设计中处理。

## 批量实验的统一入口

`python -m tools.run_paper_baseline` 统一提供 `config`、`plan`、`preflight`、`launch`、`status`、`resume`。原单次训练参数继续支持；`worker` 是清单调用的内部子命令，不需要为 N/E、模型或 seed 创建启动脚本。

两个配置用途不同：

| 配置 | 用途 |
| --- | --- |
| [paper_experiment_smoke.yml](../configs/tsc/paper_experiment_smoke.yml) | 整条并行链路的工程短测：1 个 seed、N/E 两组，各 2 回合 × 60 秒 |
| [colight_event150.yml](../configs/tsc/colight_event150.yml) | 150 回合研究候选设计：5 seeds × N/E；开发预检已执行，当前 not_ready |

这两份是**批量实验配置，传给 plan --config**，不能当作旧入口的 `--experiment-config` overlay。plan 会在输出目录生成每个运行的具体 overlay、逐回合计划和任务清单。

### 回合配置的统一维护

[paper_experiment_base.yml](../configs/tsc/paper_experiment_base.yml) 保存模型、优化器、信号、事件、种子派生、目标分组和开发预检的共同参数；两份完整 profile 通过 `extends: paper_experiment_base.yml` 继承，只覆盖各自预算与必要参数。基础片段不能单独启动实验。继承路径相对当前 YAML，字典递归合并、列表整体替换，循环继承直接拒绝。开发代码复用 `common.paper_experiment.load_config(path)`；不要自行 `yaml.safe_load` 完整 profile 后跳过继承与校验。

```bash
python -m tools.run_paper_baseline config --config configs/tsc/colight_event150.yml
```

该命令不启动仿真，输出展开后的配置、配置哈希、检查点列表和预算。研究配置每个运行为 150 回合 × 3600 秒、每 10 秒决策，共 54000 次决策；10 个 N/E 运行合计训练 1500、过程验证 1240、最终评估 1500 回合，总计 4240。检查点由同一函数供预算展示和 plan 生成使用，包含第 0 回合、每 5 回合、指定测试回合与最终回合。

`events.begin: {start: 600, stop: 1500, step: 10}` 为包含首尾的整数时间网格。`seed_policy` 明确维护训练事件种子基数、训练 SUMO 种子基数与步长，以及验证/测试计划种子；`events.validation_sumo_seeds/test_sumo_seeds` 维护评估仿真种子，`seeds` 维护训练种子。步长须覆盖全部训练回合，N/E 配对共享对应 SUMO 种子。`target_groups` 显式维护互不重叠的训练/验证/测试路口；目标存在性在生成路网计划时校验。更改参数后建立新输出目录和新清单，不修改冻结计划继续运行。

在仓库根目录、`colight` 环境运行短测：

```bash
python -m tools.run_paper_baseline plan \
  --config configs/tsc/paper_experiment_smoke.yml \
  --output data/output_data/paper_robustness/runner_smoke_local

python -m tools.run_paper_baseline launch \
  --manifest data/output_data/paper_robustness/runner_smoke_local/manifest.json \
  --train-workers 2 --eval-workers 2 --test-workers 4

python -m tools.run_paper_baseline status \
  --manifest data/output_data/paper_robustness/runner_smoke_local/manifest.json
```

plan 不启动 SUMO 或训练，输出目录必须尚不存在；launch 与 plan 分开。正式候选配置可用同一 plan 命令生成，不能因计划文件存在就把它视为完成物理标定、模型收敛或已获准执行。150 回合计划包含 1500 个训练回合、1240 个过程验证回合和 1500 个最终评估回合；本次扩展只执行工程短测。

### 调度、隔离与恢复

训练池默认 4 个进程，过程验证池默认 2 个，最终测试池默认 8 个；可用 `--max-workers` 设置三池共用上限，设为 1 可验证串行行为。每个进程使用独立 Registry、libsumo 连接、输出及 1 个 OMP/MKL/PyTorch 线程；并发只改变独立任务的执行顺序。CPU、内存、swap 和磁盘资源门禁复用已有队列。

每个训练运行按固定顺序消费自己的 episode plan。原子写完 evaluation 与 resumable 检查点后，发布含检查点哈希的 ready 文件，过程验证池即可读取该固定版本，训练继续向前执行。最终测试等待所有训练完成。验证使用原有 `TSCTrainer.train_test` 和 `EvaluationIsolationGuard`，不修改训练模型、replay、优化器、epsilon 或 RNG，也不把验证结果用于改变既定训练预算。

queue 级和逻辑任务级均有排他锁。不要同时运行两个 launch 管理同一实验；重复 launch 会拒绝。Ctrl+C/SIGTERM 停止当前队列及其工作进程，之后使用：

```bash
python -m tools.run_paper_baseline resume \
  --manifest data/output_data/paper_robustness/runner_smoke_local/manifest.json \
  --train-workers 2 --eval-workers 2 --test-workers 4
```

resume 核对源码、环境、冻结输入及检查点身份，跳过已完成任务，并接管仍存活且命令匹配的工作进程。训练从最后一个**完整发布**的检查点恢复模型、目标网络、优化器、replay、RNG、计数和事件计划位置。JSONL/CSV 回退到该检查点，TensorBoard 从保留记录重建；被回退的数据保存在 `monitor_orphans/`，不与新时间线混合。旧格式、不同协议或已变动输入不会静默续跑。

独立验证按案例保存结果、轨迹文件哈希和隔离检查；任务重试复用与当前 checkpoint/案例匹配的已完成结果。没有可恢复检查点的早期失败保留在 `failed_attempts/`，再从同一冻结计划确定性重启。恢复不允许悄悄更换事件参数来绕过物理失败；数值或事件执行异常应先调查。

### 身份、指标与看板

plan 固定模型/训练配置、训练/验证/测试计划及种子，并保存关键源码的只读审计副本；普通 RNG 不参与计划抽样。训练与 SUMO/事件 RNG 分离，N/E 对应回合共用 SUMO 种子。检查实例去重与目标分组，JSON 中的实际 kind/target/begin/end/参数决定事件身份，不能只改 event_id。

计划会生成独立场景资产，给车辆显式绑定已有 `pkw` 类型，并核验运行时确实使用该类型；原始路网与车流文件不修改。实际逐秒采集普通车辆、排队、待插入需求、系统车辆时间与事件状态，检查车辆守恒、碰撞、传送、解除以及同 checkpoint/seed 的事件前轨迹一致性。J 是固定窗口的需求归一化系统车辆时间，不是论文完整平均行程时间。逐秒 CSV 还保存阻塞车道、封路上游区域排队，以及稀疏 lane_queues_json；正常对照也保存车道排队，后续可按事件案例的 local_regions 做配对局部分析，无须再次仿真。

在另一个终端启动同一个看板入口：

```bash
python -m tools.training_dashboard serve \
  --logdir data/output_data/paper_robustness/runner_smoke_local --port 6006
```

浏览器访问 `http://127.0.0.1:6006`；远程机器使用 SSH 端口转发。训练曲线按运行保存在 `train/<run_id>/tensorboard/`；独立验证/测试按运行与来源 checkpoint 分目录；队列根下的 TensorBoard 显示任务计数和资源使用。逐决策和优化标量持续刷新，回合/案例汇总在其结束后更新。`monitor.render_every` 控制静态图生成周期，退出时仍导出。停止看板不停止训练。

主要产物：

```text
<experiment>/manifest.json                # 固定协议与全部任务、命令和输入哈希
<experiment>/experiment.yml               # 原始批量配置快照
<experiment>/assets/                      # 独立的、显式车辆类型场景
<experiment>/source_snapshot/             # 源码审计副本，不作为另一套入口执行
<experiment>/plans/、overlays/、cases.json # 数据配置，不是新的启动脚本
<experiment>/queue/status.json            # 总进度与资源门禁
<experiment>/train/<run_id>/published/    # 可供独立验证读取的检查点发布记录
<experiment>/train/<run_id>/environment/ # 逐秒 CSV 与回合事件/交通审计
<experiment>/validation|test/<run_id>/episode_*/results.json
<experiment>/validation|test/<run_id>/episode_*/cases/ # 可复用的单案例完成证据
```

### 实现位置与验证

| 位置 | 职责 |
| --- | --- |
| [run_paper_baseline.py](run_paper_baseline.py) | 唯一用户入口，复用 Runner 和既有评估循环 |
| [common/paper_experiment.py](../common/paper_experiment.py) | 严格配置、场景快照、确定性事件计划与任务清单 |
| [common/experiment_queue.py](../common/experiment_queue.py) | 从原动脉队列整理出的共享调度核心、三类资源池与依赖 |
| [run_arterial_experiment_queue.py](run_arterial_experiment_queue.py) | 原命令兼容入口，复用同一 QueueRunner |
| [world/sumo_events/episodes.py](../world/sumo_events/episodes.py) | 每回合在已关闭的 World 上重新安装不可变事件 runtime；不改旧 v1 物理实现 |
| [trainer/paper_trainer.py](../trainer/paper_trainer.py) | 训练接入、检查点发布、计划位置与监控 |

针对性测试覆盖确定性计划、输入篡改拒绝、车辆类型、完整短训练/验证、串行与并行结果一致、SIGTERM 恢复、日志一致性、案例复用及重复启动拒绝。复验：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/test_paper_experiment.py tests/test_arterial_experiment_queue.py \
  tests/test_paper_rewards_monitor.py tests/test_colight_phase.py
```

本轮主回归 45 项通过；补齐局部排队和源码快照后的相关复验 14 项通过（包含重复复验，不相加计数）。单次 10 任务短测工作跨度为串行 48.844 秒、最多 4 个 worker 并行 17.622 秒，约 2.77 倍；不能直接外推正式训练耗时。

本地证据位于 `data/output_data/paper_robustness/runner_extension_20260920/`，以该目录 `summary.json` 指向的最终验证为准。源码或依赖变化后需重新验证/建立身份；旧 TARL/规则映射清单和既有结果保持原样，不把本次扩展解释为已执行 150 回合正式研究。

### 事件强度与安全放置开发预检

[world/sumo_events/preflight.py](../world/sumo_events/preflight.py) 复用现有 World、信号适配、事件算子、逐秒记录器和共享进程队列。固定时制、MaxPressure、未训练 CoLight 使用开发种子 101/103，无优化器更新；CoLight 权重前后核对。统一命令如下，输出目录必须尚不存在：

```bash
python -m tools.run_paper_baseline preflight \
  --config configs/tsc/colight_event150.yml \
  --output data/output_data/paper_robustness/event_preflight_local --workers 4

python -m tools.run_paper_baseline status \
  --manifest data/output_data/paper_robustness/event_preflight_local/manifest.json
```

默认每个控制器/seed 执行正常与正常重放、2 个阻塞、2 个封路、3 个降雨强度和 3 个双事件组合，共 72 次，每次 3600 秒，事件窗口为 [900,1200)。局部事件目标取验证分组；不使用最终测试策略收益调参。在正常轨迹上只读探测训练及验证池共 72 条候选车道 × 91 个训练起始时刻，记录无空间和负停车距离余量。负余量是保守运动学诊断，不等于已经发生碰撞；实际仿真另外检查碰撞、传送、车辆守恒、事件边界、解除、正常 reset 重放和事件前轨迹配对。

强度阈值预先配置：阻塞/封路事件窗口的配对局部平均排队增量至少 0.5 辆，降雨的普通车辆加权平均速度降幅至少 3%。这是开发筛查条件，不是统计显著性或真实事件标定结论。只有完整覆盖、物理检查、放置诊断及全部强度样本通过，才返回 `development_checks_passed`。

退出码 **0** 表示开发检查通过，**2** 表示评估完成但 `not_ready`；工作进程失败返回非零错误码。队列 `SUCCEEDED` 只表示该控制器/seed 的评估已结束，物理失败案例仍保存在结果中；准入以 `preflight_summary.json` 或 status 的 `preflight_admission` 为准。训练 launch 不会自动引用其他目录的预检报告，启动前必须核对报告所属的配置与源码身份。

失败定向复验仍用同一命令：在配置目录保存继承研究 profile 的 YAML，设置 `preflight.controllers: [maxpressure]` 和 `preflight.case_ids: [lane_blockage_1, combined_1]`。正常与正常重放自动保留，默认 `case_ids: null` 执行全套；筛选后的诊断不能替代完整准入。中断后复用 `resume --manifest ... --max-workers 4`；它核对身份并复用已记录案例，修改实现后应新建预检目录。队列看板沿用 `tools.training_dashboard serve --logdir <预检目录>`，展示任务和资源；逐秒 CSV、单案例诊断与最终强度结果写在 `runs/<controller>_s<seed>/`，不是训练收敛曲线。

2026-09-20 实测（SUMO/libsumo 1.27.1、hz4x4、CPU）：

| 检查 | 结果 |
| --- | --- |
| 完整评估 | 72 次尝试、70 次完成、2 次于 t=901 碰撞并传送；准入 not_ready |
| 正常 reset 重放 | 三控制器 × 两开发种子全部通过 |
| 只读放置诊断 | 39312 次检查，504 次无空间、1464 次负停车余量 |
| 单车道阻塞强度 | 完成 11/12，1/11 达标 |
| 整段封路强度 | 完成 12/12，10/12 达标 |
| 全局降雨强度 | 完成 18/18，17/18 达标 |

MaxPressure seed101 的 `road_2_4_3_1` 阻塞使车辆 628 撞上新插入障碍车；seed103 的阻塞加降雨组合在 `road_2_2_2_1` 使车辆 444 碰撞。定向复验 8 次尝试、6 次完成，重现相同两例。前者在 t=900 静态空间可用，但停车距离余量为 −22.80 米，说明现有最近空隙放置不能保证动态停车安全。当前算子未因这次预检而修改；下一步应显式设计动态停车距离约束、处理无法放置的语义，再用同一预检复验并校准阻塞强度。局部排队下降不能单独视为交通改善，须结合 J、未完成车辆与待插入需求解释。

本地证据根目录为 `data/output_data/paper_robustness/event_preflight_20260920/`：`summary.json` 为本轮汇总，`development_v5/` 为完整评估，`diagnostic_v6/` 为碰撞复验。保留原始报告；v5 旧字段 `completed_simulations: 72` 实际指尝试次数，汇总已纠正为完成 70 次。更早 `physical_v3` 只验证部分物理条件，其“保留历史固定时制”限制文字已过时，不作为本次准入依据。

配置/预检与受影响队列回归 **18 passed**（105.12 秒，无跳过），日志为上述证据目录的 `final_tests.log/xml`。完整复验命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/test_paper_preflight.py tests/test_paper_experiment.py \
  tests/test_arterial_experiment_queue.py \
  tests/test_paper_p0.py::test_frozen_grounding_dependencies_preserved
```

上述为修复前的预检记录，表示预检能够正确执行与报告失败；动态放置修复见下一节。150 回合正式训练及正式测试策略评估均未执行。

### 动态放置修复与阻塞强度标定

批量实验的共享基础配置现在显式启用以下执行策略，训练、独立评估与开发预检都从冻结 episode plan 读取它：

```yaml
event_execution:
  placement:
    version: stopping-distance-v1
    clearance_m: 2
    reaction_steps: 1
    obstacle_length_m: 5
events:
  blockage_min_vehicles: 0
  block_distance: 100
  position_tolerance: 20
```

[safe_placement.py](../world/sumo_events/safe_placement.py) 在现有 runtime 实例上接入放置检查，复用 v1 的事件边界、物理插入/移除、报告和恢复；没有复制启动入口或修改冻结 v1 文件。旧固定日程入口及未携带执行策略的历史计划保留 `static-gap-v1`，不应将旧入口当作已启用新检查。

新检查对所有同车道潜在后车计算所需间距：`minGap + v × (max(tau, actionStep, simStep) + reaction_steps × simStep) + v²/(2 × decel) + clearance_m`；采用舒适减速度，额外仿真步余量覆盖瞬时插入与离散步进。候选位置同时满足前车车身间距、后车停车距离和同一时刻其他障碍物占位，在原容差内按距离确定性选择；一次批量插入前先验证全部新障碍。没有可用位置则抛出 `UnsafePlacementError`，不移动/删除普通车辆、不换目标、不延后生效。

`placement_decisions` 审计记录请求与实际位置、障碍长度、检查车辆数、最小停车余量及限制后车；失败案例记录 `placement_rejection`。汇总单列 `placement_rejections` 与 `collision_or_teleport_cases`。正常结束前仍须检查实际 SUMO 碰撞/传送和车辆守恒；运动学余量不构成对任意路网、换道与驾驶模型的安全证明。

强度参数也进入相同配置链：`block_distance` 表示障碍前端距停止线的距离，`obstacle_length_m` 表示沿车道占用长度；修改长度会同时改变真实 SUMO 障碍车长度和放置间距，不能只改报告。`blockage_min_vehicles` 为当前仿真需求窗口的静态路由需求筛选下限，默认 0 不筛选；非零时按路由连接将车辆需求分配到可行车道，筛选单车道阻塞目标，封路目标池不变。它是路由需求估计，不是保证事件窗口实际有车；`target_pools.json` 保存筛选后的 `blockage_lanes`。过高下限可能使测试事件无法去重，plan 会拒绝不足的案例空间，不能只凭预检配置可运行就跳过研究计划编译。

本轮保留原 0.5 辆局部排队增量阈值进行了开发标定，配置与产物在 `data/output_data/paper_robustness/event_gap_fix_20260920/`，均复用 `preflight --config <YAML> --output <新目录>`：

| 候选 | 单车道阻塞结果（FixedTime/MaxPressure × seed101/103 × 两目标） |
| --- | --- |
| 原 5 米障碍移至距停止线 10 米，容差 5 米 | 8/8 可执行，0/8 达到强度阈值 |
| 原 5 米障碍、路由需求下限 150 辆 | 8/8 可执行，0/8 达标；增量最大约 0.467 辆 |
| 30 米阻塞区、距停止线 1 米、容差 0.5 米、需求下限 100 辆 | 7/8 被安全检查拒绝；唯一可执行例增量约 8.47 辆 |
| 60 米阻塞区，同样位置/筛选设置 | 7/8 被拒绝；唯一可执行例增量约 8.66 辆 |

这些长阻塞区是未获准用于正式实验的开发候选，不覆盖默认 5 米物理定义，也不将单个可执行案例当成全套通过。标定汇总为 `calibration_summary.json`，失败案例完整保留。结果说明较大阻塞区可以增强效应，但不能在已被交通占用的车道上任意瞬时生成；若需要稳定强阻塞，应另行明确渐进车道封闭及实际生效时间的语义。保留原单障碍定义时，应如实视为弱扰动候选，不能通过降低阈值宣称强扰动已完成标定。

安全回归已复验原 MaxPressure 两例碰撞：相同请求/±20 米容差/种子下，8 次正常与事件仿真全部完成，无碰撞或传送。将第一例容差固定为 0 后，在 t=900 插入前明确拒绝，未生成障碍车、无碰撞或传送。`counterexamples_v1/` 和 `reject_unsafe_v2/` 保存这两条路径。

修复后全套 `full_safe_v3/` 共 72 次尝试、70 次完成；MaxPressure seed103 与未训练 CoLight seed103 的 `combined_0` 因无安全空间在插入前被拒绝，实际碰撞/传送为 0，正常重放均通过。39312 次探测中 1179 次明确拒绝，获准候选的动态停车余量均非负。强度达标为阻塞 1/12、封路 10/12、降雨 17/18；准入仍为 **not_ready**。拒绝危险插入证明检查生效，不能算作场景执行成功；原先两例碰撞已修复，也不能代替全场景可执行性与强度验收。

最终证据以本轮 `summary.json` 为准。相关回归 20 项通过，包含配置冻结、需求筛选、真实 SUMO 预检、并行/恢复及旧冻结依赖保护；最后对需求审计和停车余量报告的修正另做针对性复验，结果见 `final_recheck.log/xml`，不与主回归相加计数。

## 飞书 Wiki 工具

[feishu_wiki.py](feishu_wiki.py) 使用自建应用身份和 Python 标准库 HTTP 客户端。默认知识库与所需权限清单来自 `configs/feishu.json`；每次命令可用 `--node` 指定飞书 HTTPS Wiki 地址或节点 token。

```bash
python tools/feishu_wiki.py --help
python tools/feishu_wiki.py configure
python tools/feishu_wiki.py check
python tools/feishu_wiki.py list
python tools/feishu_wiki.py read
```

`configure` 交互读取凭证，将其保存到 Git 忽略的 `.secrets/feishu.json`（文件权限 600）；也可由环境成对提供 `FEISHU_APP_ID` 和 `FEISHU_APP_SECRET`，环境优先。`check` 检查节点与新版文档读取权限，不通过试写验证写权限；`list` 列出直属子节点。2026-09-20 已完成远端研究 Wiki 基础页面创建、写入及回读核验。该工具不依赖 colight，可直接用默认终端的 `python3`；实际实验按目标项目环境执行。

通用 `create/append` 仅用于搭建稳定目录或工作台，不用于维护实验过程：

```bash
python tools/feishu_wiki.py create --title '实验记录'
python tools/feishu_wiki.py append --node '目标节点token' --file '本地UTF-8文本文件'

# 整页替换登记的实验协议页（仅限 configs/feishu.json 的 experiment_guide_url）
python3 tools/feishu_wiki.py page-update --node '协议页URL或token' \
  --page-id 'RESEARCH-COLLABORATION-v3' \
  --file /tmp/protocol.txt \
  --archive data/output_data/<目录>/archive_protocol_before.txt
```

`create` 在目标节点下新建空白 docx 子节点；`append` 向指定 docx 末尾追加文件内容。实验任务页不要使用 `append` 累积过程记录，应使用下文的 `review-update`。普通文本不解析 Markdown，每块最多 1000 字符、单次最多 50 块。网络错误后的写入结果可能未知，应先读取核对再决定是否重试。凭证与实验产物不纳入 Git，训练入口不会调用此工具。

### Codex CLI 与实验工作台

三方向工作台及统一约定页地址保存在 `configs/feishu.json`。从本项目启动 Codex CLI 时，项目 `AGENTS.md` 提供持续维护约定。直接在其他同级项目启动的新会话，需要先明确要求它读取本项目的该约定和 Wiki 工作台；不要假定兄弟目录自动继承本项目指令。

```bash
# 读取整体实验逻辑、工作台正文与直属任务链接
python3 tools/feishu_wiki.py workbench --direction attention
# 方向另可选 semi_offline、hitl

# 一项研究实验对应一个页面；不会启动训练，初始状态为草案
python3 tools/feishu_wiki.py experiment-create --direction attention \
  --experiment-id ATT-EXAMPLE-001 --title '实验名称' --file /tmp/experiment-plan.txt

# 按 URL 读取任务；Codex 核对项目协议后执行用户已授权的实验
python3 tools/feishu_wiki.py read --node '实验页面URL'

# 用当前审核摘要替换任务正文；旧正文先归档到本地实验输出目录
python3 tools/feishu_wiki.py review-update --node '实验页面URL' \
  --review-id ATT-EXAMPLE-001-review-v1 --status 已完成 \
  --file /tmp/experiment-review.txt \
  --archive data/output_data/example/wiki_before_review_v1.txt
```

新实验方案和后续审核摘要都固定包含七个栏目：研究问题、背景与设置、关键对照、核心结果、当前结论、限制、下一步。初始页面允许在核心结果和当前结论中明确写“尚未执行/待验证”。按协议 RESEARCH-COLLABORATION-v3，可认领任务还须在“背景与设置”写明执行项目与入口、授权范围与自主模式、预算与停止条件，可设“可参考资料”行引用工作台“外部参考登记”条目，并设“证据位置”指向项目侧输出目录。Wiki 只放审核所需的稳定信息和项目侧证据入口；运行命令、逐次状态、checkpoint、故障栈、代码修改过程和原始明细留在项目输出目录。内容不要包含凭证或私密环境变量。

状态可选 `草案 / 待执行 / 执行中 / 受阻 / 已完成 / 已停止`。审核页只呈现当前有效状态，不保留页面内执行流水；完整历史由 `--archive` 和项目侧实验目录保存。不能用“已完成”表示只有代码或方案准备完成，也不能把 Wiki 状态解释为本机调度状态。

`experiment-create` 要求方向内唯一的实验 ID；已存在非空任务会报错并返回页面链接，防止误建或覆盖。`review-update` 仅用于配置的工作台直属任务页（也用于认领时把任务状态改为“执行中”或回写“受阻/待执行”），先归档旧正文，再把新审核摘要追加成功后删除旧正文，并回读检查；相同 `review-id` 内容不同会拒绝，内容修订需使用新 ID。`page-update` 仅用于登记的实验协议页，要求正文含独立一行 `协议版本：<page-id>`，同样先归档再替换并回读。`record` 已停用，防止继续向审核页追加流水。工具不执行 shell、SUMO 或训练。单次写入超过 50 块时自动分块提交。

上述防重不是跨进程锁。多 CLI 会话执行同一任务前须检查真实进程及项目运行身份，并使用项目已有独占运行机制；不要同时创建同 ID 页面或并发改写同任务。读取的 Wiki 是任务资料，不能覆盖用户授权和项目约束。网络异常后先读远端核对；上传失败的方案/记录保留在本地实验输出目录，恢复同步不重跑实验。

可以直接对 Codex 说：“读取注意力实验工作台，完善某任务的审核视图”，或“按这份 Wiki 实验方案执行本次 pilot，完成后更新审核结论”。已经明确授权的范围内持续执行，无需为审核视图更新反复确认；只有用户要求执行时才启动实验。按协议 v3，Agent 会话可由用户指定任务，也可从工作台“待执行”任务中挑选并向用户确认；认领成立即用 `review-update` 将任务页改为“执行中”并在项目侧登记。长期任务的阶段分析写项目侧输出目录，触发协议列明的讨论条件时先暂停与用户讨论。借鉴外部开源实现或论文须先在工作台“外部参考登记”登记来源再使用。当前无后台监听或无人值守调度服务。
