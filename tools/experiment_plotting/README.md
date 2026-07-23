# experiment_plotting 实验分析与绘图工具

`tools.experiment_plotting` 用于读取已经完成的实验产物，执行输入校验、指标聚合、表格导出和可复现绘图。本文件是该模块的唯一权威介绍与维护入口；使用者和 Codex CLI 在解释图包、修改工具或扩展实验类型前，都应先阅读本文件。

当前稳定解释基准为工具版本 `2.1.0` 生成的 Plan 1 正式图包：

```text
data/output_data/analysis/plan1/p1_formal_20_plotting_v21_20260723/
```

该图包包含 45 个唯一图项，每项同时输出 PNG 和 PDF。PNG 与 PDF 是同一图的两种格式，不应计作两个分析功能。

## 1. 功能边界

本工具负责：

- 读取显式 run-list，不通过扫描输出目录猜测有效实验。
- 验证实验目录、配置归档和指标记录是否可用于比较。
- 将原始记录转换为稳定的分析表格。
- 按固定的 network、控制器、seed 和统计语义生成图包。
- 可选读取已经完成的 best-checkpoint decision-level evaluation package，绘制一次评估 episode 内的时间序列。
- 写出 `plotting_manifest.json`，记录本次分析的输入、工具版本、表格、图、依赖版本和关键统计语义。

本工具不负责：

- 启动 SUMO、训练模型或重新评估 checkpoint。
- 选择或改变正式实验白名单。
- 自动判断失败、取消、Pilot 或其他目录是否可以进入正式分析。
- 根据绘图结果反向修改原始实验数据。
- 在缺少决策级记录时推测 episode 内的 reward、queue、delay 或 throughput 曲线。

SUMO 重评估属于上游数据获取流程，复用 `run.py --evaluation-manifest ... --evaluation-output ...`。plotting 只消费上游完成并通过校验的 evaluation package。无法从既有产物追溯的字段，应补充上游采集功能，不能在绘图阶段人工构造。

## 2. 稳定使用方式

在仓库根目录使用 `colight` Conda 环境运行。推荐统一使用 `analyze --profile ...` 入口：

```bash
PYTHONPATH=. conda run -n colight python -m tools.experiment_plotting.cli analyze \
  --profile plan1 \
  --run-list data/output_data/analysis/plan1/p1_formal_20_runlist_20260722.csv \
  --analysis-id <new_analysis_id> \
  --output-root data/output_data/analysis/plan1 \
  --evaluation-package \
    data/output_data/evaluations/plan1/plan1_best_checkpoint_reevaluation_v1_20260723 \
  --dpi 160
```

使用约束：

- `analysis-id` 必须是新的目录名；工具拒绝覆盖已有分析目录。
- 正式分析必须使用明确、受审计的 run-list。
- 需要 best-checkpoint 时间序列图时必须提供完整的 `--evaluation-package`。
- 不提供 evaluation package 时，仍可生成训练与 episode-level 分析，但不会生成 best-checkpoint 时间序列图及其汇总表。
- `plan1` 子命令是 Plan 1 的兼容入口；新调用优先使用具名 profile。
- `plan2` profile 用于严格校验和聚合纯离线实验。Plan 2 正式实验尚未完成时，不得把开发或空输出描述为正式 plotting 结果。

run-list 至少包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `role` | 正式运行、基线等角色 |
| `agent` | `dqn`、`fixedtime`、`maxpressure` 等控制器 |
| `network` | 场景对应的原始 network 名 |
| `training_seed` | 训练随机种子；基线也需提供可识别值 |
| `run_dir` | 实验输出目录 |
| `include` | 是否纳入本次分析 |

## 3. 输出目录契约

一个完成的 Plan 1 分析目录结构如下：

```text
<analysis_id>/
├── plotting_manifest.json
├── inputs/
│   ├── runs.csv
│   ├── evaluation_package_manifest.json       # 可选
│   └── evaluation_collection_manifest.json    # 可选
├── tables/
│   ├── metrics.csv
│   ├── run_summary.csv
│   ├── final_best_comparison.csv
│   ├── first_100_auc.csv
│   ├── learning_speed.csv
│   ├── action_distribution.csv
│   ├── efficiency.csv
│   ├── aulc.csv
│   ├── best_checkpoint_evaluation_summary.csv # 可选
│   └── config_comparison.json
└── figures/
    ├── results/
    └── diagnostics/
```

`plotting_manifest.json` 是一个图包的唯一机器入口。它记录工具版本、输入 run-list、可选 evaluation package、纳入运行数量、表格和图文件列表、依赖库版本以及统计语义。后续自动化不得只扫描目录并凭文件名猜测图包来源。

目录语义：

- `figures/results/`：用于回答预先定义的研究问题，可以在遵守本文件限制的前提下支持结果陈述。
- `figures/diagnostics/`：用于检查训练、行为、seed 差异和异常。诊断图本身不自动构成算法优劣结论。
- `tables/`：图的可追溯数值来源。需要引用精确数值时应读取表格，不应从图片像素估计。

## 4. Plan 1 固定比较语义

场景按以下顺序展示，并直接使用 network 名：

1. `sumohz1x1_config2`
2. `sumohz1x1`
3. `sumohz1x1_config4`
4. `sumohz1x1_config3`

控制器为 `DQN`、`FixedTime` 和 `MaxPressure`。Plan 1 的主排名指标是 average travel time，越低越好。reward 不作为跨控制器主排名指标。

常用统计符号：

- “单个 seed”指一个训练 seed 的独立 DQN 运行。
- 终值类图中的菱形与误差条表示 5 个训练 seed 的均值 ± 样本标准差；散点表示各 seed。
- 学习曲线通常先按 episode 对 5 个训练 seed 求均值，再以固定 bootstrap 随机种子和 1000 次重采样绘制 95% percentile bootstrap CI。
- 置信区间反映当前 seed 样本下均值的不确定性，不代表实验总体的完整概率分布。

## 5. 指标定义与禁止误读

### 5.1 Travel time 与 checkpoint

- `final` 指 episode 400 的 evaluation 结果。
- `best` 指一个训练运行已有 evaluation 中 travel time 最低的 checkpoint；平局取最早 episode。
- best-checkpoint 的新 evaluation seeds 只用于重新估计已选 checkpoint 的表现，不会反向改变 checkpoint 选择。
- `final_travel_time` 与 best-checkpoint 重评估回答不同问题，数值不要求完全相同。

### 5.2 Reward

存在两套 reward 来源，必须区分：

- `figures/diagnostics/reward_mean` 使用历史 episode-level agent reward。DQN 与 FixedTime/MaxPressure 的旧 reward generator 语义不完全一致，因此不能用它做跨控制器排名。
- `figures/results/best_checkpoint_timeseries/reward_network_mean` 和 `reward_network_sum` 使用 decision-level evaluation package 中统一的负排队车辆数语义。数值越高、越接近 0，表示排队越少；它们仍是过程解释指标，主排名保持使用 travel time。

evaluation package 同时保留原控制器 reward 以便审计，但正式跨控制器时间序列不混用旧 reward 定义。

### 5.3 Queue、delay 与 throughput

- `queue_network_mean`：路网 lane queue 的均值，默认用 60 s 滑动均值展示，越低通常越好。
- `queue_network_sum`：全路网 queue 总数，默认用 60 s 滑动均值展示，越低通常越好。
- `delay_weighted_mean`：按车辆数加权的路网 delay，使用从 episode 开始到当前时刻的累计均值，越低通常越好。
- `throughput_interval`：默认 60 s 窗口内完成车辆数，越高通常表示当前窗口疏散更多车辆，但需结合需求到达过程解释。
- `throughput_cumulative`：从 episode 开始累计完成车辆数，越高通常越好；最终值相同不代表过程拥堵相同。

### 5.4 AUC 与 AULC

- `AUC` 是 Area Under the Curve。`learning_speed_auc` 使用 episode 0 至 100 的 evaluation travel-time 曲线做梯形积分，数值受横轴区间长度影响，越低表示早期整体 travel time 越低。
- `AULC` 在本工具中指 Area Under the Learning Curve，并按横轴跨度归一：`trapz(y, x) / (x_end - x_start)`。其单位等同于纵轴，可理解为整个学习区间的平均曲线高度。
- `raw_travel_time_aulc` 越低越好。
- 归一化控制分数为 `(FixedTime - controller) / (FixedTime - MaxPressure)`：FixedTime 为 0，MaxPressure 为 1，大于 1 表示优于 MaxPressure。
- `normalized_control_aulc` 先在各场景计算归一化分数的 AULC，再对同一训练 seed 的四场景取平均，越高越好。
- AUC/AULC 衡量学习全过程或指定早期区间，不能替代最终性能。

### 5.5 数据与计算利用指标

- environment transitions 表示累计仿真交互量。
- gradient updates 表示累计参数更新次数。
- replay fill fraction 为 `replay_size / replay_capacity`。
- update-to-data ratio 为 `gradient_updates / collected_transitions`。
- replay fill fraction 和 update-to-data ratio 描述训练机制，没有统一的“越高越好”结论。
- wall time 受硬件、并发任务、系统负载和 I/O 影响，不能单独解释为算法复杂度。

## 6. 正式结果图说明（19 项）

下表路径均相对于 `figures/`，每项实际同时存在 `.png` 和 `.pdf`。

| 图项 | 回答的问题与坐标 | 统计与正确解读 | 限制 |
| --- | --- | --- | --- |
| `results/final_travel_time` | 各场景 episode 400 的 average travel time 如何；横轴为控制器，纵轴越低越好 | 展示各 seed 和均值 ± 样本标准差，并包含 FixedTime、MaxPressure | 不是 best checkpoint 重评估 |
| `results/evaluation_learning_curve` | DQN 随 completed training episode 的 evaluation travel time 如何变化 | 5 个训练 seed 均值与 95% bootstrap CI；基线为水平线 | 用于学习过程，不把单次波动解释为稳定差异 |
| `results/learning_speed_auc` | 前 100 episodes 的早期学习总体表现如何 | 每个训练 seed 的 evaluation travel-time AUC；越低越好 | 是未按跨度归一的 AUC，不是最终性能或 AULC |
| `results/action_concentration` | 最终 evaluation 是否集中使用单一动作 | 取每个运行最终 evaluation 中最大的 action fraction；越接近 1 越集中 | 低集中不自动代表策略更优，高集中需结合场景合理性判断 |
| `results/training_cost` | 400 个训练 episode 的实测墙钟成本是多少 | 汇总各训练 seed 的 TRAIN wall time，单位小时 | 受运行环境影响，不是纯算法复杂度 |
| `results/travel_time_vs_transitions` | 相同累计环境交互预算下 DQN 的表现如何 | 横轴 transitions，纵轴 travel time，5 seed 均值与 95% CI；同等预算下越低越好 | 仅适合交互定义一致的运行 |
| `results/travel_time_vs_gradient_updates` | 相同累计优化步数下学习进度如何 | 横轴 gradient updates，纵轴 travel time；同等更新数下越低越好 | 更新次数相同不保证单次更新成本或数据组成相同 |
| `results/travel_time_vs_wall_time` | 随实测训练时间增加，控制质量如何变化 | 横轴累计训练秒数，纵轴 travel time；同一硬件条件下可比较达到质量所需时间 | 跨机器或负载不同的比较不可靠 |
| `results/replay_fill_fraction` | replay buffer 在训练期间填充到什么程度 | 横轴 episode，纵轴填充比例，展示 5 seed 汇总 | 训练机制诊断，没有统一好坏方向 |
| `results/update_to_data_ratio` | 累计每条采集 transition 对应多少次梯度更新 | 横轴 episode，纵轴 updates/transitions | 训练强度描述，不是性能排名 |
| `results/raw_travel_time_aulc` | 每个场景整个交互学习区间的平均 travel time 如何 | 横轴为 DQN，纵轴为按 transitions 积分并归一后的 AULC；越低越好 | 不等于最后 episode 性能 |
| `results/normalized_control_aulc` | 归一化后跨四场景的整体学习过程如何 | 每个 seed 先跨场景平均；FixedTime=0、MaxPressure=1，越高越好 | 依赖两条基线且会隐藏场景绝对量级差异 |
| `results/best_checkpoint_timeseries/reward_network_mean` | 已选 best checkpoint 在一次 3600 s 评估内，平均 lane queue 对应的统一 reward 如何演化 | 横轴 simulation time，纵轴负 queue lane mean 的累计均值；DQN 两层 seed 汇总并与基线比较，越高越好 | 需要 evaluation package；不是 legacy agent reward |
| `results/best_checkpoint_timeseries/reward_network_sum` | 一次评估内全路网统一 reward 如何演化 | 横轴 simulation time，纵轴负 network queue sum 的累计均值；越高越好 | 路网规模变化时绝对总量可能不可直接比较 |
| `results/best_checkpoint_timeseries/queue_network_mean` | 一次评估内平均 lane queue 如何变化 | 60 s 滑动均值；纵轴越低通常越好 | 窗口会平滑短时尖峰 |
| `results/best_checkpoint_timeseries/queue_network_sum` | 一次评估内全路网排队车辆总数如何变化 | 60 s 滑动均值；纵轴越低通常越好 | 不同路网规模间需谨慎比较总量 |
| `results/best_checkpoint_timeseries/delay_weighted_mean` | 一次评估内车辆加权 delay 如何积累 | 从 episode 开始的累计加权均值；越低通常越好 | 累计均值会弱化短时变化 |
| `results/best_checkpoint_timeseries/throughput_interval` | 每个 60 s 窗口完成车辆数如何变化 | 横轴 simulation time，纵轴窗口完成数；通常越高越好 | 受到达过程与窗口边界影响 |
| `results/best_checkpoint_timeseries/throughput_cumulative` | 一次评估内累计完成车辆数增长速度如何 | 横轴 simulation time，纵轴累计完成数；通常越高越好 | 应结合 queue、delay 和总需求共同解释 |

## 7. 正式诊断图说明（26 项）

### 7.1 Episode-level 诊断（12 项）

| 图项 | 用途 | 解释限制 |
| --- | --- | --- |
| `diagnostics/individual_travel_time` | 同时展示每个训练 seed 的 TRAIN 与 evaluation travel-time 曲线及基线，定位 seed 差异和训练/评估差距 | 线多且重叠，只用于排查，不代替汇总结果图 |
| `diagnostics/reward_mean` | 查看历史 TRAIN/evaluation agent reward 的变化 | 不同控制器 reward 语义不完全一致，禁止跨控制器排名 |
| `diagnostics/loss_mean` | 查看 DQN TRAIN loss 是否发散、出现异常尖峰或长期异常 | loss 更低不等于交通控制性能更好 |
| `diagnostics/queue` | 查看 episode-level evaluation queue 随训练进度变化 | 指标来自历史 episode 汇总，不是 episode 内时序 |
| `diagnostics/delay` | 查看 episode-level approximate delay | approximate delay 与 real delay 必须区分 |
| `diagnostics/real_delay` | 查看 episode-level real delay | 用于辅助解释，主排名仍为 travel time |
| `diagnostics/throughput` | 查看 episode-level evaluation throughput | 高 throughput 需结合需求量、unfinished vehicles 和 travel time 判断 |
| `diagnostics/epsilon` | 确认探索率调度按预期变化 | epsilon 是训练配置，不是结果指标 |
| `diagnostics/replay_size` | 确认 replay buffer 增长过程 | 绝对大小需结合 capacity 和训练配置解释 |
| `diagnostics/phase_switch_frequency` | 检查 evaluation 中相位切换是否异常频繁或几乎不切换 | 没有统一越高或越低越好结论 |
| `diagnostics/action_distribution` | 以热图展示各 seed 最终 evaluation 的完整动作占比 | 动作 ID 的交通语义依赖路网与相位定义，换路网时需要局部适配 |
| `diagnostics/final_best_gap` | 比较同一运行 `final travel time - best travel time` | 接近 0 表示终点接近历史最好；大正值提示末期退化，但不能表示绝对性能优劣 |

### 7.2 Best-checkpoint 时间序列诊断（14 项）

以下 7 个指标各有两种诊断视图，共 14 项：

| 指标前缀 | 指标语义 |
| --- | --- |
| `reward_network_mean` | 负平均 lane queue 的累计均值 |
| `reward_network_sum` | 负全路网 queue 总数的累计均值 |
| `queue_network_mean` | 平均 lane queue 的 60 s 滑动均值 |
| `queue_network_sum` | 全路网 queue 总数的 60 s 滑动均值 |
| `delay_weighted_mean` | 车辆加权 delay 的累计均值 |
| `throughput_interval` | 60 s 窗口完成车辆数 |
| `throughput_cumulative` | 累计完成车辆数 |

每个前缀对应：

- `diagnostics/best_checkpoint_timeseries/<指标前缀>_training_seed_means`：DQN 每个 training seed 先对 5 个 evaluation seeds 求均值，因此显示 5 条 DQN 曲线；FixedTime 和 MaxPressure 各显示其 5 个 evaluation seeds 的均值曲线。用于判断训练 seed 间一致性。
- `diagnostics/best_checkpoint_timeseries/<指标前缀>_raw_episodes`：显示全部原始评估 episode；每场景包含 25 条 DQN 曲线以及两种基线各 5 条曲线。用于定位离群 evaluation seed 或时段，不用于直接读取总体均值。

上述 `<指标前缀>` 必须替换为以下完整路径中的一个，不能使用“reward 图”“queue 图”等模糊指代：

```text
diagnostics/best_checkpoint_timeseries/reward_network_mean_training_seed_means
diagnostics/best_checkpoint_timeseries/reward_network_mean_raw_episodes
diagnostics/best_checkpoint_timeseries/reward_network_sum_training_seed_means
diagnostics/best_checkpoint_timeseries/reward_network_sum_raw_episodes
diagnostics/best_checkpoint_timeseries/queue_network_mean_training_seed_means
diagnostics/best_checkpoint_timeseries/queue_network_mean_raw_episodes
diagnostics/best_checkpoint_timeseries/queue_network_sum_training_seed_means
diagnostics/best_checkpoint_timeseries/queue_network_sum_raw_episodes
diagnostics/best_checkpoint_timeseries/delay_weighted_mean_training_seed_means
diagnostics/best_checkpoint_timeseries/delay_weighted_mean_raw_episodes
diagnostics/best_checkpoint_timeseries/throughput_interval_training_seed_means
diagnostics/best_checkpoint_timeseries/throughput_interval_raw_episodes
diagnostics/best_checkpoint_timeseries/throughput_cumulative_training_seed_means
diagnostics/best_checkpoint_timeseries/throughput_cumulative_raw_episodes
```

## 8. Best-checkpoint 重评估统计层级

正式 v2.1 evaluation package 的统计层级固定如下：

1. 每个场景的 5 个 DQN training seeds 分别选取原训练过程中已有的 best checkpoint。
2. 每个 DQN checkpoint 使用 evaluation seeds `10000` 至 `10004` 重新评估。
3. 单个 DQN training seed 先对其 5 个 evaluation seeds 求均值。
4. 主图再对 5 个 DQN training-seed 均值求总体均值和 95% bootstrap CI。
5. FixedTime 与 MaxPressure 在每个场景分别基于 5 个 evaluation seeds 汇总。

每个 evaluation episode 为 3600 仿真秒，每 10 秒记录一次 decision，共 360 条记录。正式包共有 140 episodes 和 50,400 条 decision records。

正式包中 DQN best-checkpoint 重评估 travel time 的示例结果为：

| Network | 5 个 training-seed 均值 ± 样本标准差 |
| --- | ---: |
| `sumohz1x1_config2` | 71.0314 ± 0.5819 |
| `sumohz1x1` | 77.2028 ± 0.6015 |
| `sumohz1x1_config4` | 73.2030 ± 0.4069 |
| `sumohz1x1_config3` | 66.7438 ± 0.1440 |

这些值是当前正式数据包的可核查示例，不是代码中应硬编码的期望值。

## 9. 表格说明

| 表格 | 内容与主要用途 |
| --- | --- |
| `tables/metrics.csv` | 统一后的 TRAIN、EVALUATION 和 FINAL_EVALUATION episode-level 指标，是多数图的基础长表 |
| `tables/run_summary.csv` | 每个运行的记录数量、配置哈希、轨迹状态、final/best episode 与 travel time |
| `tables/final_best_comparison.csv` | 每个运行 final 与 best 的 travel time、queue、delay、real delay 和 throughput |
| `tables/first_100_auc.csv` | TRAIN/evaluation 在前 100 episodes 的各指标 AUC |
| `tables/learning_speed.csv` | 首次或连续 5 次达到 FixedTime、MaxPressure、final 110% 门槛的 episode |
| `tables/action_distribution.csv` | 每个运行、episode 和 action 的占比长表 |
| `tables/efficiency.csv` | evaluation quality 与 transitions、updates、累计 wall time、replay fill 和 update-to-data ratio 的对齐表 |
| `tables/aulc.csv` | 原始 travel-time AULC、单场景归一化 AULC 和跨场景归一化 AULC |
| `tables/best_checkpoint_evaluation_summary.csv` | 每个 controller、training seed、evaluation seed 的 best-checkpoint 重评估汇总；仅在提供 evaluation package 时存在 |
| `tables/config_comparison.json` | 纳入 DQN 运行的配置兼容性比较结果 |

## 10. 如何明确指代和解读图

讨论图包时至少说明以下内容：

1. 图的完整相对路径，例如 `results/final_travel_time`。
2. 数据层级，例如“episode 400 原 evaluation”或“best checkpoint 新 evaluation seeds 重评估”。
3. 聚合层级，例如“5 个 training seed 的均值 ± 样本标准差”。
4. 指标方向，例如“travel time 越低越好”。
5. 必要限制，例如“wall time 受硬件负载影响”。

推荐表述：

> `results/final_travel_time` 显示四个 network 在 episode 400 evaluation 上的 travel time；散点是 5 个训练 seed，菱形为均值，误差条为样本标准差，数值越低越好。

禁止使用以下不唯一表述：

- “reward 图”：应说明是 legacy `diagnostics/reward_mean`，还是统一 reward 的 best-checkpoint `reward_network_mean`/`reward_network_sum`。
- “最终结果”：应说明是 episode 400 final，还是已选 best checkpoint 的重新评估。
- “AUL/AULC 图”：应说明是前 100 episode 的 AUC、原始 travel-time AULC，还是归一化控制分数 AULC。
- “平均值”：应说明平均的是 evaluation seeds、training seeds、lanes、vehicles 还是 simulation time。

## 11. 扩展原则

工具设计目标是对同一实验产出协议稳定复用，而不是每次实验重新适配。扩展时遵循：

- 通用输入校验、聚合和绘图逻辑保留在本模块。
- 实验差异优先通过 profile、显式字段和配置表达，避免按实验目录名写硬编码分支。
- 场景标签默认使用原始 network 名。
- 不同路网的 action 数量、相位含义或路网级指标定义确有差异时，允许做局部适配，但适配规则必须显式、可验证且记录在本文件。
- 新图必须有唯一文件名、明确研究问题、数据来源、横纵轴、统计单位、好坏方向和禁止误读说明。
- 新增数据需求时先检查既有原始产物是否可追溯；不可追溯则修改上游采集流程，不在 plotting 中推导伪数据。
- 开发中的 S1–S4 convergence、budget sufficiency、action distribution distance 和 cross-scene diagnostics 在形成经过验证的正式图包前，不属于本文件承诺的稳定 v2.1 产出。

## 12. README 强制维护契约

任何修改 `tools/experiment_plotting/` 的提交，只要影响下列任一事项，就必须在同一提交中同步更新本 README：

- CLI 命令、参数、默认值或调用示例。
- 支持的 profile、实验计划或路网适配规则。
- run-list、实验记录、evaluation package 的输入字段或校验规则。
- 输出目录、manifest、表格或图文件名。
- 指标定义、单位、统计层级、置信区间或聚合方法。
- 图的横纵轴、平滑方式、配色语义、好坏方向或允许支持的结论。
- plotting 与训练、SUMO 评估、数据采集之间的功能边界。
- 依赖变化或会影响可复现性的行为。

维护完成条件：

1. 更新受影响章节和命令示例。
2. 在“版本与维护记录”增加一条记录；行为变化时同步更新模块版本号。
3. 使用代表性输入生成新 analysis-id，禁止覆盖旧正式图包。
4. 检查 `plotting_manifest.json` 与实际表格、图文件完全一致。
5. 核对本 README 列出的正式图项与生成结果一致。
6. 运行适用的导入、语法或最小分析验证，并在提交说明中记录。

如果模块代码已经改变而本 README 未同步，相关修改不得视为完成，也不应作为稳定功能交接给后续使用者。

## 13. 版本与维护记录

| 日期 | 工具版本 | 说明 |
| --- | --- | --- |
| 2026-07-23 | 2.1.0 | 建立模块权威介绍文档；冻结 Plan 1 正式 v2.1 图包的 45 个图项、输入输出、统计口径、解读边界及 README 同提交维护契约 |
