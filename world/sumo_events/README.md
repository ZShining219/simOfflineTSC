# SUMO 事件与同步文本报告模块

本模块负责在 SUMO 中执行可配置的交通事件，并在物理操作成功后同步发布事实文本。当前提供局部车道阻塞、整段有向道路封闭和全局降雨三类事件，已完成杭州 `hz4x4` 路网的工程验证和机制量化对比，可作为后续自然语言解析、实体映射和文本注意力实验的事件来源。

模块职责是“事件日程 → 物理作用 → 同步报告”。下游从报告文本与静态路网信息解析实体、关系和事件要素，再接入自己的文本编码器与控制策略。事件发生不会自动触发或限定注意力计算；该模块尚不包含文本学习或注意力训练逻辑。

- [事件语义与配置](#事件语义与配置)
- [接入现有 World](#接入现有-world)
- [与论文奖励训练入口的关系](#与论文奖励训练入口的关系)
- [报告与下游文本接口](#报告与下游文本接口)
- [规则实体映射 v1](#规则实体映射-v1)
- [验证与已有证据](#验证与已有证据)
- [源码职责与复用来源](#源码职责与复用来源)

## 事件语义与配置

| `kind` | 作用对象与操作 | 专用参数 | 恢复与重叠规则 |
| --- | --- | --- | --- |
| `lane_blockage` | 在一条外部机动车车道上放置静止障碍物；车辆可按 SUMO 行为在可行时换道绕行 | 必填 `lane_id`、`position`；可选 `position_tolerance` | 结束时显式移除对应障碍物；多个障碍物独立管理并检查安全间距 |
| `road_closure` | 对一个有向 edge 的全部车道调用 `lane.setDisallowed(..., ['all'])` | 必填 `edge_id` | 同时生效的封闭取并集；某车道不再被任何事件封闭时恢复原权限 |
| `global_rain` | 对静态路网中允许机动车的车道调用 `lane.setMaxSpeed`，包含路口内部连接车道 | 必填 `speed_factor`，范围 `(0, 1)` | 原始限速乘以当前最小系数；系数不连乘，全部降雨结束后恢复原限速 |

局部阻塞的 `position` 是障碍物**前保险杠距车道起点的米数**，要求至少 5 米且小于车道长度。`position_tolerance` 默认为 0；冻结 v1 在指定位置 ± 容差内选择满足静态间距的最近位置，报告和审计记录使用实际位置。无可用位置时显式报错，不删除普通车辆腾位置，也不延迟到其他时刻悄悄生成事件。**静态间距检查不保证后车有足够停车距离**：2026-09-20 跨控制器开发预检重现了两例插入后碰撞。批量实验已通过下述显式执行策略补充动态检查；旧固定日程入口仍保留 v1 行为。

道路封闭针对一个方向，双向封闭需配置两个 edge 事件。封闭不清空已在道路上的车辆，也不强制重规划路线；车辆保留原路线并等待可通行。生效时已经进入路口内部连接车道的车辆仍可能完成穿越，稍后进入被封闭道路。因此，权限生效不等于从该秒起绝对没有车辆进入。

降雨是全局**限速代理**。例如 `speed_factor: 0.7` 表示将原限速调整为 70%，不保证所有车辆的实际速度恰好降低 30%；未同时模拟能见度、制动、路面附着或驾驶人反应，也未给出经验标定的雨量—速度关系。

### 现有配置

| 文件 | 用途 | 事件窗口（秒） |
| --- | --- | --- |
| [configs/sim/hz4x4.cfg](../../configs/sim/hz4x4.cfg) | 杭州古荡 4×4 路网和车流，16 个信号路口 | 仿真配置 |
| [configs/events/hz4x4.yml](../../configs/events/hz4x4.yml) | 工程验证，包含三类事件及时间重叠 | 阻塞 `[60,120)`；封路 `[150,210)`；降雨 `[90,240)` |
| [configs/events/hz4x4_comparison.yml](../../configs/events/hz4x4_comparison.yml) | 五条件配对对比，在车流运行一段时间后施加扰动 | 三类事件均为 `[900,1200)` |

这些配置是可复现的受控场景，不代表真实事件的概率分布。模块加载固定日程，不自动随机抽样事件；同一日程随 `reset()` 重复，车流和训练种子由实验入口管理。

工程配置示例：

```yaml
schema_version: sumo-events-v1
events:
  - event_id: hz_lane_01
    kind: lane_blockage
    begin: 60
    end: 120
    lane_id: road_1_1_0_2
    position: 650
    position_tolerance: 20
  - event_id: hz_road_01
    kind: road_closure
    begin: 150
    end: 210
    edge_id: road_2_2_0
  - event_id: hz_rain_01
    kind: global_rain
    begin: 90
    end: 240
    speed_factor: 0.7
```

`load_schedule(path)` 接受 YAML 或 JSON，顶层只允许 `schema_version` 和 `events`。每个事件必须有唯一的 `event_id`（ASCII 字母、数字、下划线或连字符）、`kind`、`begin`、`end`。时间单位为 SUMO 秒，要求有限数值、`0 <= begin < end`，并与实际仿真步长对齐。未知字段、冲突的类型专用参数和无效路网目标会被拒绝；当前没有报告延迟配置。无事件日程为 `events: []`。

### 迁移到其他路网

执行器从 SUMO `.net.xml` 提取车道、道路、路口、行驶方向与车道转向连接，配置中的局部目标仍需使用该路网的真实 ID，并选择合法且能安全放置障碍物的位置。行驶方向来自车道形状首尾向量的主方向，复杂弯曲道路或特殊坐标系应单独核验文本方向。

当前集成验证覆盖 `hz4x4`，不能据此认定任意路网即插即用。更换路网时需核对路线连通性、车道权限、仿真步长、目标位置和报告定位，再运行工程验证；当前适配器要求可解析的显式路线，不能直接把尚未完成路径计算的 trips/flows 输入视为已支持。

## 接入现有 World

以下是插入已有实验循环的调用片段。`world` 为已构造且连接已关闭的 SUMO `World`；`action_sequence` 表示调用方生成的动作序列。

```python
from world.sumo_events import install_events

runtime = install_events(world, "configs/events/hz4x4.yml")
try:
    world.reset()
    catalog = runtime.network_catalog()  # 静态路网，可用于文本中的实体定位
    texts = runtime.texts(world.get_current_time(), world.intersection_ids)
    for actions in action_sequence:
        world.step(actions)
        texts = runtime.texts(world.get_current_time(), world.intersection_ids)
        # 调用方在这里解析 texts，并与常规交通状态共同构建下一次策略输入。
finally:
    world.close()
```

安装要求：

- 在 `world.reset()` 前安装，当前适配器要求 `world.step_ratio == 1`。相同日程重复安装返回同一个 runtime；已安装后切换日程应重新构造 World。
- 适配器在每次 `step_sim` 前后同步事件，使返回给调用方的报告与当前仿真时刻对齐。`world.reset()` 清空旧报告、连接状态并重放相同日程，`world.close()` 调用运行时清理。
- 不能与包含事件的旧 TARL V1/V2.1 runtime 同时使用。模块通过实例适配接入，保留 [world_sumo.py](../world_sumo.py) 与 TARL-TSC V2.1 冻结协议；不会自动替换已有 agent 持有的 runtime 引用。
- `run.py` 当前没有该模块的事件命令行开关。项目训练接入需要显式调用 `install_events`，独立验证使用下文的模块入口。

### 与论文奖励训练入口的关系

批量实验通过新增的 [episodes.py](episodes.py) 管理逐回合计划：在关闭上一回合后，恢复安装前的 World 钩子，使用预生成的种子/日程创建新的 v1 runtime，再 reset。绑定期间不会修改旧 runtime.schedule，也不会在放置失败时重新抽样目标。原 `install_events` 的固定日程接口保持原语义。计划、并行和恢复命令见 [统一入口](../../tools/README.md#批量实验的统一入口)。

开发预检由 [preflight.py](preflight.py) 提供，通过 `python -m tools.run_paper_baseline preflight --config configs/tsc/colight_event150.yml --output <新目录> --workers 4` 调用同一配置解析器与队列。回合、事件网格、种子、目标组和预检阈值归属 [共享基础配置](../../configs/tsc/paper_experiment_base.yml)，工程短测与研究预算通过继承覆盖，不新增启动脚本。普通车辆速度总和/均值与稀疏车道排队由逐秒记录器统一采集，供正常/事件配对强度分析。

本轮仅开发预检，固定时制、MaxPressure、未训练 CoLight，SUMO/libsumo、hz4x4、种子 101/103：72 次尝试中 70 次完成，2 次在障碍车插入后的第一秒碰撞并传送，定向复验重现；正常重放通过。39312 次只读放置探测发现 504 次无空间、1464 次负停车距离余量。完成的阻塞/封路/降雨案例分别有 1/11、10/12、17/18 达到预设强度阈值，当前准入 **not_ready**。负停车余量仅为风险诊断；上述两例另有真实 SUMO 碰撞记录。旧 v1 算子及历史证据保留，动态放置修正与阻塞强度校准尚待完成。

预检退出 2 表示评估完成但不满足准入；队列任务完成不等于物理检查通过。覆盖、诊断配置、复验命令及本地证据说明见 [事件开发预检](../../tools/README.md#事件强度与安全放置开发预检)。这不证明训练模型的鲁棒性，150 回合正式实验尚未执行。

上述是修复前证据。现在批量配置显式选择 `event_execution.placement.version: stopping-distance-v1`，由 [safe_placement.py](safe_placement.py) 给每个新 runtime 实例安装检查：在静态车身间距之外，计入后车速度、反应/动作/仿真步长、舒适制动距离和余量；找不到安全位置时在插入前拒绝，保存 `placement_rejection`。位置、障碍占用长度、限制后车和停车余量进入审计。训练与预检共享同一实现，原 v1 文件和旧结果不修改；无执行策略的历史计划继续使用静态检查。

修复后两例历史碰撞均在原 ±20 米容差内找到安全位置并完成仿真；固定到原危险位置的专项复验则明确拒绝，无障碍车插入、碰撞或传送。`obstacle_length_m` 与 `events.blockage_min_vehicles` 分别支持占用长度和静态路由需求筛选，属于物理定义/场景设置，不能作为隐藏的运行时修补。开发标定显示 5 米单障碍仍偏弱，30/60 米近停止线阻塞在一个可放置案例中产生明显排队，却有 7/8 例因无安全空间被拒绝，不能据此宣称正式设置已就绪。完整配置与证据见 [动态放置修复与标定](../../tools/README.md#动态放置修复与阻塞强度标定)。

新实验可在安装事件前显式调用 `world.sumo_signal_control.install_signal_control(world, {'version': 'sumo-green-yellow-v1', 'yellow_seconds': 5})`。该适配器修正实际黄灯和绿灯计时，并在每次 reset 后重新安装；`tools.run_paper_baseline` 通过 `trainer.signal_control` 配置完成同一接入。P0 配置、验证范围和证据见 [工具模块 P0 说明](../../tools/README.md#p0-工程门禁)。默认事件验证/compare 入口仍使用历史控制行为；不能将 9 月 18 日指标重新标成修正后结果。

`tools.run_paper_baseline` 是一个独立的、仅限 SUMO 的工程验证入口。它在创建 World 后调用本模块的 `install_events`，把固定日程中的活动报告数量写入训练监控，同时由 `world/paper_rewards.py` 从当前 SUMO 车辆状态计算 `paper_frap`、`paper_presslight` 和 `paper_colight` 的专属奖励。奖励计算不读取事件真值、未来日程或报告文本；事件报告也不会自动成为策略观测。

因此，事件执行、报告发布、实体映射和奖励训练是四个可组合层次：

1. 本模块负责 `[begin, end)` 物理作用、恢复和公开 `Report`；
2. `ReportGrounder` 与 `TextEntityBinder` 负责把公开报告链接到静态实体和实际观测位置；
3. `paper_*` 入口负责模型专属奖励、检查点契约和训练监控；
4. 文本实体注意力模型仍属于下一阶段设计，尚未接入 `paper_*` 策略。

奖励入口的配置、指标口径、中断边界和复验命令见 [`tools/README.md`](../../tools/README.md)。若只验证事件物理或报告映射，应使用本模块的 `__main__`、`compare` 和相应测试，不要把短训练记录解释为交通控制效果结果。

安装过程先校验显式基线路线，再加入 `--time-to-teleport -1` 和 `--ignore-route-errors true`。前者防止长时间排队的车辆被超时传送，后者允许原本有效的路线经过临时关闭的道路；基线路线中的未知 edge、断连和无法解析的路线引用仍会报错。模块不引入自动绕行。无事件对照也应安装空日程，保持这些 SUMO 选项一致：

```python
from world.sumo_events import Schedule, install_events

runtime = install_events(world, Schedule(()))
```

### 直接接入 SUMO 连接

外部调用方可用 `SumoEventRuntime(net_file, schedule)`，其中 `schedule` 是 `Schedule` 对象。在新仿真时间 0 调用 `bind(engine)`，随后每次 `simulationStep()` 后都调用 `synchronize()`，包括回合结束的最后一步，并在连接关闭前调用 runtime 的 `close()`。不能跳步、回退时间或用当前接口恢复仿真中途的快照。

直接调用方需自行配置禁止超时传送；涉及道路封闭时，先使用 `validate_routes(net_file, route_file)` 校验，再启用临时路线权限错误容忍。仓库内优先使用 `install_events`，由适配器处理这些要求。发生执行异常后，当前 runtime 失效，应关闭或重置该回合；不得继续使用先前的报告。

## 报告与下游文本接口

事件以 `[begin, end)` 为物理有效区间。开始时先施加并核验限制，再发布 `active`；结束时先移除该事件的作用，再发布 `cleared`。`updated_at` 是最近一次状态改变的仿真时刻，不是每次读取报告的时间。当前时序不包含报告滞后或延迟扰动。

| 接口 | 返回内容 | 使用边界 |
| --- | --- | --- |
| `runtime.reports()` | 已发生事件的最新 `Report` 元组，字段为 `event_id`、`status`、`updated_at`、`text` | 公开事实报告；每个事件只保留最新状态，待发生事件不可见，解除报告保留到 reset |
| `runtime.texts(now, intersection_ids)` | 按传入路口顺序返回字符串列表 | 各路口收到相同的报告拼接文本；只允许读取当前已同步时刻 |
| `runtime.network_catalog()` | 以 lane ID 为键的静态信息，含 edge、起终路口、方向、转向、长度等 | 可供实体映射使用，不包含事件或未来日程 |
| `runtime.audit()` | 完整日程、物理目标、实际障碍位置、状态转换、版本与哈希等 | 含未来信息和仿真真值，仅用于验证、记录与分析，不得作为策略输入 |

报告使用固定英文事实模板，不调用 LLM，也不包含信号动作建议。局部阻塞的文本示例（实际放置在 650 米时）：

```text
Event hz_lane_01. A stationary obstacle locally blocks lane road_1_1_0_2 (left-turn lane), on eastbound road road_1_1_0 approaching junction intersection_2_1, at 650.00 m from the lane start.
```

这里的 `left-turn lane` 明确描述**左转车道**，`eastbound` 描述所在道路的行驶方向，不将车道功能表述为“车辆正在左转时发生事故”。事件结束后的文本说明障碍物或该事件施加的限制已移除，不宣称拥堵已消散；存在重叠事件时，其他事件的限制仍可继续生效。

尚无事件发生时，`reports()` 返回空元组，`texts(...)` 为每个路口返回空字符串。全部事件结束后，文本仍包含解除报告；调用方可根据公开 `status` 管理自己的文本窗口。`texts()` 本身不附带 `updated_at` 字段，如研究需要该时间信息，应从 `reports()` 显式读取并记录输入协议。

`texts()` 不按真实事件目标预先分配到路口，也不生成实体掩码。现有规则映射入口使用 `reports()` 与静态 catalog，从公开文本得到映射结果，详见下一节。常规交通状态仍由现有 World/特征生成器提供，与文本共同进入策略的方式由下游实验定义。

### 障碍物与交通指标

人工障碍使用保留前缀 `__tsc_event__`。`TrafficConnectionView` 从现有 World 使用的车辆列表、出发/到达记录、车道观测、奖励相关统计、轨迹和等待/行程统计路径中排除这些对象，使人工障碍不会被算作普通交通需求。

这层过滤有明确范围：原始 SUMO 检测器、订阅结果和 edge 聚合接口没有被全面改写。下游若直接使用这些接口或原始连接，必须自行检查并排除障碍物，不能假定所有 SUMO 指标都已经过滤。

## 规则实体映射 v1

当前版本为 **`sumo-report-grounding-v1`**，固定的是受控报告解析、静态实体链接和观测特征绑定的行为；不包含学习型抽取、BERT 编码或注意力训练。[grounding_v1.json](grounding_v1.json) 保存当前源码、测试与相关运行依赖文件的 SHA-256，以及验证结果位置。它是文件级基线清单，不是 Git 发布标签、完整环境锁文件或历史 TARL V2.1 协议。修改其中的文件后，应新建版本身份并重新验证，不能覆盖此清单冒充同一版本。文档与下一阶段设计不纳入算法文件哈希。

### 模块职责与论文依据

| 位置 | 当前职责 | 参考依据及采用范围 |
| --- | --- | --- |
| [grounding.py](grounding.py)：`ReportGrounder` | 识别三类事件的 active/cleared 六种模板，抽取实体提及、字符位置和关系字段 | [PURE，NAACL 2021](https://arxiv.org/abs/2010.12812) 的实体与关系抽取分工；现版采用规则，未复现其神经模型 |
| [utils/text_grounding.py](../../utils/text_grounding.py)：`EntityCatalog` | 在静态目录中确认 ID，校验方向、转向、所属道路及起终点，展开作用范围 | [BLINK，EMNLP 2020](https://aclanthology.org/2020.emnlp-main.519/) 的提及—实体目录链接思路；现版使用精确 ID 与关系校验，没有候选向量检索模型 |
| [generator/text_entity.py](../../generator/text_entity.py)：`TextEntityBinder` | 根据真实观测生成器的顺序建立实体到状态特征位置的绑定 | 为 [EMMA，ICML 2021](https://proceedings.mlr.press/v139/hanjie21a.html) 式实体条件文本注意力准备关联输入；现版不学习注意力 |

可准确表述为“受控语言下基于规则的实体与关系抽取、事件要素结构化和实体链接”。不能表述为已实现自由文本理解、预训练模型实体识别，或注意力已改善信号控制。

### 输入与输出契约

输入是公开报告对象（`event_id/status/updated_at/text`）、静态路网目录和观测布局，而非仅一个裸字符串。目标 ID、事件类型、active/cleared 从文本识别；公开 `status` 仅校验一致性，冲突时报错，不覆盖文本。报告 ID 用于关联与去重，不根据 ID 名称猜测类型或目标。映射器不接收事件日程、未来结束时间、运行时真实目标或 `audit()`；测试端可以使用它们作标准答案。

| 输出 | 含义与边界 |
| --- | --- |
| `GroundedReport.text/mentions/relations` | 原文、实体字符区间 `[start,end)` 和静态关系；保留原文用于后续语义编码 |
| `event_kind/report_status/scope` 及数值字段 | 从公开文本解析出的要素，默认用于校验与诊断；直接送入策略需单列结构化事件基线。数值保留播报精度，不恢复隐藏真值 |
| `target_lane_ids` | 报告所指的车道集合，不等于所有受拥堵影响的车道 |
| `feature_bindings/unobserved_lane_ids` | 实体在当前观测中的位置，以及链接成功但不在观测中的车道 |
| `batch.texts` | 按输入报告顺序保留的文本元组；不把多报告提前合并池化 |
| `lane_report_mask` | `[目录车道数, 报告数]` 的布尔关联矩阵；行顺序由 `batch.lane_ids` 给出 |
| `feature_report_mask` | `[路口数, 最大原始车道特征数, 报告数]`；路口顺序由 `batch.intersection_ids` 给出 |
| `valid_feature_mask/feature_sizes` | 有效车道位置及各路口生成器实际输出长度；补零、跨路口补齐位置不作为实体 |

局部阻塞只链接指定车道；道路封闭展开指定有向 edge 的全部车道，并保留两端路口关系；全局降雨展开全部机动车车道，包括内部连接车道。绑定层只投影到实际可观测部分。封路上游的潜在排队不是直接目标标签，后续模型可以依据公开拓扑另行处理。

解除报告保留实体对应，不会变成空掩码；一条解除报告不能撤销另一条仍有效的事件。无报告返回空元组和报告维度为 0 的矩阵。当前不会自动淘汰解除报告、推断未来时刻或在 reset 后保留事件历史。

支持边界：六种固定英文模板及空白差异、明确实体 ID；SUMO 进口车道、`fns=['lane_count']`、`average=None`、`negative=False`。不支持任意改写、指代消解、聚合状态、相位/文本拼接后索引。BERT 空输入、无关联实体和 padding 的注意力处理属于后续模型职责。

默认 `strict=True`：未知模板、未知实体、关系冲突、元数据冲突等直接抛出带错误码的 `GroundingError`。`strict=False` 保留原文及失败原因，但不给出目标关联；绑定层拒绝失败报告，不静默忽略。重复报告 ID 在两种模式下均拒绝，因为接口接收最新快照而非历史消息流。

### 调用示例

以下片段接在 `install_events()` 之后；World 的创建和动作循环沿用已有调用方。每次 `world.reset()` 后重新构造观测生成器与 binder。

```python
from generator import LaneVehicleGenerator
from generator.text_entity import TextEntityBinder
from world.sumo_events import ReportGrounder

mapper = ReportGrounder(runtime.network_catalog())
world.reset()
generators = [
    LaneVehicleGenerator(world, world.id2intersection[node], ['lane_count'],
                         in_only=True, average=None, negative=False)
    for node in world.intersection_ids
]
binder = TextEntityBinder(mapper.catalog, world.intersection_ids, generators)
world._update_infos()  # 初始化刚订阅的 lane_count；后续 world.step() 会更新
grounded = mapper.map_reports(runtime.reports())
batch = binder.bind(grounded)
traffic = [g.generate() for g in generators]
# 将 batch.texts 与关联矩阵交给后续文本/注意力模型；当前模块不产生动作。
```

观测位置沿用 `LaneVehicleGenerator.lanes`，不按 catalog 排序猜测。输出矩阵只读，可与同一时刻观测一起存入 replay；binder 会拒绝 reset 后仍引用旧路口的生成器。调用方仍需按既有 World 接入要求负责关闭连接。

### 已完成验证与复验

2026-09-18 的四个相关测试文件合计 **66 passed**。真实链路使用 Python 3.10.18、SUMO 1.27.1/libsumo、现有 FixedTimeAgent、hz4x4、seed 7，两次各 245 秒，覆盖三类事件的发生/解除；60,852 次特征值核对通过，两次轨迹与映射哈希一致。该计数是重复时间步上的特征检查次数，不是独立文本数；该控制器仅驱动映射验证，不作为标准固定时制性能基线。

本地证据：[validation.json](../../data/output_data/sumo_events/grounding_validation_20260918_v2/test_hz4x4_report_feature_roun0/validation.json)，包含完整命令、源文件哈希、仿真命令、状态转换和每次重放结果；大规模产物不随 Git 分发。模块测试另含手工报告与小目录、错误关系、相似 ID、空报告、重叠解除、补零、顺序和 stale-generator 检查。路网目录作为共同静态输入，未声明独立测绘验证或自由文本泛化。

在仓库根目录、`colight` 环境复验：

```bash
python -m pytest -q tests/test_text_grounding.py tests/test_text_grounding_sumo.py \
  tests/test_sumo_events.py tests/test_sumo_event_comparison.py
```

pytest 默认使用临时目录存放新证据。需要指定 `--basetemp` 时必须使用新的输出位置：pytest 会清理该目录，不能指向已有证据或项目目录。

核验固定版本（路径相对仓库根目录）：

```python
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path('world/sumo_events/grounding_v1.json').read_text())
changed = [path for path, digest in manifest['files_sha256'].items()
           if not Path(path).is_file()
           or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
assert not changed, changed
```

### 后续学习型抽取替换位置

当前 `ReportGrounder.map_report()` 内同时组织模板抽取和目录链接；尚无可直接切换的模型后端。后续新增抽取接口时，由规则、预训练模型或 LLM 后端输出统一的提及、关系和事件要素，再复用 `EntityCatalog` 与 `TextEntityBinder`。不要修改 v1 语义来容纳新后端，应建立新的版本标识、失败/歧义处理与多表达测试。

第一版注意力实验继续固定规则抽取，将“抽取质量变化”与“模型是否有效利用文本”分开研究。下一阶段设计见 [文本实体注意力验证与对比实验](../../docs/text_entity_attention_experiment.md)，当前状态为待实现，不是已完成结果。

## 验证与已有证据

所有命令均在**仓库根目录、`colight` 环境**执行。已验证环境为 Python 3.10.18、SUMO 1.27.1；另做过 Python 3.9 语法兼容性检查。运行验证入口需要项目的 World/FixedTime 依赖，量化绘图还需 Matplotlib；环境安装见 [根目录 README](../../README.md)。

### 工程验证

```bash
python -m world.sumo_events \
  --sim-config configs/sim/hz4x4.cfg \
  --schedule configs/events/hz4x4.yml \
  --interface libsumo --seeds 7 17 --repeats 2 --seconds 3600 \
  --phase-seconds 20 \
  --output data/output_data/sumo_events/hz4x4_validation_local

python -m world.sumo_events \
  --interface traci --seeds 7 --repeats 2 --seconds 300 \
  --output data/output_data/sumo_events/hz4x4_traci_local
```

输出目录必须尚不存在；重复运行请更换目录。`--repeats` 至少为 2，仿真必须比所有事件的结束时刻至少多运行 1 秒。默认配置为 `hz4x4.yml`，默认时长 300 秒；`--interface` 可选 `libsumo` 或 `traci`。

每次运行保存 `seed_<seed>/repeat_<repeat>.json`，包含 SUMO 命令、边界报告、实际事件状态、车辆守恒与轨迹哈希；`summary.json` 汇总输入/源码哈希和检查结果。验证覆盖三类物理作用、同刻报告、速度/权限恢复、障碍物统计隔离、交通需求无丢失、无碰撞/传送及同种子 reset 重放。

针对模块与原 V2.1 协议的回归检查：

```bash
python -m pytest -q \
  tests/test_sumo_events.py \
  tests/test_sumo_event_comparison.py \
  reproduction/tarl_tsc/tests/test_runtime_v21.py \
  reproduction/tarl_tsc/tests/test_semantic_v21.py \
  reproduction/tarl_tsc/tests/test_sampling_v21.py
```

### 五条件量化与图表

```bash
python -m world.sumo_events.compare \
  --sim-config configs/sim/hz4x4.cfg \
  --schedule configs/events/hz4x4_comparison.yml \
  --seeds 7 17 27 --seconds 3600 --phase-seconds 20 \
  --output data/output_data/sumo_events/hz4x4_comparison_local
```

比较入口使用 libsumo，固定生成 `normal`、`lane_blockage`、`road_closure`、`global_rain`、`combined` 五个条件。输入要求每种事件恰好一个、共享起止窗口、至少两个不同种子，并保留事件前和结束后的观测时间。同种子下共享路线、控制器、仿真选项与时长，检查动作序列和事件前交通轨迹一致；各条件均安装相应日程，包括无事件对照。

| 输出 | 用途 |
| --- | --- |
| `manifest.json` | 完整调用、配置、日程、输入/源码哈希与指标定义 |
| `seed_<seed>/<condition>/timeline.csv` | 逐秒交通响应、物理状态和报告对齐数据 |
| `seed_<seed>/<condition>/result.json` | 分窗口指标、出发/到达/未完成车辆、物理边界检查和报告延迟 |
| `summary.json`、`comparison.csv` | 各条件均值、种子范围和相对同种子无事件对照的差值 |
| `figures/01_scope_and_alignment.*` | 作用范围与事件—报告时序 |
| `figures/02_traffic_response.*` | 排队、速度等随时间变化 |
| `figures/03_paired_metrics.*`、`04_comparison_table.*` | 配对指标及汇总表，均输出 PNG/PDF |
| `figures.json` | 绘图代码和数据哈希，支持追踪重新渲染 |

仅用已有数据重绘图表：

```bash
python -m world.sumo_events.compare \
  --output data/output_data/sumo_events/hz4x4_comparison_local --render-only
```

重绘会核验记录的运行代码、控制器和输入哈希；允许独立调整绘图样式，但不将修改后代码解释为原始实验所用版本。

### 指标口径

- **排队车辆数**：普通在途车辆中速度 `< 0.1 m/s` 的数量，报告窗口内逐秒计数的均值。封路局部区域包括被封闭 edge 及其上游路口的所有外部进入道路。
- **平均速度**：窗口内所有普通车辆速度之和除以车辆秒数，按车辆时间加权。
- **已完成行程耗时**：实际出发到到达的时间，仅统计仿真结束前到达的车辆；必须同时查看 `arrived`、`running`、`pending`，避免把未完成长行程排除后误判为改善。
- **时间窗口**：物理/报告状态为 `[begin,end)`；步后读取的交通样本反映刚结束的一步，因此交通指标使用 `(begin,end]`。这一区别不表示报告延迟。
- **误差带**：不同种子的最小—最大范围，不是置信区间。人工障碍不计入上述交通指标。

### 已有验证结果与解释范围

2026-09-18 留存的证据位于以下目录，均为可再生成的本地实验产物，通常不随 Git 分发；新克隆仓库可通过上述命令生成自己的结果：

```text
data/output_data/sumo_events/hz4x4_v1_20260918/
  verification.json
  libsumo/summary.json
  traci/summary.json
data/output_data/sumo_events/hz4x4_comparison_20260918/
  manifest.json
  summary.json
  comparison.csv
  figures/
  boundary_probe.json
```

工程验证为 libsumo 下种子 7、17 各重放两次、每次 3600 秒，以及 TraCI 下种子 7 重放两次、每次 300 秒。各后端内部同种子重放哈希一致；未发现需求丢失、碰撞或传送，四次完整运行均有 2983 辆普通车辆出发。模块测试 22 项、原 V2.1 回归 8 项和量化指标测试 2 项分别通过。

量化对比为 3 个种子 × 5 个条件，共 15 次、每次 3600 秒。36 次物理/报告状态转换的延迟均为 0 秒，162000 次逐事件、逐秒状态核对未发现不一致；各次运行无碰撞、无传送，2983 辆车全部出发且无等待插入的车辆。下表为三个种子的均值：

| 条件 | 事件窗口全网排队（辆） | 事件窗口全网速度（m/s） | 已完成行程耗时（秒） | 到达车辆数（辆） |
| --- | ---: | ---: | ---: | ---: |
| 无事件 | 3.3611 | 9.6718 | 328.7332 | 2744.00 |
| 局部车道阻塞 | 3.4633 | 9.6395 | 328.8745 | 2743.33 |
| 整段道路封闭 | 18.9756 | 9.2397 | 331.2785 | 2744.67 |
| 全局降雨 | 2.5067 | 6.7917 | 342.8790 | 2745.33 |
| 三类事件组合 | 13.3800 | 6.6150 | 344.0627 | 2745.67 |

道路封闭使封路局部区域的平均排队从 0.1767 增至 15.9033 辆；全局降雨条件的全网平均速度约下降 29.78%。局部阻塞目标车道的平均排队仅从 0.0144 变为 0.0156 辆，说明本配置中的交通扰动较弱，不能把“障碍物执行成功”直接解释为“产生强拥堵”。

使用这些结果时需保留以下限制：

1. **控制器行为**：验证和对比入口逐秒调用现有 `FixedTimeAgent`，设置 `t_fixed=20`。实际首路口动作变化为 `(0秒,0)、(20秒,1)、(21秒,2)、(22秒,3)…`，首次 20 秒后出现逐秒切换，不能解释成标准的每相位 20 秒固定时制。各配对条件的动作序列一致，结果用于事件机制展示；正式交通控制实验需先独立修正并验证控制基线。
2. **封路边界**：种子 7 的封路与组合条件各有一辆车在第 904 秒进入封闭道路。`boundary_probe.json` 的专项诊断确认该车在第 900 秒已位于路口内部连接车道。这符合前述封路权限语义，不代表报告存在 4 秒滞后；该诊断文件是本次调查的附加证据，并非比较入口的固定输出。
3. **研究结论范围**：证据覆盖当前 hz4x4、配置和种子，说明三类作用能执行、恢复并产生同步报告。它不证明所有路网适配完成、事件分布符合现实，或文本解析/注意力方法能够提升信号控制效果。降雨条件下排队数量下降也不能单独视为交通改善，应结合速度和行程耗时解释。

## 源码职责与复用来源

| 文件 | 职责 |
| --- | --- |
| [schema.py](schema.py) | 不可变 Event/Schedule/Report 定义、字段约束、YAML/JSON 加载 |
| [runtime.py](runtime.py) | 静态路网 catalog、物理状态转换、同步报告、恢复与审计 |
| [trex_blockage.py](trex_blockage.py) | 障碍物静态间距放置、插入、静止控制与移除；未覆盖动态停车距离 |
| [preflight.py](preflight.py) | 开发控制器物理检查、放置风险探测及正常/事件配对强度评估 |
| [safe_placement.py](safe_placement.py) | 显式版本的动态停车距离与障碍占位检查，实例级接入旧生命周期 |
| [integration.py](integration.py) | World 实例接入、显式路线验证、普通交通视图 |
| [__init__.py](__init__.py) | 公共导出：Event、Report、Schedule、load_schedule、SumoEventRuntime、install_events、validate_routes、ReportGrounder |
| [grounding.py](grounding.py)、[grounding_v1.json](grounding_v1.json) | 规则报告解析入口与固定版本清单；实体链接及观测绑定见上文 |
| [__main__.py](__main__.py) | libsumo/TraCI 工程验证入口 |
| [compare.py](compare.py)、[comparison_plots.py](comparison_plots.py) | 五条件配对实验、指标汇总与可追踪图表 |
| [test_sumo_events.py](../../tests/test_sumo_events.py)、[test_sumo_event_comparison.py](../../tests/test_sumo_event_comparison.py) | 事件机制、生命周期、接入及量化指标的针对性测试 |
| [test_text_grounding.py](../../tests/test_text_grounding.py)、[test_text_grounding_sumo.py](../../tests/test_text_grounding_sumo.py) | 模板解析、实体链接、特征绑定和真实 SUMO 重放验证 |

障碍物算子改编自 [T-REX](https://github.com/MLSM-at-DTU/T-REX)，固定来源为提交 `a91a3d8c4ac27982c75553b932eabb27acd63ca7` 的 `T_REX.py::Deployment.triggering`（674–728 行）。保留其 `route.add`、`vehicle.add/moveTo/setSpeed/setLaneChangeMode/remove` 操作，并增加连接注入、唯一 ID、显式停车、安全放置和普通车辆保护；没有引入完整 T-REX 框架、ICM 或驾驶人绕行策略。上游 MIT 许可保留于 [LICENSE.T-REX](LICENSE.T-REX)。

封路和降雨直接复用 SUMO TraCI 车道权限/限速接口，参考 [Change Lane State](https://sumo.dlr.de/docs/TraCI/Change_Lane_State.html)。定时限速语义可参考 [Variable Speed Signs](https://sumo.dlr.de/docs/Simulation/Variable_Speed_Signs.html)，道路关闭语义可参考 [Closing a Street](https://sumo.dlr.de/docs/Simulation/Rerouter.html#closing_a_street)；本模块实际执行使用 TraCI 调用，不生成 VariableSpeedSign 或 Rerouter XML。这些来源提供仿真执行机制，当前事件时长、位置和降雨系数由项目配置明确设定。
