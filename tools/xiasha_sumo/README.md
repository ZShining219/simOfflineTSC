# Xiasha SUMO 数据转换工具

该包把 `data/raw_data/xiasha1*1/原始数据/语义事件表(13min).csv` 的严格完整 OD 记录转换为 SUMO 逐车 route 和按时间窗口/OD 聚合的 flow，并把中心路口 `J` 适配为静态信号灯。默认输出到同级的 `data/raw_data/xiasha1*1/`，不会改写 `原始数据/`。

## 用法

在仓库根目录、`colight` 环境中运行：

```bash
/home/dev/miniforge3/envs/colight/bin/python -m tools.xiasha_sumo --help
/home/dev/miniforge3/envs/colight/bin/python -m tools.xiasha_sumo --all
```

可用参数包括 `--input`、`--output-dir`、`--network`、`--window`（默认 60 秒）、`--shift`（覆盖自动平移量）、`--strict-topology` 和 `--mode vehicles|flows|all`。逐车文件是 `xiasha1_sumo_vehicles.rou.xml`，flow 文件是 `xiasha1_sumo_flows.rou.xml`。

## 规则和信号周期

仅同时具有 `entry_edge`、`exit_edge`、`entry_time` 且时间可解析为浮点数的记录转换；其余记录逐条写入 skip 审计。跳过记录同时按互斥类别统计为 `missing_entry`、`missing_exit`、`invalid_time` 或 `multiple_errors`，并保留错误分量统计。`entry_time` 整体平移，使最早有效时间为 0（当前数据为 `+6.52 s`），相对顺序保持不变。`exit_time`、车道字段和 `observed_event_time` 不控制 SUMO 车辆运动。

movement 不依赖 edge 名称字符。转换器读取路网中入口 edge 的源 junction、中心路口 `J` 和出口 edge 的目标 junction，以入向/出向向量的点积和叉积判定 `straight`、`u_turn`、`left`、`right`。因此同向回转不会再被误判为直行，实际直行也不会再被误判为左转。

信号相位顺序及秒数为：南北直行绿 27、黄 3、过渡红 2；南北左转绿 22、黄 3；南北/东西全红 5；东西直行绿 34、黄 3、过渡红 2；东西左转绿 19、黄 3；东西/南北全红 5，总周期 128 秒。相位状态字符串按路网中受控 connection 的 `linkIndex` 顺序编码，`G/y/r` 分别表示绿/黄/红。原始 `dir="r"` 连接在每个相位均为 `G`，因此右转始终允许；直行和左转按对应方向相位放行。

## 输出和验证

`xiasha1_sumo_signal.net.xml` 将 `J` 改为 `traffic_light`，在 connection 之前写入有效 `tlLogic`，并为 16 个进入 `J` 的连接写入 `tl="J"` 和连续 `linkIndex`；`xiasha1_sumo_signal.add.xml` 保存同一份可移植信号定义。两个 `.sumocfg` 分别引用逐车/flow route，并包含有效的 begin/end 时间。

每次 `--all` 还会生成以下审计文件：

- `xiasha1_sumo_connection_movement_audit.csv`：`linkIndex`、边、车道、movement、每个相位状态和放行相位。
- `xiasha1_sumo_phase_matrix.csv`、`xiasha1_sumo_phase_matrix.txt`：12 行相位 × 16 列受控 connection 的 `G/y/r` 矩阵。
- `xiasha1_sumo_demand_audit.csv`：总量、OD、时间窗和 OD×时间窗的 valid/vehicle/flow 对照。
- `xiasha1_sumo_skip_audit.csv`、`xiasha1_sumo_skip_summary.csv`：跳过记录明细，以及按错误类别、进口方向和观测时间窗汇总。
- `xiasha1_sumo_route_topology_audit.csv`：观测 OD 是否存在对应 SUMO connection，并显示 `u_turn` 等 movement。

```bash
xmllint --noout data/raw_data/xiasha1*1/xiasha1_sumo_*.xml
/home/dev/miniforge3/envs/colight/bin/python -m py_compile tools/xiasha_sumo/*.py
/home/dev/miniforge3/envs/colight/bin/python3.10 /home/dev/miniforge3/envs/colight/bin/sumo -c data/raw_data/xiasha1*1/xiasha1_sumo_vehicles.sumocfg --duration-log.disable
```

报告中的需求守恒必须满足 `N_valid = N_vehicle = ΣN_flow`。当前原始数据仍包含 5 条字段完整但路网未定义 connection 的回转记录；默认模式保留这些记录以保持观测数量守恒，并在 topology audit 中标记。需要严格阻止这类记录时使用 `--strict-topology`。

路线使用入口边和出口边两条 edge，SUMO 会依据网络连接完成中间路径；不使用观测到的 `exit_time` 强制车辆结束。需要本机安装 `sumo` 才能运行动态仿真。
