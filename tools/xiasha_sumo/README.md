# Xiasha SUMO 数据转换工具

该包把 `data/raw_data/xiasha1*1/原始数据/语义事件表(13min).csv` 的严格完整 OD 记录转换为 SUMO 逐车 route 和按时间窗口/OD 聚合的 flow，并把中心路口 `J` 适配为静态信号灯。默认输出到同级的 `data/raw_data/xiasha1*1/`，不会改写 `原始数据/`。

## 用法

在仓库根目录、`colight` 环境中运行：

```bash
/home/dev/miniforge3/envs/colight/bin/python -m tools.xiasha_sumo --help
/home/dev/miniforge3/envs/colight/bin/python -m tools.xiasha_sumo --all
```

可用参数包括 `--input`、`--output-dir`、`--network`、`--window`（默认 60 秒）、`--shift`（覆盖自动平移量）和 `--mode vehicles|flows|all`。逐车文件是 `xiasha1_sumo_vehicles.rou.xml`，flow 文件是 `xiasha1_sumo_flows.rou.xml`。

## 规则和信号周期

仅同时具有 `entry_edge`、`exit_edge`、`entry_time` 的记录转换；缺失任一字段的记录严格跳过，原因写入 JSON 报告。`entry_time` 整体平移，使最早有效时间为 0（当前数据为 `+6.52 s`），相对顺序保持不变。`exit_time` 只用于统计说明，不控制 SUMO 离场。

信号相位顺序及秒数为：南北直行绿 27、黄 3、过渡红 2；南北左转绿 22、黄 3；南北/东西全红 5；东西直行绿 34、黄 3、过渡红 2；东西左转绿 19、黄 3；东西/南北全红 5，总周期 128 秒。相位状态字符串按路网中受控 connection 的 `linkIndex` 顺序编码，`G/y/r` 分别表示绿/黄/红。原始 `dir="r"` 连接在每个相位均为 `G`，因此右转始终允许；直行和左转按对应方向相位放行。

## 输出和验证

`xiasha1_sumo_signal.net.xml` 将 `J` 改为 `traffic_light` 并为 16 个进入 `J` 的连接写入 `tl="J"` 和连续 `linkIndex`；`xiasha1_sumo_signal.add.xml` 保存 `tlLogic`；两个 `.sumocfg` 分别引用逐车/flow route。`xiasha1_sumo_conversion_report.json` 记录输入、数量、跳过原因、shift、窗口和信号连接数。

```bash
xmllint --noout data/raw_data/xiasha1*1/xiasha1_sumo_*.xml
/home/dev/miniforge3/envs/colight/bin/python -m py_compile tools/xiasha_sumo/*.py
sumo -c data/raw_data/xiasha1*1/xiasha1_sumo_vehicles.sumocfg --duration-log.disable
```

已知限制：路线使用入口边和出口边两条 edge，SUMO 会依据网络连接完成中间路径；不使用观测到的 `exit_time` 强制车辆结束；需要本机安装 `sumo` 才能运行动态仿真。
