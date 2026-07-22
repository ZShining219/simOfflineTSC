# 功能 7：指标能力与基础诊断审计

本功能只审计并 characterization 当前公式，没有修改任何指标、reward 或默认口径。

## 当前代码口径

- travel time：`world/world_sumo.py::get_average_travel_time` 委托 `get_vehicles()`；`self.vehicles` 只在车辆到达时记录实际进入至到达的时长，因此当前值是已完成车辆平均旅行时间；尚无完成车辆时返回 0。
- throughput：`world/world_sumo.py::get_cur_throughput` 返回 `len(self.vehicles)`，即当前 episode 已完成车辆数，不是瞬时路网车辆数（docstring “whole roadnet at current step”易误解）。
- approximate delay：`get_lane_delay` 对每条 lane 计算 `1 - lane_avg_speed/speed_limit`，空 lane 置 0；agent 将 incoming lane delay 求和，`Metrics.delay()` 再按 decision 数和 intersection 数平均。默认 `run.py --delay_type apx`。
- real delay：`Metrics.delay()` 在未订阅 lane delay 时直接调用 `world.get_real_delay()`；该实现按车辆 lane trajectory 累加 `max(actual_lane_time - planned_lane_time, 0)` 并对已有车辆平均。当前无车辆时存在除零风险，未在 Milestone 0 修复，以免改变指标语义。
- queue：SUMO intersection observation 在 `vehicle.getWaitingTime(v) > 0` 时增加 `lane_waiting_count`；agent 对 incoming lanes 求和；`Metrics.queue()` 按 decision 数和 intersection 数平均。
- reward 聚合：`Metrics.rewards()` 对每次传入的 intersection reward 累加，最终返回所有 intersection 累积和除以 decision 数；结构化 `reward_mean` 沿用该值，`reward_sum` 只记录其底层累积和。
- DQN reward：incoming lane `lane_waiting_count` 先取全 lane 平均、取负，再乘 12。
- FixedTime/MaxPressure reward：incoming lane `lane_count` 先取全 lane 平均、取负，再乘 12。因此不同控制器的 reward 数值不是同一底层量，已登记 RD-009。

## 自动验证

- 命令：`SUMO_HOME=... python -m unittest discover -s tests -p 'test_milestone0_*.py' -v`。
- characterization：dummy world 两 intersection、两次 decision 下，当前 reward=-6、queue=4、approximate delay=2、throughput=7、travel time=12.5；real delay 分支原样委托 world 并返回 2.5。
- 静态证据命令：`rg -n 'get_average_travel_time|get_cur_throughput|get_real_delay|lane_waiting_count|lane_delay|reward_generator' world/world_sumo.py generator common/metrics.py agent/{dqn,fixedtime,maxpressure}.py`。
- `git diff` 确认本功能没有修改 `common/metrics.py`、`world/`、`agent/`、`generator/` 或 `run.py`。

## 科研边界

- RD-006 继续约束正式实验的 delay 主口径选择；Milestone 0 只记录默认和两条现有计算路径。
- RD-009 约束跨控制器 reward 的比较和报告；该事项不阻塞基础设施或 travel time/queue/delay/throughput 记录。
