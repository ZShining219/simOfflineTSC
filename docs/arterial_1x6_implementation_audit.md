# arterial_1x6 共享 DQN 实现定位审计

审计日期：2026-07-29；设备：linux4090。

## 现有实现定位

1. 单路口 DQN 状态来自进口车道 `lane_count`，并按配置拼接当前相位（原正式配置为 8 维车道数加 8 维相位 one-hot）。
2. DQN 动作数取 `len(intersection.phases)`。SUMO World 从信号程序发现合法绿相位，环境在相位切换时自动插入黄灯。
3. `TSCEnv.step` 原本返回按 agent 组织的 observation/reward；旧 DQN 每个 agent 写一条 replay。环境本身已经能够一次接收与全部路口对应的动作数组。
4. `agent/dqn.py` 是 vanilla DQN，不是 Double DQN：target network 对下一状态直接取最大 Q。
5. online/target 均为 20-20 隐层 MLP；target 按 trainer 的 `update_target_rate` 硬同步。
6. 旧 replay 是有界 `deque`，FIFO 淘汰，默认容量 5000；记录为 key 加六字段 legacy payload。
7. 纯离线线路由 `offline_run.py`、`trainer/offline_tsc_trainer.py`、`agent/offline_dqn.py` 和轨迹数据集组成，服务于既有单路口 Plan 2。
8. 仓库已有 `sequential/hybrid*.py` 的混合 replay，但其来源维度围绕旧单路口网络/阶段组织，不能直接表达六路口 `scene × intersection`。
9. 既有顺序训练支持 clear、FIFO 和 matched-wait；clear 会在场景切换时清空 replay，FIFO 保留并自然覆盖。
10. 既有顺序线路保存并恢复 epsilon；本次扩展显式提供 reset、continue、fixed-low 三种模式。
11. 旧 DQN reward 是进口车道等待车辆数的平均值取负后乘 12，聚合窗口为一个 action interval 内逐秒 reward 的平均。
12. 既有 evaluation 有冻结网络、禁止 remember、保持 RNG/optimizer/replay 不变的隔离保护，但部分状态诊断硬编码单路口 8 维形状。
13. `TSCEnv` 和 SUMO World 已有多路口遍历及联合动作下发能力；旧标准 DQN 会为每个路口建立独立网络，不符合本 Goal 的参数共享要求。
14. 四个场景位于 `data/raw_data/arterial_1x6_sumo/`，公共路网为 `arterial_1x6.net.xml`；控制路口为 `intersection_1_1` 至 `intersection_6_1`。
15. 六个控制路口的信号程序完全一致：每个路口有 8 个绿相位，XML 中每个绿相位后有 5 秒过渡相位；共享网络输出维度可稳定设为 8。

## 术语映射与实现方向

- Goal 中的“共享参数、分散决策、联合执行”映射为一个 `SharedDQNAgent`，其 `sub_agents=N`，一次批量前向产生 N 个本地动作，`TSCEnv` 仍向 SUMO 提交长度 N 的动作数组。
- Goal 中的 online replay 与 offline history 分别映射为当前场景有界 deque 和带 manifest 的只读历史池。
- 每个环境 decision 拆成 N 条 `LocalTransition`；场景 ID 只出现在 metadata，不进入网络输入。
- pressure 使用 SUMO World 已有定义：进口车道车辆数之和减出口车道车辆数之和；reward 为其负值，不做 capacity 归一化。
- 网络级 reward 仅用于报告时取局部 reward 平均；DQN 更新使用每个路口自己的局部 reward。

## 已知指标边界

SUMO World 当前能可靠提供全网完成车辆数和路口进出口车道状态，但没有维护“车辆在本窗口从某个具体路口完成放行”的 ID 级事件。因此新增路口日志将 `discharge_count` 明确记为缺失并记录原因，不以出口车道存量冒充局部吞吐量。正式实验若必须获得该指标，需要另行增加车辆越过 stop-line 的事件跟踪。
