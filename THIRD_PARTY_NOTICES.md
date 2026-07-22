# 第三方实现说明

## d3rlpy

- 项目：[`takuseno/d3rlpy`](https://github.com/takuseno/d3rlpy)
- 固定版本：`2.0.4`
- 许可证：Apache License 2.0
- 用途：Plan 2 Offline RL 的上游依赖、算法定义和实现参考。

本仓库没有复制 d3rlpy 源文件。`agent/offline_dqn.py` 提供项目适配层，保留 LibSignal 当前 DQN 的网络、RMSprop、全 Q 向量 MSE、plain-DQN target 和始终 bootstrap 语义。不能直接调用 d3rlpy 2.0.4 的 stock DiscreteDQN/DiscreteCQL loss，因为其 chosen-action Huber 和 Double-DQN CQL 语义会改变本项目既定对照。

安装、使用和再分发时应同时遵守 d3rlpy 的 Apache-2.0 许可证。其他上游来源继续见根目录 README 的“项目来源”。
