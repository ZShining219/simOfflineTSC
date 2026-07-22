# 功能 3：Seed 与可重复性验证

- 日期：2026-07-22
- README 已明确：未限定 seed 为 `training_seed=run.py --seed`；不管理或传递 SUMO seed；记录 `sumo_seed_mode=fixed_default`。
- 自动检查：Milestone 0 unittest 共 8 项通过；模型 state_dict 哈希对键顺序稳定、对 tensor 内容变化敏感；Python/NumPy 探针前后 RNG 状态完全一致。
- 独立运行：DQN / `sumohz1x1` / training_seed 19 / 每次 1 episode、训练 100 + 最终评估 100 simulation steps；prefix 分别为 `m0f3_same_seed_a_20260722`、`m0f3_same_seed_b_20260722`；两次命令除 prefix 外一致，退出码均为 0。
- 两次运行指标一致：训练 travel time 70.8571428571、reward -12.3599998、queue 9.6、delay 0.2710942、throughput 7；最终 travel time 79、reward -13.125、queue 9.9、delay 0.1776、throughput 6。
- 初始化签名一致：online hash 与 target hash 均为 `7baa6959e00be52a902457aa784d98f3ade77be28802ef4e6523e1ac3c7a6d79`，online/target 初始一致；8 个 Python random 和 8 个 NumPy random 样本逐值一致。
- 模型哈希按排序键名、dtype、shape 和 contiguous CPU tensor 原始字节生成，不使用 torch.save 文件哈希。
- 临时 DQN YAML 已恢复，未新增 SUMO seed 参数，未消耗正式运行 RNG 序列。
