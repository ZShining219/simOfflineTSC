# 功能 6：Checkpoint 语义与恢复验证

- 日期：2026-07-22
- 原项目证据：`agent/dqn.py::save_model` 仅将 target state_dict 保存到旧 `model/<episode>_<rank>.pt`；该文件不带 schema/type，不能完整恢复训练。
- 自动检查：Milestone 0 unittest 共 17 项通过；覆盖 evaluation online-only、evaluation 实际加载、resumable round-trip、非法类型、缺失字段、损坏文件和 config_hash 不匹配。
- 原子保存首轮发现：PyTorch 1.11 拒绝隐藏且无 `.pt` 后缀的临时文件；改为同目录普通 `tmp-checkpoint-*.pt` 后通过，仍使用 `os.replace` 原子发布。

## 真实保存与恢复 smoke

- agent/network/training_seed：`dqn` / `sumohz1x1` / `31`
- episodes/simulation steps：1 / 训练 700 + 最终评估 100
- prefix：`m0f6_checkpoint_resume_v2_20260722`
- 退出码：0
- 训练结果：70 decisions、6 gradient updates、1 target update、replay 70；最终评估 travel time 58.5、reward -16.08、queue 11.8、delay 0.2315、throughput 2。
- evaluation checkpoint：`checkpoints/evaluation/episode_0001.pt`；agent 字段严格为 rank 与 online_model_state_dict；online hash 为 `b8723d81a333828f0a8ef50fdf38e4a999299d79f5cc77279a54518993d28683`，等于保存后执行最终评估的 online 状态；不含 target/optimizer。
- resumable checkpoint：`checkpoints/resumable/episode_0001.pt`；保存 online、target、optimizer、epsilon、replay capacity/items、Python/NumPy/PyTorch CPU/可用 CUDA RNG 和 global/gradient/target/epoch/step counters。
- 恢复验证：主动清空 replay、修改 epsilon/counter 后加载；恢复 replay=70、global_decision_step=70、gradient_updates=6、target_updates=1；随后通过训练器统一更新入口实际执行一次 optimizer update，online hash 发生变化、gradient_updates=7、loss=689.604309。训练循环的本地 decision counter 也从恢复的 global counter 开始，避免恢复后重置 schedule。
- legacy 边界：`model/1_0.pt` 仍是旧 target-only state_dict，断言不含 `checkpoint_type`，且未被新目录覆盖或误标。
- schema：两类顶层均包含 schema_version、checkpoint_type、episode、global_decision_step、gradient_updates、config_hash、agents；类型只允许 evaluation/resumable。

临时 DQN YAML 已恢复，source simulator cfg 哈希不变。未实现 best checkpoint 选择，未改变 DQN 更新算法。
