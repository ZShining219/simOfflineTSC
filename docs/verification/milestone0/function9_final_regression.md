# 功能 9：Milestone 0 整体回归

- 日期：2026-07-22
- branch：`codex/milestone0-experiment-infrastructure`
- baseline_commit：`73d860bb3924ec15c30433a8f8b7af17787baeff`
- training_seed：三项均为 41
- world/interface/network：SUMO / libsumo / `sumohz1x1`

## FixedTime 完整环境回归

- episodes/simulation steps：1 / 3600
- 命令：`SUMO_HOME=/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo /home/dev/miniforge3/envs/colight/bin/python3.10 run.py -w sumo -a fixedtime -n sumohz1x1 --prefix m0f9_fixedtime_3600_20260722 --seed 41 --interface libsumo`
- 退出码/状态：0 / 已完成
- 指标：travel time 233.164929；reward -114.3495；queue 68.791667；delay 4.399822；throughput 1631。
- 产物：`data/output_data/tsc/sumo_fixedtime/sumohz1x1/m0f9_fixedtime_3600_20260722/`。
- 内容断言：FINAL_EVALUATION simulation_step=3600、decision_step=360、global/gradient=0；runtime controller 为 FixedTimeAgent、t_fixed=30、model/optimizer=null；6 个配置/模型文件哈希回读通过。

## MaxPressure 完整环境回归

- episodes/simulation steps：1 / 3600（临时将仅用于记录的 episodes 设为 1，运行后恢复）
- 命令：`SUMO_HOME=/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo /home/dev/miniforge3/envs/colight/bin/python3.10 run.py -w sumo -a maxpressure -n sumohz1x1 --prefix m0f9_maxpressure_3600_20260722 --seed 41 --interface libsumo`
- 退出码/状态：0 / 已完成
- 指标：travel time 80.514474；reward -32.7546；queue 10.261111；delay 4.147435；throughput 1969。
- 产物：`data/output_data/tsc/sumo_maxpressure/sumohz1x1/m0f9_maxpressure_3600_20260722/`。
- 内容断言：FINAL_EVALUATION simulation_step=3600、decision_step=360；runtime controller 为 MaxPressureAgent、t_min=10、model/optimizer=null；全部归档哈希通过。

## DQN 受控更新回归

- episodes/simulation steps：1 / 训练 700 + episode 内评估 100 + 最终评估 100
- 临时参数：learning_start=64、batch_size=64、test_when_train=true；满足 replay_size≥batch_size，保持原项目严格 `>` 条件；运行后全部恢复。
- 命令：`SUMO_HOME=/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo /home/dev/miniforge3/envs/colight/bin/python3.10 run.py -w sumo -a dqn -n sumohz1x1 --prefix m0f9_dqn_updates_20260722 --seed 41 --interface libsumo`
- 退出码/状态：0 / 已完成
- 训练指标：travel time 132.776423；reward -67.172140；queue 45.557143；delay 0.425783；throughput 246；loss 641.515930。
- 最终评估：travel time 66.857143；reward -13.4550；queue 10.3；delay 0.256329；throughput 7。
- counter：TRAIN 为 700 simulation/70 decision/6 gradient；EVALUATION 与 FINAL_EVALUATION 均保持 global=70、gradient=6；resumable checkpoint 记录 target_updates=1、replay=70。
- checkpoint：evaluation episode 1 online hash `1d2400849c92e358f093448f016c45039ebdae11a1bfff0f46ce1dc861ff98f4`；resumable schema/RNG/optimizer/counter/replay 完整；legacy `model/1_0.pt` 无 checkpoint_type。
- 产物：`data/output_data/tsc/sumo_dqn/sumohz1x1/m0f9_dqn_updates_20260722/`。

## 全局验证与清理

- `SUMO_HOME=... python -m unittest discover -s tests -p 'test_milestone0_*.py' -v`：23 项通过。
- `python -m py_compile ...`：全部受影响 Python 文件通过。
- `git diff --check` 及 baseline-to-HEAD `git diff --check`：通过。
- final artifact parser：三项 manifest/status/config hash/model/JSONL/checkpoint 逐字段断言通过。
- pre-archive 初始化失败补充 smoke：退出码 1，manifest config_hash=null，status=失败、started_at=null、error_type=RuntimeError。
- `configs/sim/sumohz1x1.cfg` 运行前后 SHA-256 均为 `314f1915c21344269dfb096f71d48741b5d16205b49b0c242a0f962796bb9dbd`。
- 临时 DQN/MaxPressure YAML 已恢复；无 simulator cfg 运行时写回；Git 不跟踪任何 `data/output_data` 文件。

已知环境警告：旧 Gym 弃用提示；可选 `torch-cluster`/`torch-spline-conv` CUDA 扩展符号不兼容而被 PyG 禁用。本次 DQN/传统控制器 CPU 路径未使用这些扩展，回归未受影响。
