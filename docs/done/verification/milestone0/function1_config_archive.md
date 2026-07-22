# 功能 1：配置与运行时模型归档验证

- 日期：2026-07-22
- baseline_commit：`73d860bb3924ec15c30433a8f8b7af17787baeff`
- 自动检查：`/home/dev/miniforge3/envs/colight/bin/python -m unittest discover -s tests -p test_milestone0_config_archive.py -v`，3 项通过；覆盖两个 prefix 并发解析、重复 prefix、归档损坏和缺失文件。
- 语法检查：`/home/dev/miniforge3/envs/colight/bin/python -m py_compile run.py common/interface.py trainer/base_trainer.py utils/logger.py agent/dqn.py tests/test_milestone0_config_archive.py`，退出码 0。
- source cfg 校验：`sha256sum configs/sim/sumohz1x1.cfg`，运行前后均为 `314f1915c21344269dfb096f71d48741b5d16205b49b0c242a0f962796bb9dbd`。

## FixedTime smoke

- agent/network/training_seed：`fixedtime` / `sumohz1x1` / `0`
- episodes/simulation steps：1 / 100（临时设置 `test_steps=100`，运行后已恢复）
- 命令：`SUMO_HOME=/home/dev/miniforge3/envs/colight/share/sumo /home/dev/miniforge3/envs/colight/bin/python3.10 run.py -w sumo -a fixedtime -n sumohz1x1 --prefix m0f1_fixedtime_20260722 --seed 0 --interface libsumo`
- 退出码：0
- 关键指标：travel time 79.5000；reward -27.1650；queue 10.0000；delay 1.9417；throughput 4。
- 产物：`data/output_data/tsc/sumo_fixedtime/sumohz1x1/m0f1_fixedtime_20260722/`。
- 内容校验：7 个归档文件全部通过 SHA-256 回读；`model_resolved.json` 记录 `model=null`、`optimizer=null`、`t_fixed=30`。

## DQN smoke

- agent/network/training_seed：`dqn` / `sumohz1x1` / `0`
- episodes/simulation steps：1 / 训练 100 + 最终评估 100（临时关闭 episode 内评估，运行后已恢复）
- 命令：`SUMO_HOME=/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo /home/dev/miniforge3/envs/colight/bin/python3.10 run.py -w sumo -a dqn -n sumohz1x1 --prefix m0f1_dqn_20260722 --seed 0 --interface libsumo`
- 退出码：0
- 关键指标：训练 travel time 59.5、reward -11.549999、queue 8.9、delay 0.256344、throughput 4；最终评估 travel time 63.4、reward -15.075、queue 11.2、delay 0.2821、throughput 5。
- 产物：`data/output_data/tsc/sumo_dqn/sumohz1x1/m0f1_dqn_20260722/`。
- 内容校验：全部归档哈希通过；实际模型为 16→20→20→8 ReLU，online/target 初始一致，optimizer 为 RMSprop，loss 为 MSELoss。

## 已知警告与行为边界

- Gym 弃用提示和可选 PyG CUDA 扩展符号警告不影响本次 CPU/SUMO 路径。
- 首次以模块名调用 unittest 时命中环境内同名 `tests` 包并触发无效 `nvidia-smi`；改用文件发现方式后测试通过。
- 未改变 state、reward、action、DQN loss、epsilon、learning-start、更新频率、replay sampling、指标公式或实验预算。
