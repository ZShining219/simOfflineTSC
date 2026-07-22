# 功能 4：结构化训练与评估日志验证

- 日期：2026-07-22
- 自动检查：`python -m unittest discover -s tests -p 'test_milestone0_*.py' -v`，11 项通过；结构化日志覆盖完整追加、缺失/额外字段、非法 record_type、缺失/空/损坏文件失败路径。
- schema：`metrics/records.jsonl` 每行一个 UTF-8 JSON 对象，字段顺序和集合严格为契约规定的 19 个字段；只允许 `TRAIN`、`EVALUATION`、`FINAL_EVALUATION`；禁止 NaN；Runner 在标记完成前重新读取验证。

## DQN 更新 smoke

- agent/network/training_seed：`dqn` / `sumohz1x1` / `23`
- episodes/simulation steps：1 / 训练 700 + 最终评估 100
- 完整命令：`SUMO_HOME=/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo /home/dev/miniforge3/envs/colight/bin/python3.10 run.py -w sumo -a dqn -n sumohz1x1 --prefix m0f4_dqn_updates_20260722 --seed 23 --interface libsumo`
- 退出码：0
- 训练记录：simulation_step=700、decision_step=70、global_decision_step=70、gradient_updates=6、loss_mean=875.1952514648438、epsilon=0.0970372509。
- 评估记录：simulation_step=100、decision_step=10；global_decision_step 和 gradient_updates 仍为 70/6；loss_mean=null。
- 更新断言：同配置 in-process Runner `m0f4_dqn_counter_assert_20260722` 直接断言 gradient_updates=6、target_updates=1。
- 关键指标：训练 travel time 172.134831、reward -78.083566、queue 52.7、delay 0.497360、throughput 267；最终评估 travel time 61.6、reward -10.26、queue 7.9、delay 0.225034、throughput 10。
- 产物：`data/output_data/tsc/sumo_dqn/sumohz1x1/m0f4_dqn_updates_20260722/metrics/records.jsonl`；原 BRF/DTL 文本日志同时存在。

## FixedTime null smoke

- agent/network/training_seed：`fixedtime` / `sumohz1x1` / `23`
- episodes/simulation steps：1 / 100
- 产物：`data/output_data/tsc/sumo_fixedtime/sumohz1x1/m0f4_fixedtime_nulls_20260722/metrics/records.jsonl`。
- 断言：仅一条 `FINAL_EVALUATION`；global_decision_step=0、gradient_updates=0；loss_mean=null、epsilon=null；travel time 79.5、throughput 4。

临时 DQN/FixedTime YAML 已恢复；simulator source cfg 哈希仍为 `314f1915c21344269dfb096f71d48741b5d16205b49b0c242a0f962796bb9dbd`。计数器仅观察既有成功调用，不改变 learning-start、更新频率或指标公式。
