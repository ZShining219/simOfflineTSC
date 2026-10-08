# SUMO 修正绿黄执行（sumo-green-yellow-v1）主线训练验证

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：在 run.py/TSCTrainer 主线启用 sumo-green-yellow-v1（5s 黄灯过渡、清空期锁相）后，DQN 在 hz4x4 上可正常训练收敛；与无过渡 legacy 臂的指标差异反映换相物理成本的口径变化（语义变更，GOVERNANCE §5.5，两臂指标不与历史无过渡 run 直接可比）。
- plan_id/队列：scv1_gy5_v1
- 设计稿/证据位置：artifacts/signal_control_v1/run_queue_dqn_gy5.json；configs/tsc/{dqn_gy5,dqn_gy5_smoke}.yml；适配器 world/sumo_signal_control.py；trainer/tsc_trainer.py signal_control 安装点
- 结论：进行中：2 臂 REGISTERED（dqn_gy5 + 同种子 legacy 对照），400ep train_frozen_eval seed=7。
- 登记单元 2（train/eval/smoke/batch/calibration = 2/0/0/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `scv1_dqn_gy5_s7` | dqn_gy5/hz4x4/s7 | 400 | REGISTERED | — | — | `data/output_data/tsc/sumo_dqn/hz4x4/scv1_dqn_gy5_s7` | REGISTERED 2026-10-08: sumo-green-yellow-v1, yellow_seconds=5, action_interval=1 |
| `scv1_dqn_legacy_s7` | dqn/hz4x4/s7 | 400 | REGISTERED | — | — | `data/output_data/tsc/sumo_dqn/hz4x4/scv1_dqn_legacy_s7` | REGISTERED 2026-10-08: 同种子无过渡对照臂(legacy pseudo_step),仅作 dqn_gy5 对比参照,非正式基线 |
