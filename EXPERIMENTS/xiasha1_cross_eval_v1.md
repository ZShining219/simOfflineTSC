# xiasha1 v2 best-checkpoint 跨场景评估

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：用 xiasha1_online_v2 的 best checkpoint 在未参与训练的需求包上评估，测量单场景专训策略的泛化差；比较 8 臂在 TT/throughput/uninserted/queue 上的跨场景退化。
- plan_id/队列：xiasha1_cross_eval_v1
- 设计稿/证据位置：docs/experiments/2026-10-22_xiasha1_cross_eval.md；8 臂 × 4 source（asym_nonly/asym_ns/dense_extreme/fluct_burst）× 3 train seeds × 4 target × 1 SUMO seed(10000)=384 eval；source=target 为同场景控制；评估-only overlay configs/tsc/eval_baseline_checkpoint.yml。
- 结论：已完成（2026-10-22）：384/384 DONE。目标相对 TT gap：FRAP 0.9%、MPLight 1.6%、DQN 家族 3.9–4.8%、SAC 7.6%、PressLight 12.0%；绝对跨源 TT：DQN 家族 13.87–13.97 最好，FRAP/MPLight 14.15–14.17，SAC 14.44，PressLight 15.33。uninserted 受同场景 checkpoint 选择影响，不能直接当泛化增益。
- 登记单元 384（train/eval/smoke/batch/calibration = 0/384/0/0/0）（追溯登记 2026-10-07）

评估 attempt 共 384 行，按包/来源聚合：

| 评估包/来源 | n DONE/FAILED/ABORTED | 条件集 | eval seeds |
|---|---|---|---|
| `frozen_best_checkpoint` | 384/0/0 | asym_nonly->asym_nonly,asym_nonly->asym_ns,asym_nonly->dense | 10000 |
