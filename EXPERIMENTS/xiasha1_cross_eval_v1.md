# xiasha1 v2 best-checkpoint 跨场景评估

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：用 xiasha1_online_v2 的 best checkpoint 在未参与训练的需求包上评估，测量单场景专训策略的泛化差；比较 8 臂在 TT/throughput/uninserted/queue 上的跨场景退化。
- plan_id/队列：xiasha1_cross_eval_v1
- 设计稿/证据位置：docs/experiments/2026-10-22_xiasha1_cross_eval.md；8 臂 × 4 source（asym_nonly/asym_ns/dense_extreme/fluct_burst）× 3 train seeds × 4 target × 1 SUMO seed(10000)=384 eval；source=target 为同场景控制；评估-only overlay configs/tsc/eval_baseline_checkpoint.yml。
- 结论：进行中：384 eval REGISTERED（2026-10-22）；结果回填后更新本行。
- 登记单元 384（train/eval/smoke/batch/calibration = 0/384/0/0/0）（追溯登记 2026-10-07）

评估 attempt 共 384 行，按包/来源聚合：

| 评估包/来源 | n DONE/FAILED/ABORTED | 条件集 | eval seeds |
|---|---|---|---|
| `frozen_best_checkpoint` | 0/0/0 | asym_nonly->asym_nonly,asym_nonly->asym_ns,asym_nonly->dense | 10000 |
