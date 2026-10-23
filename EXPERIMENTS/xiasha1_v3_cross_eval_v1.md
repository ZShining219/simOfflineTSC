# xiasha1 v3 best-checkpoint 跨场景评估

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：用 xiasha1_online_v3（280m 修复几何）的 best checkpoint 在未参与训练的需求包上评估，测量修复口径下单场景专训策略的泛化差；与 cross_eval_v1 设计逐位一致，可直接做几何前后对照。
- plan_id/队列：xiasha1_v3_cross_eval_v1
- 设计稿/证据位置：8 臂 × 4 source（asym_nonly/asym_ns/dense_extreme/fluct_burst）× 3 train seeds × 4 target × 1 SUMO seed(10000)=384 eval；eval-only overlay configs/tsc/eval_baseline_checkpoint*.yml；run_id 前缀 xi3x_。
- 结论：已完成（2026-10-22）：384/384 DONE。相对 TT gap：SAC 9.0%、PressLight 15.4%、dueling 27.5%、ddqn 38.9%、FRAP 41.6%、MPLight 46.3%、D3QN 47.6%、DQN 49.0%——v1 截断期 DQN 家族泛化'尚可'的假象被推翻；DQN 家族灾难单元如 d3qn asym_ns→asym_nonly TT 148.9（域内 36.8），饥饿策略跨场景脆断。
- 登记单元 384（train/eval/smoke/batch/calibration = 0/384/0/0/0）（追溯登记 2026-10-07）

评估 attempt 共 384 行，按包/来源聚合：

| 评估包/来源 | n DONE/FAILED/ABORTED | 条件集 | eval seeds |
|---|---|---|---|
| `frozen_best_checkpoint` | 384/0/0 | asym_nonly->asym_nonly,asym_nonly->asym_ns,asym_nonly->dense | 10000 |
