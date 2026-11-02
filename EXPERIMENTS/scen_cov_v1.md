# scen_cov_v1 场景覆盖率：半离线 vs 纯在线（pilot）

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`
> 状态：设计完成，待 P5 派发（无结果）

- 假设：同一训练预算 / 同一场景顺序 / 同一 seeds 下，半离线（`offline_ratio=0.25` 仅 st2–4 生效，全程有效混合比 ≈18.75%）控制器的场景覆盖率高于纯在线控制器；即 ΔCoverage_L1 ≥ +0.25 且逐 cell 方向 ≥3/4 一致，覆盖裕度同步更高。
- 场景与口径：[docs/scenario_taxonomy.md](../docs/scenario_taxonomy.md)（δ=0.05、η=0.10 冻结；**v1.1 已批，只批 L2**：`unfinished` 加绝对地板 `max(1.10·MP, MP+2)`，L1 原式不动，见卡 §3.3）
- 口径变更记录：v1.0 冻结于 2026-11-02 设计时；v1.1（2026-11-03 00:10 课题主管批，只批 L2）因 L2 空载格阈值退化提出（数据评测复核 2.1），**派发前 0 run，不触发重打基线**
- 实验卡：[docs/experiments/2026-11-02_scen_cov_v1.md](../docs/experiments/2026-11-02_scen_cov_v1.md)
- plan_id / 队列：`scen_cov_v1`；队列由 `dev/scen_cov/build_pilot.py` 生成（待工程开发产出），落执行运维工作树 `artifacts/scen_cov_v1/run_queue.json`（落点待裁决，见卡片 §3.9）
- 场景集（冻结）：L1 = s1 / c4 / c2 / c3（order o2，训练见过）；L2 = sparse_ultra / dense_extreme / lt_extreme / asym_wonly（训练未见，exploratory）；L3 事件层只定义不跑
- 臂：`ddqn_clear`（在线，offline_ratio=0，切场景清 replay）/ `soddqn`（半离线，offline_ratio=0.25、causal history、scene_intersection_balanced）；train seeds 0/1（确认性复跑加 2）；eval sumo seeds 101–103
- 预算：单链 ≈26 min（4 阶段 × 25 ep）；4~6 链；无 GPU（CPU 口径）；硬顶 600 train ep / 324 eval attempts
- 停止条件：卡片 §3.5 八条（S1 冒烟 / S2 链中断 / S3 饱和 / S4 全灭 / S5 分辨率 / S6 预算顶 / S7 外部守护进程 / S8 负结果）
- 结论：**待填**（P5 完成后按四层判定口径陈述，本 campaign 只覆盖优化层；负结果同样回填）
