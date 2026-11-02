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
- 执行：attempt2 `16:38:37Z→17:18:08Z`，`256/256 SUCCEEDED`、`FAILED 0`、`rc=0`、零重启零熔断；`comparable=true`、`missing=0`；全 CPU，4 卡空闲。台账 256 行全 DONE（ledger-only `3663698`）
- 结果（优化层，pilot 档 25 ep/stage，train seeds n=2）：**L1 半离线 Coverage 1.00 / Margin +0.078 vs 在线 0.25 / −0.293，Δcov=+0.75、Δmargin=+0.371、方向 4/4**；逐格看在线只在末段 c3 覆盖，s1/c4/c2 的 ATT 分别劣于 maxpressure 34.8%/53.9%/106.1%。**S5 `flag=false` 表示分辨率足够（|Δ|=0.75 ≫ 臂内 seed 极差 0），非负面。**
- 结果（L2 外推，混合信号须双报）：覆盖 0.375 vs 0.500（半离线少 1 格/两 seed 合计 3/8 格），但 margin −0.013 vs −0.240（半离线更近阈值，Δmargin=+0.227）。`guard_effective_cells=[]`：口径 v1.1 护栏本轮零翻转，对数字无影响。
- 结论：**L1 主假设成立**（同预算下在线只保住 1/4 场景、半离线 4/4）；**L2 不支持**「半离线外推覆盖更强」，如实混合报。本 campaign 只覆盖优化层，不宣称表征/语义效用/闭环层。
- 后续：A 100 ep/stage 正式档（≈3.2 CPU-h、wall 55–75 min，成本表见实验卡 §5.3）／B L3 事件层 P5b／C order o3 复现。今晚不开跑，等用户选。
