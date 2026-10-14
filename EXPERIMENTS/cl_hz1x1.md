# cl_hz1x1 同质单路口持续学习（cl1 主线 + so1 半离线 + soa1 输出锚定）

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：值迭代型 TSC-RL 在顺序场景训练下发生机制性灾难性遗忘；25% 因果历史离线混合应以架构无关方式消除遗忘且不损害新场景学习；在离线样本上加 stage 末教师输出锚定（CLEAR 式 policy cloning 的 DQN 移植）应在同回放预算下进一步压缩残余退化。
- plan_id/队列：cl_hz1x1_v1 / cl_hz1x1_anchor_{screen,main,main_o3}
- 设计稿/证据位置：artifacts/cl_hz1x1/{run_queue_cl1.json, run_queue_so1.json, run_queue_soa_*.json}；dev/cl_hz1x1/{build_queue,build_queue_so,build_queue_soa,harvest_matrix,harvest_soa,plot_so_compare,plot_soa_compare}.py；矩阵 artifacts/cl_hz1x1/matrix_report{,_so,_soa_soa1}.json；图 artifacts/cl_hz1x1/fig_{so,soa}_{retention_trajectory,worst_degradation}.png；锚定实现 agent/{dqn,frap}.py::_anchor_loss，诊断经 metrics schema v5 train_diagnostics 落盘
- 结论：cl1 1476/1476 + so1 612/612 + soa screen 306/306 + soa1 主矩阵 612/612 + soa3 复现 306/306 全 DONE。①ddqn/duel/frap 三骨干在 clear 下均崩（o2 最深退化 +597/+282/+140s），遗忘为 off-policy 机制性失败非实现特例；②结构调制易感性（duel>ddqn>frap）但不免疫；③25% 因果半离线三臂 18/18 链最深退化 ≤+9s，对角线新学习近似保持——结论=抗遗忘修复，非绝对性能提升。PPO 为 on-policy 未入 so 臂。④soa1（o2, n=3，口径=全部旧场景 cell 最深 ΔATT）：policy 锚 λ_π=1 把 worst 退化 ddqn 1.54→0.44s、duel 5.23→0.90s、frap 9.79→0.99s，对角线 ΔATT 均值变化 ≤0.4s 近似零塑性代价；λ_π=10 双指标恶化（筛除非单调剂量响应）；+value 锚 λ_q=0.1 一致略逊于纯 policy 锚（同预算下钉旧 Q 值与 TD 拉扯）。⑤o3 复现（仅 +P 臂）：worst 退化 ddqn 3.91→0.35s、duel 5.96→0.56s、frap 0.73→−0.03s（全 cell 优于基线），跨序一致；sd1 链 st3 对角线单点 +2.4s 为唯一塑性毛刺（臂均值仍 −0.10s）。⑥ρ 剂量-响应（soar, o2, n=3）：+P@0.25 全面优于 replay@0.5（0.17/0.39/0.65 vs 0.92/1.67/1.25）——锚定一半预算胜过加倍回放；+P@0.125 对 frap/duel 压过 replay@0.5（frap 1.05 vs 1.25、duel 1.45 vs 1.67）且远低于 replay@0.25 的 frap 7.03，ddqn 上则不及（1.48 vs 0.92）；+P 非单调——ρ=0.5 时 worst 回升（ddqn 0.17→1.00、duel 0.39→0.93），历史占比过高稀释当前学习信号，ρ=0.25 为甜点；对角线全程 ≤0.4s。216 行 SCOPE_DISCARD 为超范围裁决留档。
- 登记单元 4996（train/eval/smoke/batch/calibration = 612/4384/0/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `cl1_ddqn_base_c2_sd0` | ddqn_base/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c2 |
| `cl1_ddqn_base_c2_sd1` | ddqn_base/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c2 |
| `cl1_ddqn_base_c2_sd2` | ddqn_base/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c2 |
| `cl1_ddqn_base_c3_sd0` | ddqn_base/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c3 |
| `cl1_ddqn_base_c3_sd1` | ddqn_base/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c3 |
| `cl1_ddqn_base_c3_sd2` | ddqn_base/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c3 |
| `cl1_ddqn_base_c4_sd0` | ddqn_base/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c4 |
| `cl1_ddqn_base_c4_sd1` | ddqn_base/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c4 |
| `cl1_ddqn_base_c4_sd2` | ddqn_base/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c4 |
| `cl1_ddqn_base_s1_sd0` | ddqn_base/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/s1 |
| `cl1_ddqn_base_s1_sd1` | ddqn_base/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/s1 |
| `cl1_ddqn_base_s1_sd2` | ddqn_base/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/s1 |
| `cl1_ddqn_clear_o2_sd0_st1` | ddqn_clear/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_ddqn_clear_o2_sd0_st2` | ddqn_clear/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_ddqn_clear_o2_sd0_st3` | ddqn_clear/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_ddqn_clear_o2_sd0_st4` | ddqn_clear/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_ddqn_clear_o2_sd1_st1` | ddqn_clear/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_ddqn_clear_o2_sd1_st2` | ddqn_clear/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_ddqn_clear_o2_sd1_st3` | ddqn_clear/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_ddqn_clear_o2_sd1_st4` | ddqn_clear/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_ddqn_clear_o2_sd2_st1` | ddqn_clear/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_ddqn_clear_o2_sd2_st2` | ddqn_clear/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_ddqn_clear_o2_sd2_st3` | ddqn_clear/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_ddqn_clear_o2_sd2_st4` | ddqn_clear/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_ddqn_clear_o3_sd0_st1` | ddqn_clear/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_ddqn_clear_o3_sd0_st2` | ddqn_clear/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_ddqn_clear_o3_sd0_st3` | ddqn_clear/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_ddqn_clear_o3_sd0_st4` | ddqn_clear/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_ddqn_clear_o3_sd1_st1` | ddqn_clear/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_ddqn_clear_o3_sd1_st2` | ddqn_clear/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_ddqn_clear_o3_sd1_st3` | ddqn_clear/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_ddqn_clear_o3_sd1_st4` | ddqn_clear/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_ddqn_clear_o3_sd2_st1` | ddqn_clear/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_ddqn_clear_o3_sd2_st2` | ddqn_clear/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_ddqn_clear_o3_sd2_st3` | ddqn_clear/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_ddqn_clear_o3_sd2_st4` | ddqn_clear/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_ddqn_fifo_o2_sd0_st1` | ddqn_fifo/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_ddqn_fifo_o2_sd0_st2` | ddqn_fifo/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_ddqn_fifo_o2_sd0_st3` | ddqn_fifo/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_ddqn_fifo_o2_sd0_st4` | ddqn_fifo/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_ddqn_fifo_o2_sd1_st1` | ddqn_fifo/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_ddqn_fifo_o2_sd1_st2` | ddqn_fifo/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_ddqn_fifo_o2_sd1_st3` | ddqn_fifo/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_ddqn_fifo_o2_sd1_st4` | ddqn_fifo/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_ddqn_fifo_o2_sd2_st1` | ddqn_fifo/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_ddqn_fifo_o2_sd2_st2` | ddqn_fifo/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_ddqn_fifo_o2_sd2_st3` | ddqn_fifo/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_ddqn_fifo_o2_sd2_st4` | ddqn_fifo/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_ddqn_fifo_o3_sd0_st1` | ddqn_fifo/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_ddqn_fifo_o3_sd0_st2` | ddqn_fifo/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_ddqn_fifo_o3_sd0_st3` | ddqn_fifo/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_ddqn_fifo_o3_sd0_st4` | ddqn_fifo/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_ddqn_fifo_o3_sd1_st1` | ddqn_fifo/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_ddqn_fifo_o3_sd1_st2` | ddqn_fifo/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_ddqn_fifo_o3_sd1_st3` | ddqn_fifo/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_ddqn_fifo_o3_sd1_st4` | ddqn_fifo/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_ddqn_fifo_o3_sd2_st1` | ddqn_fifo/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_ddqn_fifo_o3_sd2_st2` | ddqn_fifo/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_ddqn_fifo_o3_sd2_st3` | ddqn_fifo/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_ddqn_fifo_o3_sd2_st4` | ddqn_fifo/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_duel_base_c2_sd0` | duel_base/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c2 |
| `cl1_duel_base_c2_sd1` | duel_base/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c2 |
| `cl1_duel_base_c2_sd2` | duel_base/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c2 |
| `cl1_duel_base_c3_sd0` | duel_base/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c3 |
| `cl1_duel_base_c3_sd1` | duel_base/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c3 |
| `cl1_duel_base_c3_sd2` | duel_base/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c3 |
| `cl1_duel_base_c4_sd0` | duel_base/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c4 |
| `cl1_duel_base_c4_sd1` | duel_base/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c4 |
| `cl1_duel_base_c4_sd2` | duel_base/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c4 |
| `cl1_duel_base_s1_sd0` | duel_base/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/s1 |
| `cl1_duel_base_s1_sd1` | duel_base/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/s1 |
| `cl1_duel_base_s1_sd2` | duel_base/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/s1 |
| `cl1_duel_clear_o2_sd0_st1` | duel_clear/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_duel_clear_o2_sd0_st2` | duel_clear/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_duel_clear_o2_sd0_st3` | duel_clear/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_duel_clear_o2_sd0_st4` | duel_clear/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_duel_clear_o2_sd1_st1` | duel_clear/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_duel_clear_o2_sd1_st2` | duel_clear/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_duel_clear_o2_sd1_st3` | duel_clear/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_duel_clear_o2_sd1_st4` | duel_clear/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_duel_clear_o2_sd2_st1` | duel_clear/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_duel_clear_o2_sd2_st2` | duel_clear/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_duel_clear_o2_sd2_st3` | duel_clear/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_duel_clear_o2_sd2_st4` | duel_clear/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_duel_clear_o3_sd0_st1` | duel_clear/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_duel_clear_o3_sd0_st2` | duel_clear/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_duel_clear_o3_sd0_st3` | duel_clear/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_duel_clear_o3_sd0_st4` | duel_clear/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_duel_clear_o3_sd1_st1` | duel_clear/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_duel_clear_o3_sd1_st2` | duel_clear/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_duel_clear_o3_sd1_st3` | duel_clear/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_duel_clear_o3_sd1_st4` | duel_clear/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_duel_clear_o3_sd2_st1` | duel_clear/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_duel_clear_o3_sd2_st2` | duel_clear/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_duel_clear_o3_sd2_st3` | duel_clear/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_duel_clear_o3_sd2_st4` | duel_clear/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_duel_fifo_o2_sd0_st1` | duel_fifo/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_duel_fifo_o2_sd0_st2` | duel_fifo/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_duel_fifo_o2_sd0_st3` | duel_fifo/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_duel_fifo_o2_sd0_st4` | duel_fifo/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_duel_fifo_o2_sd1_st1` | duel_fifo/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_duel_fifo_o2_sd1_st2` | duel_fifo/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_duel_fifo_o2_sd1_st3` | duel_fifo/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_duel_fifo_o2_sd1_st4` | duel_fifo/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_duel_fifo_o2_sd2_st1` | duel_fifo/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_duel_fifo_o2_sd2_st2` | duel_fifo/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_duel_fifo_o2_sd2_st3` | duel_fifo/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_duel_fifo_o2_sd2_st4` | duel_fifo/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_duel_fifo_o3_sd0_st1` | duel_fifo/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_duel_fifo_o3_sd0_st2` | duel_fifo/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_duel_fifo_o3_sd0_st3` | duel_fifo/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_duel_fifo_o3_sd0_st4` | duel_fifo/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_duel_fifo_o3_sd1_st1` | duel_fifo/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_duel_fifo_o3_sd1_st2` | duel_fifo/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_duel_fifo_o3_sd1_st3` | duel_fifo/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_duel_fifo_o3_sd1_st4` | duel_fifo/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_duel_fifo_o3_sd2_st1` | duel_fifo/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_duel_fifo_o3_sd2_st2` | duel_fifo/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_duel_fifo_o3_sd2_st3` | duel_fifo/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_duel_fifo_o3_sd2_st4` | duel_fifo/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_frap_base_c2_sd0` | frap_base/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c2 |
| `cl1_frap_base_c2_sd1` | frap_base/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c2 |
| `cl1_frap_base_c2_sd2` | frap_base/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c2 |
| `cl1_frap_base_c3_sd0` | frap_base/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c3 |
| `cl1_frap_base_c3_sd1` | frap_base/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c3 |
| `cl1_frap_base_c3_sd2` | frap_base/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c3 |
| `cl1_frap_base_c4_sd0` | frap_base/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c4 |
| `cl1_frap_base_c4_sd1` | frap_base/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c4 |
| `cl1_frap_base_c4_sd2` | frap_base/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c4 |
| `cl1_frap_base_s1_sd0` | frap_base/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/s1 |
| `cl1_frap_base_s1_sd1` | frap_base/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/s1 |
| `cl1_frap_base_s1_sd2` | frap_base/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/s1 |
| `cl1_frap_clear_o2_sd0_st1` | frap_clear/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_frap_clear_o2_sd0_st2` | frap_clear/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_frap_clear_o2_sd0_st3` | frap_clear/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_frap_clear_o2_sd0_st4` | frap_clear/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_frap_clear_o2_sd1_st1` | frap_clear/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_frap_clear_o2_sd1_st2` | frap_clear/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_frap_clear_o2_sd1_st3` | frap_clear/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_frap_clear_o2_sd1_st4` | frap_clear/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_frap_clear_o2_sd2_st1` | frap_clear/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_frap_clear_o2_sd2_st2` | frap_clear/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_frap_clear_o2_sd2_st3` | frap_clear/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_frap_clear_o2_sd2_st4` | frap_clear/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_frap_clear_o3_sd0_st1` | frap_clear/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_frap_clear_o3_sd0_st2` | frap_clear/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_frap_clear_o3_sd0_st3` | frap_clear/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_frap_clear_o3_sd0_st4` | frap_clear/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_frap_clear_o3_sd1_st1` | frap_clear/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_frap_clear_o3_sd1_st2` | frap_clear/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_frap_clear_o3_sd1_st3` | frap_clear/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_frap_clear_o3_sd1_st4` | frap_clear/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_frap_clear_o3_sd2_st1` | frap_clear/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_frap_clear_o3_sd2_st2` | frap_clear/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_frap_clear_o3_sd2_st3` | frap_clear/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_frap_clear_o3_sd2_st4` | frap_clear/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_frap_fifo_o2_sd0_st1` | frap_fifo/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_frap_fifo_o2_sd0_st2` | frap_fifo/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_frap_fifo_o2_sd0_st3` | frap_fifo/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_frap_fifo_o2_sd0_st4` | frap_fifo/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_frap_fifo_o2_sd1_st1` | frap_fifo/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_frap_fifo_o2_sd1_st2` | frap_fifo/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_frap_fifo_o2_sd1_st3` | frap_fifo/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_frap_fifo_o2_sd1_st4` | frap_fifo/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_frap_fifo_o2_sd2_st1` | frap_fifo/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_frap_fifo_o2_sd2_st2` | frap_fifo/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_frap_fifo_o2_sd2_st3` | frap_fifo/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_frap_fifo_o2_sd2_st4` | frap_fifo/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_frap_fifo_o3_sd0_st1` | frap_fifo/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_frap_fifo_o3_sd0_st2` | frap_fifo/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_frap_fifo_o3_sd0_st3` | frap_fifo/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_frap_fifo_o3_sd0_st4` | frap_fifo/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_frap_fifo_o3_sd1_st1` | frap_fifo/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_frap_fifo_o3_sd1_st2` | frap_fifo/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_frap_fifo_o3_sd1_st3` | frap_fifo/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_frap_fifo_o3_sd1_st4` | frap_fifo/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_frap_fifo_o3_sd2_st1` | frap_fifo/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_frap_fifo_o3_sd2_st2` | frap_fifo/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_frap_fifo_o3_sd2_st3` | frap_fifo/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_frap_fifo_o3_sd2_st4` | frap_fifo/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_ppo_base_c2_sd0` | ppo_base/sumohz1x1_config2/s0 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_base_c2_sd1` | ppo_base/sumohz1x1_config2/s1 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_base_c2_sd2` | ppo_base/sumohz1x1_config2/s2 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_base_c3_sd0` | ppo_base/sumohz1x1_config3/s0 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_base_c3_sd1` | ppo_base/sumohz1x1_config3/s1 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_base_c3_sd2` | ppo_base/sumohz1x1_config3/s2 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_base_c4_sd0` | ppo_base/sumohz1x1_config4/s0 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_base_c4_sd1` | ppo_base/sumohz1x1_config4/s1 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_base_c4_sd2` | ppo_base/sumohz1x1_config4/s2 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_base_s1_sd0` | ppo_base/sumohz1x1/s0 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_base_s1_sd1` | ppo_base/sumohz1x1/s1 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_base_s1_sd2` | ppo_base/sumohz1x1/s2 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o2_sd0_st1` | ppo_clear/sumohz1x1/s0 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o2_sd0_st2` | ppo_clear/sumohz1x1_config4/s0 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o2_sd0_st3` | ppo_clear/sumohz1x1_config2/s0 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o2_sd0_st4` | ppo_clear/sumohz1x1_config3/s0 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o2_sd1_st1` | ppo_clear/sumohz1x1/s1 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o2_sd1_st2` | ppo_clear/sumohz1x1_config4/s1 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o2_sd1_st3` | ppo_clear/sumohz1x1_config2/s1 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o2_sd1_st4` | ppo_clear/sumohz1x1_config3/s1 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o2_sd2_st1` | ppo_clear/sumohz1x1/s2 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o2_sd2_st2` | ppo_clear/sumohz1x1_config4/s2 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o2_sd2_st3` | ppo_clear/sumohz1x1_config2/s2 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o2_sd2_st4` | ppo_clear/sumohz1x1_config3/s2 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o3_sd0_st1` | ppo_clear/sumohz1x1_config2/s0 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o3_sd0_st2` | ppo_clear/sumohz1x1_config3/s0 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o3_sd0_st3` | ppo_clear/sumohz1x1/s0 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o3_sd0_st4` | ppo_clear/sumohz1x1_config4/s0 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o3_sd1_st1` | ppo_clear/sumohz1x1_config2/s1 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o3_sd1_st2` | ppo_clear/sumohz1x1_config3/s1 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o3_sd1_st3` | ppo_clear/sumohz1x1/s1 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o3_sd1_st4` | ppo_clear/sumohz1x1_config4/s1 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o3_sd2_st1` | ppo_clear/sumohz1x1_config2/s2 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o3_sd2_st2` | ppo_clear/sumohz1x1_config3/s2 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o3_sd2_st3` | ppo_clear/sumohz1x1/s2 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppo_clear_o3_sd2_st4` | ppo_clear/sumohz1x1_config4/s2 | 100 | SCOPE_DISCARD | — | — | `—` | v1 ppo(entropy_coef=0.001)塌缩不学习：2026-10-11 起由 ppov2(plan5 校准 entropy 0.1/reward  |
| `cl1_ppov2_base_c2_sd0` | ppov2_base/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c2 |
| `cl1_ppov2_base_c2_sd1` | ppov2_base/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c2 |
| `cl1_ppov2_base_c2_sd2` | ppov2_base/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c2 |
| `cl1_ppov2_base_c3_sd0` | ppov2_base/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c3 |
| `cl1_ppov2_base_c3_sd1` | ppov2_base/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c3 |
| `cl1_ppov2_base_c3_sd2` | ppov2_base/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c3 |
| `cl1_ppov2_base_c4_sd0` | ppov2_base/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c4 |
| `cl1_ppov2_base_c4_sd1` | ppov2_base/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c4 |
| `cl1_ppov2_base_c4_sd2` | ppov2_base/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/c4 |
| `cl1_ppov2_base_s1_sd0` | ppov2_base/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/s1 |
| `cl1_ppov2_base_s1_sd1` | ppov2_base/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/s1 |
| `cl1_ppov2_base_s1_sd2` | ppov2_base/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=single/s1 |
| `cl1_ppov2_clear_o2_sd0_st1` | ppov2_clear/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_ppov2_clear_o2_sd0_st2` | ppov2_clear/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_ppov2_clear_o2_sd0_st3` | ppov2_clear/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_ppov2_clear_o2_sd0_st4` | ppov2_clear/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_ppov2_clear_o2_sd1_st1` | ppov2_clear/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_ppov2_clear_o2_sd1_st2` | ppov2_clear/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_ppov2_clear_o2_sd1_st3` | ppov2_clear/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_ppov2_clear_o2_sd1_st4` | ppov2_clear/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_ppov2_clear_o2_sd2_st1` | ppov2_clear/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st1 |
| `cl1_ppov2_clear_o2_sd2_st2` | ppov2_clear/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st2 |
| `cl1_ppov2_clear_o2_sd2_st3` | ppov2_clear/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st3 |
| `cl1_ppov2_clear_o2_sd2_st4` | ppov2_clear/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o2/st4 |
| `cl1_ppov2_clear_o3_sd0_st1` | ppov2_clear/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_ppov2_clear_o3_sd0_st2` | ppov2_clear/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_ppov2_clear_o3_sd0_st3` | ppov2_clear/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_ppov2_clear_o3_sd0_st4` | ppov2_clear/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_ppov2_clear_o3_sd1_st1` | ppov2_clear/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_ppov2_clear_o3_sd1_st2` | ppov2_clear/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_ppov2_clear_o3_sd1_st3` | ppov2_clear/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_ppov2_clear_o3_sd1_st4` | ppov2_clear/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `cl1_ppov2_clear_o3_sd2_st1` | ppov2_clear/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st1 |
| `cl1_ppov2_clear_o3_sd2_st2` | ppov2_clear/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st2 |
| `cl1_ppov2_clear_o3_sd2_st3` | ppov2_clear/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st3 |
| `cl1_ppov2_clear_o3_sd2_st4` | ppov2_clear/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 continual-learning train order_stage=o3/st4 |
| `so1_soddqn_o2_sd0_st1` | soddqn_so/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st1 offline_ratio=0.25 |
| `so1_soddqn_o2_sd0_st2` | soddqn_so/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st2 offline_ratio=0.25 |
| `so1_soddqn_o2_sd0_st3` | soddqn_so/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st3 offline_ratio=0.25 |
| `so1_soddqn_o2_sd0_st4` | soddqn_so/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st4 offline_ratio=0.25 |
| `so1_soddqn_o2_sd1_st1` | soddqn_so/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st1 offline_ratio=0.25 |
| `so1_soddqn_o2_sd1_st2` | soddqn_so/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st2 offline_ratio=0.25 |
| `so1_soddqn_o2_sd1_st3` | soddqn_so/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st3 offline_ratio=0.25 |
| `so1_soddqn_o2_sd1_st4` | soddqn_so/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st4 offline_ratio=0.25 |
| `so1_soddqn_o2_sd2_st1` | soddqn_so/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st1 offline_ratio=0.25 |
| `so1_soddqn_o2_sd2_st2` | soddqn_so/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st2 offline_ratio=0.25 |
| `so1_soddqn_o2_sd2_st3` | soddqn_so/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st3 offline_ratio=0.25 |
| `so1_soddqn_o2_sd2_st4` | soddqn_so/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st4 offline_ratio=0.25 |
| `so1_soddqn_o3_sd0_st1` | soddqn_so/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st1 offline_ratio=0.25 |
| `so1_soddqn_o3_sd0_st2` | soddqn_so/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st2 offline_ratio=0.25 |
| `so1_soddqn_o3_sd0_st3` | soddqn_so/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st3 offline_ratio=0.25 |
| `so1_soddqn_o3_sd0_st4` | soddqn_so/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st4 offline_ratio=0.25 |
| `so1_soddqn_o3_sd1_st1` | soddqn_so/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st1 offline_ratio=0.25 |
| `so1_soddqn_o3_sd1_st2` | soddqn_so/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st2 offline_ratio=0.25 |
| `so1_soddqn_o3_sd1_st3` | soddqn_so/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st3 offline_ratio=0.25 |
| `so1_soddqn_o3_sd1_st4` | soddqn_so/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st4 offline_ratio=0.25 |
| `so1_soddqn_o3_sd2_st1` | soddqn_so/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st1 offline_ratio=0.25 |
| `so1_soddqn_o3_sd2_st2` | soddqn_so/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st2 offline_ratio=0.25 |
| `so1_soddqn_o3_sd2_st3` | soddqn_so/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st3 offline_ratio=0.25 |
| `so1_soddqn_o3_sd2_st4` | soddqn_so/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st4 offline_ratio=0.25 |
| `so1_soduel_o2_sd0_st1` | soduel_so/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st1 offline_ratio=0.25 |
| `so1_soduel_o2_sd0_st2` | soduel_so/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st2 offline_ratio=0.25 |
| `so1_soduel_o2_sd0_st3` | soduel_so/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st3 offline_ratio=0.25 |
| `so1_soduel_o2_sd0_st4` | soduel_so/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st4 offline_ratio=0.25 |
| `so1_soduel_o2_sd1_st1` | soduel_so/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st1 offline_ratio=0.25 |
| `so1_soduel_o2_sd1_st2` | soduel_so/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st2 offline_ratio=0.25 |
| `so1_soduel_o2_sd1_st3` | soduel_so/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st3 offline_ratio=0.25 |
| `so1_soduel_o2_sd1_st4` | soduel_so/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st4 offline_ratio=0.25 |
| `so1_soduel_o2_sd2_st1` | soduel_so/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st1 offline_ratio=0.25 |
| `so1_soduel_o2_sd2_st2` | soduel_so/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st2 offline_ratio=0.25 |
| `so1_soduel_o2_sd2_st3` | soduel_so/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st3 offline_ratio=0.25 |
| `so1_soduel_o2_sd2_st4` | soduel_so/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st4 offline_ratio=0.25 |
| `so1_soduel_o3_sd0_st1` | soduel_so/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st1 offline_ratio=0.25 |
| `so1_soduel_o3_sd0_st2` | soduel_so/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st2 offline_ratio=0.25 |
| `so1_soduel_o3_sd0_st3` | soduel_so/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st3 offline_ratio=0.25 |
| `so1_soduel_o3_sd0_st4` | soduel_so/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st4 offline_ratio=0.25 |
| `so1_soduel_o3_sd1_st1` | soduel_so/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st1 offline_ratio=0.25 |
| `so1_soduel_o3_sd1_st2` | soduel_so/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st2 offline_ratio=0.25 |
| `so1_soduel_o3_sd1_st3` | soduel_so/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st3 offline_ratio=0.25 |
| `so1_soduel_o3_sd1_st4` | soduel_so/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st4 offline_ratio=0.25 |
| `so1_soduel_o3_sd2_st1` | soduel_so/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st1 offline_ratio=0.25 |
| `so1_soduel_o3_sd2_st2` | soduel_so/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st2 offline_ratio=0.25 |
| `so1_soduel_o3_sd2_st3` | soduel_so/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st3 offline_ratio=0.25 |
| `so1_soduel_o3_sd2_st4` | soduel_so/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st4 offline_ratio=0.25 |
| `so1_sofrap_o2_sd0_st1` | sofrap_so/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st1 offline_ratio=0.25 |
| `so1_sofrap_o2_sd0_st2` | sofrap_so/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st2 offline_ratio=0.25 |
| `so1_sofrap_o2_sd0_st3` | sofrap_so/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st3 offline_ratio=0.25 |
| `so1_sofrap_o2_sd0_st4` | sofrap_so/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st4 offline_ratio=0.25 |
| `so1_sofrap_o2_sd1_st1` | sofrap_so/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st1 offline_ratio=0.25 |
| `so1_sofrap_o2_sd1_st2` | sofrap_so/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st2 offline_ratio=0.25 |
| `so1_sofrap_o2_sd1_st3` | sofrap_so/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st3 offline_ratio=0.25 |
| `so1_sofrap_o2_sd1_st4` | sofrap_so/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st4 offline_ratio=0.25 |
| `so1_sofrap_o2_sd2_st1` | sofrap_so/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st1 offline_ratio=0.25 |
| `so1_sofrap_o2_sd2_st2` | sofrap_so/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st2 offline_ratio=0.25 |
| `so1_sofrap_o2_sd2_st3` | sofrap_so/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st3 offline_ratio=0.25 |
| `so1_sofrap_o2_sd2_st4` | sofrap_so/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o2/st4 offline_ratio=0.25 |
| `so1_sofrap_o3_sd0_st1` | sofrap_so/sumohz1x1_config2/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st1 offline_ratio=0.25 |
| `so1_sofrap_o3_sd0_st2` | sofrap_so/sumohz1x1_config3/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st2 offline_ratio=0.25 |
| `so1_sofrap_o3_sd0_st3` | sofrap_so/sumohz1x1/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st3 offline_ratio=0.25 |
| `so1_sofrap_o3_sd0_st4` | sofrap_so/sumohz1x1_config4/s0 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st4 offline_ratio=0.25 |
| `so1_sofrap_o3_sd1_st1` | sofrap_so/sumohz1x1_config2/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st1 offline_ratio=0.25 |
| `so1_sofrap_o3_sd1_st2` | sofrap_so/sumohz1x1_config3/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st2 offline_ratio=0.25 |
| `so1_sofrap_o3_sd1_st3` | sofrap_so/sumohz1x1/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st3 offline_ratio=0.25 |
| `so1_sofrap_o3_sd1_st4` | sofrap_so/sumohz1x1_config4/s1 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st4 offline_ratio=0.25 |
| `so1_sofrap_o3_sd2_st1` | sofrap_so/sumohz1x1_config2/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st1 offline_ratio=0.25 |
| `so1_sofrap_o3_sd2_st2` | sofrap_so/sumohz1x1_config3/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st2 offline_ratio=0.25 |
| `so1_sofrap_o3_sd2_st3` | sofrap_so/sumohz1x1/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st3 offline_ratio=0.25 |
| `so1_sofrap_o3_sd2_st4` | sofrap_so/sumohz1x1_config4/s2 | 100 | DONE | — | — | `—` | REGISTERED: cl_hz1x1 semi-offline train order_stage=o3/st4 offline_ratio=0.25 |
| `soa1_soddqnp_o2_sd0_st1` | soddqnp_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa1_soddqnp_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_soddqnp_o2_sd0_st2` | soddqnp_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa1_soddqnp_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_soddqnp_o2_sd0_st3` | soddqnp_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa1_soddqnp_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_soddqnp_o2_sd0_st4` | soddqnp_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa1_soddqnp_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_soddqnp_o2_sd1_st1` | soddqnp_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa1_soddqnp_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_soddqnp_o2_sd1_st2` | soddqnp_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa1_soddqnp_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_soddqnp_o2_sd1_st3` | soddqnp_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa1_soddqnp_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_soddqnp_o2_sd1_st4` | soddqnp_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa1_soddqnp_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_soddqnp_o2_sd2_st1` | soddqnp_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa1_soddqnp_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_soddqnp_o2_sd2_st2` | soddqnp_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa1_soddqnp_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_soddqnp_o2_sd2_st3` | soddqnp_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa1_soddqnp_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_soddqnp_o2_sd2_st4` | soddqnp_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa1_soddqnp_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_soddqnpv_o2_sd0_st1` | soddqnpv_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa1_soddqnpv_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_soddqnpv_o2_sd0_st2` | soddqnpv_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa1_soddqnpv_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_soddqnpv_o2_sd0_st3` | soddqnpv_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa1_soddqnpv_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_soddqnpv_o2_sd0_st4` | soddqnpv_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa1_soddqnpv_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_soddqnpv_o2_sd1_st1` | soddqnpv_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa1_soddqnpv_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_soddqnpv_o2_sd1_st2` | soddqnpv_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa1_soddqnpv_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_soddqnpv_o2_sd1_st3` | soddqnpv_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa1_soddqnpv_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_soddqnpv_o2_sd1_st4` | soddqnpv_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa1_soddqnpv_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_soddqnpv_o2_sd2_st1` | soddqnpv_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa1_soddqnpv_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_soddqnpv_o2_sd2_st2` | soddqnpv_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa1_soddqnpv_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_soddqnpv_o2_sd2_st3` | soddqnpv_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa1_soddqnpv_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_soddqnpv_o2_sd2_st4` | soddqnpv_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa1_soddqnpv_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_soduelp_o2_sd0_st1` | soduelp_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa1_soduelp_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_soduelp_o2_sd0_st2` | soduelp_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa1_soduelp_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_soduelp_o2_sd0_st3` | soduelp_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa1_soduelp_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_soduelp_o2_sd0_st4` | soduelp_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa1_soduelp_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_soduelp_o2_sd1_st1` | soduelp_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa1_soduelp_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_soduelp_o2_sd1_st2` | soduelp_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa1_soduelp_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_soduelp_o2_sd1_st3` | soduelp_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa1_soduelp_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_soduelp_o2_sd1_st4` | soduelp_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa1_soduelp_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_soduelp_o2_sd2_st1` | soduelp_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa1_soduelp_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_soduelp_o2_sd2_st2` | soduelp_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa1_soduelp_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_soduelp_o2_sd2_st3` | soduelp_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa1_soduelp_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_soduelp_o2_sd2_st4` | soduelp_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa1_soduelp_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_soduelpv_o2_sd0_st1` | soduelpv_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa1_soduelpv_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_soduelpv_o2_sd0_st2` | soduelpv_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa1_soduelpv_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_soduelpv_o2_sd0_st3` | soduelpv_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa1_soduelpv_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_soduelpv_o2_sd0_st4` | soduelpv_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa1_soduelpv_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_soduelpv_o2_sd1_st1` | soduelpv_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa1_soduelpv_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_soduelpv_o2_sd1_st2` | soduelpv_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa1_soduelpv_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_soduelpv_o2_sd1_st3` | soduelpv_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa1_soduelpv_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_soduelpv_o2_sd1_st4` | soduelpv_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa1_soduelpv_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_soduelpv_o2_sd2_st1` | soduelpv_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa1_soduelpv_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_soduelpv_o2_sd2_st2` | soduelpv_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa1_soduelpv_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_soduelpv_o2_sd2_st3` | soduelpv_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa1_soduelpv_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_soduelpv_o2_sd2_st4` | soduelpv_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa1_soduelpv_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_sofrapp_o2_sd0_st1` | sofrapp_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soa1_sofrapp_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_sofrapp_o2_sd0_st2` | sofrapp_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soa1_sofrapp_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_sofrapp_o2_sd0_st3` | sofrapp_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soa1_sofrapp_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_sofrapp_o2_sd0_st4` | sofrapp_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soa1_sofrapp_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_sofrapp_o2_sd1_st1` | sofrapp_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soa1_sofrapp_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_sofrapp_o2_sd1_st2` | sofrapp_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soa1_sofrapp_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_sofrapp_o2_sd1_st3` | sofrapp_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soa1_sofrapp_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_sofrapp_o2_sd1_st4` | sofrapp_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soa1_sofrapp_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_sofrapp_o2_sd2_st1` | sofrapp_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soa1_sofrapp_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_sofrapp_o2_sd2_st2` | sofrapp_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soa1_sofrapp_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_sofrapp_o2_sd2_st3` | sofrapp_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soa1_sofrapp_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_sofrapp_o2_sd2_st4` | sofrapp_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soa1_sofrapp_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_sofrappv_o2_sd0_st1` | sofrappv_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soa1_sofrappv_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_sofrappv_o2_sd0_st2` | sofrappv_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soa1_sofrappv_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_sofrappv_o2_sd0_st3` | sofrappv_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soa1_sofrappv_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_sofrappv_o2_sd0_st4` | sofrappv_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soa1_sofrappv_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_sofrappv_o2_sd1_st1` | sofrappv_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soa1_sofrappv_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_sofrappv_o2_sd1_st2` | sofrappv_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soa1_sofrappv_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_sofrappv_o2_sd1_st3` | sofrappv_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soa1_sofrappv_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_sofrappv_o2_sd1_st4` | sofrappv_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soa1_sofrappv_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa1_sofrappv_o2_sd2_st1` | sofrappv_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soa1_sofrappv_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st1 offline_ratio=0.25 |
| `soa1_sofrappv_o2_sd2_st2` | sofrappv_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soa1_sofrappv_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st2 offline_ratio=0.25 |
| `soa1_sofrappv_o2_sd2_st3` | sofrappv_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soa1_sofrappv_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st3 offline_ratio=0.25 |
| `soa1_sofrappv_o2_sd2_st4` | sofrappv_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soa1_sofrappv_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor main train order_stage=o2/st4 offline_ratio=0.25 |
| `soa3_soddqnp_o3_sd0_st1` | soddqnp_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa3_soddqnp_o3_sd0_st1` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st1 offline_ratio=0.25 |
| `soa3_soddqnp_o3_sd0_st2` | soddqnp_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa3_soddqnp_o3_sd0_st2` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st2 offline_ratio=0.25 |
| `soa3_soddqnp_o3_sd0_st3` | soddqnp_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa3_soddqnp_o3_sd0_st3` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st3 offline_ratio=0.25 |
| `soa3_soddqnp_o3_sd0_st4` | soddqnp_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa3_soddqnp_o3_sd0_st4` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st4 offline_ratio=0.25 |
| `soa3_soddqnp_o3_sd1_st1` | soddqnp_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa3_soddqnp_o3_sd1_st1` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st1 offline_ratio=0.25 |
| `soa3_soddqnp_o3_sd1_st2` | soddqnp_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa3_soddqnp_o3_sd1_st2` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st2 offline_ratio=0.25 |
| `soa3_soddqnp_o3_sd1_st3` | soddqnp_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa3_soddqnp_o3_sd1_st3` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st3 offline_ratio=0.25 |
| `soa3_soddqnp_o3_sd1_st4` | soddqnp_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa3_soddqnp_o3_sd1_st4` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st4 offline_ratio=0.25 |
| `soa3_soddqnp_o3_sd2_st1` | soddqnp_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa3_soddqnp_o3_sd2_st1` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st1 offline_ratio=0.25 |
| `soa3_soddqnp_o3_sd2_st2` | soddqnp_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa3_soddqnp_o3_sd2_st2` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st2 offline_ratio=0.25 |
| `soa3_soddqnp_o3_sd2_st3` | soddqnp_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa3_soddqnp_o3_sd2_st3` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st3 offline_ratio=0.25 |
| `soa3_soddqnp_o3_sd2_st4` | soddqnp_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa3_soddqnp_o3_sd2_st4` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st4 offline_ratio=0.25 |
| `soa3_soduelp_o3_sd0_st1` | soduelp_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa3_soduelp_o3_sd0_st1` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st1 offline_ratio=0.25 |
| `soa3_soduelp_o3_sd0_st2` | soduelp_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa3_soduelp_o3_sd0_st2` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st2 offline_ratio=0.25 |
| `soa3_soduelp_o3_sd0_st3` | soduelp_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa3_soduelp_o3_sd0_st3` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st3 offline_ratio=0.25 |
| `soa3_soduelp_o3_sd0_st4` | soduelp_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa3_soduelp_o3_sd0_st4` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st4 offline_ratio=0.25 |
| `soa3_soduelp_o3_sd1_st1` | soduelp_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa3_soduelp_o3_sd1_st1` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st1 offline_ratio=0.25 |
| `soa3_soduelp_o3_sd1_st2` | soduelp_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa3_soduelp_o3_sd1_st2` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st2 offline_ratio=0.25 |
| `soa3_soduelp_o3_sd1_st3` | soduelp_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa3_soduelp_o3_sd1_st3` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st3 offline_ratio=0.25 |
| `soa3_soduelp_o3_sd1_st4` | soduelp_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa3_soduelp_o3_sd1_st4` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st4 offline_ratio=0.25 |
| `soa3_soduelp_o3_sd2_st1` | soduelp_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soa3_soduelp_o3_sd2_st1` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st1 offline_ratio=0.25 |
| `soa3_soduelp_o3_sd2_st2` | soduelp_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soa3_soduelp_o3_sd2_st2` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st2 offline_ratio=0.25 |
| `soa3_soduelp_o3_sd2_st3` | soduelp_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soa3_soduelp_o3_sd2_st3` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st3 offline_ratio=0.25 |
| `soa3_soduelp_o3_sd2_st4` | soduelp_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soa3_soduelp_o3_sd2_st4` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st4 offline_ratio=0.25 |
| `soa3_sofrapp_o3_sd0_st1` | sofrapp_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soa3_sofrapp_o3_sd0_st1` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st1 offline_ratio=0.25 |
| `soa3_sofrapp_o3_sd0_st2` | sofrapp_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soa3_sofrapp_o3_sd0_st2` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st2 offline_ratio=0.25 |
| `soa3_sofrapp_o3_sd0_st3` | sofrapp_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soa3_sofrapp_o3_sd0_st3` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st3 offline_ratio=0.25 |
| `soa3_sofrapp_o3_sd0_st4` | sofrapp_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soa3_sofrapp_o3_sd0_st4` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st4 offline_ratio=0.25 |
| `soa3_sofrapp_o3_sd1_st1` | sofrapp_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soa3_sofrapp_o3_sd1_st1` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st1 offline_ratio=0.25 |
| `soa3_sofrapp_o3_sd1_st2` | sofrapp_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soa3_sofrapp_o3_sd1_st2` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st2 offline_ratio=0.25 |
| `soa3_sofrapp_o3_sd1_st3` | sofrapp_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soa3_sofrapp_o3_sd1_st3` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st3 offline_ratio=0.25 |
| `soa3_sofrapp_o3_sd1_st4` | sofrapp_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soa3_sofrapp_o3_sd1_st4` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st4 offline_ratio=0.25 |
| `soa3_sofrapp_o3_sd2_st1` | sofrapp_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soa3_sofrapp_o3_sd2_st1` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st1 offline_ratio=0.25 |
| `soa3_sofrapp_o3_sd2_st2` | sofrapp_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soa3_sofrapp_o3_sd2_st2` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st2 offline_ratio=0.25 |
| `soa3_sofrapp_o3_sd2_st3` | sofrapp_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soa3_sofrapp_o3_sd2_st3` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st3 offline_ratio=0.25 |
| `soa3_sofrapp_o3_sd2_st4` | sofrapp_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soa3_sofrapp_o3_sd2_st4` | REGISTERED: cl_hz1x1 anchor main_o3 train order_stage=o3/st4 offline_ratio=0.25 |
| `soar_soddqn_r12_o2_sd0_st1` | soddqn_r12_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soddqn_r12_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soddqn_r12_o2_sd0_st2` | soddqn_r12_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soddqn_r12_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_soddqn_r12_o2_sd0_st3` | soddqn_r12_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soddqn_r12_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_soddqn_r12_o2_sd0_st4` | soddqn_r12_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soddqn_r12_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_soddqn_r12_o2_sd1_st1` | soddqn_r12_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soddqn_r12_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soddqn_r12_o2_sd1_st2` | soddqn_r12_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soddqn_r12_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_soddqn_r12_o2_sd1_st3` | soddqn_r12_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soddqn_r12_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_soddqn_r12_o2_sd1_st4` | soddqn_r12_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soddqn_r12_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_soddqn_r12_o2_sd2_st1` | soddqn_r12_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soddqn_r12_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soddqn_r12_o2_sd2_st2` | soddqn_r12_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soddqn_r12_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_soddqn_r12_o2_sd2_st3` | soddqn_r12_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soddqn_r12_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_soddqn_r12_o2_sd2_st4` | soddqn_r12_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soddqn_r12_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_soddqn_r50_o2_sd0_st1` | soddqn_r50_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soddqn_r50_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soddqn_r50_o2_sd0_st2` | soddqn_r50_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soddqn_r50_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_soddqn_r50_o2_sd0_st3` | soddqn_r50_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soddqn_r50_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_soddqn_r50_o2_sd0_st4` | soddqn_r50_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soddqn_r50_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_soddqn_r50_o2_sd1_st1` | soddqn_r50_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soddqn_r50_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soddqn_r50_o2_sd1_st2` | soddqn_r50_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soddqn_r50_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_soddqn_r50_o2_sd1_st3` | soddqn_r50_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soddqn_r50_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_soddqn_r50_o2_sd1_st4` | soddqn_r50_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soddqn_r50_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_soddqn_r50_o2_sd2_st1` | soddqn_r50_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soddqn_r50_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soddqn_r50_o2_sd2_st2` | soddqn_r50_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soddqn_r50_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_soddqn_r50_o2_sd2_st3` | soddqn_r50_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soddqn_r50_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_soddqn_r50_o2_sd2_st4` | soddqn_r50_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soddqn_r50_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_soddqnp_r12_o2_sd0_st1` | soddqnp_r12_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soddqnp_r12_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soddqnp_r12_o2_sd0_st2` | soddqnp_r12_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soddqnp_r12_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_soddqnp_r12_o2_sd0_st3` | soddqnp_r12_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soddqnp_r12_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_soddqnp_r12_o2_sd0_st4` | soddqnp_r12_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soddqnp_r12_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_soddqnp_r12_o2_sd1_st1` | soddqnp_r12_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soddqnp_r12_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soddqnp_r12_o2_sd1_st2` | soddqnp_r12_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soddqnp_r12_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_soddqnp_r12_o2_sd1_st3` | soddqnp_r12_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soddqnp_r12_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_soddqnp_r12_o2_sd1_st4` | soddqnp_r12_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soddqnp_r12_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_soddqnp_r12_o2_sd2_st1` | soddqnp_r12_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soddqnp_r12_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soddqnp_r12_o2_sd2_st2` | soddqnp_r12_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soddqnp_r12_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_soddqnp_r12_o2_sd2_st3` | soddqnp_r12_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soddqnp_r12_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_soddqnp_r12_o2_sd2_st4` | soddqnp_r12_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soddqnp_r12_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_soddqnp_r50_o2_sd0_st1` | soddqnp_r50_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soddqnp_r50_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soddqnp_r50_o2_sd0_st2` | soddqnp_r50_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soddqnp_r50_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_soddqnp_r50_o2_sd0_st3` | soddqnp_r50_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soddqnp_r50_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_soddqnp_r50_o2_sd0_st4` | soddqnp_r50_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soddqnp_r50_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_soddqnp_r50_o2_sd1_st1` | soddqnp_r50_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soddqnp_r50_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soddqnp_r50_o2_sd1_st2` | soddqnp_r50_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soddqnp_r50_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_soddqnp_r50_o2_sd1_st3` | soddqnp_r50_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soddqnp_r50_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_soddqnp_r50_o2_sd1_st4` | soddqnp_r50_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soddqnp_r50_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_soddqnp_r50_o2_sd2_st1` | soddqnp_r50_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soddqnp_r50_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soddqnp_r50_o2_sd2_st2` | soddqnp_r50_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soddqnp_r50_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_soddqnp_r50_o2_sd2_st3` | soddqnp_r50_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soddqnp_r50_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_soddqnp_r50_o2_sd2_st4` | soddqnp_r50_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soddqnp_r50_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_soduel_r12_o2_sd0_st1` | soduel_r12_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soduel_r12_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soduel_r12_o2_sd0_st2` | soduel_r12_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soduel_r12_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_soduel_r12_o2_sd0_st3` | soduel_r12_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soduel_r12_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_soduel_r12_o2_sd0_st4` | soduel_r12_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soduel_r12_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_soduel_r12_o2_sd1_st1` | soduel_r12_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soduel_r12_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soduel_r12_o2_sd1_st2` | soduel_r12_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soduel_r12_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_soduel_r12_o2_sd1_st3` | soduel_r12_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soduel_r12_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_soduel_r12_o2_sd1_st4` | soduel_r12_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soduel_r12_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_soduel_r12_o2_sd2_st1` | soduel_r12_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soduel_r12_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soduel_r12_o2_sd2_st2` | soduel_r12_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soduel_r12_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_soduel_r12_o2_sd2_st3` | soduel_r12_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soduel_r12_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_soduel_r12_o2_sd2_st4` | soduel_r12_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soduel_r12_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_soduel_r50_o2_sd0_st1` | soduel_r50_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soduel_r50_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soduel_r50_o2_sd0_st2` | soduel_r50_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soduel_r50_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_soduel_r50_o2_sd0_st3` | soduel_r50_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soduel_r50_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_soduel_r50_o2_sd0_st4` | soduel_r50_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soduel_r50_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_soduel_r50_o2_sd1_st1` | soduel_r50_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soduel_r50_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soduel_r50_o2_sd1_st2` | soduel_r50_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soduel_r50_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_soduel_r50_o2_sd1_st3` | soduel_r50_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soduel_r50_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_soduel_r50_o2_sd1_st4` | soduel_r50_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soduel_r50_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_soduel_r50_o2_sd2_st1` | soduel_r50_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soduel_r50_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soduel_r50_o2_sd2_st2` | soduel_r50_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soduel_r50_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_soduel_r50_o2_sd2_st3` | soduel_r50_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soduel_r50_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_soduel_r50_o2_sd2_st4` | soduel_r50_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soduel_r50_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_soduelp_r12_o2_sd0_st1` | soduelp_r12_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soduelp_r12_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soduelp_r12_o2_sd0_st2` | soduelp_r12_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soduelp_r12_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_soduelp_r12_o2_sd0_st3` | soduelp_r12_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soduelp_r12_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_soduelp_r12_o2_sd0_st4` | soduelp_r12_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soduelp_r12_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_soduelp_r12_o2_sd1_st1` | soduelp_r12_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soduelp_r12_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soduelp_r12_o2_sd1_st2` | soduelp_r12_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soduelp_r12_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_soduelp_r12_o2_sd1_st3` | soduelp_r12_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soduelp_r12_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_soduelp_r12_o2_sd1_st4` | soduelp_r12_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soduelp_r12_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_soduelp_r12_o2_sd2_st1` | soduelp_r12_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soduelp_r12_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soduelp_r12_o2_sd2_st2` | soduelp_r12_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soduelp_r12_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_soduelp_r12_o2_sd2_st3` | soduelp_r12_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soduelp_r12_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_soduelp_r12_o2_sd2_st4` | soduelp_r12_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soduelp_r12_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_soduelp_r50_o2_sd0_st1` | soduelp_r50_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soduelp_r50_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soduelp_r50_o2_sd0_st2` | soduelp_r50_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soduelp_r50_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_soduelp_r50_o2_sd0_st3` | soduelp_r50_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soduelp_r50_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_soduelp_r50_o2_sd0_st4` | soduelp_r50_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soduelp_r50_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_soduelp_r50_o2_sd1_st1` | soduelp_r50_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soduelp_r50_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soduelp_r50_o2_sd1_st2` | soduelp_r50_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soduelp_r50_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_soduelp_r50_o2_sd1_st3` | soduelp_r50_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soduelp_r50_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_soduelp_r50_o2_sd1_st4` | soduelp_r50_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soduelp_r50_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_soduelp_r50_o2_sd2_st1` | soduelp_r50_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soar_soduelp_r50_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_soduelp_r50_o2_sd2_st2` | soduelp_r50_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soar_soduelp_r50_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_soduelp_r50_o2_sd2_st3` | soduelp_r50_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soar_soduelp_r50_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_soduelp_r50_o2_sd2_st4` | soduelp_r50_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soar_soduelp_r50_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_sofrap_r12_o2_sd0_st1` | sofrap_r12_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soar_sofrap_r12_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_sofrap_r12_o2_sd0_st2` | sofrap_r12_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soar_sofrap_r12_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_sofrap_r12_o2_sd0_st3` | sofrap_r12_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soar_sofrap_r12_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_sofrap_r12_o2_sd0_st4` | sofrap_r12_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soar_sofrap_r12_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_sofrap_r12_o2_sd1_st1` | sofrap_r12_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soar_sofrap_r12_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_sofrap_r12_o2_sd1_st2` | sofrap_r12_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soar_sofrap_r12_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_sofrap_r12_o2_sd1_st3` | sofrap_r12_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soar_sofrap_r12_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_sofrap_r12_o2_sd1_st4` | sofrap_r12_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soar_sofrap_r12_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_sofrap_r12_o2_sd2_st1` | sofrap_r12_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soar_sofrap_r12_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_sofrap_r12_o2_sd2_st2` | sofrap_r12_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soar_sofrap_r12_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_sofrap_r12_o2_sd2_st3` | sofrap_r12_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soar_sofrap_r12_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_sofrap_r12_o2_sd2_st4` | sofrap_r12_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soar_sofrap_r12_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_sofrap_r50_o2_sd0_st1` | sofrap_r50_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soar_sofrap_r50_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_sofrap_r50_o2_sd0_st2` | sofrap_r50_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soar_sofrap_r50_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_sofrap_r50_o2_sd0_st3` | sofrap_r50_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soar_sofrap_r50_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_sofrap_r50_o2_sd0_st4` | sofrap_r50_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soar_sofrap_r50_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_sofrap_r50_o2_sd1_st1` | sofrap_r50_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soar_sofrap_r50_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_sofrap_r50_o2_sd1_st2` | sofrap_r50_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soar_sofrap_r50_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_sofrap_r50_o2_sd1_st3` | sofrap_r50_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soar_sofrap_r50_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_sofrap_r50_o2_sd1_st4` | sofrap_r50_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soar_sofrap_r50_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_sofrap_r50_o2_sd2_st1` | sofrap_r50_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soar_sofrap_r50_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_sofrap_r50_o2_sd2_st2` | sofrap_r50_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soar_sofrap_r50_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_sofrap_r50_o2_sd2_st3` | sofrap_r50_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soar_sofrap_r50_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_sofrap_r50_o2_sd2_st4` | sofrap_r50_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soar_sofrap_r50_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_sofrapp_r12_o2_sd0_st1` | sofrapp_r12_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soar_sofrapp_r12_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_sofrapp_r12_o2_sd0_st2` | sofrapp_r12_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soar_sofrapp_r12_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_sofrapp_r12_o2_sd0_st3` | sofrapp_r12_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soar_sofrapp_r12_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_sofrapp_r12_o2_sd0_st4` | sofrapp_r12_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soar_sofrapp_r12_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_sofrapp_r12_o2_sd1_st1` | sofrapp_r12_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soar_sofrapp_r12_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_sofrapp_r12_o2_sd1_st2` | sofrapp_r12_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soar_sofrapp_r12_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_sofrapp_r12_o2_sd1_st3` | sofrapp_r12_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soar_sofrapp_r12_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_sofrapp_r12_o2_sd1_st4` | sofrapp_r12_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soar_sofrapp_r12_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_sofrapp_r12_o2_sd2_st1` | sofrapp_r12_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soar_sofrapp_r12_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_sofrapp_r12_o2_sd2_st2` | sofrapp_r12_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soar_sofrapp_r12_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.125 |
| `soar_sofrapp_r12_o2_sd2_st3` | sofrapp_r12_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soar_sofrapp_r12_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.125 |
| `soar_sofrapp_r12_o2_sd2_st4` | sofrapp_r12_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soar_sofrapp_r12_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.125 |
| `soar_sofrapp_r50_o2_sd0_st1` | sofrapp_r50_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soar_sofrapp_r50_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_sofrapp_r50_o2_sd0_st2` | sofrapp_r50_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soar_sofrapp_r50_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_sofrapp_r50_o2_sd0_st3` | sofrapp_r50_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soar_sofrapp_r50_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_sofrapp_r50_o2_sd0_st4` | sofrapp_r50_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soar_sofrapp_r50_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_sofrapp_r50_o2_sd1_st1` | sofrapp_r50_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soar_sofrapp_r50_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_sofrapp_r50_o2_sd1_st2` | sofrapp_r50_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soar_sofrapp_r50_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_sofrapp_r50_o2_sd1_st3` | sofrapp_r50_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soar_sofrapp_r50_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_sofrapp_r50_o2_sd1_st4` | sofrapp_r50_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soar_sofrapp_r50_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soar_sofrapp_r50_o2_sd2_st1` | sofrapp_r50_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1/soar_sofrapp_r50_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st1 offline_ratio=0.0 |
| `soar_sofrapp_r50_o2_sd2_st2` | sofrapp_r50_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config4/soar_sofrapp_r50_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st2 offline_ratio=0.5 |
| `soar_sofrapp_r50_o2_sd2_st3` | sofrapp_r50_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config2/soar_sofrapp_r50_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st3 offline_ratio=0.5 |
| `soar_sofrapp_r50_o2_sd2_st4` | sofrapp_r50_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_frap/sumohz1x1_config3/soar_sofrapp_r50_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor rho train order_stage=o2/st4 offline_ratio=0.5 |
| `soas_soddqnp_l10_o2_sd0_st1` | soddqnp_l10_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soas_soddqnp_l10_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st1 offline_ratio=0.25 |
| `soas_soddqnp_l10_o2_sd0_st2` | soddqnp_l10_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soas_soddqnp_l10_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st2 offline_ratio=0.25 |
| `soas_soddqnp_l10_o2_sd0_st3` | soddqnp_l10_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soas_soddqnp_l10_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st3 offline_ratio=0.25 |
| `soas_soddqnp_l10_o2_sd0_st4` | soddqnp_l10_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soas_soddqnp_l10_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st4 offline_ratio=0.25 |
| `soas_soddqnp_l10_o2_sd1_st1` | soddqnp_l10_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soas_soddqnp_l10_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st1 offline_ratio=0.25 |
| `soas_soddqnp_l10_o2_sd1_st2` | soddqnp_l10_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soas_soddqnp_l10_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st2 offline_ratio=0.25 |
| `soas_soddqnp_l10_o2_sd1_st3` | soddqnp_l10_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soas_soddqnp_l10_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st3 offline_ratio=0.25 |
| `soas_soddqnp_l10_o2_sd1_st4` | soddqnp_l10_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soas_soddqnp_l10_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st4 offline_ratio=0.25 |
| `soas_soddqnp_l10_o2_sd2_st1` | soddqnp_l10_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soas_soddqnp_l10_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st1 offline_ratio=0.25 |
| `soas_soddqnp_l10_o2_sd2_st2` | soddqnp_l10_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soas_soddqnp_l10_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st2 offline_ratio=0.25 |
| `soas_soddqnp_l10_o2_sd2_st3` | soddqnp_l10_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soas_soddqnp_l10_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st3 offline_ratio=0.25 |
| `soas_soddqnp_l10_o2_sd2_st4` | soddqnp_l10_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soas_soddqnp_l10_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st4 offline_ratio=0.25 |
| `soas_soddqnp_l1_o2_sd0_st1` | soddqnp_l1_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soas_soddqnp_l1_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st1 offline_ratio=0.25 |
| `soas_soddqnp_l1_o2_sd0_st2` | soddqnp_l1_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soas_soddqnp_l1_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st2 offline_ratio=0.25 |
| `soas_soddqnp_l1_o2_sd0_st3` | soddqnp_l1_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soas_soddqnp_l1_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st3 offline_ratio=0.25 |
| `soas_soddqnp_l1_o2_sd0_st4` | soddqnp_l1_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soas_soddqnp_l1_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st4 offline_ratio=0.25 |
| `soas_soddqnp_l1_o2_sd1_st1` | soddqnp_l1_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soas_soddqnp_l1_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st1 offline_ratio=0.25 |
| `soas_soddqnp_l1_o2_sd1_st2` | soddqnp_l1_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soas_soddqnp_l1_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st2 offline_ratio=0.25 |
| `soas_soddqnp_l1_o2_sd1_st3` | soddqnp_l1_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soas_soddqnp_l1_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st3 offline_ratio=0.25 |
| `soas_soddqnp_l1_o2_sd1_st4` | soddqnp_l1_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soas_soddqnp_l1_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st4 offline_ratio=0.25 |
| `soas_soddqnp_l1_o2_sd2_st1` | soddqnp_l1_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soas_soddqnp_l1_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st1 offline_ratio=0.25 |
| `soas_soddqnp_l1_o2_sd2_st2` | soddqnp_l1_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soas_soddqnp_l1_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st2 offline_ratio=0.25 |
| `soas_soddqnp_l1_o2_sd2_st3` | soddqnp_l1_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soas_soddqnp_l1_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st3 offline_ratio=0.25 |
| `soas_soddqnp_l1_o2_sd2_st4` | soddqnp_l1_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soas_soddqnp_l1_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st4 offline_ratio=0.25 |
| `soas_soddqnpv_probe_o2_sd0_st1` | soddqnpv_probe_soa/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soas_soddqnpv_probe_o2_sd0_st1` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st1 offline_ratio=0.25 |
| `soas_soddqnpv_probe_o2_sd0_st2` | soddqnpv_probe_soa/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soas_soddqnpv_probe_o2_sd0_st2` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st2 offline_ratio=0.25 |
| `soas_soddqnpv_probe_o2_sd0_st3` | soddqnpv_probe_soa/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soas_soddqnpv_probe_o2_sd0_st3` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st3 offline_ratio=0.25 |
| `soas_soddqnpv_probe_o2_sd0_st4` | soddqnpv_probe_soa/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soas_soddqnpv_probe_o2_sd0_st4` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st4 offline_ratio=0.25 |
| `soas_soddqnpv_probe_o2_sd1_st1` | soddqnpv_probe_soa/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soas_soddqnpv_probe_o2_sd1_st1` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st1 offline_ratio=0.25 |
| `soas_soddqnpv_probe_o2_sd1_st2` | soddqnpv_probe_soa/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soas_soddqnpv_probe_o2_sd1_st2` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st2 offline_ratio=0.25 |
| `soas_soddqnpv_probe_o2_sd1_st3` | soddqnpv_probe_soa/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soas_soddqnpv_probe_o2_sd1_st3` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st3 offline_ratio=0.25 |
| `soas_soddqnpv_probe_o2_sd1_st4` | soddqnpv_probe_soa/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soas_soddqnpv_probe_o2_sd1_st4` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st4 offline_ratio=0.25 |
| `soas_soddqnpv_probe_o2_sd2_st1` | soddqnpv_probe_soa/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1/soas_soddqnpv_probe_o2_sd2_st1` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st1 offline_ratio=0.25 |
| `soas_soddqnpv_probe_o2_sd2_st2` | soddqnpv_probe_soa/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config4/soas_soddqnpv_probe_o2_sd2_st2` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st2 offline_ratio=0.25 |
| `soas_soddqnpv_probe_o2_sd2_st3` | soddqnpv_probe_soa/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config2/soas_soddqnpv_probe_o2_sd2_st3` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st3 offline_ratio=0.25 |
| `soas_soddqnpv_probe_o2_sd2_st4` | soddqnpv_probe_soa/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/tsc/sumo_dqn/sumohz1x1_config3/soas_soddqnpv_probe_o2_sd2_st4` | REGISTERED: cl_hz1x1 anchor screen train order_stage=o2/st4 offline_ratio=0.25 |

评估 attempt 共 4384 行，按包/来源聚合：

| 评估包/来源 | n DONE/FAILED/ABORTED | 条件集 | eval seeds |
|---|---|---|---|
| `ddqn_clear` | 234/0/0 | clear,drift_diag | 101,102,103 |
| `ddqn_fifo` | 180/0/0 | fifo | — |
| `ddqn_probe` | 16/0/0 | probe | — |
| `duel_clear` | 180/0/0 | clear | — |
| `duel_fifo` | 180/0/0 | fifo | — |
| `frap_clear` | 180/0/0 | clear | — |
| `frap_fifo` | 180/0/0 | fifo | — |
| `ppo_clear` | 0/0/0 | clear | — |
| `ppov2_clear` | 180/0/0 | clear | — |
| `ppov2_sampled_eval` | 120/0/0 | probe | — |
| `soddqn` | 54/0/0 | drift_diag | 101,102,103 |
| `soddqn_r12_soa` | 90/0/0 | semi_offline_anchor | — |
| `soddqn_r50_soa` | 90/0/0 | semi_offline_anchor | — |
| `soddqn_so` | 180/0/0 | semi_offline | — |
| `soddqnp_l10_soa` | 90/0/0 | semi_offline_anchor | — |
| `soddqnp_l1_soa` | 90/0/0 | semi_offline_anchor | — |
| `soddqnp_r12_soa` | 90/0/0 | semi_offline_anchor | — |
| `soddqnp_r50_soa` | 90/0/0 | semi_offline_anchor | — |
| `soddqnp_soa` | 180/0/0 | semi_offline_anchor | — |
| `soddqnpv_probe_soa` | 90/0/0 | semi_offline_anchor | — |
| `soddqnpv_soa` | 90/0/0 | semi_offline_anchor | — |
| `soduel_r12_soa` | 90/0/0 | semi_offline_anchor | — |
| `soduel_r50_soa` | 90/0/0 | semi_offline_anchor | — |
| `soduel_so` | 180/0/0 | semi_offline | — |
| `soduelp_r12_soa` | 90/0/0 | semi_offline_anchor | — |
| `soduelp_r50_soa` | 90/0/0 | semi_offline_anchor | — |
| `soduelp_soa` | 180/0/0 | semi_offline_anchor | — |
| `soduelpv_soa` | 90/0/0 | semi_offline_anchor | — |
| `sofrap_r12_soa` | 90/0/0 | semi_offline_anchor | — |
| `sofrap_r50_soa` | 90/0/0 | semi_offline_anchor | — |
| `sofrap_so` | 180/0/0 | semi_offline | — |
| `sofrapp_r12_soa` | 90/0/0 | semi_offline_anchor | — |
| `sofrapp_r50_soa` | 90/0/0 | semi_offline_anchor | — |
| `sofrapp_soa` | 180/0/0 | semi_offline_anchor | — |
| `sofrappv_soa` | 90/0/0 | semi_offline_anchor | — |

<!-- EXTRA-BEGIN -->

---

## 正式报告：hz1x1 同质场景持续学习（cl1 + so1 合并证据链）

### 1. 协议

顺序训练冻结评估：`train(scene_k) → freeze → eval(scene_1..k)`，产出下三角矩阵。

- 场景包（同拓扑 hz1x1、同 signal_config，仅流量/物理 ID 不同）：`s1=sumohz1x1`、`c4=sumohz1x1_config4`、`c2=sumohz1x1_config2`、`c3=sumohz1x1_config3`
- 序列：`o2 = s1→c4→c2→c3`；`o3 = c2→c3→s1→c4`
- 训练种子 0/1/2，冻结评估种子 101/102/103，每 stage 100 episode
- 单任务基线：`ddqn/duel/frap/ppov2` × 4 场景 × 3 种子（`offline_ratio=0` 与 clear 同码路，so 臂直接复用）

### 2. 批次

| 批次 | 队列 | 臂 | 任务 | 状态 |
|---|---|---|---|---|
| cl1 | `run_queue_cl1.json` | ddqn/duel/frap ×{clear,fifo} + ppov2_clear + 基线 | 1476 | 全 DONE |
| so1 | `run_queue_so1.json` | soddqn/soduel/sofrap | 612（72 训 + 540 eval） | 全 DONE |

**半离语义（so1）**：stage k 训练 batch = 75% 在线 + 25% 历史归档（`scene_intersection_balanced`）；历史可见集 = 因果 `scene_order[:k]`（不见未来包）；每 stage 归档自身 transition 供后续使用；在线 replay 切场景清空。PPO 为 on-policy，不吃离线 replay，未入 so 臂。

### 3. 最深非对角退化（相对同臂基线，全链取 max）

| 臂 | clear | fifo | 25% 因果半离线 |
|---|---:|---:|---:|
| ddqn | **+596.6**（o2_sd1 c2: 668.1） | +13.4 | **+3.8** |
| duel | **+282.8**（o3_sd0 c2: 354.2） | +166.8 | **+6.0** |
| frap | **+139.7**（o2_sd0 s1: 217.1） | +279.0 | **+9.3** |
| ppov2 | +80.2（基线方差大，非崩型） | — | — |

崩溃链计数（老包 ATT ≥120）：clear 下 ddqn 4/6、duel 3/6、frap 2/6、ppov2 6/6（PPO 按自身 1.6 倍口径为 3/6）；fifo 下 duel 3/6、frap 2/6、ddqn 0/6（但仍有 +14 同型退化）；**半离线三臂 0/18**。

### 4. 半离线终态（st4 首场景 ATT，三种子均值）

| 臂 | o2 | o3 | 基线参考 |
|---|---:|---:|---:|
| soddqn | 78.8 | 71.5 | ddqn s1≈78.1 / c2≈71.4 |
| soduel | 79.4 | 71.8 | duel s1≈78.6 / c2≈71.4 |
| sofrap | 83.7* | 70.8 | frap s1≈77.5 / c2≈71.0 |

*sofrap_o2 sd0/sd1 残留 +7~9s 漂移，为全批次唯一可见残余，量级为 fifo 同位的 ~1/20。

### 5. 结论

1. **遗忘是 off-policy 值迭代族的机制性失败**，非实现特例：三种结构迥异骨干（MLP/dueling/phase-competition）在相同毒性阶段同向退化。
2. **结构调制易感性但不免疫**：duel > ddqn > frap；soduel 证明最脆结构亦可被数据锚定完全救回（fifo 下 245s → so 下 ~79s 贴基线）。
3. **25% 因果半离线以架构无关方式消除遗忘**：失效 clear 的 +597s 退化压至 +4s；fifo（≈14ep 短窗，旧数据随即覆写）收益实为 buffer 暖启动，不等于长程旧场景锚定。
4. **性质边界**：提升的是旧场景保持度（anti-forgetting），对角线新学习近似无损；不构成"半离线普遍提升绝对性能"的证据。结论限于 hz1x1 同质拓扑；PPO 不适用同机制。

### 6. 产物

- 矩阵：`artifacts/cl_hz1x1/matrix_report.json`、`matrix_report_so.json`
- 图：`artifacts/cl_hz1x1/fig_so_retention_trajectory.png`、`fig_so_worst_degradation.png`
- 队列：`artifacts/cl_hz1x1/run_queue_cl1.json`、`run_queue_so1.json`
- 生成/收割/绘图：`dev/cl_hz1x1/{build_queue,build_queue_so,harvest_matrix,plot_so_compare}.py`
- 归档机制：`arterial/replay.py`（HistoricalPool/mixed_batch/HistoryArchiveWriter）+ `agent/{dqn,frap}.py` 移植接入

<!-- EXTRA-END -->