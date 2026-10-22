# xiasha1_compound_train_v1：复合需求包上的早收敛假说检验

> 状态：进行中（2026-10-22 派发 30/30）
> 发起日期：2026-10-22 ｜ 负责人/Agent：Z / Devin
> 最近更新：2026-10-22

## 关联

- campaign：`xiasha1_compound_train_v1`（5 臂 × 2 包 × 3 seeds = 30 train run）
- 队列文件：`artifacts/xiasha1_compound_train/run_queue.json`
  （artifacts 为 gitignore 仓外资产）
- overlay：复用 `configs/tsc/xiasha1_online_v2/{d3qn,mplight,frap,
  presslight,sac}.yml`（与 v2 完全同口径，保证收敛对比可比）
- 需求包：`data/raw_data/xiasha1_compound_{a,b}_1h/` + 对应
  `configs/sim/xiasha1_compound_{a,b}.cfg`（生成器
  `dev/demand_packs/build_xiasha_demands.py`，cat=compound）
- 前序：`xiasha1_online_v2`（288/288 DONE，~ep30 早收敛）；
  `xiasha1_v2_bestckpt_ts_v1`（96 冻结重放，时间序列）；
  `xiasha1_compound_probe_v1`（22 eval 跨评探测，复合包区分度已证实）
- 产物目录：`data/output_data/tsc/sumo_<arm>/xiasha1_compound_<x>/xi3on_*`

## 1. 改动意图

v2 在 12 个单轴消融需求包上观察到两个现象：

1. 全部 8 臂在 ep ~18–40 即收敛（conv5 中位 ~ep30），远早于
   ε 下限（ep180）——不是探索不足；
2. 多数包上学习臂间几乎无差异（DQN 家族散布 ≤0.2s）。

`dev/demand_packs/` 的 PACKS 设计是单轴消融（每包只改 scale/arm/
转向/shape 之一），单因子对策易被快速拟合。`xiasha1_compound_probe_v1`
已证明多轴复合包有区分度（dense_extreme best-ckpt 跨评：
compound_a 学习臂散布 3.31s、compound_b 2.05s，且两包臂序不同）。

本 campaign 检验的因果假说：**"v2 的顺利早收敛是需求包太简单
（单轴）所致"**。若复合包上收敛明显推迟、曲线持续波动或臂间
分化加大，则假说成立；若仍在 ep ~30 收敛，则早收敛是任务结构
（单路口、4 相位、无切换约束）的内禀性质，需求侧改动已到极限。

## 2. 改动模块

- `dev/demand_packs/build_xiasha_demands.py`：`shape` 扩为
  `(t, e0)` 双参接口，`shape_per_arm` 支持方向相关突发
  （与 normalize 互斥）；新增 `cat="compound"` 两包；
  manifest 每包补 `note` 字段。
- 新需求包（14 包集合清单 `data/raw_data/xiasha1_demand_packs.json`）：
  - `compound_a`（2463 veh）：饱和量级 + 北臂失衡（58%）+
    s55/l19/r26 均衡转向 + 1800–2400s **全臂** ×2.5 突发
    （瞬时 ~6300 veh/h，全域事件集聚）；
  - `compound_b`（2188 veh）：同量级但突发**只打北臂**（×2.5，
    窗内北臂占 88%、瞬时 N≈3560 veh/h 撞 NS 短直绿 27s；
    窗口总瞬时 ~4100 ≈ 对照论文高方差峰值，窗内考相位权重
    在线切换；窗外 ~1800 ≈ 论文低谷）。
- `dev/xiasha1_compound_train/build_queue.py`：30-run 队列生成件。
- `dev/ledger/register_xiasha1_compound_train.py`：台账登记件。
- 文档：本文件；dev/INDEX 登记。

## 3. 实验设计

### 3.1 矩阵与口径

| 项 | 值 |
| --- | --- |
| 臂 | d3qn、mplight、frap、presslight、sac（5 臂，覆盖不同观测/奖励/骨干） |
| 需求包 | compound_a、compound_b |
| 训练 seeds | 0/1/2 |
| episodes | 200（3600s/ep，action_interval 10s） |
| 评估 | 每 ep 贪心评估 ep0–200，与 v2 完全同口径 |
| 探索 | 复用 v2 overlay：d3qn/frap/presslight ε 1.0→0.05 decay 0.99995
  （~ep180 下限）；sac 原生 Categorical；mplight pfrl 线性 ε |
| 语义 | legacy pseudo_step；SUMO seed=fixed_default；saveReplay=false |

### 3.2 判定指标（对照 v2 同臂同口径）

- **收敛 episode**（conv5：TT 距 final-20 均值 ≤5% 的首个 ep）：
  v2 全臂中位 ~ep30；复合包上显著后移 → 支持假说；
- **曲线形态**：fluct 型需求下 greedy 评估曲线若持续振荡而不
  平台化，说明任务非平稳；
- **臂间散布**（final-20 TT/uninserted/queue）：probe 跨评已见
  3.3/2.1s 散布，训练后是否保持；
- **best ckpt 质量**：与 v2 dense_extreme（最高量级单轴包）对比。

### 3.3 治理

30 行 REGISTERED（code_commit 7b41114）→ sac compound_a smoke
通过（ep0–2 正常训练）→ 队列派发 parallel 12 → 完成后
`backfill_xiasha1_compound_train` 回填 DONE + 本文件 §4 补结果。

## 4. 结果

（待回填：30/30 完成后补收敛 ep 分布、final-20 指标表与 v2 对照）

## 勘误（2026-10-22，几何截断缺陷）

本文档全部指标采集自**截断几何**的 xiasha1 路网：进口车道仅 106.4m，
且路由 `departPos≈80`（车辆自距停车线 ~22-27m 处、以 10m/s 凭空插入——
源自 `tools/xiasha_sumo/converter.py` 相机视野重建遗留语义，经
`dev/demand_packs/build_xiasha_demands.py` 传播至全部 14 包）。

影响口径：
- 每车道可观测排队物理上限 ~3 辆，queue 均值/分位、waiting_time 长尾、
  uninserted 计数、travel_time 绝对值均失真；
- 结论中"臂间相对比较"仅在同几何内成立，**不可与修复后（280m、车道起点
  插入）的 v3 指标直接比较**；
- 台账 `ledger/runs.jsonl` 中全部 xiasha1* campaign 行已加
  `errata=geometry_truncated_v1v2`；正式对比口径以 `xiasha1_online_v3`
  为准。
