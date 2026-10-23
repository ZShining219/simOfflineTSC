# xiasha1_v3_cross_eval_v1（修复几何 best-ckpt 跨场景评估）

> 状态：已完成
> 发起日期：2026-10-22　｜　负责人/Agent：Devin

## 关联

- campaign：`xiasha1_v3_cross_eval_v1`（台账 384 行 DONE）
- run_id / 队列：`xi3x_<arm>_tr<src>_s<n>_te<tgt>_ss10000`；
  `artifacts/xiasha1_v3_cross_eval_v1/run_queue.json`
- 源 ckpt：`xiasha1_online_v3` 各 (arm, source, seed) best_checkpoint
- 对照基线：`xiasha1_cross_eval_v1`（同构设计，截断几何）

## 1. 改动意图

在修复几何上复测"单场景专训策略跨场景泛化差"——v1 期因
TT 被压缩到 ~13-15s 量程，泛化退化可能被低估。

## 2. 改动模块

仅评估产物：dev/xiasha1_online_v3/build_cross_eval_queue.py、
dev/ledger/register_xiasha1_v3_cross_eval.py、
backfill_xiasha1_v3_cross_eval.py。

## 3. 实验设计

与 v1 逐位一致：8 臂 × 4 source（asym_nonly/asym_ns/dense_extreme/
fluct_burst）× 3 train seeds × 4 target × 1 SUMO seed(10000)=384；
eval-only overlay；被评路网与训练路网同为 280m 修复几何。

## 4. 实验结果

384/384 DONE。臂级均值（inTT=源=目标 域内；xTT=跨源；gap=xTT/inTT）：

| arm | inTT | xTT | gap% | queue(全) | thr |
|---|---:|---:|---:|---:|---:|
| sac | 41.18 | 44.89 | **9.0** | 9.10 | 2022 |
| presslight | 42.55 | 49.11 | 15.4 | 11.26 | 2021 |
| dueling | 42.51 | 54.19 | 27.5 | 18.62 | 1987 |
| ddqn | 42.92 | 59.61 | 38.9 | 21.88 | 1966 |
| frap | 44.94 | 63.64 | 41.6 | 25.07 | 1903 |
| mplight | 44.84 | 65.59 | 46.3 | 24.90 | 1917 |
| d3qn | 42.12 | 62.18 | 47.6 | 19.41 | 1965 |
| dqn | 42.62 | 63.53 | **49.0** | 24.09 | 1962 |

灾难单元（示例）：dqn 训于 asym_ns → 评 dense_extreme TT **171.7**；
d3qn asym_ns → asym_nonly TT **148.9**（域内 36.8）。SAC/PressLight
各行平坦（38–67）。

## 5. 结论与后续

**v1 的排序在真实几何上翻转**：v1 截断期 DQN 家族 gap 仅 3.9–4.8%
（"泛化尚可"的假象），v3 实测 27.5–49.0% 最差；SAC/PressLight 反而
最稳（9.0%/15.4%）。佐证：DQN 家族学到的饥饿/过拟合策略跨场景
脆断——asym_ns 训出的相位3压制策略搬到需要左转弯服务的
asym_nonly 上直接灾难。与"v2 靠牺牲部分车流换 TT"机制判读一致，
只是其表达通道从 uninserted 变成了网内长队列/长等待。

后续：per-movement/per-lane 挖掘 v3 网内饥饿是否仍集中在
asym_nonly 相位3；评估口径若需多 SUMO seed 再扩。
