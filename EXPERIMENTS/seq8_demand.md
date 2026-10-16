# SEQ8 语义需求包 + 8 段持续/半离线 pilot（hz1x1）

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：语义化需求多样性下，「学习恢复≠控制恢复」「继续学习改写历史表现」
  「退化幅度非固定」三结论是否对序列形态（无回访/块重复回访/临近渐变）与
  历史利用策略（在线 / 自产档案动态采样）稳健。
- plan_id/队列：`build-seq8-plan`（schema_version=2）三份 manifest：
  `seq8_norep_v1`/`seq8_rep_v1`/`seq8_drift_v1` → `sequential launch`
- 设计稿/证据位置：`docs/experiments/2026-10-15_seq8_demand_suite.md`；
  需求包清单 `data/raw_data/hangzhou_1x1_demand_packs.json`；
  产物 `data/output_data/sequential/seq8_{norep,rep,drift}_formal_20261015/`
- 范围裁决（2026-10-16 用户）：
  - **SELF-CQ-R50 臂为意图错误**——Q 锚定工作池属此前已否定的技术路线，
    不纳入本 pilot 语义；SD2/SD3 六个中断 attempt 标 `cancelled` 封存，
    SD0/SD1 六个已完成 run 数据保留为产物、但不进入结论对比；
  - **seeds 只到 SD3**——SD4 波（9 run）撤销不派发；
  - 有效范围 = 3 序列 × {CONT, SELF-DHOA-R50} × seeds 0..3 = 24 run。
- 结论：（进行中——pilot 3 序列 × {CONT, SELF-DHOA-R50} × seeds 0..3
  = 24 run；SELF-CQ-R50 臂已作废）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
