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
    不纳入本 pilot 语义；SD0/SD1 六个已完成 run 数据保留为产物、但不
    进入结论对比；
  - **pilot 整体关闭**——SD2/SD3 十二个中断 run（CONT+DHOA+CQ）标
    `cancelled` 并清理半截 attempt 产物，SD4 波（9 run）未派发；
  - **SELF 半离线链路作废关闭**——远程 `cl_hz1x1` 线（`agent/dqn.py`
    教师锚定 + `arterial/replay.py` HistoricalPool 全量自产归档）为仓内
    唯一半离线口径；`sequential/self_archive.py` 与 SELF 接线已移除。
- 登记：18 train DONE + 18 train SCOPE_DISCARD + 6 batch（
  `dev/ledger/append_seq8.py` 幂等补登）。
- 结论（已完成 run 证据，仅 SD0/SD1 配对 n=42 场景格）：
  - CONT-FIFO 历史退化 mean +21.4%/med +9.8%/max +326%（灾难格 9/42）；
  - SELF-DHOA-R50 mean +1.2%/med +1.0%/max +39.1%（灾难格 1/42）——
    均匀重放自产 stage<k 数据基本压住遗忘，与 cl_hz1x1 主线方向一致；
  - REP 回访局部恢复成立（sparse_low 回访≈首次水平）但末段仍漂移；
  - 三结论在需求语义多样性下方向复现；CQ 数据仅作路线证伪留档。

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
