# hitl_live_ab

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：—
- plan_id/队列：—
- 设计稿/证据位置：—
- 结论：—
- 登记单元 6（train/eval/smoke/batch/calibration = 5/0/0/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `HITL_FULL-NAMING-CORRECTION` | NOTE/sumohz1x1/s-1 | 0 | DONE | — | — | `—` | 口径更正:HITL_FULL=示范双预训(A)+b2偏好塑形(B)双通道集成;expert_pool字段在demo_ac构造分支被静默丢弃(runtime.py |
| `HITL_AB-SD0` | AB/sumohz1x1/s0 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_AB-SD0` | A+B臂(示范双预训+b2塑形,无专家池):与FULL逐位一致(5/5 seed digest同)→代码级确认C腿在demo_ac路径未接线;FULL实为双通道 |
| `HITL_AB-SD1` | AB/sumohz1x1/s1 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_AB-SD1` | A+B臂(示范双预训+b2塑形,无专家池):与FULL逐位一致(5/5 seed digest同)→代码级确认C腿在demo_ac路径未接线;FULL实为双通道 |
| `HITL_AB-SD2` | AB/sumohz1x1/s2 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_AB-SD2` | A+B臂(示范双预训+b2塑形,无专家池):与FULL逐位一致(5/5 seed digest同)→代码级确认C腿在demo_ac路径未接线;FULL实为双通道 |
| `HITL_AB-SD3` | AB/sumohz1x1/s3 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_AB-SD3` | A+B臂(示范双预训+b2塑形,无专家池):与FULL逐位一致(5/5 seed digest同)→代码级确认C腿在demo_ac路径未接线;FULL实为双通道 |
| `HITL_AB-SD4` | AB/sumohz1x1/s4 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_AB-SD4` | A+B臂(示范双预训+b2塑形,无专家池):与FULL逐位一致(5/5 seed digest同)→代码级确认C腿在demo_ac路径未接线;FULL实为双通道 |
