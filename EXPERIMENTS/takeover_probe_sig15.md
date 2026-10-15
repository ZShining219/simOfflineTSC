# takeover_probe_sig15

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：—
- plan_id/队列：—
- 设计稿/证据位置：—
- 结论：—
- 登记单元 24（train/eval/smoke/batch/calibration = 5/18/0/1/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `takeover_probe/attn_probe_c2` | attn_probe_c2/sumohz1x1/s— | — | DONE | — | — | `data/output_data/dev_takeover_probe/attn_probe_c2` | E5响应契约探针: sga15/sga16逐tick w-θ_watch余弦对齐+反事实wL1/动作翻转; attn_text_probe.py --corri |
| `takeover_probe/sig15_S1` | sig15_S1/sumohz1x1_config2/s— | 100 | DONE | — | — | `data/output_data/dev_takeover_probe/sig15_S1` | sig15同配跨场景在训S1路网; base=p1_formal_dqn_sumohz1x1_config2_seed0 ep400 |
| `takeover_probe/sig15_S3` | sig15_S3/sumohz1x1_config4/s— | 100 | DONE | — | — | `data/output_data/dev_takeover_probe/sig15_S3` | sig15同配跨场景在训S3; base=p1_formal_dqn_sumohz1x1_config4_seed0 ep400 |
| `takeover_probe/sig15_S4` | sig15_S4/sumohz1x1_config3/s— | 100 | DONE | — | — | `data/output_data/dev_takeover_probe/sig15_S4` | sig15同配跨场景在训S4; base=p1_formal_dqn_sumohz1x1_config3_seed0 ep400 |
| `takeover_probe/sig15_corrload` | sig15_corrload/sumohz1x1/s— | 100 | DONE | — | — | `data/output_data/dev_takeover_probe/sig15_corrload` | 场景注意力+意图内化(θ_reward责任集+θ_watch观察集监督) @corridor-load三件套注入(surge+静默feeder surge+di |
| `takeover_probe/sig16_noaux` | sig16_noaux/sumohz1x1/s— | 100 | DONE | — | — | `data/output_data/dev_takeover_probe/sig16_noaux` | sig15同配 B格消融: --aux-weight 0 无观察集监督; 结论=性能不降但w对文本失明(L1 0.03 vs 0.21), aux=可审计契约而 |

评估 attempt 共 18 行，按包/来源聚合：

| 评估包/来源 | n DONE/FAILED/ABORTED | 条件集 | eval seeds |
|---|---|---|---|
| `sig15_S1_eval` | 1/0/0 | — | — |
| `sig15_S3_eval` | 1/0/0 | — | — |
| `sig15_S4_eval` | 1/0/0 | — | — |
| `sig15_dispatch_eval` | 1/0/0 | — | — |
| `sig15_eval` | 1/0/0 | — | — |
| `sig15_eval_S1` | 1/0/0 | — | — |
| `sig15_eval_S3` | 1/0/0 | — | — |
| `sig15_eval_S4` | 1/0/0 | — | — |
| `sig15_eval_delay0` | 1/0/0 | — | — |
| `sig15_eval_delay120` | 1/0/0 | — | — |
| `sig15_eval_delay60` | 1/0/0 | — | — |
| `sig15_eval_delay90` | 1/0/0 | — | — |
| `sig15_eval_distract` | 1/0/0 | — | — |
| `sig15_eval_holdout` | 1/0/0 | — | — |
| `sig15_eval_main8` | 1/0/0 | — | — |
| `sig15_eval_modcmp` | 1/0/0 | — | — |
| `sig15_eval_sev` | 1/0/0 | — | — |
| `sig16_eval` | 1/0/0 | — | — |
