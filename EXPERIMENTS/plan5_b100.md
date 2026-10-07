# Plan5-B100 跨算法锚点/校准（Phase0）

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：DDQN/CTXDDQN 在 S1-S4 场景 100ep 锚点 + 校准，建立跨算法比较底座。
- plan_id/队列：plan5_b100 logical_run_id=P5-{ANCHOR,CAL}-<ALGO>-<场景>-SD<seed>
- 设计稿/证据位置：data/output_data/cross_algorithm/plan5_b100/{runs,manifest,audit}
- 结论：Phase0 基础设施（fixedtime 参照/probe/resume 等价/并发门禁/环境/provenance）与 DDQN/CTXDDQN 锚点、校准全部完成；Phase0 不构成正式批次结论。
- 登记单元 68（train/eval/smoke/batch/calibration = 40/0/0/10/18）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `P5-PHASE0/environment` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/environment` | Phase0 基础设施/门禁证据批次: plan5 micromamba 环境冻结 |
| `P5-PHASE0/fixedtime` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/fixedtime` | Phase0 基础设施/门禁证据批次: FixedTime 参照（S1-S4×repeat×2） |
| `P5-PHASE0/probe_evaluation_runs` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/probe/evaluation_runs` | Phase0 基础设施/门禁证据批次: 探针评估执行 |
| `P5-PHASE0/probe_plan5_fixed_probe_v1` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/probe/plan5_fixed_probe_v1` | Phase0 基础设施/门禁证据批次: plan5_fixed_probe_v1 冻结探针 |
| `P5-PHASE0/provenance` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/provenance` | Phase0 基础设施/门禁证据批次: 源码 provenance/bundle |
| `P5-PHASE0/resource_gate` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/resource_gate` | Phase0 基础设施/门禁证据批次: 1/4/8 并发门禁 |
| `P5-PHASE0/resource_gate_attempt_1_failed` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/resource_gate_attempt_1_failed` | Phase0 基础设施/门禁证据批次: 并发门禁失败 attempt |
| `P5-PHASE0/resource_gate_attempt_2` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/resource_gate_attempt_2` | Phase0 基础设施/门禁证据批次: 并发门禁 attempt2 |
| `P5-PHASE0/resource_gate_attempt_3` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/resource_gate_attempt_3` | Phase0 基础设施/门禁证据批次: 并发门禁 attempt3 |
| `P5-PHASE0/resume_equivalence` | phase0/—/s— | — | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/resume_equivalence` | Phase0 基础设施/门禁证据批次: resume 等价性验证 |
| `P5-CAL-PPO-S2-LR1E4-EC001-SD100` | PPO_CAL/sumohz1x1/s100 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR1E4-EC001-SD100` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR1E4-EC001-SD101` | PPO_CAL/sumohz1x1/s101 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR1E4-EC001-SD101` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR1E4-EC001-SD102` | PPO_CAL/sumohz1x1/s102 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR1E4-EC001-SD102` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR1E4-EC01-SD100` | PPO_CAL/sumohz1x1/s100 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR1E4-EC01-SD100` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR1E4-EC01-SD101` | PPO_CAL/sumohz1x1/s101 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR1E4-EC01-SD101` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR1E4-EC01-SD102` | PPO_CAL/sumohz1x1/s102 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR1E4-EC01-SD102` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR25E5-EC001-SD100` | PPO_CAL/sumohz1x1/s100 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR25E5-EC001-SD10` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR25E5-EC001-SD101` | PPO_CAL/sumohz1x1/s101 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR25E5-EC001-SD10` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR25E5-EC001-SD102` | PPO_CAL/sumohz1x1/s102 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR25E5-EC001-SD10` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR25E5-EC01-SD100` | PPO_CAL/sumohz1x1/s100 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR25E5-EC01-SD100` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR25E5-EC01-SD101` | PPO_CAL/sumohz1x1/s101 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR25E5-EC01-SD101` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR25E5-EC01-SD102` | PPO_CAL/sumohz1x1/s102 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR25E5-EC01-SD102` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR5E4-EC001-SD100` | PPO_CAL/sumohz1x1/s100 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR5E4-EC001-SD100` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR5E4-EC001-SD101` | PPO_CAL/sumohz1x1/s101 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR5E4-EC001-SD101` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR5E4-EC001-SD102` | PPO_CAL/sumohz1x1/s102 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR5E4-EC001-SD102` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR5E4-EC01-SD100` | PPO_CAL/sumohz1x1/s100 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR5E4-EC01-SD100` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR5E4-EC01-SD101` | PPO_CAL/sumohz1x1/s101 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR5E4-EC01-SD101` | attempts=1; type=CAL |
| `P5-CAL-PPO-S2-LR5E4-EC01-SD102` | PPO_CAL/sumohz1x1/s102 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-CAL-PPO-S2-LR5E4-EC01-SD102` | attempts=1; type=CAL |
| `P5-ANCHOR-CTXDDQN-S1-SD0` | CTXDDQN_ANCHOR/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S1-SD0` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S1-SD1` | CTXDDQN_ANCHOR/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S1-SD1` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S1-SD2` | CTXDDQN_ANCHOR/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S1-SD2` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S1-SD3` | CTXDDQN_ANCHOR/sumohz1x1_config2/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S1-SD3` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S1-SD4` | CTXDDQN_ANCHOR/sumohz1x1_config2/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S1-SD4` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S2-SD0` | CTXDDQN_ANCHOR/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S2-SD0` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S2-SD1` | CTXDDQN_ANCHOR/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S2-SD1` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S2-SD2` | CTXDDQN_ANCHOR/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S2-SD2` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S2-SD3` | CTXDDQN_ANCHOR/sumohz1x1/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S2-SD3` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S2-SD4` | CTXDDQN_ANCHOR/sumohz1x1/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S2-SD4` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S3-SD0` | CTXDDQN_ANCHOR/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S3-SD0` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S3-SD1` | CTXDDQN_ANCHOR/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S3-SD1` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S3-SD2` | CTXDDQN_ANCHOR/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S3-SD2` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S3-SD3` | CTXDDQN_ANCHOR/sumohz1x1_config4/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S3-SD3` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S3-SD4` | CTXDDQN_ANCHOR/sumohz1x1_config4/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S3-SD4` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S4-SD0` | CTXDDQN_ANCHOR/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S4-SD0` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S4-SD1` | CTXDDQN_ANCHOR/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S4-SD1` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S4-SD2` | CTXDDQN_ANCHOR/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S4-SD2` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S4-SD3` | CTXDDQN_ANCHOR/sumohz1x1_config3/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S4-SD3` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-CTXDDQN-S4-SD4` | CTXDDQN_ANCHOR/sumohz1x1_config3/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-CTXDDQN-S4-SD4` | attempts=1; type=ANCHOR |
| `P5-ANCHOR-DDQN-S1-SD0` | DDQN_ANCHOR/sumohz1x1_config2/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S1-SD0` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S1-SD1` | DDQN_ANCHOR/sumohz1x1_config2/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S1-SD1` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S1-SD2` | DDQN_ANCHOR/sumohz1x1_config2/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S1-SD2` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S1-SD3` | DDQN_ANCHOR/sumohz1x1_config2/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S1-SD3` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S1-SD4` | DDQN_ANCHOR/sumohz1x1_config2/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S1-SD4` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S2-SD0` | DDQN_ANCHOR/sumohz1x1/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S2-SD0` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S2-SD1` | DDQN_ANCHOR/sumohz1x1/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S2-SD1` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S2-SD2` | DDQN_ANCHOR/sumohz1x1/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S2-SD2` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S2-SD3` | DDQN_ANCHOR/sumohz1x1/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S2-SD3` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S2-SD4` | DDQN_ANCHOR/sumohz1x1/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S2-SD4` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S3-SD0` | DDQN_ANCHOR/sumohz1x1_config4/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S3-SD0` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S3-SD1` | DDQN_ANCHOR/sumohz1x1_config4/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S3-SD1` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S3-SD2` | DDQN_ANCHOR/sumohz1x1_config4/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S3-SD2` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S3-SD3` | DDQN_ANCHOR/sumohz1x1_config4/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S3-SD3` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S3-SD4` | DDQN_ANCHOR/sumohz1x1_config4/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S3-SD4` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S4-SD0` | DDQN_ANCHOR/sumohz1x1_config3/s0 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S4-SD0` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S4-SD1` | DDQN_ANCHOR/sumohz1x1_config3/s1 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S4-SD1` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S4-SD2` | DDQN_ANCHOR/sumohz1x1_config3/s2 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S4-SD2` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S4-SD3` | DDQN_ANCHOR/sumohz1x1_config3/s3 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S4-SD3` | attempts=3; type=ANCHOR |
| `P5-ANCHOR-DDQN-S4-SD4` | DDQN_ANCHOR/sumohz1x1_config3/s4 | 100 | DONE | — | — | `data/output_data/cross_algorithm/plan5_b100/runs/P5-ANCHOR-DDQN-S4-SD4` | attempts=3; type=ANCHOR |
