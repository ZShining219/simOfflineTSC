# seq8_demand

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：—
- plan_id/队列：—
- 设计稿/证据位置：—
- 结论：—
- 登记单元 42（train/eval/smoke/batch/calibration = 36/0/0/6/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `SEQ8-DRIFT-RESOURCE-GATE/seq8_drift_formal_20261015` | evidence/—/s— | — | DONE | — | — | `data/output_data/sequential/seq8_drift_formal_20261015/resource_gate.json` | 派发资源门禁：slots=4 valid=True |
| `SEQ8-DRIFT-SHARED-EVAL/seq8_drift_formal_20261015` | evidence/—/s— | — | DONE | — | — | `data/output_data/sequential/seq8_drift_formal_20261015/_shared_evaluation` | seq8_drift_formal_20261015 冻结评估去重存储：6550 aliases / 6549 physical cells |
| `SEQ8-NOREP-RESOURCE-GATE/seq8_norep_formal_20261015` | evidence/—/s— | — | DONE | — | — | `data/output_data/sequential/seq8_norep_formal_20261015/resource_gate.json` | 派发资源门禁：slots=4 valid=True |
| `SEQ8-NOREP-SHARED-EVAL/seq8_norep_formal_20261015` | evidence/—/s— | — | DONE | — | — | `data/output_data/sequential/seq8_norep_formal_20261015/_shared_evaluation` | seq8_norep_formal_20261015 冻结评估去重存储：6393 aliases / 6389 physical cells |
| `SEQ8-REP-RESOURCE-GATE/seq8_rep_formal_20261015` | evidence/—/s— | — | DONE | — | — | `data/output_data/sequential/seq8_rep_formal_20261015/resource_gate.json` | 派发资源门禁：slots=4 valid=True |
| `SEQ8-REP-SHARED-EVAL/seq8_rep_formal_20261015` | evidence/—/s— | — | DONE | — | — | `data/output_data/sequential/seq8_rep_formal_20261015/_shared_evaluation` | seq8_rep_formal_20261015 冻结评估去重存储：6434 aliases / 6402 physical cells |
| `CONT-D1-SD0` | CONT-D1/hz1x1:st_dom>base>lt_high>lt_extreme>lt_wsing>asym_wonly>asym_we>asym_ns/s0 | 800 | DONE | 2026-10-15 | 75.7/2195/47 | `data/output_data/sequential/seq8_drift_formal_20261015/CONT-D1-SD0` | attempts=1; stage=formal; 冻结评估cell×835（去重存储于_shared_evaluation） |
| `CONT-D1-SD1` | CONT-D1/hz1x1:st_dom>base>lt_high>lt_extreme>lt_wsing>asym_wonly>asym_we>asym_ns/s1 | 800 | DONE | 2026-10-15 | 75.4/2186/56 | `data/output_data/sequential/seq8_drift_formal_20261015/CONT-D1-SD1` | attempts=1; stage=formal; 冻结评估cell×835（去重存储于_shared_evaluation） |
| `CONT-D1-SD2` | CONT-D1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_drift_formal_20261015/CONT-D1-SD2` | attempts=2（有效attempt_2） |
| `CONT-D1-SD3` | CONT-D1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_drift_formal_20261015/CONT-D1-SD3` | attempts=2（有效attempt_2） |
| `CONT-N1-SD0` | CONT-N1/hz1x1:sparse_ultra>fluct_burst>lt_extreme>asym_wonly>dense_extreme>st_dom>asym_we>fluct_ramp/s0 | 800 | DONE | 2026-10-15 | 86.9/1782/93 | `data/output_data/sequential/seq8_norep_formal_20261015/CONT-N1-SD0` | attempts=1; stage=formal; 冻结评估cell×835（去重存储于_shared_evaluation） |
| `CONT-N1-SD1` | CONT-N1/hz1x1:sparse_ultra>fluct_burst>lt_extreme>asym_wonly>dense_extreme>st_dom>asym_we>fluct_ramp/s1 | 800 | DONE | 2026-10-15 | 89.9/1915/106 | `data/output_data/sequential/seq8_norep_formal_20261015/CONT-N1-SD1` | attempts=1; stage=formal; 冻结评估cell×835（去重存储于_shared_evaluation） |
| `CONT-N1-SD2` | CONT-N1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_norep_formal_20261015/CONT-N1-SD2` | attempts=2（有效attempt_2） |
| `CONT-N1-SD3` | CONT-N1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_norep_formal_20261015/CONT-N1-SD3` | attempts=2（有效attempt_2） |
| `CONT-R1-SD0` | CONT-R1/hz1x1:sparse_low>lt_high>fluct_burst>dense_high>sparse_low>lt_high>fluct_burst>dense_high/s0 | 800 | DONE | 2026-10-15 | 111.4/2622/114 | `data/output_data/sequential/seq8_rep_formal_20261015/CONT-R1-SD0` | attempts=1; stage=formal; 冻结评估cell×825（去重存储于_shared_evaluation） |
| `CONT-R1-SD1` | CONT-R1/hz1x1:sparse_low>lt_high>fluct_burst>dense_high>sparse_low>lt_high>fluct_burst>dense_high/s1 | 800 | DONE | 2026-10-15 | 113.1/2590/108 | `data/output_data/sequential/seq8_rep_formal_20261015/CONT-R1-SD1` | attempts=1; stage=formal; 冻结评估cell×825（去重存储于_shared_evaluation） |
| `CONT-R1-SD2` | CONT-R1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_rep_formal_20261015/CONT-R1-SD2` | attempts=2（有效attempt_2） |
| `CONT-R1-SD3` | CONT-R1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_rep_formal_20261015/CONT-R1-SD3` | attempts=2（有效attempt_2） |
| `SELF-CQ-R50-D1-SD0` | SELF-CQ-R50-D1/hz1x1:st_dom>base>lt_high>lt_extreme>lt_wsing>asym_wonly>asym_we>asym_ns/s0 | 800 | DONE | 2026-10-15 | 75.9/2194/49 | `data/output_data/sequential/seq8_drift_formal_20261015/SELF-CQ-R50-D1-SD0` | attempts=1; stage=formal; 冻结评估cell×835（去重存储于_shared_evaluation）; SELF自产档案 method |
| `SELF-CQ-R50-D1-SD1` | SELF-CQ-R50-D1/hz1x1:st_dom>base>lt_high>lt_extreme>lt_wsing>asym_wonly>asym_we>asym_ns/s1 | 800 | DONE | 2026-10-15 | 77.0/2194/48 | `data/output_data/sequential/seq8_drift_formal_20261015/SELF-CQ-R50-D1-SD1` | attempts=1; stage=formal; 冻结评估cell×835（去重存储于_shared_evaluation）; SELF自产档案 method |
| `SELF-CQ-R50-D1-SD2` | SELF-CQ-R50-D1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_drift_formal_20261015/SELF-CQ-R50-D1-SD2` | attempts=1 |
| `SELF-CQ-R50-D1-SD3` | SELF-CQ-R50-D1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_drift_formal_20261015/SELF-CQ-R50-D1-SD3` | attempts=1 |
| `SELF-CQ-R50-N1-SD0` | SELF-CQ-R50-N1/hz1x1:sparse_ultra>fluct_burst>lt_extreme>asym_wonly>dense_extreme>st_dom>asym_we>fluct_ramp/s0 | 800 | DONE | 2026-10-15 | 92.0/1857/106 | `data/output_data/sequential/seq8_norep_formal_20261015/SELF-CQ-R50-N1-SD0` | attempts=1; stage=formal; 冻结评估cell×835（去重存储于_shared_evaluation）; SELF自产档案 method |
| `SELF-CQ-R50-N1-SD1` | SELF-CQ-R50-N1/hz1x1:sparse_ultra>fluct_burst>lt_extreme>asym_wonly>dense_extreme>st_dom>asym_we>fluct_ramp/s1 | 800 | DONE | 2026-10-15 | 90.7/1869/91 | `data/output_data/sequential/seq8_norep_formal_20261015/SELF-CQ-R50-N1-SD1` | attempts=1; stage=formal; 冻结评估cell×835（去重存储于_shared_evaluation）; SELF自产档案 method |
| `SELF-CQ-R50-N1-SD2` | SELF-CQ-R50-N1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_norep_formal_20261015/SELF-CQ-R50-N1-SD2` | attempts=1 |
| `SELF-CQ-R50-N1-SD3` | SELF-CQ-R50-N1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_norep_formal_20261015/SELF-CQ-R50-N1-SD3` | attempts=1 |
| `SELF-CQ-R50-R1-SD0` | SELF-CQ-R50-R1/hz1x1:sparse_low>lt_high>fluct_burst>dense_high>sparse_low>lt_high>fluct_burst>dense_high/s0 | 800 | DONE | 2026-10-15 | 121.1/2579/112 | `data/output_data/sequential/seq8_rep_formal_20261015/SELF-CQ-R50-R1-SD0` | attempts=1; stage=formal; 冻结评估cell×825（去重存储于_shared_evaluation）; SELF自产档案 method |
| `SELF-CQ-R50-R1-SD1` | SELF-CQ-R50-R1/hz1x1:sparse_low>lt_high>fluct_burst>dense_high>sparse_low>lt_high>fluct_burst>dense_high/s1 | 800 | DONE | 2026-10-15 | 111.8/2624/117 | `data/output_data/sequential/seq8_rep_formal_20261015/SELF-CQ-R50-R1-SD1` | attempts=1; stage=formal; 冻结评估cell×825（去重存储于_shared_evaluation）; SELF自产档案 method |
| `SELF-CQ-R50-R1-SD2` | SELF-CQ-R50-R1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_rep_formal_20261015/SELF-CQ-R50-R1-SD2` | attempts=1 |
| `SELF-CQ-R50-R1-SD3` | SELF-CQ-R50-R1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_rep_formal_20261015/SELF-CQ-R50-R1-SD3` | attempts=1 |
| `SELF-DHOA-R50-D1-SD0` | SELF-DHOA-R50-D1/hz1x1:st_dom>base>lt_high>lt_extreme>lt_wsing>asym_wonly>asym_we>asym_ns/s0 | 800 | DONE | 2026-10-15 | 75.9/2195/47 | `data/output_data/sequential/seq8_drift_formal_20261015/SELF-DHOA-R50-D1-SD0` | attempts=1; stage=formal; 冻结评估cell×835（去重存储于_shared_evaluation）; SELF自产档案 method |
| `SELF-DHOA-R50-D1-SD1` | SELF-DHOA-R50-D1/hz1x1:st_dom>base>lt_high>lt_extreme>lt_wsing>asym_wonly>asym_we>asym_ns/s1 | 800 | DONE | 2026-10-15 | 75.3/2192/50 | `data/output_data/sequential/seq8_drift_formal_20261015/SELF-DHOA-R50-D1-SD1` | attempts=1; stage=formal; 冻结评估cell×835（去重存储于_shared_evaluation）; SELF自产档案 method |
| `SELF-DHOA-R50-D1-SD2` | SELF-DHOA-R50-D1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_drift_formal_20261015/SELF-DHOA-R50-D1-SD2` | attempts=2（有效attempt_2） |
| `SELF-DHOA-R50-D1-SD3` | SELF-DHOA-R50-D1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_drift_formal_20261015/SELF-DHOA-R50-D1-SD3` | attempts=2（有效attempt_2） |
| `SELF-DHOA-R50-N1-SD0` | SELF-DHOA-R50-N1/hz1x1:sparse_ultra>fluct_burst>lt_extreme>asym_wonly>dense_extreme>st_dom>asym_we>fluct_ramp/s0 | 800 | DONE | 2026-10-15 | 90.1/1857/109 | `data/output_data/sequential/seq8_norep_formal_20261015/SELF-DHOA-R50-N1-SD0` | attempts=1; stage=formal; 冻结评估cell×835（去重存储于_shared_evaluation）; SELF自产档案 method |
| `SELF-DHOA-R50-N1-SD1` | SELF-DHOA-R50-N1/hz1x1:sparse_ultra>fluct_burst>lt_extreme>asym_wonly>dense_extreme>st_dom>asym_we>fluct_ramp/s1 | 800 | DONE | 2026-10-15 | 91.5/1909/106 | `data/output_data/sequential/seq8_norep_formal_20261015/SELF-DHOA-R50-N1-SD1` | attempts=1; stage=formal; 冻结评估cell×835（去重存储于_shared_evaluation）; SELF自产档案 method |
| `SELF-DHOA-R50-N1-SD2` | SELF-DHOA-R50-N1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_norep_formal_20261015/SELF-DHOA-R50-N1-SD2` | attempts=2（有效attempt_2） |
| `SELF-DHOA-R50-N1-SD3` | SELF-DHOA-R50-N1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_norep_formal_20261015/SELF-DHOA-R50-N1-SD3` | attempts=2（有效attempt_2） |
| `SELF-DHOA-R50-R1-SD0` | SELF-DHOA-R50-R1/hz1x1:sparse_low>lt_high>fluct_burst>dense_high>sparse_low>lt_high>fluct_burst>dense_high/s0 | 800 | DONE | 2026-10-15 | 121.2/2523/107 | `data/output_data/sequential/seq8_rep_formal_20261015/SELF-DHOA-R50-R1-SD0` | attempts=1; stage=formal; 冻结评估cell×825（去重存储于_shared_evaluation）; SELF自产档案 method |
| `SELF-DHOA-R50-R1-SD1` | SELF-DHOA-R50-R1/hz1x1:sparse_low>lt_high>fluct_burst>dense_high>sparse_low>lt_high>fluct_burst>dense_high/s1 | 800 | DONE | 2026-10-15 | 111.7/2557/107 | `data/output_data/sequential/seq8_rep_formal_20261015/SELF-DHOA-R50-R1-SD1` | attempts=1; stage=formal; 冻结评估cell×825（去重存储于_shared_evaluation）; SELF自产档案 method |
| `SELF-DHOA-R50-R1-SD2` | SELF-DHOA-R50-R1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_rep_formal_20261015/SELF-DHOA-R50-R1-SD2` | attempts=2（有效attempt_2） |
| `SELF-DHOA-R50-R1-SD3` | SELF-DHOA-R50-R1/hz1x1/s— | — | SCOPE_DISCARD | 2026-10-15 | — | `data/output_data/sequential/seq8_rep_formal_20261015/SELF-DHOA-R50-R1-SD3` | attempts=2（有效attempt_2） |
