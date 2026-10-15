# hitl_live_gate

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：—
- plan_id/队列：—
- 设计稿/证据位置：—
- 结论：—
- 登记单元 30（train/eval/smoke/batch/calibration = 25/0/0/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `HITL_GATE_MARGIN_LIVECOLLECT` | GATE_MARGIN/sumohz1x1/s-1 | 0 | DONE | — | — | `data/output_data/hitl_probe/gate_modes/margin` | gate消融live采集: margin模式×10seed配对会话+打包池pools/hitl_gate_margin |
| `HITL_GATE_ON_DEMAND_LIVECOLLECT` | GATE_ON_DEMAND/sumohz1x1/s-1 | 0 | DONE | — | — | `data/output_data/hitl_probe/gate_modes/on_demand` | gate消融live采集: on_demand模式×10seed配对会话+打包池pools/hitl_gate_on_demand |
| `HITL_GATE_PERIODIC_LIVECOLLECT` | GATE_PERIODIC/sumohz1x1/s-1 | 0 | DONE | — | — | `data/output_data/hitl_probe/gate_modes/periodic` | gate消融live采集: periodic模式×10seed配对会话+打包池pools/hitl_gate_periodic |
| `HITL_GATE_RANDOM_LIVECOLLECT` | GATE_RANDOM/sumohz1x1/s-1 | 0 | DONE | — | — | `data/output_data/hitl_probe/gate_modes/random` | gate消融live采集: random模式×10seed配对会话+打包池pools/hitl_gate_random |
| `HITL_GATE_WINDOW_LIVECOLLECT` | GATE_WINDOW/sumohz1x1/s-1 | 0 | DONE | — | — | `data/output_data/hitl_probe/gate_modes/window` | gate消融live采集: window模式×10seed配对会话+打包池pools/hitl_gate_window |
| `HITL_GATE_MARGIN-SD0` | GATE_MARGIN/sumohz1x1/s0 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_MARGIN-SD0` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_MARGIN-SD1` | GATE_MARGIN/sumohz1x1/s1 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_MARGIN-SD1` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_MARGIN-SD2` | GATE_MARGIN/sumohz1x1/s2 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_MARGIN-SD2` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_MARGIN-SD3` | GATE_MARGIN/sumohz1x1/s3 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_MARGIN-SD3` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_MARGIN-SD4` | GATE_MARGIN/sumohz1x1/s4 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_MARGIN-SD4` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_OD-SD0` | GATE_OD/sumohz1x1/s0 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_OD-SD0` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_OD-SD1` | GATE_OD/sumohz1x1/s1 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_OD-SD1` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_OD-SD2` | GATE_OD/sumohz1x1/s2 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_OD-SD2` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_OD-SD3` | GATE_OD/sumohz1x1/s3 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_OD-SD3` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_OD-SD4` | GATE_OD/sumohz1x1/s4 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_OD-SD4` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_PERIOD-SD0` | GATE_PERIOD/sumohz1x1/s0 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_PERIOD-SD0` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_PERIOD-SD1` | GATE_PERIOD/sumohz1x1/s1 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_PERIOD-SD1` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_PERIOD-SD2` | GATE_PERIOD/sumohz1x1/s2 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_PERIOD-SD2` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_PERIOD-SD3` | GATE_PERIOD/sumohz1x1/s3 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_PERIOD-SD3` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_PERIOD-SD4` | GATE_PERIOD/sumohz1x1/s4 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_PERIOD-SD4` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_RAND-SD0` | GATE_RAND/sumohz1x1/s0 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_RAND-SD0` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_RAND-SD1` | GATE_RAND/sumohz1x1/s1 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_RAND-SD1` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_RAND-SD2` | GATE_RAND/sumohz1x1/s2 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_RAND-SD2` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_RAND-SD3` | GATE_RAND/sumohz1x1/s3 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_RAND-SD3` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_RAND-SD4` | GATE_RAND/sumohz1x1/s4 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_RAND-SD4` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_WIN-SD0` | GATE_WIN/sumohz1x1/s0 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_WIN-SD0` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_WIN-SD1` | GATE_WIN/sumohz1x1/s1 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_WIN-SD1` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_WIN-SD2` | GATE_WIN/sumohz1x1/s2 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_WIN-SD2` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_WIN-SD3` | GATE_WIN/sumohz1x1/s3 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_WIN-SD3` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
| `HITL_GATE_WIN-SD4` | GATE_WIN/sumohz1x1/s4 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_GATE_WIN-SD4` | gate触发策略消融: 五模式同seed同规模纠偏池(r=0.25), 唯一变量=触发策略; margin唯一全指标无显著伤害 |
