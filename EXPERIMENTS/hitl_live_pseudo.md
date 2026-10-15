# hitl_live_pseudo

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：—
- plan_id/队列：—
- 设计稿/证据位置：—
- 结论：—
- 登记单元 11（train/eval/smoke/batch/calibration = 10/0/0/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `HITL_APISW_LIVECOLLECT` | APISW/sumohz1x1/s-1 | 0 | DONE | — | — | `data/output_data/hitl_probe/weak_student` | 弱学生(ep0快照)live采集10会话: tt=127.8/thr=168.7/unfin=158.8,打包pools/student_weak_ep0+de |
| `HITL_APISD-SD0` | APISD/sumohz1x1/s0 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_APISD-SD0` | 伪专家对照臂: 学生自产转移做demo双预训(1213条); 结果≈专家示范APIQB,证明A通道收益来自"离线转移联合预训"机制而非专家质量; 专家数据的独特 |
| `HITL_APISD-SD1` | APISD/sumohz1x1/s1 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_APISD-SD1` | 伪专家对照臂: 学生自产转移做demo双预训(1213条); 结果≈专家示范APIQB,证明A通道收益来自"离线转移联合预训"机制而非专家质量; 专家数据的独特 |
| `HITL_APISD-SD2` | APISD/sumohz1x1/s2 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_APISD-SD2` | 伪专家对照臂: 学生自产转移做demo双预训(1213条); 结果≈专家示范APIQB,证明A通道收益来自"离线转移联合预训"机制而非专家质量; 专家数据的独特 |
| `HITL_APISD-SD3` | APISD/sumohz1x1/s3 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_APISD-SD3` | 伪专家对照臂: 学生自产转移做demo双预训(1213条); 结果≈专家示范APIQB,证明A通道收益来自"离线转移联合预训"机制而非专家质量; 专家数据的独特 |
| `HITL_APISD-SD4` | APISD/sumohz1x1/s4 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_APISD-SD4` | 伪专家对照臂: 学生自产转移做demo双预训(1213条); 结果≈专家示范APIQB,证明A通道收益来自"离线转移联合预训"机制而非专家质量; 专家数据的独特 |
| `HITL_APISW-SD0` | APISW/sumohz1x1/s0 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_APISW-SD0` | 坏源臂: 未训练(ep0)学生自产轨迹双预训(1217条); 显著有害——管线忠实注入源质量,外部人源价值在源不可自产时(冷启动)显式成立 |
| `HITL_APISW-SD1` | APISW/sumohz1x1/s1 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_APISW-SD1` | 坏源臂: 未训练(ep0)学生自产轨迹双预训(1217条); 显著有害——管线忠实注入源质量,外部人源价值在源不可自产时(冷启动)显式成立 |
| `HITL_APISW-SD2` | APISW/sumohz1x1/s2 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_APISW-SD2` | 坏源臂: 未训练(ep0)学生自产轨迹双预训(1217条); 显著有害——管线忠实注入源质量,外部人源价值在源不可自产时(冷启动)显式成立 |
| `HITL_APISW-SD3` | APISW/sumohz1x1/s3 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_APISW-SD3` | 坏源臂: 未训练(ep0)学生自产轨迹双预训(1217条); 显著有害——管线忠实注入源质量,外部人源价值在源不可自产时(冷启动)显式成立 |
| `HITL_APISW-SD4` | APISW/sumohz1x1/s4 | 20 | DONE | — | — | `data/output_data/hitl_probe/pilot5/runs/HITL_APISW-SD4` | 坏源臂: 未训练(ep0)学生自产轨迹双预训(1217条); 显著有害——管线忠实注入源质量,外部人源价值在源不可自产时(冷启动)显式成立 |
