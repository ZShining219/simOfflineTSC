# hitl_warmloop

> 索引：[EXPERIMENTS.md](../EXPERIMENTS.md)；事实源 `ledger/runs.jsonl`

- 假设：—
- plan_id/队列：—
- 设计稿/证据位置：—
- 结论：—
- 登记单元 28（train/eval/smoke/batch/calibration = 25/0/0/0/0）（追溯登记 2026-10-07）

| run_id | 臂/net/seed | ep | 状态 | 起始 | TT/th/unfinished | 产物路径 | 备注 |
|---|---|---|---|---|---|---|---|
| `HITL_WARMLOOP-A2-DESIGN` | NOTE/sumohz1x1/s-1 | 0 | DONE | 2026-10-14 | — | `—` | A2设计:通道串接C->A同血脉;采集A1控盘介入8会话/seed(gate_qg=3补偿强学生低命中);demo_pretrain只取hitl_expert_ |
| `HITL_WARMLOOP-A2-V1FIND` | NOTE/sumohz1x1/s-1 | 0 | REGISTERED | 2026-10-14 | — | `data/output_data/hitl_probe/warmloop/p2/runs_v1` | A2预声明决策: WA2 v1(critic=true)ep2起塌/sw0/thr28, v2(actor-only)ep12起塌/sw0/thr285; 两版 |
| `HITL_WARMLOOP-DESIGN` | NOTE/sumohz1x1/s-1 | 0 | REGISTERED | 2026-10-14 | — | `—` | 实验设计:已有模型M0(20ep未收敛)→probe student冻结基线→同档hitl臂MP介入采集→per-seed池→warm-init续训(exper |
| `WA2-SD0` | WA2/sumohz1x1/s0 | 20 | DONE | 2026-10-14 | 109.6 (vs A0=95.5/A1=100.9/MP=83.8)/522 (vs A0=425/A1=493/MP=644)/— | `data/output_data/hitl_probe/warmloop/p2/runs/WA2-SD0` | warmloop A2臂:M1暖启动+DemoAC示范预训练(actor+critic)续训20ep产A2;判据=queue轨迹贴近MP程度(vs A0/A1/ |
| `WA2-SD1` | WA2/sumohz1x1/s1 | 20 | DONE | 2026-10-14 | 109.6 (vs A0=95.5/A1=100.9/MP=83.8)/522 (vs A0=425/A1=493/MP=644)/— | `data/output_data/hitl_probe/warmloop/p2/runs/WA2-SD1` | warmloop A2臂:M1暖启动+DemoAC示范预训练(actor+critic)续训20ep产A2;判据=queue轨迹贴近MP程度(vs A0/A1/ |
| `WA2-SD2` | WA2/sumohz1x1/s2 | 20 | DONE | 2026-10-14 | 109.6 (vs A0=95.5/A1=100.9/MP=83.8)/522 (vs A0=425/A1=493/MP=644)/— | `data/output_data/hitl_probe/warmloop/p2/runs/WA2-SD2` | warmloop A2臂:M1暖启动+DemoAC示范预训练(actor+critic)续训20ep产A2;判据=queue轨迹贴近MP程度(vs A0/A1/ |
| `WA2-SD3` | WA2/sumohz1x1/s3 | 20 | DONE | 2026-10-14 | 109.6 (vs A0=95.5/A1=100.9/MP=83.8)/522 (vs A0=425/A1=493/MP=644)/— | `data/output_data/hitl_probe/warmloop/p2/runs/WA2-SD3` | warmloop A2臂:M1暖启动+DemoAC示范预训练(actor+critic)续训20ep产A2;判据=queue轨迹贴近MP程度(vs A0/A1/ |
| `WA2-SD4` | WA2/sumohz1x1/s4 | 20 | DONE | 2026-10-14 | 109.6 (vs A0=95.5/A1=100.9/MP=83.8)/522 (vs A0=425/A1=493/MP=644)/— | `data/output_data/hitl_probe/warmloop/p2/runs/WA2-SD4` | warmloop A2臂:M1暖启动+DemoAC示范预训练(actor+critic)续训20ep产A2;判据=queue轨迹贴近MP程度(vs A0/A1/ |
| `WA3-SD0` | WA3/sumohz1x1/s0 | 30 | DONE | 2026-10-14 | — | `data/output_data/hitl_probe/warmloop/p3/runs/WA3-SD0` | A3通道回转:C->A->C串接;A2 actor权重经weights_from拼接warm-init;判据=queue终末贴近MP(9)程度 ／ 场内达MP量 |
| `WA3-SD1` | WA3/sumohz1x1/s1 | 30 | DONE | 2026-10-14 | — | `data/output_data/hitl_probe/warmloop/p3/runs/WA3-SD1` | A3通道回转:C->A->C串接;A2 actor权重经weights_from拼接warm-init;判据=queue终末贴近MP(9)程度 ／ 场内达MP量 |
| `WA3-SD2` | WA3/sumohz1x1/s2 | 30 | DONE | 2026-10-14 | — | `data/output_data/hitl_probe/warmloop/p3/runs/WA3-SD2` | A3通道回转:C->A->C串接;A2 actor权重经weights_from拼接warm-init;判据=queue终末贴近MP(9)程度 ／ 场内达MP量 |
| `WA3-SD3` | WA3/sumohz1x1/s3 | 30 | DONE | 2026-10-14 | — | `data/output_data/hitl_probe/warmloop/p3/runs/WA3-SD3` | A3通道回转:C->A->C串接;A2 actor权重经weights_from拼接warm-init;判据=queue终末贴近MP(9)程度 ／ 场内达MP量 |
| `WA3-SD4` | WA3/sumohz1x1/s4 | 30 | DONE | 2026-10-14 | — | `data/output_data/hitl_probe/warmloop/p3/runs/WA3-SD4` | A3通道回转:C->A->C串接;A2 actor权重经weights_from拼接warm-init;判据=queue终末贴近MP(9)程度 ／ 场内达MP量 |
| `WA4-SD0` | WA4/sumohz1x1/s0 | 1 | DONE | 2026-10-14 | 86.2 (vs MP=83.8)/636 (vs MP=644)/— | `data/output_data/hitl_probe/warmloop/p4/runs/WA4-SD0` | A4通道回转:C->A->C->A;A3控盘surge介入采集(takeover目标);stage_episodes=[1]对齐A2口径(ep1=pretrai |
| `WA4-SD1` | WA4/sumohz1x1/s1 | 1 | DONE | 2026-10-14 | 86.2 (vs MP=83.8)/636 (vs MP=644)/— | `data/output_data/hitl_probe/warmloop/p4/runs/WA4-SD1` | A4通道回转:C->A->C->A;A3控盘surge介入采集(takeover目标);stage_episodes=[1]对齐A2口径(ep1=pretrai |
| `WA4-SD2` | WA4/sumohz1x1/s2 | 1 | DONE | 2026-10-14 | 86.2 (vs MP=83.8)/636 (vs MP=644)/— | `data/output_data/hitl_probe/warmloop/p4/runs/WA4-SD2` | A4通道回转:C->A->C->A;A3控盘surge介入采集(takeover目标);stage_episodes=[1]对齐A2口径(ep1=pretrai |
| `WA4-SD3` | WA4/sumohz1x1/s3 | 1 | DONE | 2026-10-14 | 86.2 (vs MP=83.8)/636 (vs MP=644)/— | `data/output_data/hitl_probe/warmloop/p4/runs/WA4-SD3` | A4通道回转:C->A->C->A;A3控盘surge介入采集(takeover目标);stage_episodes=[1]对齐A2口径(ep1=pretrai |
| `WA4-SD4` | WA4/sumohz1x1/s4 | 1 | DONE | 2026-10-14 | 86.2 (vs MP=83.8)/636 (vs MP=644)/— | `data/output_data/hitl_probe/warmloop/p4/runs/WA4-SD4` | A4通道回转:C->A->C->A;A3控盘surge介入采集(takeover目标);stage_episodes=[1]对齐A2口径(ep1=pretrai |
| `WBASE-SD0` | WBASE/sumohz1x1/s0 | 20 | DONE | 2026-10-14 | — | `data/output_data/hitl_probe/warmloop/p0/runs/WBASE-SD0` | warmloop验证链Phase0:ep0短程20ep产未收敛M0(续训起点) |
| `WBASE-SD1` | WBASE/sumohz1x1/s1 | 20 | DONE | 2026-10-14 | — | `data/output_data/hitl_probe/warmloop/p0/runs/WBASE-SD1` | warmloop验证链Phase0:ep0短程20ep产未收敛M0(续训起点) |
| `WBASE-SD2` | WBASE/sumohz1x1/s2 | 20 | DONE | 2026-10-14 | — | `data/output_data/hitl_probe/warmloop/p0/runs/WBASE-SD2` | warmloop验证链Phase0:ep0短程20ep产未收敛M0(续训起点) |
| `WBASE-SD3` | WBASE/sumohz1x1/s3 | 20 | DONE | 2026-10-14 | — | `data/output_data/hitl_probe/warmloop/p0/runs/WBASE-SD3` | warmloop验证链Phase0:ep0短程20ep产未收敛M0(续训起点) |
| `WBASE-SD4` | WBASE/sumohz1x1/s4 | 20 | DONE | 2026-10-14 | — | `data/output_data/hitl_probe/warmloop/p0/runs/WBASE-SD4` | warmloop验证链Phase0:ep0短程20ep产未收敛M0(续训起点) |
| `WEXP-SD0` | WEXP/sumohz1x1/s0 | 20 | DONE | 2026-10-14 | +5.41 CI[-7.45,+14.13]/+68.0 CI[-5.76,+145.88] p≈0.04/— | `data/output_data/hitl_probe/warmloop/p1/runs/WEXP-SD0` | warmloop验证链Phase3:M0暖启动+per-seed介入池expert_ratio=0.15续训20ep产M1;判据=M1-M0配对Δ；expert |
| `WEXP-SD1` | WEXP/sumohz1x1/s1 | 20 | DONE | 2026-10-14 | +5.41 CI[-7.45,+14.13]/+68.0 CI[-5.76,+145.88] p≈0.04/— | `data/output_data/hitl_probe/warmloop/p1/runs/WEXP-SD1` | warmloop验证链Phase3:M0暖启动+per-seed介入池expert_ratio=0.15续训20ep产M1;判据=M1-M0配对Δ；expert |
| `WEXP-SD2` | WEXP/sumohz1x1/s2 | 20 | DONE | 2026-10-14 | +5.41 CI[-7.45,+14.13]/+68.0 CI[-5.76,+145.88] p≈0.04/— | `data/output_data/hitl_probe/warmloop/p1/runs/WEXP-SD2` | warmloop验证链Phase3:M0暖启动+per-seed介入池expert_ratio=0.15续训20ep产M1;判据=M1-M0配对Δ；expert |
| `WEXP-SD3` | WEXP/sumohz1x1/s3 | 20 | DONE | 2026-10-14 | +5.41 CI[-7.45,+14.13]/+68.0 CI[-5.76,+145.88] p≈0.04/— | `data/output_data/hitl_probe/warmloop/p1/runs/WEXP-SD3` | warmloop验证链Phase3:M0暖启动+per-seed介入池expert_ratio=0.15续训20ep产M1;判据=M1-M0配对Δ；expert |
| `WEXP-SD4` | WEXP/sumohz1x1/s4 | 20 | DONE | 2026-10-14 | +5.41 CI[-7.45,+14.13]/+68.0 CI[-5.76,+145.88] p≈0.04/— | `data/output_data/hitl_probe/warmloop/p1/runs/WEXP-SD4` | warmloop验证链Phase3:M0暖启动+per-seed介入池expert_ratio=0.15续训20ep产M1;判据=M1-M0配对Δ；expert |
