# semi_offline/ — 半离线 / 持续学习研究线

原先分散在仓根 `sequential/`、`plan5/`、`arterial/` 的半离线相关代码，2026-11-12 收拢到这里。

| 位置 | 内容 | 状态 | 台账 campaign |
|---|---|---|---|
| （见下）cl_hz1x1 当前线 | 自产历史归档 + 混合采样 + 教师锚定（经 `run.py`） | ✅ 当前主线 | `cl_hz1x1`、`seq8_anchor` |
| `offline/` | 纯离线训练（Plan 2 batch_dqn / cql_dqn）：入口 `cli.py`（`python -m semi_offline.offline`）及智能体、trainer、task、数据集实现（原在框架层） | 在用（离线基线） | `hz1x1_offline_v1`（`xiasha1_offline_v1/v2` 已作废） |
| `sequential/` | Plan 3/4 Sequential DQN、HA-SODQN、seq8 与 HITL 通道训练，`python -m semi_offline.sequential` | ❄️ 冻结，仅复跑底稿 | `ha_sodqn`、`seq8_demand`、HITL 通道训练 |
| `plan5/` | 8 月跨算法半离线实验（自带一套基座副本），`python -m semi_offline.plan5` | ❄️ 已结束 | `plan5_b100` |
| `arterial/` | arterial 1×6 走廊的场景定义、评估矩阵与入口（`python -m semi_offline.arterial`，原仓根 `arterial_run.py`；通用回放已迁入 `common/`） | ❄️ 已结束 | `arterial_1x6` |

## 当前主线 cl_hz1x1 的代码在哪

cl_hz1x1 复用框架层的 DQN/FRAP，没有独立的包：

- 机制：`agent/dqn.py`、`agent/frap.py` 的 `_anchor_loss`（教师锚定）；`common/history_replay.py`（历史归档与 `mixed_batch` 采样）
- 派发：`tools/run_experiment_queue.py` + `dev/cl_hz1x1/`（队列生成、收割、出图）；overlay 在 `artifacts/cl_hz1x1/overlays/`
- 回归检查：`bench/golden/cl_hz1x1_two_stage.json`（`python -m lab golden`）
- 设计稿：`docs/plan_qanchor_clhz1x1.md`；实验记录与结论：`EXPERIMENTS/cl_hz1x1.md`

## 配置在哪

配置目录沿用历史名称（被代码、历史队列与台账引用，不改名）：`configs/offline_tsc/`（纯离线）、`configs/sequential/`（sequential/HA-SODQN）、`configs/arterial/`（arterial）；cl_hz1x1 的 overlay 由 `dev/cl_hz1x1/` 生成到 `artifacts/cl_hz1x1/overlays/`，智能体默认配置在 `configs/tsc/`。

## 依赖关系

本线可依赖 `infra/`（运行基座）与框架层；`hitl/` 的训练通道复用 `sequential/` 的方法件。
历史 checkpoint 中的旧模块路径（`sequential.*`、`plan5.*`、`arterial.*`）由 `infra/_legacy.py` 解析，可照常加载。
