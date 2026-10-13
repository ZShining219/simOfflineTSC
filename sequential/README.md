# sequential — 半离线/持续学习训练包

> 包形状约定见 GOVERNANCE §3.1。本包为**方法包**。

## 问什么

LibSignal 基线之上的两个研究问题：

1. **持续学习**：多场景顺序切换时的 replay/遗忘问题（Plan 3/4 顺序训练）；
2. **半离线**：顺序在线交互 + Plan 1 冻结历史档案的混合利用（HA-SODQN，`mode=semi_offline`）；HITL 数据面产物（偏好对/示范集/专家池）在本包消费（`mode=hitl_channel`）。

## 入口面表

| 命令 | 用途 |
|---|---|
| `python -m sequential <verb>` | 包门面（manifest 驱动）：`import-parents` → `build-plan`/`build-ha-plan` → `launch` → `run-child` → `status`/`analyze` 等，全 verb 见 `--help` |
| `python -m sequential.offline <verb>` | Plan 2 纯离线面：`prepare-plan2`/`validate-dataset`/`benchmark-dataset`/`train` |
| `python -m sequential.legacy_calibration` | Plan 1 冻结评估器直跑（校准对照件） |

兼容 shim（过渡，将移除）：`python sequential_run.py` → `-m sequential`；`python offline_run.py` → `-m sequential.offline`。

## 工单与产物

- 工单：`configs/sequential/*.yml`（plan34/ha_sodqn_b100 等）、`configs/offline_tsc/`
- 产物：`output_data/sequential/`、`output_data/ha_sodqn/`、`output_data/offline_tsc/`、`output_data/hitl_probe/`（HITL 轮次训练）

## 边界

- 本包消费 HITL 数据面产物（pairs/demos/pool/RM ckpt），**不认 HITL 语义**；实时引擎与数据面管线归 `hitl/`（`hitl/README.md`）。
- 内部模块自由组织（cli/manifest/launcher/trainer/ha_agent/adjudication/reward_model/evaluator/analysis 等约 40 件），跨包只许被以下层契约引用。

## 与基线层的关系

复用 LibSignal `agent/`（DQNNet 等）、`world/`、`common/`、`dataset/`、`utils/`；`run.py` 在线入口与本包互不调用，靠 `offline.py` 的延迟 import 保持基线 import 图不变。
