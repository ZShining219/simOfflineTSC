# hitl/ — 人在回路（HITL）方法包

本仓 HITL 线的家：**实时会话 + 数据面管线**。升格自原 `tools/sumo_live_broadcast`、
`dev/hitl_live`、`dev/action_ranking_audit`（2026-10-13 整理）。

## 子包职责

| 子包 | 职责 | 对应通道 |
|---|---|---|
| `live/` | 实时引擎：SUMO 转播、真人专家席、问询门（periodic/on_demand/random/margin/window）、逐决策 transitions 录制 | C 通道载体 |
| `pipeline/` | 数据面：会话打包→片段切割→偏好对/示范集/专家池→RM 拟合 | A 示范、B 偏好 |
| `plans/` | 正式 plan 生成器与派发器（喂 `sequential/` 的 run-child） | 全通道 |
| `probe/` | 配对探针（student/hitl/expert 同 seed 对照）+ 判官/RM 归一化校准 | B′、C |
| `figs/` | 出图与汇总表 | — |
| `audit/` | B′ 判官审计（错序/分歧/反事实/rank_aux） | B′ |
| `topology/` | 跨场景路网拓扑配置（sumohz1x1 系） | — |

## 边界（重要）

- **训练态件在 `sequential/`**：`adjudication`/`reward_model`/`reward_shaping`/`demo_ac`/`expert_pool`
  是 sequential 训练协议的 B′ 通道臂，由训练环直接消费，**不属本包**。
- **本包只做产物生产**：`pipeline/` 可引 `sequential.io`/`sequential.reward_model`（模型类），
  但不实现训练逻辑；`sequential/` 只按 manifest/schema 消费产物，不知 HITL 语义。
- **依赖单向**：`hitl/` → 下层包公共接口（agent/world/sequential.io）；
  `sequential/`、骨架层、`live/` 引擎**禁止** `import hitl.*`。
- **agent 接入走 adapter**：`live/controllers` 为注册表面，HITL 不 import agent 具体类；
  新控制器在各自方法包侧注册（跨模块改动按 mechanism→宿主、policy→hitl 分家）。

## 产物约定

- 根：`output_data/hitl_probe/`（存量 24 桶不动，桶地图见其 README）；
- 会话录制落 `output_data/hitl_probe/sessions/`（勿再默认 /tmp）；
- 新产出按 `output_data/hitl_probe/<stage>/` 分区；台账新 run 用 `line=hitl`。

## 入口

- 实时会话：`python -m hitl.live`（原 `python -m tools.sumo_live_broadcast`）
- plan 生成：`python hitl/plans/build_r3_plan.py`；派发：`bash hitl/plans/dispatch_r3.sh`
- 判官/RM 校准：`python hitl/probe/calibrate_*.py`

## 相关仓

`hitl_traffic_spec_v0_4`：已收官的正式矩阵仓（v1/v2 45-run/ingo 证据），冻结作
正式证据仓，不合并；本包为实时线现役管线。
