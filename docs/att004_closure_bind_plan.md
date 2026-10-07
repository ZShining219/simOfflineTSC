# closure_bind 臂设计（w34 批，T3）

日期：2026-10-07 · 作者：w34_1 · campaign `att_entity_004` · 分支 `feat/w34-eval-v2-closure`

## 目的与裁决依据

t34_devin/t34_cc 两批已定位 road_closure 覆盖缺口：边界边 closure（to_junction 为
priority 非信号节点）的 lanes 不属于任何受控节点观测集，`TextEntityBinder` 无可绑定
节点，事件对策略 z_task 完全不可见；其效用严格为零（t34_cc 实测配对差恒 0）。
Z 裁决采用方案 (b)：扩展绑定语义——**road_closure 类事件绑定到"最近受控上游节点"**。

## 覆写项（相对基配置 `configs/tsc/att_entity_004/generated/att004_fixG_200.yml`）

| 项 | fixG_200 | closure_bind_200 |
|---|---|---|
| `model.closure_binding` | 不设（None） | `nearest_upstream` |
| `model.closure_bind_max_hops` | — | 默认 4（未显式写） |
| 其余 model/trainer/logger 全项 | — | **逐字相同**（fusion_gate、starvation_guard、huber、structaux、episodes 200、plan `hz4x4_random_v1`） |

训练配置：`att004_closure_bind_200.yml`；smoke：`att004_closure_bind_smoke.yml`（2ep，同模型段）。

## 语义边界（什么改了、什么没改）

- 只改 **grounding→policy 绑定层**：`generator/text_entity.py` 的 `TextEntityBinder` 新增
  `closure_binding` 模式。edge-scope、`event_kind=road_closure`、且特征级零绑定的报告，
  经目录拓扑（catalog lanes 的 from/to_junction）BFS 上游，绑定到最近受控节点
  （同深度多个候选取 policy 节点序最小者，确定性）。
- fallback 落在新增的 `node_fallback_mask`（节点级），与 `feature_report_mask`
  分离：车道级物理 referent 审计不失真；`to_gnode` 将其 OR 进 `node_report_mask`。
- **不改**：冻结 eval 事件计划本体（`configs/events/plans/*` 零改动）、奖励定义、
  指标口径、其余臂行为（默认 `closure_binding=None` 时输出逐位不变）。
- 遮蔽语义不变：同一节点多事件仍按 report_id 先到先得；fallback 绑定只填空位。

## 审计

绑定命中时 `run_dir/closure_bind_log.jsonl` 追加 `{t, report_id, edge_id, bound_node}`；
Gnode 同时携带 `node_fallback_mask` 供离线区分 lane 绑定与 fallback 绑定。

## 执行

- s7，200ep，CPU（CUDA_VISIBLE_DEVICES=-1），队列路径
  `artifacts/att004_eval_v2_w34/run_queue_w34_closure.json`，tmux `w34_closure`。
- 先 2ep smoke（`att004_closure_bind_smoke.yml`）验证链路与日志，再起正式 run。
- 状态流：REGISTERED→SMOKE→RUNNING→DONE，台账 `ledger/runs.jsonl`。
