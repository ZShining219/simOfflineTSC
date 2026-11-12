# docs/ — 文档目录规则

本目录承载以下内容，职责互不相同：

| 子域 | 语义 | 生命周期 |
|---|---|---|
| `docs/plan*.md` 等计划文档 | 与用户讨论形成的方案/goal，执行期参考 | 失效即更新或清理，不作长期执行依据 |
| `docs/experiments/` | **实验记录**——每次实验一份固定格式文档（见下） | 长期保留，否决/归档实验只改状态不删 |
| `docs/done/` | 里程碑验收、决策登记等归档 | 冻结，只追加不修改 |
| `docs/briefs/` | 对外汇报、展示与 PPT 素材 | 按汇报场次保留；结论以台账与 experiments 为准 |
| `docs/refs/` | 论文、外部资料与文献清单 | 长期保留 |

## 实验记录规则（`docs/experiments/`，强制）

一次实验 = 「实验相关记录 + 相关改动」的整体，必须回答四问：**改动意图、
改动模块、实验设计、实验结果**。

- **E1 一实验一档**：`docs/experiments/YYYY-MM-DD_<slug>.md`，模板用
  `TEMPLATE.md`；slug 确定后不改名。
- **E2 四节必填**：意图 / 模块 / 设计 / 结果。发起时先写前三节，
  收割后补结果与结论；任何一节空缺 = 记录未完成。
- **E3 延续即更新**：同一实验的延续迭代更新同一文档并刷新
  「最近更新」日期；研究问题变了 → 开新文档。
- **E4 分工不重复**：docs/experiments = 叙事层（为什么/怎么设计/结论）；
  `ledger/runs.jsonl` = 机读事实；`EXPERIMENTS/` = 台账自动索引层
  （`dev/ledger/gen_experiments.py` 重生成）。实验文档引用 run_id/
  campaign，不复制指标全表。
- **E5 随码入库**：实验文档与对应代码/配置改动同批 commit。
- **E6 否决保留**：被否决或归档的实验文档改状态保留，不删除——
  负结论也是证据。

## 计划文档规则

- 仅放与用户确认过的方案/goal 类文件；
- 失效内容及时标注或移入 `done/`，避免过期方案继续被当执行依据。

## 入口地图（2026-11-02 治理盘点补）

| 入口 | 用途 | 状态 |
|---|---|---|
| `run.py` | 单次实验主入口（参数定义、训练/评估） | 正式入口 |
| `python -m sequential` / `python -m sequential.offline` | 顺序训练 / 纯离线训练面 | 正式入口 |
| `tools/run_experiment_queue.py` | 队列派发（批量 run 与恢复） | 正式入口 |
| `arterial_run.py` | arterial 1x6 线专用入口 | 正式入口 |
| ~~`offline_run.py` / `sequential_run.py`~~ | 旧入口兼容 shim | 已于 2026-11-12 移除，改用 `python -m sequential[.offline]` |

完整派发/恢复流程见 `AGENTS.md`；台账回填规则见团队侧的 git 作业规程（不在本仓分发）。

## 计划文档状态索引（盘点时点：2026-11-02）

| 文档 | 定位 | 自述状态 / 盘点判断 |
|---|---|---|
| `plan0721.md` | Plan 0–4 总计划与状态索引 | 持续更新，现行总索引 |
| `plan1_formal_online_dqn.md` | Plan1 正式 online DQN 子计划 | 已通过（历史依据，不改） |
| `plan1_support_and_pilot.md` | Plan1 支持/基线/Pilot 子计划 | 已通过（历史依据，不改） |
| `plan0726.md` | HA-SODQN 两阶段实施计划 | 早期实施依据（历史） |
| `plan2_offline_support.md` | Plan2 纯 offline DQN 支持 | 工程能力已完成（历史） |
| `plan_hitl_live.md` | HITL × 半离线设计参考 | 设计定稿（历史执行参考） |
| `plan_qanchor_clhz1x1.md` | cl_hz1x1 Q-锚定实验计划 | 已收官（历史证据） |
| `plan_cross_algorithm_semi_offline.md` | 半离线跨算法实验计划 | 进行中（2026-09-09 建立，需 owner 确认是否仍有效） |
| `plan_trajectory_scene_semantics.md` | 轨迹场景语义标注探索 | 进行中（2026-10-08 建立，需 owner 确认是否仍有效） |
| `plan.md` | **Plan5** 跨算法机制定位正式规范 | 现行规范（注意：文件名未带 Plan 编号，易误读） |
| `plan5_goal_prompt.md` | Plan5 Codex CLI 执行 prompt | 一次性消费（历史） |

规则：新增计划文档须在此表登记；状态变化时同步改本表，不靠文件名猜。
