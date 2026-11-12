# 整理与规范计划（2026-11）

> 状态：执行中（分支 `gov/refactor`，锚点标签 `pre-refactor-20261112`）。最近更新：2026-11-12。
> 本文是整理工作的执行依据与进度记录；完成后移入 `docs/done/`。

## 1. 目标

用户的核心诉求：**看到一个结果时，能确定它对应哪次实验、哪版代码、哪份配置、哪套数据，并且能判断 agent 找到的东西对不对**。
整理（目录、入口、文档）服务于这个目标；约束必须由机制执行，而不是只写在文档里。

原则：

1. 不改实验语义：只动位置、名称、入口、记录；同种子改动前后指标逐条一致。
2. 历史产物原地冻结，不搬不删；新产物进新规范。删除任何产物须用户逐批授权。
3. 代码可以搬：台账 8090 行记录了 `code_commit`，证据可经 git 历史追溯。
4. 小步提交，每步中文提交信息；合入 `main`、推送远端均须用户确认。
5. 整理期间其他 agent 暂停写仓（用户 2026-11-12 决定）。

## 1.1 ⚠ 注意力线代码缺失（2026-11-12 发现，最高优先级）

`tarl_*` 智能体（sensor/gat/concat/selfattn/attention/gating/crossq）与 `agent/scene_attention.py` 的文本编码器，
实现都在 `reproduction/tarl_tsc/`。该目录约 3GB、**只有 2 个文件入过库**，其余（`models/tarl.py`、`text/encoder.py`、
`events/runtime.py` 等）从未入库；2026-10-15 提交 `8bae003` 将其当作"未入库残留"整体删除。

影响：`att_entity_003/004`（数千 run，面板所称"当前有效设计栈"）在本仓库**无法复现、无法继续开发**；
正式验收工具 `tools/run_tarl_struct_cf.py`（bench/GOVERNANCE 引用）同样不可用。此前 `agent/__init__.py` 静默吞掉导入失败，
现已改为显式警告。本机（73）、各工作树与 `/projects` 下其他项目均未找到副本；注意力线 run 多在机器 34 上执行，**34 上很可能仍有完整目录**。

恢复步骤（需能访问机器 34）：
1. 在 34 上确认 `<仓库>/reproduction/tarl_tsc/` 存在，只同步源码与小配置（`*.py/*.json/*.yml`），排除 checkpoint 与产物；
2. 放回本仓同一路径（保持 `from reproduction.tarl_tsc...` 导入不变），在 `lab/checks.py` 仓根白名单登记 `reproduction`；
3. 验证 `python -c "import agent.tarl"` 无警告，并为一个 tarl 臂新增黄金 smoke 用例后入库。

## 2. 诊断（2026-11-12，机器 73 视角，`python -m lab audit` 可复现）

| 问题 | 数据 |
|---|---|
| 运行时代码版本从未被记录 | `utils/logger.py` 的 `BASELINE_COMMIT` 是写死常量，所有 manifest 都自称来自 `73d860bb`；队列 `git_commit` 恒为 None；台账 `code_commit` 是登记时提交号。代码可信等级：verified 0 / registered 5416 / constant 2966 / none 7333 |
| 产物大面积不在本机 | 台账 15,735 行中 10,449 行的 `run_dir` 在本机与各工作树都找不到，仅 19 行标记 `evidence_lost`；推测多在机器 34（本机无法连通），也可能已丢失 |
| 当前主线最不可追溯 | `cl_hz1x1` 5,584 行：有指标 0 行、本机在盘 0 行、2,305 行无 `run_dir` |
| 指标缺失且口径不明 | 77% 的行 `metrics` 为空；台账值与评估摘要的 final/best 都不相等，未注明取值口径 |
| **台账指标混入训练回合（已证实）** | `xiasha1_online_v3`（288 行）与 `hz1x1_sanity_v1`（60 行）的回填脚本声明取"末 20 条评估记录均值"，实际按字段筛选，混入 TRAIN（含探索）与 FINAL_EVALUATION 记录。单 run 偏差 −8.5～+8.1 s；各车流包最优臂不变，但 v3 有 6/12、sanity 有 1/4 个车流包的臂排名改变。台账数值暂未改写，待用户定口径 |
| 指标口径各 campaign 不一 | 在盘且有指标的 run 中，只有上述两个 campaign 与 `xiasha1_compound_train_v1`（最终评估）能确认公式，其余各用不同公式 |
| 配置哈希对不上 | 1,426 行台账 `config_sha` 与 run 目录 `config_hash` 不一致（约定的"DONE 时替换"未执行） |
| 台账重复与绝对路径 | 20 个 run_id 重复登记；48 行使用绝对路径 |
| 登记逻辑分散 | `dev/ledger/` 下 26 个按 campaign 的一次性登记脚本，`_config_sha` 被实现 12 次 |
| 数据同名不同内容 | xiasha1 路网曾修正（`errata=geometry_truncated_v1v2`），同一 net 名不同时期内容不同，历史 run 未记数据指纹 |
| 产物落点分散 | 仓根 `analysis/`（1.3G）、`final_result/`、`artifacts/`（含 3,904 个 run 目录）、`data/output_data/` 下 23+ 随意目录、各工作树；代码中 266 处硬编码产物路径 |
| 分层倒置 | 核心层 6 处 import 研究线（`arterial` 实为通用回放/归档库） |
| 约束只在文档 | 无 git hook、无 CI；AGENTS.md 写 Python 3.9（实为 3.10）、写"无测试套件"（实有约 600 用例） |
| 运维 | 根磁盘 100%（剩 1.6G）；`/tmp` 下挂 3 个工作树；本地 `main` 领先 `origin/main` 37 个提交 |

## 3. 已完成

| 提交 | 内容 |
|---|---|
| `bccacf1` | `lab/` 溯源层与 `python -m lab audit/show/find`；manifest 记录运行时 `code` 与 `machine`；队列写 `git_commit/git_dirty` |
| `ce25a94` | 提交闸门 `.githooks/`（符号链接、产物入库、仓根白名单、大文件、内网 IP/绝对路径、中文提交信息） |
| `64c833f` | manifest 记录 `data` 指纹；`python -m lab data` 数据注册表 |
| `908f3c7` | `lab.paths` 产物落点收口（`TSC_OUTPUT_ROOT`）；架构棘轮测试 |
| `335d2bf` | 仓根散件归入 `docs/briefs/`、`docs/refs/`；AGENTS.md 纠错并加入 lab 使用规范；本计划文档 |

| `5444895` | 记录台账指标混入训练回合的已证实问题 |
| `9814bc7` | 通用回放/路口排序/文件锁迁入 `common/`，核心层对研究线的依赖由 6 处降至 1 处 |

验证：`hz1x1` `dqn_smoke` 在改动前后同种子运行，指标记录逐条一致；新 manifest 的数据指纹与 `lab data` 一致。

## 4. 待办

### A. 台账可信（优先）
- [x] 统一登记/回填：`python -m lab register` / `harvest`（端到端经正式队列 runner 验证），取代 26 个一次性脚本
- [x] 按既定约定回填 `config_sha ← run_manifest.config_hash`：1,426 行修正（旧值存 `config_sha_registered`）、548 行补齐，配置哈希不符 1,426 → 0
- [x] 标准口径定为最佳 checkpoint 评估（用户 2026-11-12）：`lab metrics` 回填 3,291 行，946 行改为标准值，旧值保留（`4813e27`）
- [x] 48 行绝对路径改为相对路径（`4813e27`）
- [x] 20 处重复登记均为逐字段相同的重复追加，已删除（`945ff9d`）；48 行绝对路径改为相对路径
- [ ] 机器 34 可连通后，在 34 上运行 `lab audit`，合并两机视角，区分"在 34 上"与"真丢失"

### B. 产物落点
- [x] 产物地图 `python -m lab outputs`（2026-11-12 快照：4 个产物区 52 个目录；`ha_sodqn` 379G + `sequential` 287G 占 91%，
  其中 `sequential` 仅 42 行台账引用；31 个目录 2.5G 未被台账引用，26 个可追溯到写入脚本）
- [ ] 历史散目录（仓根 `analysis/`、`final_result/`，`data/output_data/` 下非 run 目录）放置冻结说明
- [ ] 分析/出图脚本逐步改用 `lab.paths.report_dir()`，随迁随收紧棘轮基线
- [ ] 历史产物"永留 / 可弃"清单（含冻结线 `ha_sodqn` 381G、`sequential` 290G）交用户裁决

### C. 代码分层
- [x] `arterial/` 的通用回放与归档迁入核心层，原位置保留兼容层（`9814bc7`）
- [x] `common/experiment_queue.py` 对 `sequential` 的依赖反转（`9814bc7`）
- [x] `tools/` 中 57 个 campaign 一次性脚本迁至 `dev/<campaign>/`（对照表在 `dev/INDEX.md`）；`tools/` 根目录 81 → 24
- [ ] `scene_attention/_legacy/router/` 仍被 `scene_attention/figs/report_effect_fig.py` 引用且被汇报材料称为 HITL 调度工具，移出前需确认
- [x] 仓根兼容 shim（`offline_run.py`、`sequential_run.py`）已移除（`4dcd01d`）

### D. 配置与文档
- [x] `configs/tsc/`：评估后不搬迁（智能体默认配置按名加载且有 include 关系）；改为 `lab configs` 生成索引 + 根目录 overlay 只许减少
- [x] 规则文档去重：变更日志迁至 `docs/architecture_changelog.md`；上手清单只留 AGENTS.md 一份；GOVERNANCE 与新机制对齐（`57b2ed2`）
- [x] `dev/INDEX.md`：评估后保留手写（含大量历史裁决，34 个子目录仅 3 个有 README）；改为测试校验每个 `dev/<目录>` 均已登记，补登 `hitl_warmstart`
- [ ] `EXPERIMENTS.md` 面板按台账刷新

### E. 测试与 CI
- [x] 黄金 smoke：在线主路径 + cl_hz1x1 半离线两阶段，重构前后逐条一致（`python -m lab golden`）
- [ ] 为注意力线（scene_attention）与纯离线（sequential.offline）补黄金用例（需先让其入口遵守 `TSC_OUTPUT_ROOT`）
- [ ] 修复或标注历史失败用例（见第 5 节），拆出快速门禁集
- [ ] 推送打通后接入 GitHub Actions：`lab check --all` + 快速门禁

## 4.1 研究侧备忘（不属于本次整理，不安排执行）

- 指标口径改为最佳 checkpoint 后，`xiasha1_online_v3` 有 2/12、`hz1x1_sanity_v1` 有 1/4 个车流包的最优臂改变。
  该口径选择与报告同源、偏向训练曲线波动大的算法；若要据此下结论，需对最佳 checkpoint 做独立种子冻结复评。

## 5. 测试基线

在 `pre-refactor-20261112` 的干净工作树中运行全量 `tests/`（colight 环境，Python 3.10）：
**541 通过 / 51 失败 / 2 跳过，耗时 19 分钟**。干净工作树中没有 gitignored 的本机产物，部分用例依赖这些产物。

| 失败类别 | 数量 | 说明 |
|---|---|---|
| 断言失败 | 24 | 其中 `test_paper_agents_sumo` 20 个，需逐个排查是代码回归还是测试过时 |
| 缺少本机产物/文件 | 12 | 依赖 `artifacts/`、`data/output_data/` 等未入库内容，应改为自带夹具或标记为本机集成测试 |
| 正式实验 seed 门禁 | 9 | 测试未按门禁传入 `seed/sumo_seed`，门禁生效后测试未同步 |
| 依赖/其他 | 6 | `test_concat_colight` 导入问题、事件生成器确定性等 |

判定规则：重构分支上的失败若在下列名单中，视为既有问题；不在名单中即为新引入，必须修复。

<details><summary>重构前失败用例名单（51）</summary>

- `tests.test_arterial_stage1_manifest::test_stage1_manifest_has_frozen_twenty_run_matrix`
- `tests.test_concat_colight::test_concat_empty_scene_is_zero_and_finite`
- `tests.test_concat_colight::test_concat_forward_is_trainable_and_legal`
- `tests.test_cross_algorithm_assets.CrossAlgorithmAssetPlanTests::test_asset_chain_has_frozen_twenty_run_matrix`
- `tests.test_event_plans::test_generator_is_deterministic`
- `tests.test_feishu_wiki.ExperimentWorkflowTest::test_large_plan_rejected_before_network`
- `tests.test_hitl_packager.HitlPackagerTest::test_broken_state_chain_quarantines_episode`
- `tests.test_hitl_packager.HitlPackagerTest::test_dataset_manifest_passes_loader_validation`
- `tests.test_hitl_packager.HitlPackagerTest::test_expert_intervention_metadata_preserved`
- `tests.test_hitl_packager.HitlPackagerTest::test_invalid_action_quarantines_episode`
- `tests.test_hitl_packager.HitlPackagerTest::test_missing_field_quarantines_and_labels`
- `tests.test_hitl_packager.HitlPackagerTest::test_semantic_hashes_present`
- `tests.test_hitl_packager.HitlPackagerTest::test_valid_session_produces_loadable_dataset`
- `tests.test_hitl_packager.HitlPackagerTest::test_yellow_phase_index_quarantines`
- `tests.test_milestone0_run_state.RunStateTest::test_completed_lifecycle_has_fixed_identity_and_times`
- `tests.test_milestone0_run_state.RunStateTest::test_explicit_evaluation_seed_is_recorded_without_changing_default`
- `tests.test_milestone0_run_state.RunStateTest::test_failure_is_terminal_and_redacts_environment_values`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[legacy-events-colight]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[legacy-events-colight_phase_index]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[legacy-events-colight_phase_one_hot]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[legacy-events-frap]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[legacy-events-presslight]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[legacy-normal-colight]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[legacy-normal-colight_phase_index]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[legacy-normal-colight_phase_one_hot]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[legacy-normal-frap]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[legacy-normal-presslight]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[signal_v1-events-colight]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[signal_v1-events-colight_phase_index]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[signal_v1-events-colight_phase_one_hot]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[signal_v1-events-frap]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[signal_v1-events-presslight]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[signal_v1-normal-colight]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[signal_v1-normal-colight_phase_index]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[signal_v1-normal-colight_phase_one_hot]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[signal_v1-normal-frap]`
- `tests.test_paper_agents_sumo::test_hz4x4_training_readiness[signal_v1-normal-presslight]`
- `tests.test_paper_experiment::test_interrupted_training_resumes_exactly`
- `tests.test_paper_experiment::test_serial_parallel_match_and_dashboard`
- `tests.test_paper_p0::test_frozen_grounding_dependencies_preserved`
- `tests.test_paper_preflight::test_preflight_cli_finishes_assessment_without_training`
- `tests.test_paper_rewards_monitor::test_corrected_training_reproducible[colight]`
- `tests.test_paper_rewards_monitor::test_corrected_training_reproducible[colight_phase]`
- `tests.test_paper_rewards_monitor::test_corrected_training_reproducible[frap]`
- `tests.test_paper_rewards_monitor::test_corrected_training_reproducible[presslight]`
- `tests.test_paper_rewards_monitor::test_real_reward_training_and_dashboard[signal_v1-colight]`
- `tests.test_paper_rewards_monitor::test_real_reward_training_and_dashboard[signal_v1-colight_phase]`
- `tests.test_paper_rewards_monitor::test_real_reward_training_and_dashboard[signal_v1-frap]`
- `tests.test_paper_rewards_monitor::test_real_reward_training_and_dashboard[signal_v1-presslight]`
- `tests.test_plan1_trajectory.Plan1TrajectoryTest::test_online_replay_tuple_is_not_expanded`
- `tests.test_strong_training::test_formal_plan_balances_joint_profiles_and_keeps_holdouts`

</details>

## 6. 待用户裁决

1. 推送：GitHub SSH 认证可用（账号 ZShining219）；`gov/refactor` 分支是否推送、何时合入 `main`。
2. 机器 34 无法从本机连通；10,449 行产物是否在 34 上需要在 34 上核查。
3. 仓根未跟踪图片：`a1俯视底图.png` 是 xiasha 照片叠加工具的输入，是否移到 `data/raw_data/xiasha_photo/`（仍不入库）。
4. 台账重复登记 20 处以哪一行为准。
5. ~~台账指标标准口径~~：已定为最佳 checkpoint 评估（2026-11-12）。
