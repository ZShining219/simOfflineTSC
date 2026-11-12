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
| 本提交 | 仓根散件归入 `docs/briefs/`、`docs/refs/`；AGENTS.md 纠错并加入 lab 使用规范；本计划文档 |

验证：`hz1x1` `dqn_smoke` 在改动前后同种子运行，指标记录逐条一致；新 manifest 的数据指纹与 `lab data` 一致。

## 4. 待办

### A. 台账可信（优先）
- [ ] `lab` 统一登记 API（schema 校验、唯一 `config_sha` 实现、幂等追加），替代 26 个一次性登记脚本
- [ ] 按既定约定回填 `config_sha ← run_manifest.config_hash`（1,426 行，可 git 回滚）
- [ ] 用户定标准口径后：`lab` 提供唯一指标汇总函数并写 `metric_source`；修正 xiasha1_online_v3 / hz1x1_sanity_v1 的混入训练回合问题（旧值保留在 `metrics_legacy`）；其余 campaign 逐一核对公式
- [ ] 20 处重复登记交用户裁决；48 行绝对路径改为相对路径
- [ ] 机器 34 可连通后，在 34 上运行 `lab audit`，合并两机视角，区分"在 34 上"与"真丢失"

### B. 产物落点
- [ ] 历史散目录（仓根 `analysis/`、`final_result/`，`data/output_data/` 下非 run 目录）放置冻结说明
- [ ] 分析/出图脚本逐步改用 `lab.paths.report_dir()`，随迁随收紧棘轮基线
- [ ] 历史产物"永留 / 可弃"清单（含冻结线 `ha_sodqn` 381G、`sequential` 290G）交用户裁决

### C. 代码分层
- [ ] `arterial/` 的通用回放与归档迁入核心层，原位置保留兼容层（消除 5 处倒置）
- [ ] `common/experiment_queue.py` 对 `sequential` 的依赖反转
- [ ] `scene_attention/_legacy/` 移出活跃包；`tools/` 中 campaign 一次性脚本归 `dev/<campaign>/`
- [ ] 仓根兼容 shim（`offline_run.py`、`sequential_run.py`）按 README 承诺移除

### D. 配置与文档
- [ ] `configs/tsc/` 根目录 111 个平铺 yml 按 `baselines/` 与 `<campaign>/` 分层
- [ ] 收敛规则文档：README（门户）/ AGENTS.md（必须与禁止）/ GOVERNANCE.md（实验流程），去重
- [ ] `dev/INDEX.md`（826 行手写）改为由各目录 README 生成
- [ ] `EXPERIMENTS.md` 面板按台账刷新

### E. 测试与 CI
- [ ] 修复或标注历史失败用例（见第 5 节），拆出快速门禁集
- [ ] 推送打通后接入 GitHub Actions：`lab check --all` + 快速门禁

## 5. 测试基线

见下方（在 `pre-refactor-20261112` 干净工作树中运行全量 `tests/`）。

## 6. 待用户裁决

1. 推送：GitHub SSH 认证可用（账号 ZShining219）；`gov/refactor` 分支是否推送、何时合入 `main`。
2. 机器 34 无法从本机连通；10,449 行产物是否在 34 上需要在 34 上核查。
3. 仓根未跟踪图片：`a1俯视底图.png` 是 xiasha 照片叠加工具的输入，是否移到 `data/raw_data/xiasha_photo/`（仍不入库）。
4. 台账重复登记 20 处以哪一行为准。
5. 台账指标标准口径：评估末 20 次均值 / 最终评估 / 最佳 checkpoint 评估，三选一（或并存并注明）；是否据此修正 348 行已证实有误的数值。
