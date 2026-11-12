# 仓库协作指南

> 本仓实验治理见 GOVERNANCE.md + EXPERIMENTS.md + bench/（台账 ledger/runs.jsonl 为事实源；一次性脚本登记见 dev/INDEX.md）。

## 项目概述

本仓库是基于 LibSignal 整理的 Python 交通信号控制研究框架，提供与 Gym 风格兼容的交通环境，以及适用于 CityFlow 和 SUMO 仿真器的传统控制与强化学习智能体。

## 项目结构

三层，只许上层依赖下层（`tests/test_architecture.py` 校验）：

- **框架层（LibSignal）**：`run.py` 训练入口；`agent/` 智能体；`world/` 仿真器适配；`trainer/`、`task/` 训练循环；
  `common/`、`utils/`、`generator/` 共享工具；`data/raw_data/` 路网与车流输入，`data/dataset/` 数据集接口。
- **实验基座**：`lab/` 管理（台账、溯源、核查、黄金 smoke、提交闸门，纯标准库）；`infra/` 运行（checkpoint、事务、调度、独立评估）。
- **研究线**：`semi_offline/`（半离线/持续学习：当前主线 cl_hz1x1、纯离线训练 offline/、冻结的 sequential/plan5/arterial）；
  `scene_attention/`（场景引导注意力）；`hitl/`（人在回路）。各包 README 说明状态与入口。
- **配套**：`configs/` 配置（`configs/tsc/README.md` 为索引）；`tools/` 可复用工具；`dev/<topic>/` 一次性脚本（登记于 `dev/INDEX.md`）；
  `bench/` 验收口径与黄金 smoke；`ledger/`、`EXPERIMENTS/` 台账；`docs/` 计划、实验记录与归档；`tests/` 测试。
- **产物（不入库）**：`data/output_data/`、`artifacts/`；仓根 `analysis/`、`final_result/` 为冻结的历史产物。

## 开发流程

- 运行本项目时使用名为 colight 的 Conda 环境；执行项目命令前可运行 conda activate colight 激活环境。
- colight 环境当前为 Python 3.10（原 LibSignal 为 3.9）。
- 使用 pip install -r requirements.txt 安装核心依赖；仿真器依赖需要单独安装。
- 使用 python run.py 启动实验，并通过命令行参数选择仿真器、智能体和路网。
- 仿真器依赖可用时，使用 python run.py --help 查看支持的参数。
- Git 提交信息必须使用中文，并清晰说明本次提交的主要修改内容。
- 不要将生成的输出、缓存、IDE 文件和大型实验产物提交到 Git。
- 每个克隆执行一次 `git config core.hooksPath .githooks` 启用提交闸门（`python -m lab check`）。

## 实验基础设施（lab/）

查结果、查数据、查证据一律用 `python -m lab`，不要临时写脚本扫目录：

| 命令 | 回答的问题 |
|---|---|
| `python -m lab show <run_id>` | 这个结果是哪个 run：台账行、产物位置、代码版本可信等级、配置快照、数据来源、指标（台账值与评估摘要并列） |
| `python -m lab find --campaign C --arm A --net N --status DONE` | 某条件下有哪些 run |
| `python -m lab audit [--campaign C]` | 各 campaign 证据是否在盘、代码版本是否可信、配置哈希是否对得上、有无重复登记 |
| `python -m lab data [NET]` | 某个路网/车流对应哪些文件、当前指纹、被多少 run 使用 |
| `python -m lab metrics [--campaign C] [--apply]` | 按标准口径复算并回填台账指标（默认只预览） |
| `python -m lab outputs [--root R]` | 每个产物目录多大、属于哪个 campaign、台账是否引用、未登记的由哪个脚本写出 |
| `python -m lab configs [--write]` | configs/tsc 中哪些是智能体默认配置、哪些是实验 overlay、被谁引用（新增配置后 `--write` 更新索引） |
| `python -m lab register QUEUE` / `python -m lab harvest --campaign C` | 派发前登记、收割时回填台账（取代各 campaign 自写的登记/回填脚本；默认预览，`--apply` 写入） |
| `python -m lab golden [NAME] [--update]` | 改了代码，主线训练结果变没变（固定种子小规模运行，逐条比对） |

- 新 run 的 `run_manifest.json` 自动记录 `code`（运行时 HEAD/分支/dirty/改动指纹）、`data`（输入文件指纹）、`machine`。
  在每台机器的 shell 配置中设置 `TSC_MACHINE=34` 或 `73`。
- 代码版本可信等级：`verified`（运行时记录）> `registered`（仅登记时提交号）> `constant`（写死常量，等于没有）> `none`。
  引用结果做结论时注明等级；2026-11-12 之前的 run 均不是 `verified`。
- **台账指标标准口径 = 最佳 checkpoint 评估**（trainer 写入 `evaluation/summary.json` 的选择：travel_time 最小、并列取最早）。
  回填/登记指标一律 `python -m lab metrics --campaign C`（预览）→ `--apply`，禁止在登记脚本里自写公式。
  每行 `metric_source` 标明口径：`best_ckpt` 标准；`eval_attempt` 冻结评估行；`legacy:*` 无法复算的旧值，比较前先核对。
  该值取自训练期逐回合评估、选择与报告同源，偏乐观；需无偏比较时对最佳 checkpoint 另做冻结复评。
- 产物路径一律经 `lab.paths`（`output_root()`/`report_dir(topic)`/`queue_dir(campaign)`）获取；分析、图表、汇报产物写
  `<output_root>/reports/<topic>/`，不得再在仓根或 output_data 下新开散目录。`TSC_OUTPUT_ROOT` 可让多工作树共用产物根。
- `tests/test_architecture.py` 是棘轮：核心层（agent/world/trainer/task/common/utils/generator/data）不得新增对研究线
  （semi_offline/arterial/sequential/scene_attention/hitl/plan5/tools/dev）的 import，不得新增硬编码产物路径；存量登记在
  `tests/architecture_baseline.json`，修掉一处就删一处。

## 修改规范

- 保持 common/registry.py 中基于注册器的架构和现有注册约定。
- 将仿真器特有行为限制在 world/ 中相应的实现内。
- 实验参数优先通过 configs/ 调整，避免硬编码。
- 除非实验数据或结果体积较小、确有必要且有助于复现，否则不要提交生成的数据或结果。
- 修改智能体或训练器后，至少使用受影响仿真器的一项代表性配置进行验证。

## Agent 工作原则

本仓库既是实验代码仓库，也是持续演化的科研工作空间。完成当前任务固然重要，但在实现过程中还应关注代码、实验和文档是否能够被后续工作理解与复用。

### 先理解任务所处的阶段

开始工作前，应结合用户目标和仓库现状，判断当前任务主要属于哪种情况：

- 探索一个尚未确定的想法；
- 验证某项工程能力是否成立；
- 将已经确认的方案实现为稳定功能；
- 执行正式科研实验；
- 整理已有结果和证据。

不同阶段可以采用不同的实现方式。

探索阶段允许使用轻量、局部和临时性的实现，以尽快获得判断依据（时候需要及时清理或者整理避免堆积）；当方案被确认并准备长期使用时，应再考虑将其整理进稳定模块、配置体系和统一入口。

不要将尚处于探索阶段的内容过早包装为成熟能力，也不要让已经反复使用的能力长期停留在一次性脚本中。

### 将当前任务放回已有项目结构中思考

在新增代码、配置、脚本或文档前，先了解仓库中是否已有相近实现，以及当前任务与已有模块的关系。

优先考虑：

- 是否可以直接复用已有能力；
- 是否可以通过配置支持新的实验条件；
- 是否可以对已有模块做小范围扩展；
- 是否确实需要新增独立实现。

这里的目标不是避免新增文件，而是避免出现多个职责相近、边界不清、以后难以判断应使用哪一个的实现。

如选择新增实现，应使其用途和边界清楚，使后续使用者能够理解：

- 它解决什么问题；
- 它与已有实现有什么区别；
- 它是临时验证，还是计划长期保留的能力。

### 关注任务的沉淀方式

Agent 的产出可能包括代码、配置、实验数据、图表、日志、结论和文档。不同产物的生命周期不同，不必全部长期保留。

工作过程中应主动判断：

- 哪些内容是后续实验仍会复用的项目能力；
- 哪些内容属于一次正式实验的可追踪证据；
- 哪些内容只是帮助当前判断的临时产物。

对于临时产物，应避免让其与正式入口、正式结果混杂。任务结束后，可根据实际价值选择删除、归档，或在确认有复用价值后再整理为正式能力。

不要为了证明任务完成而制造不必要的文档、脚本或中间文件。已有文档能够承载新信息时，优先更新已有内容；只有当新内容具有独立、持续的用途时，再创建新的正式文件。

### 区分工程完成、实验完成和科研结论

本仓库中的任务通常包含多个层次：

- 代码或功能已经实现；
- 功能已经通过最小运行验证；
- 正式实验已经执行并得到结果；
- 实验结果足以支持某项科研判断。

这些状态不能相互替代。

代码能够运行，并不说明实验设计合理；实验成功输出指标，也不说明科研假设得到支持。

在汇报任务结果时，应如实说明当前完成到哪个层次，哪些内容已经验证，哪些仍属于推测或待实验确认。

### 让验证与任务目标相匹配

验证的重点不是执行尽可能多的命令，而是提供足以判断当前修改是否可信的证据。

对于局部代码修改，可以使用导入检查、语法检查或最小单元验证；对于运行行为修改，应执行能够覆盖关键路径的最小实验；对于正式实验，应记录配置、场景、随机种子、命令和结果位置。

验证范围应与任务风险相匹配。不要为了形式完整而运行无关测试，也不要在修改了关键训练、采样、状态、动作或评估逻辑后，仅以代码检查代替运行验证。

### 控制项目理解成本

项目复杂度不仅来自代码数量，也来自重复入口、相似配置、过期文档、难以追踪的结果和无法解释的历史版本。

完成任务时，应简单回顾：

- 这次工作是否引入了新的理解负担；
- 后续人员是否能够找到正确入口；
- 是否存在容易混淆的相似实现；
- 当前有效状态是否比任务开始前更明确。

如本次任务不可避免地产生了临时性或重复性内容，应在汇报中明确指出，而不是默认其已经成为项目长期结构的一部分。

### 在效率和长期维护之间保持平衡

科研项目需要快速试错，也需要逐步沉淀。不要因为追求长期整洁而阻碍必要探索，也不要因为追求当前速度而不断积累难以维护的一次性实现。

默认采用以下思路：

- 不确定时，先做能够回答关键问题的最小验证；
- 方向确认后，再将反复使用的能力整理为稳定实现；
- 正式实验开始前，确保入口、配置和结果记录足够清楚；
- 完成阶段性工作后，适当清理已经失效的探索内容。

Agent 应帮助降低后续任务成本，而不仅是完成当前指令。

## 用户沟通与文档产出
- 每轮对话前都需要附带“Dear Z“，这是对用户的称呼
- 交流内容必须清晰、具体，避免使用可能产生多种理解的模糊或歧义表述。
- 如果发现需求、目标、范围或关键决策存在模糊，应先结合用户已提供的信息、仓库现状和已有文档尝试消除歧义。对不影响主要方向的细节，可采用保守且可逆的处理方式，并在结果中说明所作假设；只有当不同理解会导致明显不同的实现路线、破坏已有行为或产生较高返工成本时，再向用户确认。
- 用户可能尚未想清楚具体方案；遇到这种情况时，应结合现有信息给出可执行的建议，说明推荐理由、主要取舍和可能影响，帮助用户进一步明确需求并作出决定。
- 与用户讨论过程中，避免未经确认直接创建或修改文档，优先在对话中说明相关内容。
- 如确有需要将内容落成文档，应先向用户说明拟创建或修改的文档及目的；只有在获得用户明确允许后，才能执行文档创建或修改。
- 与用户讨论的内容如需产出 plan、goal 或其他非永久但需要作为目标执行依据的参考文件，统一存放在 docs/ 目录中；文件不再有效时应及时更新或清理，避免将过期内容继续作为执行依据。

## 验证要求

`tests/` 有约 600 个用例，全量运行耗时较长且存在历史失败（见 `docs/plan_refactor_202611.md` 测试基线一节）。
验证分三层，按改动范围选用：

1. **快速门禁（每次提交，约 3 秒）**：`python -m pytest tests/test_lab.py tests/test_lab_metrics.py tests/test_ledger_schema.py tests/test_lab_checks.py tests/test_architecture.py`
2. **黄金 smoke（改动 agent/world/trainer/common/utils 或任何训练路径时必跑，约 15 秒）**：`python -m lab golden`。
   用例在 `bench/golden/`，覆盖在线训练主路径与 cl_hz1x1 半离线两阶段（历史混合采样、教师锚定）；同种子指标须逐条一致。
   有意改变行为时用 `--update` 重新生成期望值，并在提交信息中写明原因与影响。新的研究线上线时同步新增用例。
3. **全量测试（按需）**：`tests/` 中多数用例绑定具体 campaign 资产，51 个在重构前即失败（名单见
   `docs/plan_refactor_202611.md`）；只用于确认"没有搬坏"，不作为结果正确性的证据。

修改运行行为时，另需记录所用仿真器、智能体、路网、随机种子和完整命令。

## 新 agent 上手 checklist

0. 先跑 `python -m lab audit` 了解证据现状；引用任何结果前先 `python -m lab show <run_id>` 核对。
1. 读 `EXPERIMENTS.md` 顶部「当前面板」（现状/当前设计栈/代号地图）→ 本文件；`GOVERNANCE.md` 按需查节；campaign 明细在 `EXPERIMENTS/<campaign>.md`（勿通读索引层以外内容）；机读事实源 `ledger/runs.jsonl`。
2. 认领登记见 `artifacts/wiki_claims/`（历史追溯登记不进 claims）。
3. 新实验：设计定稿→台账 REGISTERED→SMOKE→远端 tmux 自治派发→收割三指标+过程记录→回填台账；**同时在 `docs/experiments/` 按模板建/更实验叙事记录**（四节必填：意图/模块/设计/结果，规则见 `docs/README.md` 与 GOVERNANCE §4.5）。正式 run 只允许 队列 runner + `run_queue_*.json` 或 `bench/` profile 两条路径；禁止新写一次性驱动脚本直拉正式 run（一次性件归 `dev/<topic>/` 并登 `dev/INDEX.md`）。
4. 注意力线口径：三指标（travel_time/throughput/unfinished_vehicles）强制同报 + 事件窗分层 + 四层判定（表征/优化/语义效用/闭环）；阶段①分类关卡不过线不进阶段②，见 `bench/BENCH.md`。
5. 本仓 GitHub public：入库文件禁写内网 IP/主机名/跳板拓扑/绝对路径，机器用代号 34/73，路径一律仓内相对路径。
6. 改架构 = commit + 在 `docs/architecture_changelog.md` 追加一行；改判据/口径属变更，台账开新 campaign，新旧口径不混排。
7. 拿不准 → 记"受阻"+原因，等裁决，不猜。

## 架构变更日志

见 `docs/architecture_changelog.md`（只追加）。
