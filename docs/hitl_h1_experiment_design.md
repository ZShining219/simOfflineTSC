# HITL-H1 专家策略制定与融入：实验设计与模块规划

日期：2026-09-24。状态：设计文档；H1-E0 审计已完成，H1-E1/E2 已实现并验收（见 §5 结果列），H1-E3 未执行。
对应 Wiki 任务页：[HITL-H1] 专家策略制定与融入（人在回路工作台）。
执行载体：同级项目 `../hitl_traffic_spec_v0_4`（SPEC V0.4，2026-09-15）；本文档存放于本仓库 `docs/` 作为协作设计依据，实验代码与产物仍在执行项目内。

**执行方式变更（2026-09-24 用户确认）**：不再恢复冻结队列 `formal_5ce4f9b1d65b466a`（含中断的 C2 与剩余 41 jobs）。H1 作为新实验线在既有代码、数据、artifact 上重新规划执行；原冻结计划与已完成 run 保留为历史证据。原"验证间隔 10→20"协议修订议题随之失效——新计划中验证间隔作为普通设计参数直接设定。

## 1. 定位与命名约定

H1 目标：把"专家知道怎么做"变成机器可保存、可调用、可进入策略的知识。只研究"专家已被调用以后怎样输入、保存和作用"；触发条件研究归属 H3。

**命名冲突警示**：SPEC V0.4 §13 的 H1–H4 是执行门禁（资源接口/功能闭环/pilot/正式证据），与本框架的 HITL-H1/H2/H3 研究阶段同名不同义。本文档及后续沟通中，门禁写作 "SPEC-H1~H4 gate"，研究阶段写作 "HITL-H1/H2/H3"。

## 2. 子目标与现有实现映射（已现场核验）

| 子目标 | V0.4 已有实现 | 状态 |
|---|---|---|
| H1.1 ExpertAdapter.propose(obs, mask) | `hitl_tsc/feedback_control.py::SOTL.propose(raw, mask)` → {action, reason, green_queue, red_queue}；`expert_version` 已写入 transition | IMPLEMENTED，pilot/formal 已用 |
| H1.2 ExpertCaseLibrary | `hitl_tsc/demo_data.py::select_demos` → `demos.jsonl`（1800 起点+10 步后缀）；专家标定轨迹；各 run `proposals.jsonl` | 部分覆盖：有示范存储，无按 scenario/goal 检索的库抽象 |
| H1.3 策略融入 | `hitl_tsc/demo_ac.py::DemoAC.pretrain(actor_enabled, critic_enabled)`；A0/Aπ/AQ/AπQ 消融 | IMPLEMENTED，pilot v3 通过；formal 正式证据缺 |
| H1.4 ActionArbiter + 执行绑定 | `hitl_tsc/feedback_control.py::arbitrate()`；transition 含 agent/safe_agent/expert/accepted + expert_queried/intervened；SPEC T03 覆盖回报归属 | IMPLEMENTED，`verified_formal/formal_C1_11` 有真实干预数据（ep1：20 查询、15 干预） |

记录链数据模型已满足框架验收形态：`transitions.jsonl`（obs/agent_action/accepted_action/next_obs/mask/t0/t1）⨝ `proposals.jsonl`（transition_id → expert action+reason）。

## 3. H1-E0 顶层审计结果（2026-09-24 已执行）

`runs/` 顶层存在 12 个标记 completed 的 formal-profile run（各 72000 transitions/200 episodes）：A0/A_pi/A_Q/A_piQ（seed 11）、B0_11、B0_11_575b102c、B0_22、B1_11、B1_22、B2_11、C1_11、C2_11。

**结论：全部为失效旧计划产物，不能继承为冻结计划 `formal_5ce4f9b1d65b466a` 的正式证据。**

依据：

1. **数据身份不同**：run 的 transition 记录 case 为 `BC_TYC_FIXED_train_1000`，属 `artifacts/invalidated_legacy/data_manifest.json` 的 18-case 体系（10/3/5 划分）；冻结计划锁定的当前 manifest 为 54 个 `reference_NS/EW/BAL_*` case（30/9/15），sha256=bb8a599c…。case 家族与规模完全不同。
2. **代码身份不同**：run 执行于 2026-09-15 晚、v1–v7 计划时代（plan_id=formal_1789474289…789681），代码为未提交 worktree（v7 源哈希 39538bdc… ≠ 当前 44f5fb24…）；冻结计划锁定 commit `0b08b3b`（09-16 02:22 提交，含输出根修复与审计训练实现）。
3. **计划与输出结构不同**：冻结 run_root 为 `runs/verified_formal/`；顶层 run 无 `versions.json`、`scientific_results` 标记与 episode 级目录结构，继承检查亦不通过。
4. 附带事实：这些 run 的 `config.resolved.json` 与当前 `load_bundle(base,'formal',<exp>)` 逐叶子比较 0 差异——配置语义相同，失效在数据与代码身份，不在配置内容。

处置建议（未执行）：它们是 A 管线在 formal 规模端到端跑通的开发级证据，可用于链审计抽样与学习曲线粗查，但**不得计入正式对照**；建议后续移入 `invalidated_legacy/` 或加标记文件避免误用，需用户确认。顶层 `formal_C2_11` 是唯一完整 C2，同样仅限开发证据。

## 4. 缺口清单

- G1 ExpertCaseLibrary：需把示范转移/专家标定轨迹/干预记录组织为可按 `scenario + goal` 检索并关联 outcome 的 case 库。纯数据组织，不需新仿真。
- G2 验收证据装配：缺一个审计器把记录链与 case 链装配成可审核产物。
- G3 formal A 正式证据：A 系正式对照在任何计划下均未产出（旧队列未执行到，且已放弃恢复）；需新建 H1 实验计划产出。
- G4 文档过期：`artifacts/method_diff.md` 有"A 未实现"旧描述（评审报告已点名）。

## 5. 实验设计

| ID | 内容 | 产出 | 依赖/规模 |
|---|---|---|---|
| H1-E0 身份审计 | 已完成（见 §3） | 审计结论 | 只读，已执行 |
| H1-E1 记录链验收 | ✅ 已执行（2026-09-24）：`hitl_tsc/record_chain_audit.py`，支持 flat 与 episode 两种布局 | `artifacts/h1_acceptance/record_chain_*.json`：verified_formal C1 **passed**（200 eps/72000 转移/3773 查询/2837 干预/0 违规）；顶层 dev A_piQ passed（0 查询，符合 A 组无专家语义）；顶层 dev C1 **failed**（23 条 queried_without_proposal_row——dev 时代无 proposals 表，且查询计数语义与现行不同） | 只读，已完成 |
| H1-E2 ExpertCaseLibrary 构建与检索验收 | ✅ 已执行（2026-09-24）：`hitl_tsc/expert_case_library.py`，CLI `build/query/audit` | `artifacts/expert_case_library_v1/`：**25013 case**（19440 标定轨迹 + 1800 示范 + 3773 C1 干预），唯一键、0 缺 outcome、审计 passed；`query(scenario=, family=, goal=, source_kind=)` 已验证 | 只读装配，已完成 |
| H1-E3 策略融入正式对照 | **新建 H1 实验计划（缩减版正式，2026-09-24 用户选定）**：A0/Aπ/AQ/AπQ 对照，沿用 V0.4 已验证训练栈（`a_training.py`/`demo_ac.py`）、54-case manifest、env_binding；建议默认 100 episodes × 3 seeds（11/22/33）、验证间隔 20（每 run 5 个候选点）、9 case 验证集与 15 case 测试集不变；执行顺序 smoke→pilot→缩减正式 | 正式对照结果 + 新计划锁定文件 | 参数可按资源再调；不恢复旧队列 |
| H1-E4 示范溯源审计 | ✅ 已执行（2026-09-24，详见 §5.1）：逐条核对 1800 条 demo 的 record 级 `config_id`、suffix `run_id`、源轨迹路径与锁文件 `ranked_configs` | 结论：**无错标**。5 个源配置各 360 条，`config_id` 权威正确；suffix 转移本身无 `expert_version` 字段，溯源经 `demo_id`/`run_id`/文件成员关系隐式承载，E2 库已显式物化 | 只读，已完成 |
| H1-E5 专家资格验证 | ✅ 已执行（2026-09-24，v2 含 max-pressure）：`hitl_tsc/expert_qualification.py`，在 9 个**未参与选参**的验证 case 上冻结评估 9 个 SOTL 配置 + mp_q/mp_n + fixedtime_30s + random_legal，共 117 episode 真实仿真 | `artifacts/expert_qualification_v2/expert_qualification.json`：**qualified**（g0_r06 赢 fixed/random），但 `best_controller_id=mp_n_m0`——MP 领先 SOTL 约 18%，专家质量梯度成立 | 已执行，约 11 分钟 |
| H1-E6 第二专家示范 | ✅ 已执行（2026-09-24）：`hitl_tsc/mp_demos.py` 采集 mp_q/mp_n 两变体在 6 校准 case 的完整轨迹（12 episode），从中选 1800 条 mp_q_m0 示范（与 v1 同规模同选取机制）；轨迹转移自含 `expert_type`/`expert_version` 显式戳 | `artifacts/mp_calibration_v1/mp_expert.lock.json`（mp_n 44.36 / mp_q 45.49）+ `artifacts/demos_v2/`（1800 条 MP 示范）+ `artifacts/expert_case_library_v2/`（31133 case，含 MP 两类来源，审计 passed） | 已执行，约 2 分钟 |

判定要点：H1-E1/E2/E4 不需要新仿真；H1-E5 为一次性轻量仿真（已完成）；H1-E3 是唯一需要真实训练的部分，走"新计划"路径，不再依赖旧队列。

### 5.1 专家溯源模型（H1-E4 核查结论）

A 与 C 使用**不同**的专家版本语义，现有数据已正确区分：

| 链路 | 专家版本 | 载体 | 状态 |
|---|---|---|---|
| A 示范（demos_v1） | 实际产出配置：`g0_r06`/`g3_r06`/`g6_r06`/`g0_r03`/`g3_r03` 各 360 条 | demo record 的 `config_id`（权威）+ suffix `run_id=calibration_<config>_<case>` + `expert_proposal` | ✅ 无错标；suffix 转移无独立 `expert_version` 字段，溯源为隐式（经成员关系），case 库已显式物化 |
| C 干预（verified_formal） | 锁定最优 `g0_r06`（`bc_runner.py:117`，凡装有专家的转移一律盖章，含未查询步） | transition `expert_version` + `expert_queried` 标志区分是否实际使用 | ✅ 语义=「已安装专家」；因 C 只装 g0_r06，不构成错标 |
| 遗留 runner.py | `sotl_v1`（仅查询时盖章） | 旧 dev 路径约定 | ⚠️ 与 bc_runner 口径不同，仅影响失效 dev 数据 |

核查确认：五条示范来源各自的 `config_id` 在 demos.jsonl、标定轨迹目录、锁文件 ranked_configs 三处一致，未被统一标为 g0_r06。E2 库中 demonstration 类 case 的 `source.config_id` 逐条等于真实产出配置。

补充（E6 后）：demos_v2 的 MP 轨迹与 demo 记录已改为**自含显式溯源戳**（转移级 `expert_type`/`expert_version` + record 级同名字段），隐式溯源缺口在新产物上闭合；v1 的隐式形态保留，由 case 库物化兜底。

### 5.2 专家资格门（H1-E5）

新增「资格验证」环节回应"calibration 上最优 ≠ 合格专家"的问题：冻结 g0_r06 后，在未参与选参的 validation split 上与基线对照。结果：`qualification_status=qualified`（开发判据：0 异常且 mean_RRT ≤ fixedtime 且 ≤ random，非论文主张）。

需要如实记录的两个事实：(1) r06 三配置（g0/g3/g6_r06）在**标定集与验证集上指标均完全相同**——红阈值主导切换决策、绿阈值不具区分度，`g0_r06` 的"selected"本质是按 config_id 字典序的并列取首，语义上代表「r06 配置族」而非严格唯一最优；(2) 验证集上 r06 族仍并列第 1（55.23 vs r03 族 ~56.4 vs r12 族 61.7），排序方向在两集合间一致。

**专家质量梯度（v2 结果，`artifacts/expert_qualification_v2/`）**：新增第二个虚拟专家 MaxPressure（相位选择式，开边界化简 pressure=服务车道排队和）后，验证集排序变为 mp_n 45.07 > mp_q 45.28 > sotl_r06 族 55.23 > sotl_r03 族 ~56.4 > sotl_r12 族 61.7 > fixedtime 68.30 > random 105.05。**锁定专家 g0_r06 并非可得最强规则专家**——max-pressure 在其未见 case 上领先约 18%。这使"专家质量"成为 H1.3 可用的实验变量（SOTL≈中等质量、MP≈较高质量、fixed/random 作对照底），但也带来两个必须正视的问题：(a) 以 SOTL 为示范源的预训练可能把策略锚在次优水平；(b) C 路径当前用 g0_r06 干预，干预质量低于可得的 MP 干预——是否升级 C 专家属 H3 范畴的决策，需单独确认。

## 6. 模块规划

**新增（薄层，不改训练语义）**

| 文件（执行项目内） | 职责 |
|---|---|
| `hitl_tsc/expert_case_library.py` | ExpertCaseLibrary：从既有 artifacts 建只读索引，`query(scenario=, family=, goal=, source_kind=)`；不重新仿真、不改写原数据。**已实现** |
| `hitl_tsc/record_chain_audit.py` | 记录链审计：字段完整、引用一致、intervened/动作一致、accepted∈mask、episode 边界与终止；输出验收 JSON 与抽样链记录。**已实现** |
| `hitl_tsc/expert_adapter.py` | ExpertAdapter 协议层：统一 `propose(obs, mask)` 签名并盖 source/version/config_id 戳；`sotl_adapter()` 从 expert_config.lock 构造。**已实现** |
| `hitl_tsc/expert_qualification.py` | 专家资格门：冻结选定配置，在 validation split 上与全部 SOTL 候选 + max-pressure 两变体 + fixedtime_30s + random_legal 对照，输出 `expert_qualification.json`。**已实现并执行（v2）** |
| `hitl_tsc/max_pressure.py` | 第二虚拟专家 MaxPressure：合法相位中按服务车道排队压力 argmax，tie→保持，`demand∈{q,n}`、`switch_margin` 可调；`max_pressure_adapter()` 接入 ExpertAdapter 溯源戳。**已实现** |
| `hitl_tsc/mp_demos.py` | MP 轨迹采集 + 示范选取（复刻 calibration/select_demos 机制），转移级显式溯源戳；CLI `python -m hitl_tsc.mp_demos`。**已实现并执行** |
| `hitl_tsc/expert_case_library.py` | 扩展 `extra_demos`/`extra_calibrations` 入口，demo/轨迹的 `expert`/`version` 从记录字段或 spec 读取，不再写死 sotl。**已实现** |

**改动**

| 文件 | 改动 |
|---|---|
| `hitl_tsc/pilot_verification.py` / `formal_evidence.py` | 接入记录链审计为 H1 适用项 |
| `artifacts/method_diff.md` | 更新"A 未实现"过期描述（属冻结输入附属文档，改动前需确认） |
| `tests/test_h1_modules.py` | 10 项：adapter 溯源戳/合法性、链审计双布局与违例检测、case 库构建/检索/审计、资格控制器（fixedtime 周期切换/illegal 保持、random 合法性+确定性、注册表 selected 标记）。**已实现，49 项全仓测试通过** |

**不改动**：`feedback_control.py`、`demo_ac.py`、`demo_data.py`、DQN/trainer 核心——语义已被 pilot/formal 验证，H1 只做组织与装配。

## 7. 验收与状态登记

- 记录链：transitions ⨝ proposals 按 transition_id 已能产出 obs_t→agent→expert→accepted→step→obs_{t+1} 完整链；E1 审计器负责断言与抽样。
- Case 链：ExpertCase ID→scenario/goal→expert 方案→outcome 由 E2 库装配；示范 case 的 `source.config_id` 为真实产出配置（E4 已核）。
- 专家资格：E5 门确认「被称作 expert 的控制器在未见 case 上是合理、稳定、高质量的策略来源」——这是把 calibration 最优者升格为 expert 的前置条件，已过。
- 状态口径：E1/E2/E4/E5 完成后 H1.1/H1.2/H1.4 可登记 SMOKE_PASS（有真实仿真数据支撑的审计通过可视同 EXPERIMENT_PASS 的证据形态，但登记前需逐条注明所依据 run 的身份）；H1.3 的 EXPERIMENT_PASS 明确等新 H1 计划的正式对照完成。
- H1 定位提醒：本阶段是任务书功能闭环，不强行做新算法贡献。

## 8. 待决事项

1. ~~新 H1 计划参数~~：已选"缩减版正式"（100 eps × 3 seeds、验证间隔 20 为建议默认）；执行前仍需确认 wall-clock 预算与并发上限。E3 消融可考虑增加"专家来源"因子（demos_v1 SOTL vs demos_v2 MP）。
2. 顶层 12 个失效 run 的处置（标记/移动需确认）。
3. `method_diff.md` 是否允许更新。
4. Wiki 工作台状态修订：当前标记"受阻"系因旧队列中断；任务切换后建议改为新框架下的状态描述，待新计划参数确认后一并 `review-update`。
5. 【已转 H3】C 路径干预专家是否从 g0_r06 升级为 MP——属 H3 触发+干预语义决策，H1 阶段不动 `expert_config.lock.json` 的 selected。
