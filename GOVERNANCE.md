# GOVERNANCE — simOfflineTSC 仓内实验治理规范

> 仓内实例化版本：**实验治理规范 v1.1**（治理母版的仓内落地；与母版冲突时回修本文件）。
> 配套：`EXPERIMENTS.md`（人读台账）、`ledger/runs.jsonl`（机读台账）、`bench/`（测试集规范）、
> `dev/INDEX.md`（一次性脚本登记）。

## 0. 一句话原则

**任何实验行为都必须留下可追溯身份；任何在跑实验必须能在不看会话记录的情况下被完全还原。**
agent 会话是临时的，仓库登记是永久的。验收判据：换一个全新 agent 只读仓库，能回答
"跑过什么、为什么跑、每臂何义、什么状态、产物在哪、结论是什么"。

## 1. 身份模型

| 概念 | 本仓落地 |
|---|---|
| line | `attention` / `semi_offline` / `engineering`（本仓现有三类事实口径） |
| campaign | `att_entity_002/003/004`、`arterial_1x6`、`plan1_dqn`、`plan5_b100`、`tarl_reproduction`、`tarl_v21_formal`、`milestone0_infra`、`paper_infra_validation`、`misc_probes` |
| run | 最小登记单位：一次训练 / 单次冻结评估 attempt / smoke / 工程验证批次 |
| run_id | 既有命名即身份（如 `att004_fixG_s17`、`flx_eval_v1/<pkg>/attempts/<name>`）；全局唯一，重名时以产物路径补全 |
| plan_id | 队列文件/审计链身份（如 `run_queue_att004_formal.json`、`P5-ANCHOR-DDQN-S2-SD7`） |
| code_commit | 启动时工作树 commit；dirty 必须记录 |

run 身份五元组：`run_id + campaign + plan_id/config_sha + code_commit + runtime_env`。

## 2. 状态机

```
DESIGN → REGISTERED → SMOKE → RUNNING → DONE
   ↘        ↘            ↘        ↘ ABORTED（必填 abort_reason）
   任意态 → FAILED（必填 error） / SUPERSEDED（同身份重跑替代，填 superseded_by）
                                  / RESUMED（中断恢复，记 resume_from_ckpt + 事件）
                                  / SCOPE_DISCARD（范围废弃，如队列 DROPPED）
                                  / LEGACY_UNCLEAR（历史追溯还原失败，禁止冒充 DONE）
```

- ABORTED 是正常终态，必填原因 + 已产证据位置；残留 `运行中` 状态且无存活进程者按 ABORTED 登记并在 notes 记已写训练条数。
- `evidence_lost:true` = 产物已失（如队列 FAILED 且无产物），不许留空。
- `partial_*`/`quarantine`/`packages_INVALID_*` 桶内 attempt 按隔离语义登记 ABORTED/FAILED。
- 追溯登记一律按真实终态；分不清 → LEGACY_UNCLEAR。

## 3. 本仓目录规范

```
GOVERNANCE.md            # 本文件
AGENTS.md                # agent 上手入口 + 架构变更日志
EXPERIMENTS.md           # 人读台账主表
ledger/runs.jsonl        # 机读台账（仓根 ledger/，不放 artifacts/——那层被 gitignore）
bench/                   # 测试集规范（BENCH.md + profiles/*.yml）
configs/                 # 实验配置（tsc/、events/plans/、sim/）
artifacts/               # 队列/状态/评估包/设计稿/wiki_claims（不入 git）
data/output_data/        # run 产物树（不入 git）
dev/                     # 一次性脚本唯一合法堆放处 + INDEX.md
dev/_legacy/<date>/      # 历史散件归档（移动非删除）
```

## 4. 台账（两级，强制）

### 4.1 `ledger/runs.jsonl` 字段口径

每行 JSON，字段全集：

```
run_id line campaign arm alias_of net seed
episodes_total episodes_increment resume_from ngpu
status plan_id code_commit dirty queue_file run_dir tmux started
eval_mode behavior_source metrics abort_reason error evidence_lost
notes run_kind config_sha sumo_seed_mode sumo_seed eval_seed condition
superseded_by runtime_env
```

- episode 三字段：`episodes_total`（终点）、`episodes_increment`（本次新增）、`resume_from`。
- `alias_of` 非空 = 别名/隔离副本，统计 n 时去重。
- `eval_mode`：本仓现有口径 `train_frozen_eval`（训练内联 eval）、`frozen_eval_attempt`（独立冻结评估）、`engineering_validation`。
- `behavior_source`：eval attempt 回指训练 run/包名。
- `run_kind`：`train` / `eval` / `smoke` / `calibration` / `batch`。
- `metrics`：训练行填末次完整 eval 的 `travel_time/throughput/unfinished_vehicles/queue/delay`；eval attempt 行指标留在包 manifest，不重复展开。
- 更正用追加新行 + `superseded_by`，不改旧行。

### 4.2 `EXPERIMENTS.md`

按 campaign 分节；节首：假设一句 + plan_id + 设计稿位置 + 结论（含负结论）；
训练/smoke/批次行全表，eval attempt 按包聚合；追溯批次标 `（追溯登记 YYYY-MM-DD）`。

### 4.3 时机

设计定稿→campaign 节(DESIGN)；派发前→runs.jsonl REGISTERED；迁移→两行同步；
收官 24h 内→metrics+结论回填。

### 4.4 seed 贯通自检

同配置不同 seed 终值指标完全相同 → flag 并记入 campaign 备注。
追溯批已执行一次：DONE 训练行按 (campaign,arm) 分组比对，未发现触发（见 EXPERIMENTS.md 附注）。

## 5. 注意力线口径（本仓硬约定）

### 5.1 三指标强制

`travel_time` / `throughput` / `unfinished_vehicles` 三指标强制同报；
**禁止只报 action-change 类过程量**。附加：queue、delay、waiting_time、reward_mean。

### 5.2 分层指标

事件评估除整 episode 三指标外，必须给事件窗分层：事件窗内（当前口径 900–1200s）/
窗口外分别统计 queue 类量（`tools/att004_cond_eval.py` 与 `tools/att004_baseline_eval.py` 已实现该口径）。

### 5.3 四层判定惯例

结论按四层分别陈述，不可互相替代：

1. **表征层**：z_task 语义探针（`tools/run_tarl_repr_probe.py` + `tools/analyze_repr_probe.py`：event_kind / event_junction / node_is_target / movement / speed_factor，balanced accuracy/F1，按 event_id 分组切分）。
2. **优化层**：冻结 eval 三指标（`tools/att004_cond_eval.py` / `tools/run_tarl_struct_cf.py`）+ 经典基线对照（`tools/att004_baseline_eval.py`）。
3. **语义效用层**：反事实条件分歧（canonical vs empty/wrong_location/wrong_type/both_wrong/foreign；`run_tarl_struct_cf.py` / `run_tarl_text_cf.py`）+ 文本效用（`run_tarl_text_utility.py`）。
4. **闭环/控制层**：事件条件下的真实决策收益（条件 eval 三指标差 + 事件窗分层）。

### 5.4 bench 把关口径

见 `bench/BENCH.md` 与 `bench/profiles/att_hz4x4.yml`：阶段①场景区分分类关卡不过线不进阶段②决策优化。

## 6. 启动与运行规范

1. 正式实验只允许两条路径：(a) 队列 runner + `run_queue_*.json`（历史工具：`tools/run_experiment_queue.py`、`tools/run_arterial_experiment_queue.py`、`tools/att004_build_queue.py` 产物 + 对应 `run_state_*` 看门狗）；(b) `bench/` profile 流程。**禁止新写一次性 `*_driver.sh`/`run_*.sh` 直拉正式 run。**
2. 一次性辅助脚本只进 `dev/<topic>/` 并登 `dev/INDEX.md`；历史散件归档 `dev/_legacy/`（移动非删除）。
3. run 必须在远端 tmux 内自治 + watchdog resume；断链不触发远端动作。
4. 正式 run 前必须 commit，禁止 dirty 树起正式跑；plan/lock 变更 → 新 plan_id + superseded_by 记录。

## 7. 资源档位

| 档位 | 机器 | 约束 |
|---|---|---|
| `gpu-heavy-34` | 34 单卡 8G | replay 峰值 ~20G RAM、ep200 ckpt ~3.6G、stagger≥420s、parallel≤5、mem 护栏 |
| `cpu-sim-73` | 73 四卡 | SUMO/CPU 瓶颈型训练，并发盯 load>18 错峰 |
| `eval-light` | 两机 | eval/探针类轻负载，可插缝 |

## 8. git 纪律

1. 架构优化/新建 = 必 commit（特性分支）+ AGENTS.md「架构变更日志」追加：日期/commit/改动/动机/影响臂与 profile。
2. 治理文件随仓 git；分支 `gov/<主题>`；push 主线需用户明确指令。
3. 新 tmux 会话名不得撞 `devin`/`devin2`；建议 `d34_*`/`d73_*`/任务前缀。
4. commit message 用中文。
5. **符号链接零容忍（P1 条款，2026-10-07 hitl 仓事故立）**：严禁符号链接入库
   （git mode 120000）；提交前 `git ls-files -s | awk '$1=="120000"'` 必须为空；
   merge/checkout/pull 前 `git ls-tree -r <ref> | awk '$1=="120000"'` dry-run，命中即停手
   ——gitignored 目录（runs//artifacts//data/output_data/）在 merge 下零保护，会被树内
   符号链接静默覆盖。
6. **产物快照**：runs//artifacts//data/output_data/ 等 gitignored 产物树须定期 rsync
   快照到仓外目录（orchestration/bin/snapshot_artifacts.sh）；快照是 run 级明细最后防线。

## 9. wiki 联动

仓侧 ledger/EXPERIMENTS.md = 事实源；wiki = 批次收官时的渲染投影（review-update 整页替换+归档）。
认领登记见 `artifacts/wiki_claims/README.md`（claims 是执行认领，历史补登不进 claims）。

## 10. 公网与安全

- 本仓 GitHub **public**：入库文档禁写内网 IP/主机名/跳板拓扑/含用户目录细节的绝对路径；
  机器只用代号 34/73；台账与文档一律仓内相对路径。
- `.secrets/`、凭据文件永不入库（`.gitignore` 已含 `/.secrets/`）。
- 保留/删除：永留 = plan/lock/summary/终评 ckpt/台账；可弃 = 中间 ckpt/wasted/临时日志；
  删除须授权；台账要能回答"哪些产物可弃"。

## 11. 旧约定作废对照表（追溯期执行）

| 旧约定 | 新约定 |
|---|---|
| docs/plan.md "remote policy = no push" | 以用户最新指令为准（特性分支可 commit；push 待指令） |
| 散装 `*_driver.sh`/`*.sh` 直跑 | 仅 queue+bench 两条合法路径（§6.1） |
| `wasted/` 废 episode | 并入原 run 的 notes 登记 |
| 工具脚本散落 tools/ 根目录 | 一次性脚本归 dev/ + INDEX.md；工具性可复用件留 tools/ 并在 tools/README.md 登记 |

## 12. 新 agent 上手 checklist

1. 读 `GOVERNANCE.md` + `AGENTS.md` + `EXPERIMENTS.md`（台账即状态）。
2. 认领登记（`artifacts/wiki_claims/`）→ 台账 REGISTERED → §6 合法路径执行 → §4 回填。
3. 改架构 = commit + 变更日志；改判据/口径属变更，台账开新 campaign，新旧口径不混排。
4. 拿不准 → "受阻"+原因，等裁决。

## 13. 边界

本规范只管工程流程与记录；**奖励定义/事件物理语义/评价指标/研究问题 = 科研边界，变更需用户确认**。
