# BENCH — simOfflineTSC 测试集 / 基准规范

> 可调用、可扩展的验证配置规范。不要求自动化框架；要求任何 agent 读规范即可正确调用。
> 训练过程记录（收敛曲线/干预统计/进度日志）与 eval 终值同为基准产物。
> 环境约定：conda env `colight`（AGENTS.md 开发流程节）；历史 run 另有冻结 env 指针，
> 见各 run manifest / 队列 command 字段。

## 调用契约（固定动作序列）

1. 身份五元组定稿（run_id + campaign + plan_id/config_sha + code_commit + runtime_env）；
2. `ledger/runs.jsonl` 追加 REGISTERED（派发前）；
3. SMOKE：≤2ep 冒烟通过后才进正式；
4. 远端 tmux 自治派发（会话名不撞 `devin`/`devin2`）；
5. 收割三指标 + 过程记录 → 回填台账 → 对照表；
6. 结果不达标不过线 → 不进入下一阶段。

## 注意力线两阶段（`bench/profiles/att_hz4x4.yml`）

### 阶段① 场景区分分类关卡

- **z_task 语义探针**：`tools/run_tarl_repr_probe.py --checkpoint <ckpt> --config <cfg>
  --agent tarl_attention --network hz4x4 --prefix <p> --output <dir> --seed <s>
  [--event-plan configs/events/plans/att004_t3_eval_v1.yml]` 产出
  `repr_records.npz` + `repr_labels.jsonl`；再用 `tools/analyze_repr_probe.py
  --dirs <dir1> [dir2 ...] --out report.csv` 得各任务 balanced-accuracy/F1
  （present/type/movement 见任务集 event_kind/event_junction/node_is_target/movement/speed_factor）。
  `TARL_TEXT_CONDITION=empty` 跑同型探针给"非文本通道"上界。
- **反事实区分度**：`tools/run_tarl_struct_cf.py` 在 canonical vs wrong_type /
  wrong_location / both_wrong / foreign 变体下产出逐决策 Q/action 分歧率。
- **held-out kind 泛化**：`--event-plan configs/events/plans/att004_t3_eval_v1.yml`
  与 `att004_t4_eval_v1.yml`（训练未见的事件布局/组合）。
- **产出 = 场景分类报告卡（按 event kind 分解）**：每 kind 一行 P/R/F1 + 探针 acc/F1 +
  反事实分歧率。**①不过线 → 不进阶段②。**

### 阶段② 决策优化

- **驱动**：`tools/att004_cond_eval.py --run-dir data/output_data/tsc/<agent>/hz4x4/<run>
  --episode <ep> --arm-config configs/tsc/att_entity_004/generated/a3s_condeval_e0.yml
  --reps 3 --sumo-base <seed> --seed <seed> --tag <tag>`。
- **条件**：`canonical / empty / wrong_location / wrong_type / both_wrong`（经
  `TARL_TEXT_CONDITION` 注入）+ `normal`（经 `TARL_EVENT_CONDITION`）——即
  canonical/empty/wrong-*/normal 全集。
- **固定 sumo seeds**：reps=3 → `{seed, seed+400000, seed+800000}`（沿用 att004 既有口径）。
- **ckpt 口径**：`<run_dir>/checkpoints/resumable/episode_0<EP>.pt`；cond eval 标准档位
  ep∈{20,50,100}，e200 臂另有 200。
- **三指标**：travel_time / throughput / unfinished_vehicles（强制同报）+
  事件窗（900–1200s）分层 queue。
- **基线对照**：`tools/att004_baseline_eval.py --agent {fixedtime,maxpressure,colight,mplight}
  --config configs/tsc/att_entity_004/generated/baseval_<agent>.yml --seed <s>
  --sumo_seed <ss> --prefix <p> --output <dir>`（RL 基线需 `--checkpoint` 其训练 run 的
  resumable ckpt；fixedtime/maxpressure 不需要）。

### 扩展方法

- 新 arm = 新训练 run + 复用同一 eval 条件矩阵（profile 不变，台账记 arm）。
- 新条件 = `tools/run_tarl_struct_cf.py` VARIANTS 增项或 `TARL_*_CONDITION` 新值 +
  profile yml `eval.conditions` 追加 + commit message 与 AGENTS.md 变更日志记录；
  新旧口径不混排，台账开新 campaign。
- 新事件计划 = `configs/events/plans/` 新增 plan（含 manifest）+ profile 引用。

## 既有证据位置（追溯期）

- 阶段①类：`data/output_data/analysis/att_entity_004/{probes,repr*}`、
  `tools/analyze_repr_probe.py` 产出的 report.csv 类文件、att003 `tarlrepr_*`/`*_probe` 目录。
- 阶段②类：`data/output_data/analysis/att_entity_004/{baseline_cmp_probe,single_cmp*}`、
  `artifacts/att_entity_004/{run_state_condeval,run_state_fixG_eval,...}`、
  `artifacts/sga_concat_colight_200_v1/eval_v{1,2}/`、
  `artifacts/att_entity_003/eval_*/`。

## 规则

- 比对只允许同 net + 同 eval 条件矩阵 + 同 seed 集 + 同 ngpu 档位；
  跨口径结果并排出现时必须注明口径差。
- profile 变更 = 改 yml + 本文件补节 + commit；进 AGENTS.md 架构变更日志。
- 阶段①与阶段②的证据不可互相替代（分类对 ≠ 决策好）。
