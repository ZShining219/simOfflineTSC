# cl_hz1x1 Q-锚定（Q-Anchor）半离线实验计划

状态：已收官（screen/main/o3/ρ/vdqn/snap3/snap3o/forget 全批次 DONE）。
设计定稿 2026-10-14，与用户确认λ筛选={1,10}两档、o2 先行。
正式实验记录见 `docs/experiments/2026-10-14_cl_hz1x1_qanchor.md`。

## 动机

cl_hz1x1 so1 已证明 25% 因果历史回放将跨场景最深退化压到 ≤9.4s，但残余未清零；
且 HA-SODQN E4 显示保持力随离线比例上升而适应代价上升（稳定性-可塑性绑定）。
参考 CLEAR（Rolnick et al. 2019）的行为克隆项，把"保持旧策略"从回放数量解耦为
对回放状态的输出锚定：TD 只约束被采动作，锚定约束该状态全部动作分布。

## 机制

stage>1 的训练 run 通过 overlay `model.teacher_paths`（scene→前一 stage
episode_0100.pt）加载各可见场景的冻结教师网络，逐 batch 在离线样本上加：

    loss = TD(online ∪ offline)
         + λ_π · mean_off KL(softmax(Q_teacher/T) ‖ softmax(Q_θ/T))
         + λ_q · mean_off Huber(Q_θ − Q_teacher)   # 仅非执行动作维（mask）

实现：`agent/dqn.py::_anchor_loss` / `agent/frap.py::_anchor_loss`；
教师加载 `_load_anchor_teachers`（读 ckpt `agents[rank].online_model_state_dict`）。
诊断：metrics records.jsonl 新增 schema v5 `train_diagnostics`（td_loss /
anchor_policy_loss / anchor_value_loss / offline_count）。

## 矩阵与判据

- screen：soddqnp λ_π∈{1,10} + soddqnpv_probe（λ_π=1, λ_q=0.1），o2 × seed{0,1,2}
  → `run_queue_soa_screen.json`（306 任务）。选 λ 依据：最深退化最小且对角线
  ΔATT 不恶化（>+1s 视为锚过强）。
- main：{ddqn,duel,frap}×{+P,+P+V}@r25×o2×3seed（18 链/612 任务），λ 用 screen 结果。
- 对照基线直接复用 so1（soddqn/soduel/sofrap，λ=0 同协议）。
- 呈现：`matrix_report` 同构指标——worst-case ΔATT、retention trajectory、
  对角线 ΔATT、AULC；机制面板补 anchor_loss 曲线 + drift/action-flip。

判读协议（预先声明）：worst 退化压向 ≤3s 且对角线不变=有效；drift 降但 ATT 无差
=ATT 已饱和，转低比例 stress 臂；对角线恶化=λ 过大降档；与纯回放不可区分=λ 过小。

## 执行

builder：`dev/cl_hz1x1/build_queue_soa.py`（smoke/screen/main 阶段化）；
台账行 `artifacts/cl_hz1x1/ledger_rows_soa_<phase>.jsonl`；
队列经 `tools/run_experiment_queue.py` 派发（该 runner 已补 depends_on/requires
门控，此前本地 runner 不检查依赖会导致抢跑）。
