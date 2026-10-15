# cl_hz1x1 半离线输出锚定（Q-Anchor）战役

> 状态：已完成
> 发起日期：2026-10-14　｜　负责人/Agent：Devin
> 最近更新：2026-10-15（对齐远程整理线时补录正式记录）

## 关联

- campaign：`EXPERIMENTS/cl_hz1x1.md`（台账行 `line=cl_hz1x1`，批次
  `soas/soa1/soa3/soar/soav/snap3/snap3o/drift` 共 5619 行 DONE）
- 队列文件：`artifacts/cl_hz1x1/run_queue_soa_{screen,main,main_o3,rho}.json`、
  `run_queue_soa_snap3.json`；manifest 清单
  `artifacts/cl_hz1x1/{snap3o_eval,forget_eval}/collection_manifest.json`
- code_commit：39f29f1（机制）→ e7a21ed（soa1 收官）→ 3415b64（o3 复现）→
  5928d41（ρ 剂量）→ b321206（vdqn 对照）→ 2c6c32d（适应分析）→
  859ab8f+e582ef5（snap3）→ 02e790a+0ed1d42+06cc456（snap3o 微观）→
  b3b09bb（forget_eval + frap 白名单）→ b6e5140（动机图）
- 产物目录：`artifacts/cl_hz1x1/`（metrics json / fig_* / 评估包）

## 1. 改动意图

cl1+so1 已证明 25% 因果半离线回放把跨场景最深退化从 +597s 压到 ≤9.3s，
但残余未清零，且回放比例本身是稳定性-可塑性绑定的粗旋钮。本实验引入
CLEAR（Rolnick et al. 2019）式**输出锚定**：把"保持旧策略"从回放数量
解耦为对回放历史样本的分布级约束，验证三件事——

1. 锚定能否把残余退化压向零且不以新场景塑性为代价；
2. 回放比例 ρ 与锚定强度 λ 的剂量-响应结构；
3. 机制层面：入场冲击与旧场景遗忘各自是什么微观形态，锚定如何改变它。

## 2. 改动模块

- `agent/dqn.py` / `agent/frap.py`：`anchor_lambda_policy/value`、
  `anchor_temperature`、`teacher_paths`、`_load_anchor_teachers`、
  `_anchor_loss`（KL(softmax(Q_teacher/T)‖softmax(Q_θ/T)) 于历史样本 +
  可选非执行动作维 Huber 值锚定；TD 照常作用于全 batch）
- `utils/logger.py`：metrics schema v5 `train_diagnostics`；
  eval-manifest 清单校验与加载白名单补 `frap`（b3b09bb）
- `run.py`：`EvaluationManifestRunner`（冻结 ckpt 同 eval seed 逐决策步
  DECISION_METRICS 通道）+ `model_snapshot` overlay 键（裸 model/{e}_0.pt
  eval-only 加载）
- `trainer/tsc_trainer.py`：`load_model_snapshot`、per-decision 记录回调
- `tools/run_experiment_queue.py`：depends_on/requires 门控修复
- `arterial/replay.py`：历史归档/混合采样为 so1 既有设施，本实验复用
- `dev/cl_hz1x1/`：队列 builder（build_queue_soa、build_snap3o_manifest、
  build_forget_manifest）、收割（harvest_soa）、绘图（plot_soa_*、
  plot_snap3o_micro、plot_forget_micro、plot_motivation）、诊断
  （drift_diag、grad_diag、fqe_probe）；全部登记 `dev/INDEX.md`

## 3. 实验设计

- 场景/序列：hz1x1 同质 4 场景包 `s1/c4/c2/c3`；`o2=s1→c4→c2→c3`，
  `o3=c2→c3→s1→c4` 跨序复现；每 stage 100 episode
- 臂：`clear`（纯在线）/`so`（ρ=0.25 因果回放）/`+P`（回放+λπ 策略锚）/
  `+P+V`（再加 λq=0.1 值锚）；screen 筛 λπ∈{1,10}；ρ∈{0.125,0.25,0.5}
  剂量臂；vanilla DQN 对照（soav）
- 骨干：ddqn / duel / frap；训练种子 {0,1,2}；冻结评估种子 {101,102,103}
- 指标口径：stage 间冻结评估矩阵（worst_old_degradation / 对角线 ΔATT /
  AULC）、ep0-20 入场成本、drift 诊断（公共参考池 q_l1/qc_l1/贪心翻转率）、
  逐 10s 微观时序（queue/累计吞吐/逐 lane 热力）
- 判定（预声明，docs/plan_qanchor_clhz1x1.md）：worst 退化 →≤3s 且对角线
  不恶化（>+1s 判锚过强）；与纯回放不可区分判 λ 过小
- ckpt 口径：DQN 系 `model/{e}_0.pt`=target 网快照（snap3 注明），
  resumable payload=online 权重（snap3o/forget_eval 使用），frap=online

## 4. 实验结果

**主矩阵（o2，相对同臂基线 worst 旧场景退化）**：

| 臂 | worst ΔATT | 对角线最大 ΔATT |
|---|---:|---:|
| so（回放） | +3.8（ddqn）/ +6.0（duel）/ +9.3（frap） | ~0 |
| **+P（λπ=1）** | **≤+0.99**（9 链最大 soddqnpv_o2_sd1 +1.11 属 +P+V 臂；+P 臂最大 +0.99） | ≤+1.4 |
| +P+V | ≤+1.50 | ≤+1.5 |

- **o3 跨序复现**：+P 全链 worst ≤+0.56s，结论不依赖顺序
- **ρ 剂量-响应**：锚定可省回放预算，ρ=0.25 为甜点（ρ12 残余略升、ρ50
  塑性税加重）
- **入场冲击**：o3 st3（c3→s1）clear ep0 冻结 ATT 236-243s——微观解剖
  （snap3o，online 口径同 seed 逐 10s 时序）定位为**单方向相位饿死持续
  整 episode、吞吐 −37%**；回放/锚定臂消除入场灾难，但新场景轴存在
  间歇性中段回弹（诚实边界，非单调占优）
- **遗忘微观解剖**（forget_eval，s1@es101，39 controller）：frap sd1
  +9.79s 退化**全部发生在 st4 前 50 episode**（77.2→86.8→87.0），形态为
  进口道组相位**平衡漂移**（吞吐无损 1974 vs 1978）；+P 臂同链位全平
- **漂移机制**（drift_diag）：公共池 |ΔQ|1 沿链 clear 197 vs so 122，
  翻转率 .60 vs .47；去中心化 qc_l1 两臂接近——回放压住漂移幅度与
  翻转率而非消除形状漂移
- **佐证图**：`fig_soa_{compare,panels,episodes,rho,adapt_*}.png`、
  `fig_snap3o_{queue,throughput,laneheat,conv}.png`、
  `fig_forget_{queue,throughput,laneheat,att}.png`、`fig_cl_motivation.png`

## 5. 结论与后续

判定：**支持**。+P 锚定（λπ=1, ρ=0.25）在既有半离线链路上把最坏旧场景
退化压至 ≤1s 量级且对角线近乎无损；机制图显示其作用面是切断"分布漂移
→相位分配失衡→性能崩塌"的传导，而非阻止漂移本身。

边界与待办：
- 结论限于 hz1x1 同质拓扑 1×1；PPO 为 on-policy 不吃离线回放（未入臂）
- `agent/shared_dqn.py` 未接 anchor——arterial 泛化线若启用需补实现
- 锚定配置仅存在于 dev builder overlay，未沉淀 `configs/` 示例 yml
- 新场景中段回弹、qc_l1 形状漂移两臂接近——caption 措辞按
  "contains/bounds" 而非 "prevents"
