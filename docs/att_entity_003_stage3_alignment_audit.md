# ATT-ENTITY-003 阶段三设计审计：Event–Traffic / Event–Transition 对齐目标

依据：任务页 review-v5「下一步」的设计审计前置门槛；用户设计稿（CAREL-style + GRIF-style 两条对齐机制）。本稿只做设计审计，不写实现代码；通过后按实验 A → 验证 → 实验 B 顺序执行。

状态：draft 待用户/审核确认。

## 0. 论文目标复核（对原论文机制的事实性确认）

### CAREL（Saghafian et al., TMLR 2025, arXiv:2411.19787；代码 github.com/ArminS03/CAREL）

- 场景：instruction-following RL（BabyAI/MiniGrid/SHELM）。问题陈述与设计稿一致：纯 RL reward 不足以学出语言↔环境 grounding。
- 实际机制（比设计稿更具体，影响我们的实现选择）：
  - 只对**成功轨迹**施加辅助损失（reward ≥ max 的比例阈值过滤）；RL 主循环仍用全部数据。
  - 相似度不是单一 InfoNCE，而是 X-CLIP 式**四粒度组合**：episode↔instruction（全局）、episode↔word、observation↔instruction、observation↔word（带 AOSM 双层注意力聚合），s = 四项均值，再做对称 InfoNCE（ep→instr 与 instr→ep 双向）。
  - observation 表示里**加 action embedding**（positional 式）；instruction 表示逐 token。
  - instruction tracking 用对齐分数在 episode 内遮盖已完成指令部分——**不迁移**（我们的事件不是可逐步完成的子任务序列）。
- 设计稿中"instruction↔traffic trajectory 对齐 + InfoNCE"的概括与原文相符；我们只需要其**episode↔instruction 全局粒度 + 对称 InfoNCE**骨架，逐 token/逐帧细粒度在我们的场景下没有对应物（事件文本是单条报告，不是多 token 指令序列——先按整句向量处理；若 BERT token 级对齐值得做，列为可选增强而非首版）。

### GRIF（Myers et al., CoRL 2023, PMLR 229:3894；arXiv:2307.00117；代码 rail-berkeley.github.io/grif）

- 实际机制：双向 InfoNCE，cosine 相似度 C(s,g,ℓ)=cos(f_φ(ℓ), h_ψ(s0,g))；f_φ 语言编码器，h_ψ 为 **transition encoder**（输入 (s0, g) 状态对，实现上用改造 CLIP 图像编码器吃两张叠图）；正样本=同条轨迹的 (start, end, 标注)，负样本双侧采样（随机他轨迹 transition 对 + 随机他指令），各采 k 个。
- 对齐后策略在所有（含无标注）数据上以该 task embedding 为条件训练。
- 我们的语义反转成立：GRIF 是"指令=想造成的状态变化"，我们是"事件=外生造成的状态变化"；对齐对象同为"状态变化"而非绝对状态。policy-conditioning 框架不迁移（我们的事件表示进 FLX 支路的既有路径不变）。

## 1. 项目侧可用张量/数据盘点

| 数据 | 位置 | 形态 |
|---|---|---|
| 事件表示 z_e | `EventSceneRepresentation.z_events` / `SceneCondition.event_embeddings`（`encode_context_events` 产出，BERT text + structured meta + fusion） | [B, M, 128]，事件级 |
| 事件→节点归属 | `node_event_mask` [N,M]（直接接地）+ `node_event_relation` [N,M,3] | 已有，Gnode 绑定器 |
| 每步节点交通特征 | `obs`（`_network_input`：每路口 lane 特征行 + phase 块）；原始口径还有 `world.get_lane_vehicle_count / lane_waiting_count / lane_delay` | obs 行 [N, lane_dim]；phase 块在尾部需剔除 |
| 逐步 transition | replay 项含 (obs_t, obs_t1, scene_t, scene_t1) | 单步 10s，无多步窗口 |
| 评估期窗口特征 | eval 包 records.jsonl 的 DECISION_METRICS（queue_intersections、delay、lane_vehicle_counts、pressure 等逐决策步） | 离线 retrieval 探针可直接用 |
| 事件时间窗 | SceneContext/events 有 begin/end；episode 内 sim_time 可得 | event_steps 口径已用 |

**关键结构事实：replay buffer 只有单步 transition，没有事件窗口轨迹。** 因此 z_τ（交通窗口表示）不能从 replay 采样构造，只能在 **episode 在线滚动时累积**（agent 在 remember/每步已拿到 scene_t 与 obs），episode 结束汇成 (事件, 窗口特征序列) 对。这是本设计最重要的落点约束。

## 2. 实验 A：CAREL-style Event–Traffic Alignment 设计

### 2.1 数据构造（episode 侧在线累积器）

- 每决策步（10s）：若 scene 有活动事件，记录 {event_id, event payload(text/meta 引用), 受影响节点集合(node_event_mask 为真的节点), 该节点集合的 lane 特征行（剔除 phase 块）, sim_time}。
- episode 结束：对每个事件 e 取窗口 W_e = [begin_e, min(end_e, begin_e + 600s)]（对齐上限 600s，与事件时长上限一致），窗口内该事件受影响节点的特征序列 → TrafficWindowEncoder → z_τ。
- z_e 侧：episode 末用**当前参数重编码**事件文本/meta（payload 已存），保证梯度经 SceneContextEncoder 回流；窗口特征为常量输入。

### 2.2 正负样本

- 正样本：(e 的 z_e, e 的受影响节点窗口 z_τ)。
- hard negatives（同一 InfoNCE 分母内全部使用）：
  1. **同事件错位位置**：e 的窗口换到未受影响节点（wrong-location 空间版）；
  2. **同位置无事件期**：同节点在不重叠的事件窗口外取等长窗口（时间版负样本，防"位置识别"shortcut）；
  3. **同 batch 他事件窗口**（跨事件负样本）；
  4. **同类型不同位置**事件对（跨 episode/batch，防"位置→文本"查表）；
  5. 乱序语义：wrong-location 文本变体 ↔ 原窗口（复用探针既有扰动生成器）。
- multi-event 归属：每个事件只与其直接接地节点的窗口配正样本；两个事件共享节点时按窗口内该节点特征对两事件**同时**配对（信息真实共享），但在 retrieval 评估中把该节点计为两事件的合法答案，避免假阴性。
- 全局事件（global_rain，scope=network）：接地节点=全网，窗口=全网均值特征序列；单独成组评估，不与局部事件混排（其"位置"语义不同）。

### 2.3 损失

z_e 与 z_τ 分别 L2 归一，cosine 相似度、温度 τ=0.07；对称 InfoNCE（z_e→z_τ 与 z_τ→z_e）：

L_align = −mean_i [ log exp(s_ii/τ) / Σ_j exp(s_ij/τ) + log exp(s_ii/τ) / Σ_j exp(s_ji/τ) ]

L = L_TD + λ · L_align，λ ∈ {0.1, 1.0} 起步（设计稿 {0.01,0.1,1.0}，0.01 档预期太弱，先两档；若 L_align 量级与 L_TD 差 >2 个数量级，按梯度范数比校准 λ）。

### 2.4 施加节奏（对 CAREL"仅成功轨迹"的适配决策）

CAREL 的成功过滤在我们场景没有直接对应物（TSC 无"成功 episode"阈值）。**首版：所有含事件 episode 都产对**（事件在场=该轨迹确实携带事件信息，等价于 CAREL 的"轨迹确实与指令匹配"前提）；每个 `train()` 从对齐缓冲（近 K=32 episode 的对）采样构造 InfoNCE batch，与 TD 更新同节奏。备选：按 episode reward 分位过滤——列为审计项 A-4 的开问题，不作为首版默认。

## 3. 实验 B：GRIF-style Event–Transition Alignment 设计（仅 A 完成后启动）

- z_ΔH = TransitionEncoder(H_before, H_after)：H_before=事件 begin 前一个决策步的受影响节点特征，H_after=事件 end（或 begin+600s 截断）时同节点特征；encoder=MLP([H_b, H_a−H_b, H_a])。
- **剔除动作/phase 特征**：H 只用 lane 交通量（count/waiting/delay 行），不碰 phase 块与 action——防"对齐到策略响应"混淆。
- 负样本同 A 结构，另加**同事件无对应变化**（no-event 窗的 Δ）显式为负。
- 风险注记（审计项 B-1）：ΔH 同时含事件物理效应与信号控制响应；若 ΔH 学到"该位置策略习惯动作的后果"而非"事件语义"，retrieval 会虚高。审计手段：对比 z_ΔH 在"同事件不同 strategy 阶段 checkpoint"下的稳定性，以及对固定相位 baseline 数据的重放检验。

## 4. Leakage / Shortcut 审计清单（实现期必过）

| 风险 | 检验 | 处置 |
|---|---|---|
| 位置查表：z_τ 靠"哪个节点"识别事件 | negatives 含同型异位 + 同位无事件；retrieval 测试在 held-out (位置,时刻) 组合上评 | 若 retrieval 在 held-out 位置掉到随机→判为 shortcut，负样本补同位无事件权重 |
| 时刻泄漏：窗口特征含绝对时间 | TrafficWindowEncoder 输入剔除时间戳；窗口内只放交通量 | 实现期断言输入维度 |
| phase/动作混入 ΔH | z_ΔH/z_τ 输入维度白名单=lane count/waiting/delay | 同上断言 |
| 事件身份经 text 侧泄漏到窗口侧 | 窗口特征不含事件 id/文本；正确性由归属 mask 保证 | 代码评审点 |
| 跨 episode 负样本不足 | 对齐缓冲 K≥32 episode 且覆盖 ≥3 事件类型 | 缓冲统计日志 |

## 5. 验证口径（四层，全部产出后才下结论）

1. **Representation**：held-out eval episode 上做 retrieval——z_e 在候选窗口集（含 hard negatives）中 Top-1 / Top-5 / margin；同时报 alignment loss 轨迹。retrieval≈随机 → 判负，不进入闭环解读。
2. **语义反事实**：沿用现有探针三变体，增报"correct vs wrong_location 的窗口对齐分数差"（语义方向性），不只报动作变化率。
3. **闭环**：randeval_v1 12 条件 + none/all，ep200，5 seeds；事件窗口/affected 节点/single|multi 分层；Δ vs CoLight 与 vs 纯 FLX 的配对差 + CI。
4. **Normal 安全性**：none 条件 + 无事件 episode 下 ≈ CoLight（复用现有判定线）。

预算建议（阶段三 A 实验）：λ pilot 2 seeds×2 档=4 短训定位量级 → 主矩阵 C2 λ* ×5 seeds×200ep=5 训 + 评估 5 集合；C0/C1 复用阶段二已有结果，零成本。

## 6. 代码落点清单（预计改动，实现阶段执行）

- 新增 `agent/scene_alignment.py`：`TrafficWindowEncoder`（均值池化+MLP→128d）、`AlignmentPairBuffer`、对称 InfoNCE。
- `agent/colight.py`：rollout 侧每步累积窗口特征（remember 路径已知 scene_t/obs）；`train()` 加 λ·L_align（复用 `_log_branch_diagnostics` 的日志通道，新键 align_loss/retrieval_top1）；配置键 `sga_align_*`。
- `utils/logger.py`：评估白名单对 align 变体名的接纳（如沿用 sga_flx_colight + 配置开 关，则无需动）。
- 配置：`configs/tsc/sga_flx_colight_text_event_random_200_align*.yml`（自包含）。
- 测试：`tests/test_scene_alignment.py`（encoder 形状、InfoNCE 单调性、无事件空对、归属规则、白名单输入断言）。
- 探针复用：`tools/run_sga_counterfactual.py` 增报窗口对齐分数字段；retrieval 评估新脚本 `tools/align_retrieval_eval.py`（读 eval records.jsonl + 冻结 checkpoint）。

## 7. 不做/延后

- CAREL instruction tracking、逐 token/frame 细粒度对齐：不迁移/可选增强。
- GRIF 的 policy-conditioning 框架与 CLIP 改造：不迁移。
- CAREL+GRIF 同叠：禁止。
- 预告文本（B）与难例分布（C）：保留为"alignment 显著但闭环零收益"后的转向分支。

## 8. 判负边界（预注册）

- retrieval Top-1 在 held-out（位置,时刻）组合上 ≈ 随机 → 表示层失败。
- retrieval 显著 + 反事实语义方向性显著 + 闭环事件窗口 Δ CI 跨零 → "grounding 成功未转化"，记负结果并转 B/C 分支。
- normal 条件退化超既有噪声线 → 融合干扰回归，回退 λ 或归因到对齐目标与 TD 的梯度冲突。

---

## 附录：实现落点与 pilot 早期观察（执行记录）

### 实现落点（已验证）

- `agent/scene_alignment.py`：`EpisodeAlignAccumulator`（rollout 内 remember() 逐步累积原始 lane obs 行 + 每事件接地节点 mask + 活跃 step 列表；episode 末产出 `EventAlignPair`：正样本=事件活跃窗口×接地节点均值池化，显式负样本=wrong_node（同窗口未接地节点）+ no_event（同节点的连续无事件段））、`TrafficWindowEncoder`（mean/max/std/last−first 统计 → MLP → L2 归一）、`symmetric_infonce`、`AlignmentBuffer`（32 episode 滑动池）。
- `agent/colight.py`：`ColightNet` 在 `sga_align_lambda>0` 时挂 `align_window_encoder`（仅辅助损失使用）；`SGAFlxColightAgent.remember` 驱动累积器、`train()` 中 `loss += λ·align`、诊断行新增 `align_loss/align_pairs`；`_encode_events_for_align` 用当前参数重编码 text+meta→fusion→L2。λ=0/缺省时行为与基线完全一致（已回归）。
- 配置：`sga_flx_colight_text_event_random_align_l{0p1,1p0}.yml`（与 stage-2 random-200 全同，唯一变量 sga_align_*）、smoke 配置与 `hz4x4_align_smoke_v1` 计划、探针配置 `sga_flx_align_probe.yml`。
- 测试：`tests/test_scene_alignment.py` 19 项全过（窗口归属、多事件分离、global rain、连续无事件负样本、窗口上限、双编码器梯度回传、无事件空产出、缓冲上界）。
- 工具：`tools/run_align_retrieval.py`（冻结检查点检索探针：Top-1 / margin / 分负样本类型）；`tools/flx_eval_report.py` 扩展 align 方法名与对照。

### Pilot 早期事实（待 ep200 + 检索确认）

- align_loss 在训练缓冲上 ~几十 episode 内饱和至 ≤0.01（λ 两档同）；事件对按 episode 累积正常（4→8）。
- ep0→ep50 权重移动：window_enc |ΔW|=42.4（仅对齐梯度可达）；fusion |ΔW|=49.7（init 范数 14.1，+250%）——对齐梯度通路真实、强度大。
- 冻结 ep50 检查点在 all.yml 留出 episode 检索：**top1=0/3、margin_all=−0.41**（ep0 未训练基线 −0.06）——缓冲内饱和但留出检索退化，提示存在记忆化/shortcut 成分；待 ep200 与更多 episode 复核后定性。
