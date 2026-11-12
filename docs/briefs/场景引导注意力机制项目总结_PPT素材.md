# 场景引导注意力机制项目总结（PPT 素材版）

> 更新日期：2026-10-29  
> 用途：把仓库中与“场景引导注意力”有关的研究线、代码结构、实验结果和边界整理成可以直接拆分为 PPT 页面的素材。  
> 证据口径：代码说明“做了什么、怎么做”；smoke/导入验证说明“能否运行”；正式 run 和冻结评估说明“得到什么结果”；四层判定说明“结果能支持到什么程度”。

## 1. 一页执行摘要

项目要解决的问题是：交通信号控制器通常只看到车道队列、车辆数和等待时间，无法可靠利用“某一条道路发生了什么、事件处于什么阶段、应该关注哪一组上下游节点”等场景语义。项目把事件播报变成结构化、可定位、可审计的场景条件，并用注意力将场景信息连接到路口节点或车道 token，最后让控制器在原有交通状态决策基础上进行受约束的场景适应。

当前有两条需要在汇报中同时交代的线：

| 研究线 | 主要问题 | 当前状态 | 最适合 PPT 的说法 |
|---|---|---|---|
| `att_entity_002–004` | 多路口事件语义怎样进入 TARL/CoLight/SGA，是否真的被定位、识别并改变行为 | `att_entity_004` 为仓库当前注意力主线，参考臂 `a3s`，最新修复臂 `fixG` | “从文本注入推进到结构化 `z_task`、辅助监督和反事实验证” |
| `scene_attention/` `sig9–16` | 单个信号控制器怎样把场景注意力直接用于相位决策 | `SceneJointPolicy + SGA + 有界残差` 为 canonical；`sig15` 为主要证据包 | “从调度器选择控制器，收敛到注意力直接修正控制器 Q 值” |

当前最稳妥的结论有四层：

1. **表征层已成立**：结构化事件槽位被写入目标节点，held-out 探针可以识别事件身份；文本/事件条件确实进入模型，而不是完全被交通状态替代。
2. **优化层出现正向信号**：`fixG` 相比 `a3s` 收敛更快，`sig15` 相比同架构 no-text 的 corridor queue integral 下降约 8.6%。
3. **语义效用层尚未完全翻正**：事件语义能改变注意力和部分动作，但“有文本”相对 empty 的净收益仍小，某些事件还会带来窗内代价；强基线 MaxPressure 在 throughput 上仍更高。
4. **闭环层尚未完成**：road closure 的绑定覆盖和事件效用仍有边界，不能把当前结果写成“场景注意力已经普遍优于传统控制器”或“已经证明跨路网泛化”。

## 2. 背景、问题和研究假设

### 2.1 为什么只用交通状态不够

交通状态是结果变量：队列变长、等待时间增加时，拥堵已经发生。事件播报可能在车辆状态完全显现之前提供类型和位置，例如“道路 A 发生拥堵”“道路 B 将出现流量激增”“某路段正在恢复”。控制器需要回答三个问题：

- **看哪里**：报告中的道路、路口或事件实体对应哪些控制节点/车道；
- **理解什么**：事件类型、活动/恢复状态、严重度、剩余时间等语义；
- **怎样用**：是否改变相位、向上游馈入边扩散关注、保持下游排空，或在不可信时保持原控制器行为。

如果把事件信息简单拼在所有节点后面，模型可能学到事件存在，却不知道事件位置；如果把事件作为硬 mask，则可能阻止模型表达上游、下游和走廊效应。项目因此采用“结构化字段 + 软 grounding + 注意力 + 原始交通残差”的组合。

### 2.2 研究假设的演进

| 阶段 | 假设 | 主要检验 |
|---|---|---|
| ATT-ENTITY-002 | SGA/concat 将事件文本送入 CoLight/MPLight 后，event 条件应优于 normal 条件 | 文本注入矩阵、event/normal、反事实条件、H2/H3 |
| ATT-ENTITY-003 | TARL 的文本-实体 alignment 和传感矩阵能提升事件场景鲁棒性 | alignment 权重、随机事件分布、留出检索和反事实 |
| ATT-ENTITY-004 | 结构化 `z_task` 加语义辅助监督，比纯文本注入更能同时保持场景区分和决策优化 | node-level `z_task`、SemanticAuxHead、holdout、canonical/empty/wrong_type、eval_v2 |
| `scene_attention` sig9–16 | 场景注意力应直接进入单控制器的相位 Q 值；初始时保持已训练 DQN 的行为，事件相关时再有限度修正 | adapter 蒸馏、原始 reward TD、no-text、lie/none、attention probe、跨场景 S1/S3/S4 |

## 3. 方法演进时间线：从“注入文本”到“控制器内化场景”

### 3.1 ATT-ENTITY-002：先回答“文本是否进入模型”

这一阶段建立了 `sga_colight`、`concat_colight`、`sga_mplight` 等臂，比较文本事件、无事件和不同注入方式。正式登记 2291 个单元，其中包括 24 个有效训练和 48 个评估包，结论记录为 **H2 支持、H3 未支持**。

这一步的价值是证明接口和实验矩阵可运行，同时暴露两个问题：

- 文本事件进入网络，不等于控制器可靠地使用了事件的类型和位置；
- 如果 lane token 自己已经带有由报告生成的 event flag，文本位置会被重复编码，模型可能依赖 shortcut。

### 3.2 ATT-ENTITY-003：加入 alignment 和传感矩阵

这一阶段围绕 TARL 做 `align_trans`、`align_trans_meta`、FLX/GRIF-style alignment 和随机事件分布矩阵，共 20 个训练 run 和 20×14 评估包，并完成文本反事实探针。

alignment 机制已经实现并通过 19 项工程测试，但早期留出检索出现 top1=0/3，说明模型可能记住训练分布或事件模式，不能把 alignment 写成“已证明提升泛化”。这一步的正确表述是：**alignment 作为可训练、可检查的语义约束已经落地；其科研收益需要看后续严格的 held-out 和跨场景证据。**

### 3.3 ATT-ENTITY-004：结构化 `z_task`、SemanticAuxHead 和四层判定

这一阶段把事件报告拆成 per-node 的结构化槽位，直接写入节点特征，同时增加语义辅助头。配置族位于 `configs/tsc/att_entity_004/`，当前参考臂为 `a3s`，修复线为 `fixG`。

`z_task` 的基本思想是：对每个控制节点保留一行事件条件，非目标节点为零或无事件编码，目标节点携带事件生命周期/类型/位置相关字段。事件计划、目标节点和控制输入分开记录，便于离线审计。事件族扩展时通过 `model.ztask_vocabs` 显式扩大词表，例如 surge 臂使用 `[3, 5, 4, 5, 6]`。

SemanticAuxHead 不直接替代控制损失，而是要求隐藏表示保留可解码的事件语义。仓库台账跟踪 `aux_present_acc`、`aux_type_acc` 和 `aux_loss`；配置中的典型设置为 `semantic_aux: {enabled: true, lambda: 1.0}`。顶层 `agent/tarl.py` 是注册 shim，实际 TARL 实现由项目的 `reproduction.tarl_tsc.parent_adapter` 接入；因此 PPT 中应把它画成“结构化事件输入 → TARL attention → 控制头 + 语义辅助头”。

### 3.4 `scene_attention`：从 router 试探收敛到单控制器 SceneJointPolicy

早期 `DispatchRouter` 是“两层决策”：第一层根据场景选择控制器，第二层由被选控制器输出相位。该 router 代际已被否决，代码保留在 `scene_attention/_legacy/router/` 作为历史快照和 HITL 调度工具，不能作为当前训练主线。

当前主线把注意力放进一个信号控制器：

```text
车道物理状态 + 车道身份 + 全局相位状态 + 事件报告
        │
        ├─ lane self-attention
        ├─ report → lane cross-attention
        ├─ edge identity binding
        └─ 原始交通残差旁路
                 ↓
              场景表示 z、注意力 w
                 ↓
        已训练 DQN 的 q_base + 有界 gated residual
                 ↓
              相位动作
```

`sig9_joint_bc` 用 adapter-teacher 产生带场景注意力的动作示范，`sig10_joint_rl` 再用原始控制 reward 做 Double-DQN TD 微调；`sig11` 是无意图对照，`sig12–14` 探索单边意图和走廊意图，`sig15` 形成 corridor-load 主链，`sig16_noaux` 去掉辅助监督。

## 4. 当前实现的输入和数据流

### 4.1 事件报告不是自由文本，而是结构化报告

`scene_attention/dispatch_state.py` 中的 `FeatureBuilder` 统一训练、评测和 live-broadcast adapter 的特征定义。报告字段包括：

| 字段 | 含义 | 是否直接可见 |
|---|---|---|
| `kind` | 事件类型，例如 surge、blockage、closure | 是，one-hot |
| `edge` | 报告涉及的道路边 | 是，与 edge identity 共用索引空间 |
| `active/recovering` | 事件生命周期 | 是 |
| `onset_age_s` | 报告已持续时间 | 是 |
| `severity` | 严重度 | 是 |
| `est_remaining_s` | 估计剩余时间 | 可选；在无 forecast 版本中关闭 |
| `objective` | 内部意图/目标标记 | 由 intent 编译器使用，不应被解释成自然语言指令 |

报告同时渲染成人类可读句子用于日志和 UI，但模型主要读结构化字段。`report_delay` 可模拟播报延迟；`include_forecast=False` 可避免把未来信息作为报告内容泄漏给模型。

### 4.2 车道 token 与身份绑定

每个 lane token 至少包含归一化的 queue、count、waiting-time；`phys='queue'` 只保留 queue，`phys='full'` 保留完整三类物理特征。`layout='identity'` 时，lane token 追加 edge one-hot，报告也在同一个 edge 索引空间中编码。这样“报告指向哪条边”只来自报告字段和共享身份空间，而不是由报告生成一个重复的 lane flag。

旧的 `layout='flag'` 会把 report-derived event flag 再写回 lane token，形成渠道重复；它保留用于历史 checkpoint 和消融，不是当前推荐设计。

### 4.3 走廊意图的两个集合

`FeatureBuilder.intent_vec()` 将现象报告编译成边级意图 `theta_reward`：报告边负责处理，启用 `intent_coord_w` 时向 feeder 和 discharge 边扩散权重。`intent_watch()` 构造更宽的观察集合 `theta_watch`，包含事件边及其上游 approach 边。

两者有意分开：

- `theta_reward` 是控制目标，进入 reward 或评估时的责任集；
- `theta_watch` 是辅助监督目标，告诉注意力“哪些边应该被观察”，不等于必须对这些边采取同一个动作。

这一分离是 sig15 的关键：注意力需要观察走廊，但奖励不应把所有相邻节点都当成同等责任目标。

## 5. 注意力模块的具体计算

### 5.1 `agent/scene_attention.py`：通用场景到节点 SGA

这一模块适用于 CoLight/MPLight/TARL 的节点特征。给定交通节点表示 (H\in\mathbb{R}^{B\times N\times D}) 和场景表示 (z\in\mathbb{R}^{B\times D})：

\[
q=W_q\operatorname{LN}(z),\quad k_i=W_k\operatorname{LN}(h_i),\quad
\alpha_i=\operatorname{softmax}_i(s\,\cos(q,k_i)/T).
\]

注意力上下文为 (c=\sum_i\alpha_iW_vh_i)。门控残差再根据节点状态和上下文生成：

\[
h'_i=h_i+\rho\,g_i(h_i,c)\,W_o h_i.
\]

这里的 `alpha` 是可记录的节点权重；`residual_scale` 控制场景分支的最大影响。无事件时，`MultiEventSceneGuidedAttention` 返回交通特征的精确 identity，避免空报告改变原控制器。

多事件版本对每个 event-node pair 计算独立 relevance，不把所有事件强制归一化成一个概率分布。grounding mask 作为 soft bias，relation embedding 表达上游/下游关系，模型仍然可以把影响传播到未被报告直接点名的节点。

`SceneCrossAttentionResidual` 使用“节点为 query、事件为 key/value”的多头 cross-attention，并将输出线性层零初始化：

\[
U=\operatorname{softmax}(QK^T/\sqrt d+b_{ground}+b_{relation})V,\quad
H'=H+P(\operatorname{LN}(U)).
\]

零初始化使模型刚开始等价于原交通表示，但输出头仍可在第一步获得梯度；这比把整个分支乘以零标量更安全，因为后者可能让分支在初期没有梯度。

### 5.2 `scene_attention/dispatch_model.py`：canonical SGA 编码器

`SceneAttention` 的输入是 `lanes [B,L,F]`、`text [B,T]` 和 `global [B,G]`。其主要步骤如下：

1. `lane_proj` 把 lane token 映射到 `d_model`；
2. `text_proj` 把结构化报告映射成 query；
3. `geom` 存在时，计算 lane identity 与报告 edge identity 的匹配；
4. 把 `bind_gain × identity_match` 加到 cross-attention logits；
5. 用 `residual_pool='mean'` 或 `mms(mean+max+std)` 汇总原始交通状态；
6. 将 cross-attention context 与交通残差融合成 (z)，同时返回 lane 权重 (w)。

canonical `CompositeSceneEncoder`（variant=`sga`）在此基础上加入 lane self-attention，并把报告语义 embedding 作为第三路输入：

\[
K_0=\operatorname{lane\_proj}(H),\quad K=K_0+\operatorname{SelfAttn}(K_0,K_0,K_0),
\]
\[
q=\operatorname{text\_proj}(x),\quad
w=\operatorname{softmax}(qK^T/\sqrt d+b_{identity}),
\]
\[
z=\operatorname{MLP}([\operatorname{CrossAttn}(q,K,K),
\operatorname{RawTraffic}(H,g),\operatorname{text\_proj}(x)]).
\]

因此 SGA 同时具备：场景内部节点交互、报告到节点的定位、报告语义进入表示、交通状态残差旁路。

### 5.3 `SceneJointPolicy`：让注意力直接影响相位 Q 值

`SceneJointPolicy` 把已收敛的 DQN 保留在计算图中。底座从 lane count 和 phase one-hot 重建输入，得到 (q_{base})。场景编码器得到 (z,w)，零初始化的 `delta` 头输出动作方向 (d=\tanh(\Delta(z)))。

报告绑定质量由目标 edge 的 attention mass 计算：

\[
m=\max_e\sum_{i\in e}w_i,
\quad gate=\operatorname{clip}\left(\frac{m-\tau}{1-\tau},0,1\right).
\]

最终动作价值为：

\[
q=q_{base}+\eta\,(\max q_{base}-\min q_{base})\,gate\,d.
\]

默认设计中的 `dose_frac=0.35` 是残差剂量上限，`bind_tau=0.4` 是绑定阈值。没有可靠 edge binding 时，gate 为 0 或很小，控制器保持原 DQN；绑定成立时，注意力最多在当前 Q spread 的有限比例内调整相位排序。这种设计解决了两个工程风险：从零训练场景控制器容易破坏基础驾驶能力，未绑定文本则不应任意改变动作。

## 6. 监督、训练和评测

### 6.1 TARL 结构化线的训练信号

ATT-ENTITY-004 的主控制损失仍是原 TARL/RL 目标，语义辅助头增加结构化标签损失。概念上可以写为：

\[
L=L_{RL}+\lambda_{aux}L_{semantic},
\]
\[
L_{semantic}=L_{present}+L_{type}+L_{location/identity}+\cdots.
\]

辅助头只在标签有效的事件/节点上计算，并把 `present/type` 等可审计指标写入日志。它的目标是保留事件语义，不是直接替代交通 reward。

### 6.2 SceneJointPolicy 的两阶段训练

**阶段 A：adapter-teacher 蒸馏。** 已训练的场景 adapter 在原 DQN 之上产生每个决策 tick 的动作。记录动作是否偏离冻结 DQN 的 argmax；偏离 tick 被赋予 `1 + distill_dev_weight` 的更高 CE 权重。可选 `distill_filter` 只保留 adapter 在同一随机种子上优于 vanilla DQN 的 episode，减少把失败示范写入学生。

**阶段 B：原始 reward Double-DQN 微调。** 学生从蒸馏权重继续训练，target network 每 `target_sync` 个 episode 同步；TD 梯度同时更新 DQN 底座（小 `base_lr`）和场景分支。每个 replay batch 还可加入注意力对 `intent_lane_w`/`intent_lane_idx` 的交叉熵辅助项：

\[
L_{aux}=-\sum_i t_i\log w_i,
\qquad L_{total}=L_{HuberTD}+\lambda_{aux}L_{aux}.
\]

sig15 的 `aux_weight` 用于将注意力拉向可解释的观察集合；sig16 将其设为 0，专门检查性能和注意力审计能力是否分离。

### 6.3 评测条件和指标

每个事件条件固定 12 个 rollout（4 个事件位置/场景 × 3 个随机条件的组合口径），决策间隔通常为 10 秒，单次评测 180 ticks。主要指标为：

- `queue_int`：全网队列积分，越低越好；
- `corr_queue_int`：事件走廊队列积分，越低越好；
- `ev_queue_int`：事件边责任集队列积分，越低越好；
- `throughput`：完成车辆数，越高越好；
- `peak_queue`：峰值队列，越低越好；
- `clear_s`：清空事件影响的时间，越低越好；
- `bind_mass`/`corr_bind_mass`：注意力在事件边/走廊边上的质量，用于机制审计，不能单独当作性能指标。

条件至少包括 canonical、no-text/empty、lie/wrong-type、none，以及 fixedtime、MaxPressure 等控制基线。分析时要按事件窗、恢复窗和窗外分层，避免把全 episode 的交通差异误判成文本效用。

## 7. 结果一：ATT-ENTITY-004（TARL 多路口主线）

### 7.1 fixG 相对 a3s 的优化和语义结果

`fixG` 收官评估使用 5 个训练 seed、checkpoint `{ep50, ep100, ep150, ep200}`、5 个文本条件和 3 个 eval seed，共 300 个配对 run。相同训练 seed、相同 eval seed 的 canonical travel time 结果为：

| 判定层 | 结果 | 解释 |
|---|---:|---|
| 优化层 | ep50 配对 TT **−24.9 s** | fixG 收敛更快 |
| 优化层 | ep100 配对 TT **−8.9 s** | fixG 仍优于 a3s |
| 表征层 | `wrong_type − canonical` 中位 **+1.34 s** | 事件类型改变了行为，语义不是完全旁路 |
| 效用层 | `canonical − empty` 中位 **+0.58 s** | 文本相对 empty 尚无稳定净收益；正值代表 canonical 更差 |
| 效用层 | 同一差值均值 **+2.27 s** | 被单点约 +28.7 s 拉高，不能只报均值 |

因此当前正确的结论是“结构化语义已被模型读取，fixG 优化更好，但事件信息尚未转化为稳定的闭环收益”。

### 7.2 closure binding 后续结果

`closure_bind` 用 `TextEntityBinder(closure_binding='nearest_upstream')` 经拓扑 BFS 把原本无法直接特征绑定的 closure 报告送到最近受控上游节点，并用独立的 `node_fallback_mask` 记录 fallback。后续 eval_v2 审计覆盖从 42/46 提升到 46/46，窗内 queue gap 从旧臂约 0 变为约 +1.04，说明报告确实送达了模型；但全集合 TT 差距仍约 +0.44 s，canonical/empty 和强基线差距没有闭合。

这说明“覆盖补全”和“覆盖带来收益”是两个问题。PPT 中可把它画成左边 coverage 提升、右边 utility 仍未翻正的双栏结果。

### 7.3 强基线差距

eval_v2 同集基线中，`fixG_can` 的 TT 仍高于 MPLight 约 6.28 s、MaxPressure 约 2.49 s；MPLight 40/40 胜过 fixG，MaxPressure 40/40 胜过 fixG。该结果是当前主线必须保留的负结果：场景语义表征和可解释注意力已经形成，但还不能宣称超过成熟交通启发式控制器。

## 8. 结果二：scene_attention sig15 主链

### 8.1 sig15 主评测汇总

数据源：`data/output_data/dev_takeover_probe/sig15_eval/results.json`，每个臂 `n=12`。

| 臂 | queue_int ↓ | corridor queue_int ↓ | event queue_int ↓ | throughput ↑ | bind_mass |
|---|---:|---:|---:|---:|---:|
| `joint_sga15` | **83,847.5** | **40,142.5** | 20,453.3 | 1,019.3 | 0.3290 |
| `joint_notext` | 88,355.0 | 43,910.8 | 20,805.8 | 1,020.5 | 0.2746 |
| `joint_lie` | 84,415.8 | 41,201.7 | **20,087.5** | 1,014.2 | 0.2852 |
| `joint_base` | 86,649.2 | 42,510.0 | 21,214.2 | 1,010.4 | — |
| `maxpressure` | 82,816.7 | 40,987.5 | 21,162.5 | **1,134.3** | — |
| `fixedtime` | 130,801.7 | 53,521.7 | 44,077.5 | 849.2 | — |

相对同架构 no-text，`joint_sga15` 的 corridor queue integral 下降：

\[
(43,910.8-40,142.5)/43,910.8\approx\mathbf{8.6\%}.
\]

全网 queue integral 下降约 5.1%，event queue integral 下降约 1.7%，throughput 基本持平（约 −0.1%）。相对 `joint_base`，全网 queue 下降约 3.2%，corridor queue 下降约 5.6%。相对 MaxPressure，SGA corridor queue 略低约 2.1%，但 throughput 低约 10.1%，所以这是“局部走廊压力更低、服务量仍不足”的混合结果。

### 8.2 结构对照：为什么要保留 self-attention、cross-attention 和 joint training 的区分

`sig15_eval_modcmp/results.json` 中，所有对照也使用 12 个条件：

| 臂 | queue_int ↓ | corridor queue_int ↓ | event queue_int ↓ | throughput ↑ |
|---|---:|---:|---:|---:|
| `concat_st` | 131,899.2 | 57,515.8 | 45,501.7 | 656.0 |
| `selfattn_st` | 130,858.3 | 58,847.5 | 38,058.3 | 714.5 |
| `sga_st` | 136,052.5 | 66,157.5 | 32,051.7 | 673.8 |
| `joint_sga12` | 87,635.0 | 45,621.7 | 23,194.2 | 1,083.2 |
| `joint_sga15` | **83,847.5** | **40,142.5** | 20,453.3 | 1,019.3 |

最值得讲的不是“SGA 单独编码器一定最好”，而是 **joint training + 已训练 DQN 底座 + 有界残差** 明显优于从零训练的 `*_st` 对照。`joint_sga15` 相比 `sga_st` 的 corridor queue integral 下降约 39.3%，throughput 提升约 51.3%。

### 8.3 sig16：辅助监督更像“可审计语义契约”

`sig16_noaux` 去掉 `aux_weight`，性能仍接近 sig15：

| 臂 | corridor queue_int ↓ | event queue_int ↓ | throughput ↑ |
|---|---:|---:|---:|
| `joint_sga15` | 40,142.5 | 20,453.3 | 1,019.3 |
| `joint_sga16`（noaux） | 40,140.0 | 19,716.7 | 1,027.5 |

但注意力对文本/意图的敏感度显著下降。反事实探针汇总中，sig15 的 `w_l1(true,lie)=0.2095`、`w_l1(true,none)=0.2150`；sig16 下降为 0.0270 和 0.0153。也就是说，辅助监督暂时没有表现为明显的额外交通性能，却保留了一个重要的语义审计契约：报告改变时，注意力分布应当改变。

## 9. 结果三：反事实注意力探针

数据源：`data/output_data/dev_takeover_probe/attn_probe_c2/attn_probe.json`，sga15/sga16 各 324 个 decision ticks。探针把同一交通状态下的报告替换为：真实报告 `true`、错误道路/类型 `lie`、无报告 `none`。

| 指标 | sig15 | sig16 noaux | 解释 |
|---|---:|---:|---|
| `align_true` | 0.3447 | 0.3337 | 真实报告与责任/观察集合的对齐 |
| `align_lie` | 0.3243 | 0.3352 | 错误报告下的对齐 |
| `align_none` | 0.3024 | 0.3323 | 无报告时的对齐 |
| `w_l1_true_lie` | 0.2095 | 0.0270 | 改变报告后注意力变化量 |
| `w_l1_true_none` | 0.2150 | 0.0153 | 报告移除后注意力变化量 |
| `flip_lie_rate` | 1.23% | 0.31% | 错误报告导致动作翻转的比例 |
| `flip_none_rate` | 28.70% | 23.77% | 移除报告导致动作翻转的比例 |

探针说明三件事：

1. sig15 的权重确实会随着报告条件变化；
2. 报告位置改变并不稳定地导致动作翻转，说明注意力变化与动作效用之间还隔着底座 Q 值、残差剂量和相位约束；
3. 去掉 auxiliary loss 后，交通性能可保持，但“模型是否看到了文本”变得难以审计。

因此 PPT 应同时画 attention heatmap 和 action flip rate，避免只展示漂亮的热图就把它当作控制收益。

## 10. 跨场景和其他证据

`takeover_probe_sig15` 还包含 S1、S3、S4 三个场景的 100 episode 训练/接管探针，以及 delay 0/60/90/120、holdout、distract、severity 等冻结评估包。它们支持把 sig15 描述为“多场景 pilot 和机制探针已完成”，但当前可复用的完整数值主表仍以 `sig15_eval`、`sig15_eval_modcmp` 和 `attn_probe_c2` 为主。

正式报告必须沿用仓库治理要求：travel time、throughput、unfinished vehicles 三指标同报，并按事件窗分层；不能用 queue integral 单独替代三指标。sig15 产物记录的是 queue/throughput 体系，若要对外形成正式交通控制结论，应补齐同口径的 travel time 和 unfinished 汇总。

## 11. 已验证内容、负结果和边界

### 11.1 可以明确写成“已完成”的内容

- 事件报告的结构化字段、edge identity、报告延迟和生命周期已接入统一 FeatureBuilder。
- `SceneAttention`、`CompositeSceneEncoder` 和 `SceneJointPolicy` 已实现，能返回场景表示和 attention weights。
- identity binding、交通残差旁路、zero-init residual、bounded gate 都有明确代码路径。
- adapter-teacher → 加权 CE 蒸馏 → 原始 reward TD 的训练链路已实现并产生 sig15 产物。
- `z_task` 语义探针、SemanticAuxHead 日志字段、反事实条件和 attention probe 已纳入证据链。

### 11.2 结果支持的正向判断

- `fixG` 相对 `a3s` 的优化层明显改善，且 wrong type 条件能产生非零行为差异。
- sig15 相对 no-text 的 corridor queue integral 下降约 8.6%，说明文本/意图通道在该 pilot 中有局部控制效用。
- sig15 相对 noaux 的 attention L1 差异明显更大，辅助监督提高了语义可审计性。
- 场景注意力与原 DQN 的联合训练比从零训练的场景编码器更稳定。

### 11.3 必须保留的负结果

- ATT-ENTITY-004 的 canonical 相对 empty 的中位差值仍为 +0.58 s，文本通道没有稳定净收益。
- closure 的绑定覆盖补全后，utility 仍有正残差；事件送达不等于事件决策正确。
- MaxPressure 在 sig15 中 throughput 明显高于 joint_sga15；fixG 在 eval_v2 中仍落后 MPLight/MaxPressure。
- ATT-ENTITY-003 的早期留出检索 top1=0/3，alignment 可能存在 shortcut/记忆化风险。
- lie/wrong-type 只在少数 tick 造成动作翻转；不能用“attention 改变”直接推出“策略知道了事件”。
- road closure 的上游 fallback 语义、边覆盖和事件效用仍需要继续裁决，不能写成闭环已解决。

## 12. PPT 页面规划（建议 14–16 页）

| 页码 | 页面标题 | 核心内容 | 推荐图/表 |
|---:|---|---|---|
| 1 | 为什么需要场景引导 | 交通状态滞后、事件报告有先验位置和类型 | “状态通道 vs 场景通道”双输入图 |
| 2 | 研究问题 | 看哪里、理解什么、怎样用 | 三个问题图标/流程 |
| 3 | 方法演进 | 002 → 003 → 004 → sig15 | 时间线 |
| 4 | 当前两条主线 | TARL 多路口线与 SceneJointPolicy 单控制器线 | 代码/实验地图 |
| 5 | 输入定义 | lane physical、edge identity、global、event fields | 特征矩阵 |
| 6 | `z_task` 结构化语义 | per-node 槽位、目标节点、生命周期和类型 | 16 节点网格热图 |
| 7 | 通用 SGA | scene query 对 node key/value，soft grounding | 公式 + 节点权重热图 |
| 8 | CompositeSceneEncoder | self-attention → cross-attention → identity binding → residual | `scene_attention/figs/sga_arch_fig.py` |
| 9 | SceneJointPolicy | DQN base、gate、bounded residual | Q 值残差示意 |
| 10 | 训练流程 | adapter teacher、winner filter、weighted CE、TD | 两阶段流程图 |
| 11 | ATT-ENTITY-004 结果 | fixG vs a3s 三层判定 | ep50/100 paired TT + canonical/empty |
| 12 | sig15 主结果 | SGA/no-text/base/MaxPressure/fixedtime | queue integral/throughput 分组柱状图 |
| 13 | 结构对照 | `*_st` vs joint | modcmp 热图 |
| 14 | 辅助监督消融 | sig15 vs sig16 noaux | attention L1 与交通指标双轴图 |
| 15 | 反事实审计 | true/lie/none，attention change 与 action flip | heatmap + flip-rate 柱状图 |
| 16 | 结论与下一步 | 表征达标、局部效用、闭环未过线 | 四层判定表 |

### 12.1 推荐直接生成的图

- **架构图**：`scene_attention/figs/sga_arch_fig.py`、`joint_attn_fig.py`；
- **sig15 结果图**：`scene_attention/figs/sig15_panels.py`、`sig15_table.py`、`corridor_delta_fig.py`；
- **机制证据图**：`scene_attention/figs/evidence_fig.py`、`report_effect_fig.py`；
- **事件语义图**：`tools/att004_focus_score.py`、`tools/att004_focus_null.py`、`tools/run_tarl_repr_probe.py`；
- **反事实图**：`scene_attention/probe/attn_text.py` 和 `data/output_data/dev_takeover_probe/attn_probe_c2/attn_probe.json`。

## 13. 可复用数据表（绘图输入）

### 13.1 sig15 主结果

```csv
arm,n,queue_int,corr_queue_int,ev_queue_int,throughput,bind_mass
joint_sga15,12,83847.5,40142.5,20453.3,1019.3,0.3290
joint_notext,12,88355.0,43910.8,20805.8,1020.5,0.2746
joint_lie,12,84415.8,41201.7,20087.5,1014.2,0.2852
joint_base,12,86649.2,42510.0,21214.2,1010.4,
maxpressure,12,82816.7,40987.5,21162.5,1134.3,
fixedtime,12,130801.7,53521.7,44077.5,849.2,
```

### 13.2 sig15/sig16 attention audit

```csv
arm,n_ticks,align_true,align_lie,align_none,w_l1_true_lie,w_l1_true_none,flip_lie_rate,flip_none_rate
sga15,324,0.3447,0.3243,0.3024,0.2095,0.2150,0.0123,0.2870
sga16_noaux,324,0.3337,0.3352,0.3323,0.0270,0.0153,0.0031,0.2377
```

### 13.3 ATT-ENTITY-004 关键差值

```csv
comparison,metric,delta,interpretation
fixG_vs_a3s_ep50,paired_canonical_travel_time,-24.9,fixG faster convergence
fixG_vs_a3s_ep100,paired_canonical_travel_time,-8.9,fixG remains better
wrong_type_vs_canonical,median_travel_time,+1.34,type semantics affect behavior
canonical_vs_empty,median_travel_time,+0.58,text has no stable net utility yet
canonical_vs_empty,mean_travel_time,+2.27,mean inflated by one large pair
```

## 14. 证据路径和复现入口

### 14.1 主要代码

- [agent/scene_attention.py](agent/scene_attention.py)：通用 SceneEncoder、SceneGuidedAttention、多事件 SGA、cross-attention residual、MPLight adapter；
- [agent/tarl.py](agent/tarl.py)：TARL 方法注册入口；
- [scene_attention/dispatch_state.py](scene_attention/dispatch_state.py)：报告字段、lane token、identity layout、intent_vec/intent_watch；
- [scene_attention/dispatch_model.py](scene_attention/dispatch_model.py)：`SceneAttention`、`CompositeSceneEncoder`、identity binding；
- [scene_attention/signal_model.py](scene_attention/signal_model.py)：`SceneResidualPolicy`、`SceneJointPolicy`；
- [scene_attention/train.py](scene_attention/train.py)：蒸馏、Double-DQN、aux alignment、配置参数；
- [scene_attention/eval.py](scene_attention/eval.py)：冻结事件条件评测；
- [scene_attention/README.md](scene_attention/README.md)：sig9–16 机制谱系和当前 canonical 说明。

### 14.2 实验与结果

- [EXPERIMENTS.md](EXPERIMENTS.md)：当前面板、代号地图、campaign 索引；
- [EXPERIMENTS/att_entity_002.md](EXPERIMENTS/att_entity_002.md)：SGA/concat/MPLight 文本注入矩阵；
- [EXPERIMENTS/att_entity_003.md](EXPERIMENTS/att_entity_003.md)：TARL alignment/传感矩阵；
- [EXPERIMENTS/att_entity_004.md](EXPERIMENTS/att_entity_004.md)：`z_task + SemanticAuxHead + 反事实` 主线、fixG、eval_v2 和 closure binding；
- [EXPERIMENTS/takeover_probe_sig15.md](EXPERIMENTS/takeover_probe_sig15.md)：sig15/sig16 训练和评测登记；
- `data/output_data/dev_takeover_probe/sig15_eval/results.json`：sig15 主评测；
- `data/output_data/dev_takeover_probe/sig15_eval_modcmp/results.json`：结构对照；
- `data/output_data/dev_takeover_probe/attn_probe_c2/attn_probe.json`：sig15/sig16 反事实 attention audit。

### 14.3 推荐汇报时的证据声明

建议在 PPT 页脚统一标注：

> “结果来自仓内冻结 pilot/评估包；不同 campaign 的指标、事件协议和仿真场景不自动可比。三指标正式结论需回到 ledger 和对应 campaign 明细核对。”

这样既可以展示已经完成的工程和实验，也能避免把局部 attention 变化、queue integral 改善或单场景 pilot 误写成普适科研结论。

