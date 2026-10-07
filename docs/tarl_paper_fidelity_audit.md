# TARL-TSC 论文复现保真度审计（ATT-ENTITY-003 子任务）

任务定位：把 TARL-TSC（Al-Quhfa/Mothana/Song, EAAI 182:116083）按论文公式
复现后作为 baseline 跑我们的 benchmark（hz4x4 + hz4x4_random_v1 事件计划 +
randeval_v1 留出评估），与 CoLight / SGA 各版本对照，判断"文本引导注意力
无收益"是我们架构设计的问题还是更普遍的现象。

## 一、论文架构规范（from TARL.pdf 原文）

| 组件 | 论文规范 |
|---|---|
| 文本编码 | 冻结 Sentence-BERT（详节指定 all-mpnet-base-v2），768 维，逐事件计算一次 |
| 交通嵌入 | s_it = MLP_traffic(s_raw)，128 维（Eq.9） |
| 共享隐空间 | e′=W_e e_t, s′=W_s s_it，d_h=256（Eq.11） |
| TARL-ATT | o_text=Attn(e′,S′,S′)；o_traffic=Attn(s′,e′,e′)；concat(512)→256→ReLU+Drop→128+s_it 残差（Eq.12-13，图2） |
| TARL-AGM | g=σ(W_g[e′;s′])，h=g⊙e′+(1−g)⊙s′（256）→Linear(256→128)→ReLU+Drop（Eq.14-15，图3） |
| GAT | 2 层标准 GAT，K=8 头，邻居集 N(i)∪{i}（Eq.16-17） |
| Q 网络 | 轻量两层 MLP（Eq.18），全节点共享参数（IQL-PS） |
| 训练 | TD loss；target 每 200 步同步；buffer 50k 均匀采样；batch 128；Adam 1e-4；γ0.99；ε 1.0→0.01 前 300ep |

## 二、实现位置与接入方式

- 模型：`reproduction/tarl_tsc/models/paper_tarl.py`（PaperTARLPolicy）
- 适配：`reproduction/tarl_tsc/parent_adapter.py`（`tarl_impl: paper` 时启用；
  `policy_texts()` 优先取 `world._sumo_event_runtime` 的项目事件 runtime，
  texts() 把公开报告广播给全部节点 = 论文全局 e_t 语义；TARL 自带 runtime
  保持 normal 空计划，不重复注入物理事件）
- 配置：`configs/tsc/tarl_paper_{sensor,gat,attention,gating}_rande_200.yml`（+ `_smoke`）
- 注册名沿用 `tarl_sensor / tarl_gat / tarl_concat / tarl_attention / tarl_gating`，
  `tarl_impl: paper|legacy` 配置项选择实现版本
- 测试：`reproduction/tarl_tsc/tests/test_paper_tarl.py`（19 项）

## 三、对论文的偏离与未指明参数的取值（必须随结果一并声明）

| 项 | 处理 | 影响评估 |
|---|---|---|
| 文本编码器 | bert-base-uncased（本地资产）替代 all-mpnet-base-v2 | 同为冻结 768 维 SBERT 系；语义粒度近似，属有界偏离；mpnet 权重外网不可得 |
| o_traffic 注意力 | 单 key（节点自身文本 token）→ softmax 退化返回 e′；MHA 投影仍学习 | 论文 e_t 无节点下标的字面化实现；广播文本下与全局语义一致 |
| Dropout 率 | p=0.1（论文图标 Dropout 未给率） | 常规取值；仅梯度更新时激活，rollout/评估确定性 |
| GAT 层间激活 | ELU（标准 GAT 惯例，论文未指明） | 常规取值 |
| Reward | 保留框架本地 waiting×12（与本仓全部学习类基线一致） | **故意偏离**：保证与 CoLight/SGA/MPLight 的可比性优先于公式逐字 |
| 优化器 | 框架 RMSprop(lr=1e-4) 替代 Adam 1e-4 | 框架统一组件；lr 已按论文设 |
| ε 计划 | 框架乘性衰减 0.8/0.99995/0.05（与其他 200ep 臂一致） | 训练控制对齐 benchmark 而非论文日程 |
| 交通观测 | 节点 lane_count + one-hot 相位（框架生成器） | benchmark 特定，维度自适应投影 |
| 事件物理 | 我们的 SUMO 计划书（blockage/closure/rain），非 CityFlow 注入 | benchmark 定义，本任务就是"换 model 跑我们的 benchmark" |

## 四、代码保真修复记录

- `self.impl` 赋值移到 `super().__init__` 之前（`_build_model` 在其内被调用，原位置会 AttributeError）。
- 融合头 Dropout 改为 `F.dropout(training=self.training and train)`：否则评估期
  dropout 消耗 torch RNG，触发 EvaluationIsolationGuard "mutated training state"。
- `last_fusion_weights`（attention）改为记录 o_text 的 16-key 权重——o_traffic
  单 key 权重恒 1 无诊断价值。
- `policy_texts()` 文本源切到 `world._sumo_event_runtime`：原实现只读 TARL 自带
  空 runtime，接我们的事件系统后文本会恒为空串（隐性"无文本"臂）。
- eval 白名单（utils/logger.py）补 tarl_* 五名 + DQN 族 checkpoint 校验。

## 五、已完成的端到端验证（smoke: tarlp_att_smoke, 4ep×300s）

- 事件按计划逐 episode 注入（event_plan_log.jsonl，各 episode sha256 独立）
- text_source=project_runtime；事件窗 [60,200) 内 67/92 决策带真实事件文本，窗外为空
- impl=paper / protocol=tarl_graph_v2 正确记录；评估隔离保护通过；双类 checkpoint 落盘
- 注意力权重初始均匀 1/16（未训练基线，后续训练应分化）

## 六、stage-5 矩阵（进行中）

`artifacts/att_entity_003/run_queue_stage5_tarl.json`：4 臂 × 5 seeds(7/17/27/37/47)
×200ep，parallel 5。评估：与其他臂同一 randeval_v1 + fixed 条件集、同一评估种子。

## 七、既有证据定位

`reproduction/tarl_tsc/` 内已有 v2.1 协议复现（legacy 实现，简化维度/GraphMix/
单方向注意力）：其报告 FAIL——门控与 canonical 文本均未显示稳定收益。本阶段
论文保真版用于回答：在**我们的**事件分布与评估协议下，贴近论文结构的
ATT/AGM 是否产生增量，从而区分"我们架构的问题 vs 该方向普遍无效"。

## 八、评估链路修复记录（stage-4 eval 推进中发现）

- `run.py` schema-v2 eval：`compose_evaluation_world_config` 只保留 4 个 runtime
  字段，会丢弃 agent 必需的 `world.signal_config`（mplight/frap 系在 agent init
  即崩 KeyError）。修复：compose 后从源 config 补回 signal_config（同网络评估下
  安全，网络身份由 validate_cross_scene_traffic_identity 另行校验）。
- `agent/mplight.py`：`MPLight_InerAgent.batch_act` 的 `test` 参数从不使用，
  探索与否取决于 PFRL `self.training`。训练内 eval 因 explorer 已衰减侥幸正确；
  独立 eval 进程 explorer 全新（eps_start=1.0）→ 动作近随机（TT≈483 vs 训练 347）。
  修复：`get_action(test=True)` 时临时置 `agents_iner.training=False` 并在返回后
  恢复。**此前所有 MPLight 独立 eval 结果无效**；已重跑（normal TT 334.3）。
- MPLight 无 audited checkpoint（self.target_model=None → save_checkpoint 早退）：
  `tools/convert_mplight_ckpt.py` 把 legacy `model/{ep}_0.pt` 封装为 evaluation
  型 checkpoint（仅 online weights；target/optimizer/replay 无源可封 → resumable
  role 按设计不可用），manifest 走 `checkpoint_role=best`。
- `run.py` EvaluationManifestRunner 的检查点加载白名单漏了 tarl_*（utils/logger.py
  侧的解析/审计白名单已补，此处漏改）：checkpoint 被解析、审计、记录，但
  `load_online_checkpoint` 从未执行 → 独立 eval 跑的是**随机初始化网络**。
  症状：吞吐 ~700-900（vs 正常 ~2700，~900 辆车无法进入路网）、相位切换
  360/episode（argmax 近恒定）、幸存者偏差 TT≈190-350 完全不可信；
  `online_hash_match` 只比对两个 checkpoint 文件而非加载后模型，故未报错。
  修复：run.py 加载分支补 tarl_* 五名，并新增加载后哈希校验
  （`hash_torch_state_dict(agent.model.state_dict())` vs manifest
  `checkpoint_audit.online_model_state_hashes`）——今后任何 agent 被静默
  跳过加载都会直接报错而非产出假结果。**修复前全部 TARL eval 结果无效**，
  已移至 `eval_stage5_tarl/packages_INVALID_no_ckpt_load/` 并重跑。
