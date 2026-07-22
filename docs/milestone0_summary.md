# Milestone 0：公共实验基础设施总结

## 1. 范围与结论

Milestone 0 已完成公共实验基础设施，不包含 100 episode Pilot、20 次正式 Online 训练、Offline 训练或 Sequential 正式实验，也未合并 main。行为基线固定为 `73d860bb3924ec15c30433a8f8b7af17787baeff`。

所有未阻塞验收项均有自动检查、真实产物或代码审计证据。FixedTime、MaxPressure 和 DQN 最终真实 SUMO 回归均通过。

## 2. 原项目主要差距与代码证据

- `common/interface.py` 和 `trainer/base_trainer.py` 原先共同使用 `configs/sim/<network>.cfg`，`utils/logger.py::modify_config_file` 会直接写回共享源 cfg，存在并发串扰和源码副作用。
- `agent/dqn.py::save_model` 原先只保存 target state_dict 到旧 model 目录，不能表示当前 online 策略或恢复训练。
- `run.py` 原先没有 manifest/status 状态机、配置完整性终态校验或失败原因证据。
- `trainer/tsc_trainer.py` 原先只有文本日志，没有固定 JSON schema、global decision/gradient counter、评估隔离或 checkpoint 顶层状态。
- 配置归档原先缺少实际 runtime model/optimizer/loss/controller 描述和规范 state hash。
- 项目没有跨运行闭合 allowlist 比较器；alternate simulator cfg 还存在等价值的类型表示差异。
- 指标来源分散，且 DQN 与传统控制器的同名 reward 使用不同底层量。

## 3. 功能实现与工程意图

1. 配置归档与 simulator 隔离：从启动时捕获的源字节生成运行专属 resolved cfg，SUMO 只读取运行文件；归档源 YAML、source/resolved cfg、最终配置、实际 model/controller 和 SHA-256。
2. 运行身份与状态：固定 run_id、manifest/status 四状态原子生命周期；完成、task 失败和 pre-archive 初始化失败均可追溯。
3. Seed 与复现签名：README 固定 training_seed/SUMO seed 边界；按 tensor 内容规范哈希 online/target；随机探针保存并恢复 RNG。
4. 结构化日志：`metrics/records.jsonl` 固定 19 字段和三种 record_type；真实记录 simulation/decision/global/gradient counter，保留旧文本日志。
5. 评估隔离：每次评估前后比较 online/target、optimizer、epsilon、replay、dataset、gradient/global counter，并阻断 remember。
6. Checkpoint：新增 online-only evaluation 与完整 resumable schema；原子保存、严格加载校验和恢复后 optimizer 更新；legacy target-only 文件保持独立。
7. 指标审计：记录 travel time、queue、delay、throughput、reward 的实际代码口径，以 characterization test 冻结现状。
8. 配置一致性：闭合 allowlist CLI 比较 resolved config 与 runtime model；input/action dimension 不一致立即失败；四个真实 network 归档比较通过。
9. 整体回归：FixedTime/MaxPressure 各 3600 steps，DQN 覆盖真实更新、target update、两次隔离评估和两类 checkpoint。

## 4. 算法与科研语义边界

Milestone 0 未改变 state、reward、action mapping、DQN loss、terminal/truncated bootstrap、epsilon schedule、learning-start 条件、预热动作策略、train/target update frequency、replay replacement/sampling、指标公式或正式实验预算。

`agent/dqn.py` 相对 baseline 的唯一修改是增加 `activation_name='relu'` 供运行时归档，forward 计算不变。Trainer 的新 counter/guard/checkpoint 代码观察或保存既有状态；fresh run 的 global counter 初值仍为 0。三个 alternate cfg 的字符串布尔改为等价 JSON boolean，`world_sumo.py` 的现有解析结果和 SUMO 命令不变。

## 5. 验证概况

- 自动检查：23 项 Milestone 0 unittest 全通过，覆盖成功与失败路径。
- 语法：受影响 Python 文件 py_compile 全通过。
- 真实 SUMO：Function 1–6/8 的针对性 smoke 和 Function 9 三类最终回归均通过。
- 产物：manifest/status、config hash、model description、JSONL、evaluation/resumable checkpoint 均实际读取断言。
- 配置：四场景 runtime input_dim=16、action_dim=8、hidden=[20,20]、RMSprop、MSELoss，规范化比较无非法差异。
- 清理：无临时 YAML 或 simulator source cfg 变化；大型运行产物保持 Git ignore。

详细命令和结果位于 `docs/verification/milestone0/function1_config_archive.md` 至 `function9_final_regression.md`。

## 6. 提交与远端分支

- `1eb3bd9` 建立Milestone 0实验基础设施执行规范
- `442f2c0` 实现运行配置归档与仿真配置隔离
- `60a2759` 增加运行身份与原子状态管理
- `fd0f7c4` 固化训练种子边界与初始化复现签名
- `1c28fc8` 增加训练评估结构化指标日志
- `e9e6232` 强化训练与评估状态隔离
- `214486c` 实现评估与可恢复检查点语义
- `cd7fc54` 审计并固化现有指标计算口径
- `c871bcb` 增加跨运行配置一致性校验
- `0e864a1` 同步Milestone 0功能提交证据
- `4040f97` 补全初始化失败运行状态
- `98cbc00` 完成Milestone 0整体回归与总结

远端分支：`origin/codex/milestone0-experiment-infrastructure`。未 force push，未改写已推送历史，未自动合并 main。

## 7. 已知警告与剩余风险

- Gym 已停止维护；当前警告不影响已验证路径，但后续依赖升级需单独兼容性工作。
- 可选 PyG CUDA 扩展与当前 Torch ABI 不兼容并被禁用；本次路径未使用。
- `world_sumo.py::get_real_delay` 在无车辆轨迹时可能除零；因指标公式受保护，本 Milestone 只登记风险。
- throughput docstring 容易被理解为瞬时路网车辆数，实际是已完成车辆数。
- 短 smoke 中 travel time=0 可能只表示尚无完成车辆，不能解释为零旅行时间或性能最优。
- Milestone 0 不提供失败目录内 resume；重试必须新 prefix。

## 8. Milestone 0 后续科研决定

- RD-002 已决定：保持 `total_decision_num > learning_start` 的严格大于边界。
- RD-003 已决定：learning start 前保持完全随机预热。
- RD-004 已决定：保持按成功梯度更新次数执行现有 epsilon 乘法衰减。
- RD-005 已决定：TD target 保持始终 bootstrap，同时在 trajectory 中分别记录 terminated/truncated。
- RD-006 已决定：同时记录 approximate/real delay，以 approximate 为主、real 为辅助。
- RD-007 已决定：final checkpoint 为主；诊断性 best 按最低 average travel time 选择，并列时取较早节点。
- RD-009 已推迟：保持各控制器 reward 现状，在后续实验启动前专项讨论跨控制器比较边界。

RD-001 与 RD-008 已实现并验证；RD-002 至 RD-007 的决定需要在 Plan 1 启动门中逐项落实和验收。

## 9. 下一 Milestone 进入条件

1. 将 RD-002 至 RD-007 的已确认语义落实到 Plan 1 冻结配置、轨迹、指标和 checkpoint 选择协议，并完成针对性验收。
2. 冻结 Plan 1 正式 DQN 配置、评估节点、主指标与 best 规则；RD-009 在后续实验启动前另行专项讨论，讨论完成前禁止跨控制器直接比较 reward。
3. 实现并验收 append-only trajectory，届时再证明 evaluation transition count=0。
4. 用户审阅本分支并自行决定是否合并；Codex 不自动合并 main。
5. 上述条件满足后才运行 100 episode Pilot；Pilot 通过后才允许 20 次正式 Online 训练。
