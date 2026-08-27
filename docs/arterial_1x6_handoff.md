# arterial_1x6 共享 DQN 扩展交接报告

日期：2026-07-29；执行设备：linux4090。仓库审计详见 `docs/arterial_1x6_implementation_audit.md`。

## 实现结论

本扩展增加了一个同时兼容 1 个和 N 个路口的 `shared_dqn` 控制器。六路口 arterial 场景只创建一套 online network、target network、optimizer、epsilon schedule 和 replay；每个路口用本地观测独立选动作，六个动作在一个 SUMO control step 内共同提交。代码没有构造 `8^6` 联合动作，也没有把 scene ID 输入网络。

原 DQN 算法主体保持不变：20-20 MLP、vanilla target DQN、RMSprop、full-Q-vector MSE、硬 target 同步和梯度裁剪。多路口扩展默认每个 environment decision 执行一次梯度更新，而不是因产生六条 transition 而执行六次更新。默认 3600 秒、10 秒 action interval 时，单路口每 episode 产生 360 条 transition，六路口产生 2160 条；稳定训练期约 360 次梯度更新，按局部 transition 计算的 UTD 约为 1/6。

## 修改文件

- `agent/shared_dqn.py`：共享参数的分散式 DQN、本地 observation/reward、position encoding、in/out 状态、action mask、mixed batch、TD 诊断、epsilon 模式、历史导入导出。新增文件，不改变旧 `dqn` 类。
- `arterial/control.py`：自然排序的稳定路口顺序、action mask 和 ID 到动作映射。
- `arterial/replay.py`：可追溯本地 transition、在线 FIFO replay、历史池、causal/full 可见性、scene×intersection 分层采样、mixed batch、分块历史档案。
- `arterial/experiment.py`：四场景映射、三种顺序、配置校验、stage overlay、跨场景 evaluation manifest 与矩阵行协议。
- `arterial_run.py`：从实验配置生成或执行单个顺序 stage，显式传入历史路径与父 checkpoint。
- `environment.py`：MultiDiscrete 声明改为每个受控路口一个离散维度；单路口仍为单维。
- `trainer/tsc_trainer.py`：共享 transition 写入、更新就绪条件、纯离线模式、updates-per-decision、流式 history archive、动作/路口/时间窗口日志、共享 checkpoint、stage 导入、冻结评估扩展。
- `world/world_sumo.py`：稳定路口顺序、entered/exited 事件记录、teleport 场景下可靠 unfinished 口径；SUMO_HOME 不再是导入模块的硬条件。
- `run.py`：YAML experiment overlay、stage checkpoint 导入、shared-DQN 跨场景 checkpoint evaluation。
- `utils/logger.py`：shared-DQN 指标 schema v4、evaluation v2 六路口字段及 checkpoint 支持。
- `world/__init__.py`、`agent/__init__.py`、`agent/utils.py`：可选 CityFlow、PyG/PFRL 依赖缺失时不阻止 SUMO shared-DQN 入口导入；相关算法自身仍要求各自依赖。
- `tests/test_plan1_metrics.py`：unfinished 车辆口径回归。
- `tests/test_arterial_*.py`、`tests/test_shared_dqn_*.py`、`tests/test_tsc_env_shared_actions.py`：本 Goal 的自动测试。
- `configs/tsc/shared_dqn.yml`、`configs/arterial/*.yml`：模型基础配置和实验示例。

## 配置入口

- 单路口兼容：`configs/arterial/single_intersection_compat.yml`
- 六路口独立在线：`configs/arterial/independent_online.yml`
- 顺序在线 clear/FIFO：`configs/arterial/sequential_online.yml`，修改 `replay_clear_on_scene_switch` 选择策略
- 纯离线：`configs/arterial/pure_offline.yml`
- causal semi-offline：`configs/arterial/semi_offline_causal.yml`
- full-history 非因果上界：`configs/arterial/semi_offline_full_history.yml`
- 最小 smoke：`configs/arterial/smoke.yml`

三种顺序以 `order_1`、`order_2`、`order_3` 表达。`arterial_run.py` 会解析顺序、场景、evaluation/checkpoint 周期并生成 resolved overlay。例如：

```bash
python arterial_run.py \
  --config configs/arterial/semi_offline_causal.yml \
  --stage 1 \
  --history-path /absolute/path/to/300_0.6/history_archive \
  --stage-checkpoint /absolute/path/to/stage0/checkpoints/resumable/episode_0100.pt \
  --output-overlay /absolute/path/to/run_overlays/order1_stage1.yml
```

确认输出的 command 和 metadata 后再加 `--execute`。full-history 配置会在 metadata 中写 `full_history_noncausal_upper_bound=true`。

## 数据协议

### Transition

每条 `LocalTransition` 含 `state, phase, action, reward, next_state, next_phase, terminated, truncated`。metadata 含：

```text
transition_id, scene_id, intersection_id, episode_id, decision_step,
source, training_stage, policy_version
```

`source` 为 `online_current` 或 `offline_history`。scene ID 不进入神经网络。

### History manifest

历史档案按 episode 分块写 `.pt`，manifest 明确记录：

```text
roadnet_id, scene_id, intersection_ids, state_schema, action_schema,
reward_schema, num_episodes, num_decision_steps, num_transitions,
training_seed, source_policy, collection_stage, created_at, transition_files
```

加载时会验证路口 ID/顺序、状态、动作和 reward schema；不兼容档案会立即失败。

### Replay 与 mixed batch

在线 replay 是有界 FIFO，并在结构化日志中按 scene 和 intersection 统计组成。offline ratio 表示最终 batch 的历史样本比例；0、0.25、0.5、0.75 和纯离线 1.0 均受支持。每次更新记录实际 online/offline 数量、两侧 loss/TD error/Q、target Q、gradient norm 和 offline 来源分布。ratio=0 时不会访问历史池。

### Metrics

- `metrics/records.jsonl`：网络级训练/evaluation 和 replay/UTD/epsilon 诊断。
- `actions.jsonl`：episode、simulation/decision step、scene/intersection、selected/executed action、epsilon。
- `intersection_metrics.jsonl`：六路口 local reward、queue、delay、进出口车辆数、pressure、phase、switch、occupancy。
- `time_window_metrics.jsonl`：默认 5 分钟窗口的网络 queue/delay/throughput/entered/exited/finished/unfinished 和 `intersection × window` 指标。
- evaluation package：每个 decision 含六路口 pressure、进出口车辆数、phase 和切换信息；summary 含 travel time、delay、平均/最大 queue、throughput、completed/unfinished vehicles。

路口级 `discharge_count` 当前显式为 null：SUMO World 尚无可靠 stop-line ID 事件，不能用出口车道存量冒充局部 throughput。

### Checkpoint

evaluation checkpoint 保存 online network。resumable checkpoint 另含 target、optimizer、epsilon、完整 online replay、replay 利用率、global decision/gradient/target counters、Python/NumPy/Torch RNG，以及：

```text
current_scene, scene_stage, intersection_ids, visible_history_scenes,
policy_version, epsilon_mode, replay_saved, resolved_config_snapshot
```

stage 导入保留网络/target/optimizer；clear 清空 replay，FIFO 保留；epsilon 按 reset/continue/fixed-low 明确处理。

## Reward 定义

- `original`：每路口进口车道等待车辆数的平均值取负，再乘 12；与旧 DQN 一致。
- `pressure`：`r_i = -(sum incoming lane vehicle count - sum outgoing lane vehicle count)`。按 lane 聚合，当前不做 capacity 归一化；网络报告取六个局部 reward 的平均，DQN 更新始终使用局部 reward。

## 自动测试结果

完整命令：

```bash
LD_LIBRARY_PATH=/tmp/sumo-runtime-libs/root/usr/lib/x86_64-linux-gnu \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

结果：`177 passed`。覆盖六路口顺序/动作、ID-keyed reset/step、每步六 transition、共享输出维度、action mask、mixed ratio、ratio=0、causal/full、分层采样、evaluation isolation、单路口、三种 order、四场景文件及信号程序等。

## Smoke test 结果

所有训练 smoke 使用 SUMO/libsumo、training seed 0，目的仅为链路验证。

1. 单路口回归：既有 `sumohz1x1` + `dqn`，1 episode、40 秒、batch 2；训练、checkpoint、episode 0/1 evaluation 成功。
2. 六路口在线：`300_0.6`，1 episode、40 秒、4 decisions；产生 24 条局部 transition，执行 3 次共享梯度更新，history manifest、evaluation 和 checkpoint 成功。
3. 六路口半离线：当前 `300_0.3`，causal 历史仅 `300_0.6`，1 episode、40 秒、batch 8、offline ratio 0.5；每次实际 4 online + 4 offline，offline 来源未出现当前/未来场景，online/offline loss 均有记录。
4. 纯离线：读取 24 条固定历史，3 gradient steps，无新环境 transition；checkpoint 和冻结 evaluation 成功。
5. 两场景切换：`300_0.6 -> 300_0.3`，网络 hash 在导入点完全一致，replay 从 24 清为 0，epsilon 从 0.0985074875 按 reset 恢复为 0.1，切换后正常收集与更新。
6. 四场景加载：`300_0.3`、`300_0.6`、`700_0.3`、`700_0.6` 均完成真实 SUMO reset、六状态、六动作和 step。
7. 跨场景 evaluation：stage-0 checkpoint 在四场景各冻结运行 3600 秒，共 1440 条 decision records；package 哈希校验通过，4 次 isolation 均证明 online/target/optimizer/epsilon/replay/RNG 未改变。

调试中修复了初始化前读取 lane_count、指标 schema 版本、pressure 诊断订阅、clear replay 的过早更新、YAML scene ID 数值化、evaluation 六路口字段、SUMO teleport unfinished 统计等问题。失败尝试不计入成功结果。

## 正式实验启动建议

先从 `300_0.6` 做独立在线 pilot，因为需求较稳定，便于核对学习曲线、动作占比和 replay/UTD。建议初始 pilot 为 50–100 episodes；若 travel time、queue、throughput 和 TD loss 仍未稳定，再扩到 200–400 episodes。四场景正式比较必须保持相同 episode/decision/gradient budget、状态、reward、backbone、optimizer 和 epsilon mode。

建议每 25 episodes evaluation/checkpoint；正式批量前先确认：

1. Python 3.9 `colight` 环境及 SUMO/libsumo 版本；本机本轮使用 Python 3.10、libsumo 1.27.1，并因系统缺 `libXrender.so.1` 临时设置了 `LD_LIBRARY_PATH`。
2. 是否允许 SUMO teleport；本轮完整评估观测到 teleport，unfinished 口径已正确，但科研协议应固定并记录该设置。
3. lane capacity/长度来源；启用归一化 in/out 状态前必须确定真实 capacity，不能沿用默认 1。
4. 正式 training seeds、evaluation seeds、存储预算和 history archive 保留策略。
5. 是否需要精确路口 discharge；若需要，应先实现 stop-line crossing 事件。

正式监测至少包括 travel time、average/maximum queue、delay、throughput、unfinished、六路口 pressure 热图、action/phase switch、online/offline TD 冲突、history 来源分布和实际 offline ratio。不得根据本次 smoke 的短时数值宣称半离线方法有效。
