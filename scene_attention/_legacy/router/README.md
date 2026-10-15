# _legacy/router/ — router 时代探针归档（非当前主线）

本目录收编 2026-10 期间「场景引导注意力 → 控制器路由」架构时代的
脚本。该架构是**双层决策**：上层 DispatchRouter 按播报选择控制器，
下层控制器选信号相位——已被否决（等价于训两层策略，上层动作
空间退化为控制器选择）。

**当前主线在包根**：单层场景条件信号控制器
（`SceneJointPolicy` = dqn_s1 底座 + sga 注意力模块 + 有界残差），
训练链路 = adapter 教师蒸馏（`sig9_joint_bc`）→ 原装 reward RL
微调（`sig10_joint_rl`）。入口：`python -m scene_attention.train` /
`python -m scene_attention.eval`。

## 目录内文件性质

- `train_router.py`：router 训练入口。其事件常量
  （`TRAIN_EDGES/TRAIN_ONSETS/TRAIN_DURS/TRAIN_PERIODS/EDGE_TARGETS/FEEDER_OF/DISCHARGE_OF/DISTRACTOR_OF`）
  已抽出为活包唯一常量源 **`scene_attention/events.py`**；
  本文件内同名定义为历史快照，冻结不动、不作事实源；
- `eval_router.py` / `router_runtime.py`：router 评测与运行时；
- `probe_repr.py` / `eval_objective.py` / `channel_ablate.py` /
  `collect_*.py` / `fork_probe.py` / `scan_pairs.py` /
  `stage1_event_physics.py` / `make_figs.py` / `scene_distill.py`：
  router/编码器时代的表征探针、消融、出图脚本——历史实验证据，
  不属于当前主线。文件内 import 已改写至新包路径，
  属 best-effort 可跑，不保证随活包演化继续可用。

新工作不要以本目录文件为架构参照；编码器活组件
（`CompositeSceneEncoder` 等）在 `../../dispatch_model.py`，
`DispatchRouter` 类随之作废，仅被本目录脚本引用。
