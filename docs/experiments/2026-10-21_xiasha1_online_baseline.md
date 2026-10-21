# xiasha1_online_baseline：xiasha1 需求包上的纯在线 off-policy 基线

> 状态：进行中（准备阶段；正式 run 未启动）
> 发起日期：2026-10-21 ｜ 负责人/Agent：Z / Claude Code
> 最近更新：2026-10-21

## 关联

- campaign：`xiasha1_online_v1`（登记与提交随本轮配置定稿一并完成）
- 队列文件：`artifacts/xiasha1_online_v1/run_queue.json`（随登记生成）
- 正式 overlay：`configs/tsc/xiasha1_online/{dqn,ddqn,dueling,d3qn,
  presslight,frap,mplight,sac}.yml`（经 `--experiment-config` 叠加在
  各臂 canonical 配置之上）
- code_commit / config_sha：登记时以提交 sha 回填
- 产物目录：`data/output_data/tsc/sumo_{dqn,presslight,frap,mplight,sac}/xiasha1_*`
  （已有工程冒烟；正式 run 前缀 `xi1on_<arm>_<pack>_s<seed>`）

## 1. 改动意图

在 xiasha1*1 单受控路口 J 上，用同一套环境与评估条件比较纯在线
off-policy 控制器在不同需求包上的收敛情况与训练后性能。关注：

- 从零初始化、纯在线训练，不加载预训练、不使用离线/历史数据、不使用教师锚定；
- 各控制器保留原生算法超参，统一环境、训练预算和评估条件；
- 训练过程看收敛速度/稳定性，训练后看冻结 checkpoint 的
  travel_time / throughput / unfinished_vehicles。

不纳入：CoLight（单路口图退化）、PPO（on-policy，按本轮口径排除）、
TARL / HA-SODQN / offline（非纯在线）。规则控制器只作参照。

定位边界：本实验每个 run 固定单一需求包、从零初始化、仅用在线采样
replay——回答「控制器本身的可训性/收敛性」。区别于仓内半离线线
（ha_sodqn/cl_hz1x1：混入 history_archive 历史数据与锚定）与顺序
训练线（seq8/cl_hz1x1 stage<k>：跨场景切换研究迁移与遗忘）。本
campaign 的产出是那两条线在 xiasha1 上的**纯在线 control 参照**。

## 2. 改动模块

- `agent/sac.py`：注册 `sac`；增加 `model` / `target_model` / `optimizer`
  兼容 bundle；`get_action(test=True)` 改为策略 argmax；`train()` 增加
  minibatch 保护、twin-Q min、soft target 更新；`save_model/load_model`
  保存完整 bundle。
- `agent/__init__.py`：导入并注册 `SACAgent`。
- `configs/tsc/sac.yml`：SAC 基线配置（gamma 0.99、lr 3e-4、batch 256、
  temperature 0.2、tau 0.005、buffer 10000、16 维观测、reward_scale 1.0）。
- `agent/mplight.py`：修复 `load_model`；增加 `checkpoint_eval_only = True`。
- `trainer/tsc_trainer.py`：支持 evaluation-only checkpoint；
  `evaluation/summary.json` 改为按实际 checkpoint 文件计数，并新增
  `checkpoint_protocol` 字段（`full` / `evaluation_only`）。
- `utils/logger.py`、`run.py`：把 `presslight`、`sac` 加入 evaluation
  manifest 的 agent 白名单与 checkpoint 加载集合。
- `trainer/tsc_trainer.py`：`EpisodeTrajectoryWriter` 白名单由
  `{dqn, shared_dqn}` 扩至 `{dqn, shared_dqn, presslight, frap,
  mplight, sac}`——8 个正式臂训练过程全量记录 transition npz
  （schema A），供后续离线/半离线训练复用。
- `configs/tsc/frap.yml`、`configs/tsc/mplight.yml`：新增
  `world.signal_config.xiasha1`（4 条 `phase_pairs`）。
- `configs/sim/xiasha1_*.cfg` ×12：`"network"` 统一为 `xiasha1`；
  `dev/demand_packs/build_xiasha_notrain_queue.py` 同步。
- `tools/eval_baseline_checkpoint.py`：per-agent 训练后评估加载器，
  支持标准 `checkpoints/evaluation/*.pt` 与 `model/{e}.pt`。
- 新增工程冒烟配置：`configs/tsc/sac_smoke.yml`、
  `presslight_smoke.yml`、`frap_smoke.yml`、`mplight_smoke.yml`、
  `dqn_smoke_{ddqn,dueling,d3qn}.yml`；评估 overlay
  `configs/tsc/eval_baseline_checkpoint{,_gy5}.yml`。
- `tests/test_plan1_evaluation_schedule.py`：mock `save_checkpoint`
  同步创建文件，适配文件计数语义。

## 3. 实验设计

### 3.1 场景与需求

- 路网：xiasha1*1，单受控路口 J，4 个绿相位（EW 直行、EW 左转、
  NS 直行、NS 左转），12 个进口车道。
- 需求：12 个 1 h 包（sparse / dense / turn / asym / fluct），
  正式矩阵直接使用全部 12 包（2026-10-21 确认，跳过 pilot）。

### 3.2 臂设置

| 臂 | 启动 / 开关 | 类型 |
| --- | --- | --- |
| DQN | `-a dqn` | off-policy value-based |
| DDQN | `-a dqn` + `model.double_dqn: true` | DQN 变体 |
| Dueling | `-a dqn` + `model.dueling: true` | DQN 变体 |
| D3QN | 两个 flag 同时 true | DQN 变体 |
| PressLight | `-a presslight` | off-policy + pressure reward |
| FRAP | `-a frap` | off-policy + phase competition |
| MPLight | `-a mplight` | off-policy + FRAP backbone（pfrl） |
| SAC | `-a sac` | off-policy actor-critic |
| 规则参照 | `fixedtime` / `maxpressure` / `sotl` | 非训练 |

共 8 个 off-policy 臂。冒烟门：7 臂已完成工程冒烟（§4）；
vanilla DQN 未单独冒烟（其链路 = DDQN 变体去掉 flag，且历史长期在用），
正式派发前补一条 1-episode 冒烟通过即放行，失败则该列撤下。

### 3.3 统一口径与原生超参

统一：SUMO/libsumo、xiasha1 需求包、`action_interval=10`、
`steps=3600`、训练 episodes、训练 seeds、评估 SUMO seeds、评估节点、
checkpoint 选择规则、三指标口径。

已冻结（2026-10-21 确认）：

- **信号执行语义 = legacy `pseudo_step`**（不安装 `signal_control`），
  与已完成的 36 个规则基线及原始程序臂同口径；gy5 语义变更臂不在
  本 campaign 范围，跨口径指标不可比；
- **训练预算 = 200 episodes**（`trainer.episodes: 200`）；
- **训练 seeds = 3**（0/1/2，经 `--seed`；训练期 SUMO seed 保持
  `fixed_default`，与规则基线一致）；
- **评估节奏 = 每 episode 一次贪心评估**（显式
  `evaluation_episodes: [0..200]`，含 ep0 基线点）；resumable
  checkpoint 取子集 `[0, 25, 50, 100, 150, 200]`；
- **`world.saveReplay: false`**（控制 288-run 盘的 replay 产物体积；
  指标与 checkpoint 不受影响）；
- **transition 全量记录**：8 臂训练 ep 每 ep 一个
  `trajectory/episodes/episode_XXXX.npz`（schema A：state/phase/
  action/reward/next_state/done/epsilon/behavior_mode/queue/delay/
  throughput/waiting + manifest/index/sha256 校验），评估 ep 不记。
  约 288×200×360 ≈ 2070 万条 transition、~10GB 盘；直接可作为
  cl_hz1x1 式 `history_archive` 半离线数据源或离线 RL
  （batch_dqn/cql_dqn）训练集。观测维数各臂不同（16/22/13），
  数据只同臂内复用；SAC 的 `behavior_mode` 标签沿用
  `epsilon_greedy` 字面（实为随机策略采样），以 `epsilon=0.0` 识别；
- 训后评估 = `tools/eval_baseline_checkpoint.py` 加载
  `evaluation/summary.json` 选定的 `best_checkpoint`，×3 个 SUMO seeds
  （10000/10001/10002，`--sumo_seed` 显式传入）；`final` checkpoint
  作诊断对照。

保留原生：各 agent 的网络结构、学习率、batch、gamma、replay、
epsilon/探索 schedule、奖励定义。DQN 家族内部共享同一套 DQN 超参，
只切换 `double_dqn` / `dueling`。

### 3.4 观测 / 动作对齐

| 臂 | 观测 | 动作 | 奖励 |
| --- | --- | --- | --- |
| DQN 家族 / SAC | 12 进口 lane + 4 相位 one-hot = 16 | 4 | `-queue` |
| PressLight | 12 进口 + 6 出口 lane + 4 相位 one-hot = 22 | 4 | `-pressure` |
| FRAP / MPLight | 12 进口 lane + 1 相位标量 = 13 | 4 | `-queue` |

- 所有臂输出 0..3，执行 `world.id2intersection['J'].green_phases[action]`；
- FRAP/MPLight **必须 `one_hot: false`**：FRAP backbone 在
  `one_hot: true` 时要求 `num_movements == len(phase_pairs)`，xiasha1 的
  12 车道 / 4 相位会越界；
- `reward`/`loss` 不跨控制器比较；主指标 `travel_time`，强制同报
  `throughput`、`unfinished_vehicles`。

### 3.5 xiasha1 signal_config

`configs/tsc/frap.yml` / `mplight.yml`：

```yaml
xiasha1:
  phase_pairs: [[4, 10], [5, 11], [1, 7], [2, 8]]
  valid_acts: null
```

- 绿相位顺序：0 EW 直行、1 EW 左转、2 NS 直行、3 NS 左转；
- 进口车道顺序：`N0,N1,N2,E0,E1,E2,S0,S1,S2,W0,W1,W2`；
- pair 分别使用对向两臂的专用直行/左转车道；
- 信号程序或路网变更后必须重新标定。

### 3.6 需求包 network 身份

12 个 `configs/sim/xiasha1_<pack>.cfg` 的 `"network"` 均为 `xiasha1`；
CLI `-n xiasha1_<pack>` 仍是需求包身份，run 目录与
`run_manifest.network` 仍用 pack 名。不要改回 pack 名，否则
FRAP/MPLight 查 `signal_config[network]` 会 miss。

### 3.7 Checkpoint 与评估路线

- DQN 家族 / PressLight / FRAP / SAC：`checkpoint_protocol: full`；
  标准 `run.py --evaluation-manifest ... --evaluation-output ...`，
  `checkpoint_role` 可用 `best` / `final` / `resumable`。
- MPLight：`checkpoint_protocol: evaluation_only`；有
  `checkpoints/evaluation`，无 `checkpoints/resumable`；标准 manifest
  只能用 `best` / `final`。需要按模型文件评估时使用
  `tools/eval_baseline_checkpoint.py` 加载 `model/{e}.pt`。
- MPLight 的 `model/{e}.pt` 保存节奏由 `logger.save_rate` 控制；
  DQN/SAC/PressLight/FRAP 由 `evaluation_episodes` /
  `resumable_checkpoint_episodes` 控制。
- `evaluation/summary.json` 的 checkpoint 计数按实际文件计；
  MPLight 的 `resumable_checkpoint_count=0` 是设计限制，不是缺文件。

### 3.8 收敛与验收口径

- 过程：EVALUATION（贪心）曲线，每 episode 一点（ep0–ep200 共 201 点）；
  TRAIN 曲线只作诊断。
- 收敛：10-episode 滑动平均连续 20 次落在 final-20 均值 ±5% 内；
  同时报 ±3%。
- 其他：first-100 AUC、final-20 mean ± std、收敛 episode。
- 训练后：**best checkpoint 为主**（`evaluation/summary.json` 按
  min travel_time、同值取早 ep 自动选定，`best_episode` 已记录在案）、
  final 为诊断；每个选中 ckpt 跑 3 个 SUMO seeds（10000–10002）；
  可选跨包评估。
- 统计单位：训练 seed（3 个）；先按 pack 报，再汇总相对规则基线的改善。

### 3.9 推进阶段

1. 7 个 off-policy 臂各 1 episode 工程冒烟（已完成，见第 4 节；
   vanilla DQN 在派发前补冒烟门）；
2. ~~pilot~~ 跳过（2026-10-21 确认，冒烟通过即直接正式）；
3. 正式训练：8 臂 × 12 包 × 3 seeds = 288 run，队列 runner +
   `artifacts/xiasha1_online_v1/run_queue.json`；
4. 训后评估：每 run 取 `best_checkpoint` ×3 SUMO seeds = 864 eval
   （`tools/eval_baseline_checkpoint.py` + `eval_baseline_checkpoint.yml`
   legacy overlay）；
5. 正式 run 前登记台账（campaign `xiasha1_online_v1`，状态
   REGISTERED）；收割后回填台账与本文件第 4/5 节。

## 4. 实验结果

**正式结果：尚未开始。** 本轮只完成控制器可用性、checkpoint 与
训练后评估链路的工程验证。以下冒烟均为 300 步、少量更新的工程 run，
不是性能证据，也不进入台账正式统计。

| 臂 | 冒烟 run | 状态 | checkpoint / 评估验证 |
| --- | --- | --- | --- |
| DDQN | `sumo_dqn/xiasha1_sparse_low/dqn_ddqn_smoke_*` | 完成 | `full` |
| Dueling | `sumo_dqn/xiasha1_sparse_low/dqn_dueling_smoke_*` | 完成 | `full` |
| D3QN | `sumo_dqn/xiasha1_sparse_low/dqn_d3qn_smoke_*` | 完成 | `full`；标准 manifest 评估通过 |
| PressLight | `sumo_presslight/xiasha1_sparse_low/presslight_smoke_*` | 完成 | `full`；标准 manifest 评估通过 |
| FRAP | `sumo_frap/xiasha1_sparse_low/frap_smoke_*` | 完成 | `full` 2/2；标准 manifest 评估通过 |
| MPLight | `sumo_mplight/xiasha1_sparse_low/mplight_smoke_*` | 完成 | `evaluation_only` 2/0；标准 manifest 评估通过 |
| SAC | `sumo_sac/xiasha1_sparse_low/sac_smoke_*` | 完成 | `full`；标准 manifest 与 per-agent loader 结果一致 |

其他验证：

- `py_compile` 通过；
- `git diff --check` 通过；
- `tests/test_plan1_evaluation_schedule.py` 2 项通过；
- FRAP/MPLight 的 `checkpoints/evaluation` 与 `model/{e}.pt` 均能加载，
  标准 manifest 在 xiasha1 上完成评估。

## 5. 结论与后续

**结论（准备阶段）**：xiasha1 上 8 个 off-policy 臂的代码/配置链路可用
（vanilla DQN 派发前补冒烟门）；DQN 家族、PressLight、FRAP、SAC 使用
full checkpoint，MPLight 使用 evaluation-only checkpoint；动作空间统一为
4 个绿相位；观测编码和奖励保留各控制器原生差异。

**已锁定决策（2026-10-21）**：legacy `pseudo_step` 信号语义、
200 episodes、每 episode 评估、best-ckpt×3 SUMO seeds 训后评估、
12 包全量、3 训练 seeds、跳过 pilot。

**后续**：

1. 提交本轮 working tree（代码 + 冒烟/正式配置 + 数据包 + 文档），
   取得 code_commit；
2. 台账补 `xiasha1_online_v1` REGISTERED 行 + 生成
   `artifacts/xiasha1_online_v1/run_queue.json` +
   `EXPERIMENTS/xiasha1_online_v1.md` 明细页；
3. 补 vanilla DQN 1-ep 冒烟门；同时任选一条非 dqn 臂（如 sac）
   短冒烟复验 trajectory npz 写入与 `validation.json` 通过
   （白名单刚扩臂，属新路径）；
4. 派发正式矩阵（队列 runner，tmux 会话）；收割后回填台账与本文件
   第 4/5 节。

**已知边界**：FRAP/MPLight `one_hot: true` 不可用；PressLight 观测为
22 维；MPLight 无 resumable checkpoint；跨控制器不能比较 reward/loss；
xiasha1 signal_config 在信号程序/路网变更后必须重新标定；
transition 记录的 `behavior_mode` 对 SAC 为字面标签（非 epsilon 探索）。
