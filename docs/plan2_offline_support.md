# Plan 2 纯 Offline DQN 支持说明

## 状态与边界

截至 2026-07-23，Plan 1 四场景 × 五个 behavior training seed 的 20 个正式运行及 trajectory 白名单已经通过。Plan 2 工程能力已完成算法、数据、恢复、SUMO 生命周期、日志、汇总和并发 I/O 门禁，但正式 Plan 2 科研训练尚未开始；Pilot、失败或未完成轨迹仍禁止进入正式数据集。

Online 与 Offline 是两个明确隔离的入口：

| 模式 | 入口 | 配置目录 | 训练数据 | 输出根目录 |
|---|---|---|---|---|
| Online TSC | `run.py` | `configs/tsc/` | 仿真交互与 replay | `data/output_data/tsc/` |
| Plan 2 Offline TSC | `offline_run.py` | `configs/offline_tsc/` | Plan 1 只读 NPZ 索引 | `data/output_data/offline_tsc/` |

不得用 `run.py --agent batch_dqn/cql_dqn` 启动离线训练，也不得用 `offline_run.py` 替代 Online DQN。`run.py`、`configs/tsc/` 和 Online trainer 的默认导入链未修改。

## 架构

- `offline_run.py`：独立的准备、校验和训练入口；准备/校验数据时不初始化 SUMO。
- `dataset/offline_trajectory_dataset.py`：校验 Plan 1 白名单运行并生成只读索引，不复制 trajectory；提供阶段切分和场景均衡采样。
- `agent/offline_dqn.py`：复用现有 DQN 网络与评估观测，提供 Batch-DQN/CQL-DQN loss。
- `trainer/offline_tsc_trainer.py`：执行固定数据更新、固定节点 SUMO 评估、结构化指标和完整 checkpoint/resume。
- `task/offline_tsc_task.py`：沿用仓库 Registry 任务编排。
- `configs/offline_tsc/`：离线公共配置和两种算法的差异配置。
- `tools.experiment_plotting plan2`：严格验证正式矩阵并生成 Plan 2 汇总表。

## d3rlpy 适配决定

默认依赖固定为 `d3rlpy==2.0.4`，来源见 `THIRD_PARTY_NOTICES.md`。项目使用语义适配层而不直接采用 stock loss，原因是 d3rlpy 2.0.4 的默认行为与既有 Online DQN 不同：

- stock DQN 使用 chosen-action Huber；项目要求全 Q 向量 MSE；
- stock DiscreteCQL 基于 Double-DQN；项目要求 plain DQN target；
- 项目要求 `terminated/truncated` 均不屏蔽 bootstrap；
- leave-one-out 要求先均匀选源场景，再从该场景抽取完整 batch。

适配层的两种算法共享网络、batch size、RMSprop、gamma、target 更新和 144,000 次更新。CQL-DQN 唯一新增项为 `alpha * mean(logsumexp(Q)-Q(data_action))`，正式 `alpha=1`。

若已安装的 d3rlpy 版本不兼容，可显式选择 `--backend native`；正式运行必须在 `offline_run_metadata.json` 中保留后端和回退原因，不能静默改变 Torch、Gym 或 NumPy。

## 安装

在 `colight` 环境中先安装项目核心依赖，再以不解析依赖的方式加入固定 Offline 包，避免 pip 替换现有 Torch/Gym/NumPy：

```bash
conda activate colight
python -m pip install -r requirements.txt
python -m pip install --no-deps -r requirements-offline.txt
python -c "import d3rlpy; assert d3rlpy.__version__ == '2.0.4'"
```

## 数据准备与验证

CSV 白名单字段固定为：

```text
run_path,network,behavior_training_seed
```

必须精确列出四场景 × seeds 0～4 的 20 个正式 Plan 1 运行目录。准备器会拒绝不以 `p1_formal_` 标记的 prefix，并交叉校验运行、配置归档和 trajectory 的 config hash，防止 Pilot 或伪完成目录混入。准备命令：

```bash
python offline_run.py prepare-plan2 \
  --run-list /absolute/path/plan1_formal_runs.csv \
  --dataset-id plan2_formal_v2 \
  --source-root /absolute/plan1/root \
  --source-root-id plan1_formal_root
```

输出位于 `data/output_data/offline_datasets/plan2/<dataset_id>/`，包括源运行列表、带 SHA-256 的 shard 索引、Q1～Q4/full 和四折 leave-one-out manifest。源 NPZ 不复制、不修改。schema v2 同时保存 `source_root_id`、相对路径、原绝对路径、builder/feature/trajectory schema、创建 commit、路网/信号程序/动作映射/特征/奖励语义哈希及去重前后计数；transition 唯一键固定为 `run_id + shard_sha256 + episode_id + decision_step`。旧 v1 manifest 不做静默解释，必须使用新 dataset ID 重建。

单个 manifest 验证：

```bash
python offline_run.py validate-dataset \
  --manifest /path/to/plan2_formal_v2/datasets/sumohz1x1/full/manifest.json \
  --source-root /new/mount/plan1/root
```

读取策略为训练启动时每个所需 NPZ shard 只打开、解压一次，随后关闭文件并合并为连续内存数组；`sample_batch` 不再访问磁盘。正式运行前可执行：

```bash
python offline_run.py benchmark-dataset --manifest /path/to/manifest.json \
  --source-root /new/mount/plan1/root --batch-size 64 \
  --sample-batches 1000 --workers 8
```

## 离线训练

同场景 Batch-DQN 示例：

```bash
python offline_run.py train \
  --agent batch_dqn \
  --network sumohz1x1 \
  --dataset-manifest /path/to/plan2_formal_v2/datasets/sumohz1x1/full/manifest.json \
  --source-root /new/mount/plan1/root \
  --prefix p2_full_batch_seed1000 \
  --seed 1000 \
  --backend d3rlpy \
  --ngpu -1
```

CQL 只把 `--agent` 改为 `cql_dqn`。正式 base/offline seeds 固定为 1000～1004；默认派生 model=`base`、sampler=`base+10000`、evaluation=`base+20000`，SUMO 保持 fixed-default。四者均有独立 CLI 参数和结构化记录。`--total-updates` 仅供开发冒烟；正式汇总会拒绝不是 144,000 updates 和固定评估节点的运行。

`training_update=N` 明确表示已完成 N 次梯度更新。初始化时 target 与 online 同步；update 0 评估随机初始化策略，不做 target update；interval=10 时完成第 10、20…次梯度更新后同步。恢复于 update 9 或 10 时沿用同一口径。

恢复必须使用新的 prefix，避免覆盖不可变运行目录：

```bash
python offline_run.py train ... \
  --prefix p2_full_batch_seed1000_resume1 \
  --resume /absolute/path/checkpoints/resumable/update_072000.pt
```

resumable checkpoint 保存 online/target、optimizer、update/target 计数、数据采样 RNG、Python/NumPy/Torch RNG、CQL alpha、logical run ID、metrics 和已完成评估。评估成功写入 metrics/evaluation results 后才保存 resumable checkpoint；因此 update 0 和其他固定节点恢复时不会重复执行或产生重复日志。

每次 SUMO 评估使用 `evaluation/runs/update_N/attempt_M/` 独立目录，默认 300 秒超时、失败后重试一次。默认 libsumo 是进程内接口，不使用端口或子进程；选择 TraCI 时自动申请空闲端口，并将 stdout/log/error-log 分离。连接始终在 `finally` 清理，TraCI 子进程若未退出会依次 terminate/kill；清理失败会使 run 标记为失败，不会写 completed evaluation 或 completed run。

## 结果汇总

Plan 2 run-list 字段固定为：

```text
algorithm,dataset_id,dataset_kind,dataset_stage,evaluation_network,offline_training_seed,run_dir,include
```

正式汇总：

```bash
python -m tools.experiment_plotting plan2 \
  --run-list /absolute/path/plan2_runs.csv \
  --analysis-id plan2_formal_v1
```

默认强制每个实验单元包含 seeds 1000～1004、144,000 updates、六个固定评估点、两类 checkpoint、完成状态、有效配置归档、无目标场景泄漏和零训练环境交互。它还从结构化 metadata 校验 feature/reward/action/tl/topology hashes、模型/优化器、gamma/batch/target/alpha、dataset manifest、evaluation config、backend 和代码 commit；同名但配置不同的运行会被拒绝。prefix 仅供人读，不承担属性解析。metrics 必须以 update 单调、无重复、完整换行的 JSONL 存在，evaluation 还必须匹配 checkpoint SHA-256 与 evaluation seed。`--allow-incomplete` 只用于工程冒烟。

## 已执行验证

- 20 项 Plan 2 专项测试：全 Q target/梯度、batch 1/64、可变 action dim、Batch 与 CQL alpha=0 的 loss/梯度/step/target 完全等价、CQL 极值有限性、初始化 target 同步、target 0/1/9/10/11 与真实 resume 参数边界。
- trajectory/schema：terminated/truncated bool、episode 内链、末 transition、Q1/Q4 无丢失重复、dtype/NaN/Inf/动作范围/one-hot、source-root 重定位、v1 拒绝和 run/shard/content 去重。
- 正式数据重建：schema v2 校验 8,000 shards、2,880,000 transitions，raw 与 deduplicated 计数一致。
- I/O 基准（full 单场景 720,000 transitions）：单进程加载约 7.14 秒、驻留数组约 125.3 MB、10,000 batch 采样约 13,123 batch/s；8 workers 最大加载约 7.74 秒、合计驻留约 1.00 GB；16 workers 最大加载约 8.55 秒、合计驻留约 2.00 GB，均无打开 shard 文件。
- d3rlpy 兼容性：Python 3.10、Torch 1.13.1、Gym 0.26.2 下导入和项目适配后端通过。
- TraCI SUMO 冒烟：`sumohz1x1`、Batch-DQN、offline seed 1000、1 update；update 0/1 均完成，checkpoint hash 匹配，attempt 独立日志与 close report 有效，结束后无残留 SUMO 进程。一次故障注入运行因日志 schema 错误被正确标记为失败且无残留进程，修正后的新 prefix 运行完成。
- resume 冒烟：从 `update_000001.pt` 在新 prefix 恢复；恢复后旅行时间 `454.31529850746267`，与原运行最终评估一致。

上述冒烟只验证工程路径，不是 Plan 2 科研结果，也不能进入正式统计。
