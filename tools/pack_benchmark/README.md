# tools.pack_benchmark — pack × controller 性能记录与作图

本模块只为“需求包 × 控制器”的**端到端性能记录和作图**服务。它读取已经完成的 `run.py` run 目录和 `orig` 直跑，输出一张 canonical 数据表、固定指标定义的图和 provenance manifest。

它**不训练模型、不修改控制器、不重新评估 checkpoint**。需要重跑控制器时，仍走 `run.py` + 队列 runner；本模块只消费其结果。

## 为什么需要它

同一批实验里，`queue`、`waiting`、`real_delay` 很容易在不同脚本里被重新定义。本模块把定义收敛到 `metrics.py` 的 `METRIC_REGISTRY`：

- 每个指标有唯一 `metric_id`、单位、定义、统计人群、时间口径和 `cross_arm_comparable` 标记。
- 图和表只读 canonical dataset，不在绘图脚本里重算指标。
- 每个 cell 记录 `source_kind`、`run_id`、`run_dir`、`config_sha256`、`records_sha256`、`control_semantics`、`sumo_seed_mode` 和输入文件 sha256。
- `manifest.json` / `captions.md` 自动记录指标定义、输入 hash、代码 HEAD/dirty、SUMO 版本和图文件 hash。

## 快速使用（xiasha1 现有 36 个 run）

```bash
conda run -n colight python -m tools.pack_benchmark all \
  --spec configs/benchmarks/xiasha1_notrain.yaml \
  --analysis-id xiasha1_notrain_canonical_20261021 \
  --orig-mode rerun \
  --legacy-matrix data/output_data/analysis/xiasha1_notrain/notrain_matrix.json
```

也可以分步：

```bash
python -m tools.pack_benchmark collect  --spec ... --analysis-id ...
python -m tools.pack_benchmark validate --analysis-id ... --legacy-matrix ...
python -m tools.pack_benchmark plot     --analysis-id ...
```

`--orig-mode`：

- `rerun`：重新跑 12 个原生程序包，保留 `statistic.xml` / `tripinfo.xml` / `queue_samples.csv`。
- `reuse`：复用已经存在的 `raw/orig_prog/<pack>/orig_metrics.json`（适合调试）。
- `skip`：不采 orig；通常只用于开发，validate 会因缺少 cell 失败。

## Episode 内逐仿真步曲线

现有 36 个 ctl run 只有 `FINAL_EVALUATION`，没有 episode 内采样。要得到逐仿真步曲线，需要一次**只加采样、不改控制器语义**的重放：

```bash
# 1) 生成 36 个 replay task（prefix=xi1ts_*，overlay 只开 eval_step_log）
python -m tools.pack_benchmark timeseries-queue \
  --spec configs/benchmarks/xiasha1_notrain_timeseries.yaml \
  --queue artifacts/xiasha1_timeseries/run_queue.json

# 2) 用仓库队列 runner 跑 replay
conda run -n colight python tools/run_experiment_queue.py \
  --queue artifacts/xiasha1_timeseries/run_queue.json \
  --state artifacts/xiasha1_timeseries/run_state \
  --parallel 6 --poll 5

# 3) 收集 step log，并与 canonical dataset 的 FINAL_EVALUATION 校验
python -m tools.pack_benchmark timeseries-collect \
  --spec configs/benchmarks/xiasha1_notrain_timeseries.yaml \
  --reference-dataset data/output_data/analysis/pack_benchmark/<canonical-id>/benchmark.json \
  --analysis-id <timeseries-id> --orig-mode rerun

# 4) 出图
python -m tools.pack_benchmark timeseries-plot --analysis-id <timeseries-id>
```

- overlay：`configs/tsc/xiasha1_timeseries.yml`，仅设置 `trainer.eval_step_log: true` 和 `eval_step_log_interval: 1`。控制器、信号执行、seed 均不变。
- ctl 侧 1 s 采样（`evaluation/steps_ep0001.jsonl`），orig 侧 10 s 采样（`raw/orig_prog/<pack>/queue_samples.csv`）；图按 10 s 抽样绘制，CSV 保留原始分辨率。
- `timeseries-collect` 会自动对比 replay 的 FINAL_EVALUATION 与 canonical cell（ATT/throughput/unfinished/queue/waiting/real_delay/delay/phase_switches）；不一致时非零退出。
- 当前 xiasha 数据：`xi1ts_*` 36 个 replay，288 个终值对比 0 mismatch；`tables/timeseries_long.csv` 共 133,920 行（36×3600 + 12×360）。
- 曲线图：`fig06` queue、`fig07` throughput、`fig08` backlog、`fig09` 累计 ATT、`fig10` waiting。

## 指标定义（canonical）

| metric_id | 定义 | 人群 / 时间口径 | 跨臂 |
|---|---|---|---|
| `att_arrived_s` | 到达车辆 depart→arrive 均值 | arrived / per vehicle | 是 |
| `throughput_arrived_veh` | 到达车辆数 | arrived / 终点累计 | 是 |
| `unfinished_departed_veh` | 已出发未到达（含 teleport） | departed / 终点快照 | 是 |
| `uninserted_loaded_veh` | loaded - inserted；ctl 由 orig loaded 派生 | loaded / 终点快照 | 是（ctl 标 derived） |
| `backlog_total_veh` | `unfinished + uninserted` | loaded / 终点快照 | 是 |
| `queue_decision_mean_veh` | 每 10s 决策步 Σ12 进口道 wait>0 车辆数均值 | 受控进口道 / 决策步均值 | 是 |
| `waiting_end_in_net_s` | 终点在网车辆累计等待均值 | in-net / 终点快照 | 是（带 `waiting_n`） |
| `real_delay_world_s` | world lane 轨迹近似时延 | trajectory / per vehicle | 否（ctl only） |
| `real_delay_tripinfo_s` | SUMO tripinfo `timeLoss` | arrived / per vehicle | 否（orig only） |
| `lane_delay_index` | `1-v̄/vmax` 决策步均值 | 受控进口道 / 决策步均值 | 否（ctl only） |
| `teleports_count` | statistic teleports | network / 终点累计 | 否（orig only） |
| `phase_switches` / `phase_switch_frequency` | 动作切换次数/频率 | controller decisions | 否（诊断） |

注意：

- `ATT` 只统计**已到达车辆**；必须和 throughput、backlog 一起看。
- `queue` 是时间平均，`waiting_end_in_net_s` 是终点快照，二者不是同一时间口径。
- `real_delay_world_s` 与 `real_delay_tripinfo_s` 不是同一个估计量，不能混在同一张跨臂图里。
- 旧 `notrain_matrix.json` 的 `real_delay_s` 是一列混装：ctl 映射到 `real_delay_world_s`，orig 映射到 `real_delay_tripinfo_s`。

## 当前 xiasha 数据的一个已知语义问题

现有 36 个 ctl run 走 legacy `Intersection.pseudo_step`：`current_phase_time` 在相位切换后不重置，`t_fixed=30` 实际几乎每个决策都换相位；本批 `phase_switches` 为 fixedtime 357/360、sotl 357/360、maxpressure 316/360。`cells.csv` 的 `control_semantics` 和 `warnings` 会标出这一点。

本模块**不修复**控制器，也不重跑。若之后启用 `trainer.signal_control: sumo-green-yellow-v1` 重跑，必须作为新 benchmark/campaign，旧数据保留为 legacy，新旧不在同一对照表混排。

## 输出目录

```text
data/output_data/analysis/pack_benchmark/<analysis_id>/
  benchmark.json
  captions.md
  manifest.json
  figures_manifest.json
  tables/{cells.csv,metrics_long.csv,metric_definitions.csv,packs.csv,arms.csv}
  raw/orig_prog/<pack>/{statistic.xml,tripinfo.xml,summary.xml,
                         queue_samples.csv,orig_metrics.json,command.json}
  manifests/{collection_manifest.json,validation.json}
  figures/fig01_overview_three_metrics.{png,pdf}
  figures/fig02_queue_waiting.{png,pdf}
  figures/fig03_metric_heatmaps.{png,pdf}
  figures/fig04_backlog_decomposition.{png,pdf}
  figures/fig05_tradeoff_scatter.{png,pdf}
```

`analysis-id` 必须是新目录；模块默认拒绝覆盖已有的非空目录。旧分析目录不会被修改。

## Spec

`configs/benchmarks/xiasha1_notrain.yaml` 声明：

- `horizon_s` / `decision_interval_s` / `seed` / `episode`
- `pack_manifest` 和 `pack_dir_template`
- `in_edges` / `out_edges`（orig queue 采样用）
- `arms`：一个 `direct_sumo` 臂 + 若干 `run_dir` 臂；每个 `run_dir` 臂声明 `agent`、`run_dir_template` 和 `expected_control_semantics`。

新增一个控制器或一批包时，只改 spec，不改绘图代码。

## 验证

- `validate` 检查：pack×arm 完整性、`loaded = throughput + unfinished + uninserted`、`backlog = unfinished + uninserted`、必填 metric 有限。
- `--legacy-matrix` 会逐值对比旧 `notrain_matrix.json`；当前 xiasha 数据为 384 个值、0 mismatch。
- 测试：`conda run -n colight python -m pytest -q tests/test_pack_benchmark.py`

## 后续（不在本版）

- 用 `world/sumo_signal_control.py` 的 `sumo-green-yellow-v1` 重跑规则控制器，作为新 benchmark。
- 上游 `run_manifest.json` 记录真实启动 commit/dirty；上游 `FINAL_EVALUATION` 增加 `loaded/inserted/teleports` 和 `waiting_n`，这样 ctl 的 backlog/waiting 不必派生。
- 把 `common/plotting_style.py` 抽成共享样式层，供本模块和 `tools.experiment_plotting` 共用。
