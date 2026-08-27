# arterial_1x6 Stage 0 资源与预算报告

日期：2026-07-30；设备：`linux4090`。本报告只用于冻结正式实验预算与并发度，pilot 数据不进入正式科研结果。完整机器可读证据位于 `artifacts/arterial_experiments/stage0/analysis.json`。

## 结论

- 正式 run-level 并发数：`max_workers=8`。
- 顺序在线、causal semi-offline 与 full-history upper-bound 的统一每场景预算：`B=200 episodes`。
- Stage 1 独立在线 collector 预算保持补充协议冻结值：`400 episodes`，不因 B=200而缩短。
- 每个 run 固定数值库线程为 1，不启用进程内 BLAS 线程扩张。
- 所有 Stage 0 运行仅改变 training seed，SUMO simulator config seed 保持 0。

## 资源剖析设置

代表性场景使用高负载 `700_0.6`，每个 run 为 25 episodes、每 episode 3600 秒、10 秒 action interval，即每 run 9000 decisions 与 54000 个六路口局部 transitions。依次实测 1、2、4、8 和 12 workers；每轮均对 checkpoint、history manifest、结构化指标和 NaN/Inf进行完整性检查。

| Workers | 最慢 run wall time (s) | 总吞吐 (episodes/min) | 相对单 run 变慢 | CPU 平均 | RSS 峰值 | Swap 增量 | 判定 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 646.84 | 2.32 | 0.0% | 5.17% | 基线监控版本未记录进程 RSS | 0 | 通过 |
| 2 | 669.56 | 4.48 | 3.5% | 8.12% | 1.37 GB | 0 | 通过 |
| 4 | 719.47 | 8.34 | 11.2% | 16.51% | 2.70 GB | 0 | 通过 |
| 8 | 766.30 | 15.66 | 18.5% | 32.37% | 5.40 GB | 0 | 通过 |
| 12 | 1023.89 | 17.58 | 58.3% | 46.11% | 7.70 GB | 0 | 淘汰 |

12 workers 虽然总吞吐仍略高，但最慢 run 相对单 worker 变慢 58.3%，超过预先规定的约 25%降速门槛。采用 8 workers 可保留 CPU/内存安全余量，并避免把超线程/调度竞争造成的严重单 run 延迟带入正式实验。磁盘写入峰值不超过约 22.5 MB/s，未观察到持续 I/O拥塞；所有轮次 swap 增量为 0，未发生 OOM、仿真退出或输出冲突。

## 预算 pilot

两个独立 pilot：

- `300_0.6`，training seed 9100，100 episodes；
- `700_0.6`，training seed 9101，100 episodes。

两者均生成：100 个 TRAIN records、5 次冻结 evaluation、100-episode resumable checkpoint、36000 decisions、216000 transitions；全部指标为有限值。wall time 分别为 1150.22 秒和 2025.85 秒。

稳定判据在运行完整结果前已固定：

1. 同时检查 travel time、queue、delay、throughput；
2. 比较相邻两个 20-episode 窗口；
3. 四项指标的窗口均值相对变化均不超过 10%；
4. 连续 3 个 episode endpoint 满足才判为稳定。

结果：

| 场景 | 首次稳定 episode | 最后 20 episodes travel time | queue | delay | throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| `300_0.6` | 61 | 91.65 | 2.77 | 0.154 | 4354.60 |
| `700_0.6` | 85 | 102.33 | 9.65 | 0.306 | 8799.95 |

统一预算按预先规则计算：

```text
latest_stable_episode = 85
round_up_25(1.5 × 85) = 150
B = max(200, 150) = 200
```

因此正式顺序方法每个场景使用 200 episodes。该预算不低于附件规定的 200 episodes下限，并对最晚稳定 episode 保留超过 100%的余量。

## 运行入口与证据

- 队列器：`tools/run_arterial_experiment_queue.py`
- 资源 manifest 生成器：`tools/build_arterial_stage0_manifest.py`
- 预算 manifest 生成器：`tools/build_arterial_stage0_budget_manifest.py`
- 分析器：`tools/analyze_arterial_stage0.py`
- 资源状态：`artifacts/arterial_experiments/stage0/profile_w{1,2,4,8,12}_state/`
- 预算状态：`artifacts/arterial_experiments/stage0/budget_state/`
- 汇总证据：`artifacts/arterial_experiments/stage0/analysis.json`

## 阶段门禁

Stage 0 已完成。Stage 1 可在以下启动门禁通过后执行：

- 最新代码完整回归通过；
- collector seed metadata 与 transition ID 唯一性测试通过；
- 最新代码/环境快照重新冻结；
- 20-run Stage 1 manifest 校验通过；
- 8 workers 的资源门禁仍满足磁盘、内存和 swap 阈值。
