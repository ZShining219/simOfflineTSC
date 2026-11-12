# infra/ — 实验运行基座

所有研究线共用的运行能力，2026-11-12 从 `sequential/` 抽出（模块名不变）。

| 模块 | 能力 |
|---|---|
| `io.py` | 原子写 JSON、文件 sha256 |
| `core.py` | 回放记录与训练载荷结构、参数/优化器/RNG/回放的规范摘要、RNG 捕获 |
| `checkpoint.py` | 完整 checkpoint 的保存、校验、恢复，滚动恢复点管理 |
| `transaction.py` | 回合级事务（中断可恢复） |
| `state.py` | 运行阶段日志（journal） |
| `launcher.py` | 资源闸门、并发控制、尝试谱系、失败分类与自动恢复 |
| `evaluator.py` | 与训练隔离的独立评估器、在线快照 |

## 分层

```
框架层  agent/ world/ trainer/ task/ common/ utils/ generator/ data/   （LibSignal）
基座    infra/（本目录，运行）   lab/（管理：台账、溯源、核查、闸门，纯标准库）
研究线  semi_offline/  scene_attention/  hitl/
```

只许上层依赖下层：infra 可依赖框架层，不得依赖任何研究线；框架层不得依赖 infra（`tests/test_architecture.py` 校验）。

## 新实验用哪一套

- 单个训练 run：`python run.py`；批量派发：`python tools/run_experiment_queue.py`；登记/回填：`python -m lab register|harvest`。
- 需要多阶段 checkpoint 链、事务恢复、独立评估的 manifest 驱动实验：用本目录。
- `semi_offline/plan5/` 里有一套同名的 checkpoint/launcher/evaluator（8 月 plan5 实验自带的副本，已冻结），**新工作不要复用**。

## 历史 checkpoint 兼容

2026-11-12 之前保存的 checkpoint 以 `sequential.core.ReplayRecord` 等旧路径 pickle 了类对象。
导入本包时会注册旧模块名到新位置的解析（`_legacy.py`），旧 checkpoint 照常 `torch.load`，摘要与改名前一致。
旧名字只用于读取历史数据，代码里一律写新路径（架构测试禁止 import 旧名字）。
