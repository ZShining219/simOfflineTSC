"""实验运行基座：checkpoint 保存/校验/恢复、回合事务、运行日志、资源闸门与调度、独立评估、原子读写与摘要。

2026-11-12 自 sequential/ 抽出（原模块名不变：io/core/checkpoint/transaction/state/launcher/evaluator），
供所有研究线共用。分层：可依赖 LibSignal 框架层（agent/world/common …），不得依赖任何研究线（semi_offline/
scene_attention/hitl）。

兼容：历史 checkpoint 以 ``sequential.core.ReplayRecord`` 等旧模块路径 pickle 了类对象。本包导入时注册
旧模块名到新位置的解析（见 ``_legacy``），旧 checkpoint 可继续 torch.load；新代码一律使用新路径
（tests/test_architecture.py 禁止 import 旧名字）。
"""
from . import _legacy

_legacy.install()
