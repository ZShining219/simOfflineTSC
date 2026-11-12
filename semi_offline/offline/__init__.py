"""纯离线训练（Plan 2：batch_dqn / cql_dqn）。

入口 ``python -m semi_offline.offline <prepare-plan2|validate-dataset|benchmark-dataset|train>``；
智能体、trainer、task、数据集实现自框架层迁入（2026-11-12），导入即向注册表登记。
"""
from .cli import build_parser, main  # noqa: F401
