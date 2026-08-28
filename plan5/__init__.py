"""Plan5 cross-algorithm experiment infrastructure.

The package is deliberately independent of the legacy agent implementations:
Plan5 has a frozen observation/action/reward contract and must not inherit
their incompatible defaults.
"""

from .config import Plan5Config, load_config
from .algorithms import ddqn_target
from .context import ContextHistory, context_from_events, direction_from_geometry

__all__ = [
    "Plan5Config", "load_config", "ddqn_target",
    "ContextHistory", "context_from_events",
    "direction_from_geometry",
]
