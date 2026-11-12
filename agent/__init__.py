from .base import BaseAgent
from .rl_agent import RLAgent
from .maxpressure import MaxPressureAgent
try:
    from .colight import CoLightAgent
except ModuleNotFoundError as error:
    if error.name not in {'torch_scatter', 'torch_geometric'}:
        raise
from .dqn import DQNAgent
from .sotl import SOTLAgent
from .frap import FRAP_DQNAgent
try:
    from .ppo_pfrl import IPPO_pfrl
except ModuleNotFoundError as error:
    if error.name != 'pfrl':
        raise
# from .maddpg import MADDPGAgent
from .maddpg_v2 import MADDPGAgent
from .magd import MAGDAgent
from .presslight import PressLightAgent
from .fixedtime import FixedTimeAgent
try:
    from .mplight import MPLightAgent
except ModuleNotFoundError as error:
    if error.name != 'pfrl':
        raise
from .shared_dqn import SharedDQNAgent
from .sac import SACAgent
try:
    from . import tarl
except ModuleNotFoundError as error:
    # reproduction/tarl_tsc was removed upstream (8bae003); keep the TARL
    # shim optional so the rest of the agent registry stays importable.
    if error.name != 'reproduction':
        raise
    # 不再静默：tarl_* 智能体与 scene_attention 文本编码器的实现（reproduction/tarl_tsc）
    # 从未完整入库且已于 8bae003 删除，注意力线 att_entity_003/004 当前无法复现。
    # 恢复办法见 docs/plan_refactor_202611.md「注意力线代码缺失」。
    import logging as _logging
    _logging.getLogger(__name__).warning(
        'tarl_* 智能体不可用：reproduction/tarl_tsc 缺失（见 docs/plan_refactor_202611.md）')
try:
    from .colight import SGAColightAgent, ConcatColightAgent
except ModuleNotFoundError as error:
    if error.name not in {'torch_scatter', 'torch_geometric'}:
        raise
try:
    from .mplight import SGAMPLightAgent
except ModuleNotFoundError as error:
    if error.name not in {'torch_scatter', 'torch_geometric', 'pfrl'}:
        raise

# from .ppo_pfrl import IPPO_pfrl
