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
try:
    from . import tarl
except ModuleNotFoundError as error:
    # reproduction/tarl_tsc was removed upstream (8bae003); keep the TARL
    # shim optional so the rest of the agent registry stays importable.
    if error.name != 'reproduction':
        raise
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
