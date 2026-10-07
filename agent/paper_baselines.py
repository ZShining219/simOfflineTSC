"""Explicit new reward variants; never overwrite legacy registered models.

Import through tools.run_paper_baseline. Model/training structures are inherited
unchanged; only get_reward is replaced under an owned, frozen contract. These
are author-reward SUMO adaptations, not author-exact full reproductions.
"""
import numpy as np

from common.registry import Registry
from common.paper_rewards import load_profile
from .frap import FRAP_DQNAgent
from .presslight import PressLightAgent


class PaperRewardMixin:
    def __init__(self, world, rank):
        config = Registry.mapping['model_mapping']['setting'].param
        self.reward_profile = load_profile(config['reward_profile'], self.reward_owner)
        if not hasattr(world, 'paper_reward_source'):
            raise ValueError('paper_* agents require the explicit paper baseline runner')
        super().__init__(world, rank)
        self.reward_nodes = (tuple(world.intersection_ids) if self.sub_agents > 1
                             else (world.intersection_ids[rank],))

    def get_reward(self):
        values = [self.world.paper_reward_source.reward(self.reward_profile, node)['reward']
                  for node in self.reward_nodes]
        result = np.asarray(values, dtype=np.float32)
        return result if self.sub_agents > 1 else result[0]


@Registry.register_model('paper_frap')
class PaperFRAP(PaperRewardMixin, FRAP_DQNAgent):
    reward_owner = 'frap'


@Registry.register_model('paper_presslight')
class PaperPressLight(PaperRewardMixin, PressLightAgent):
    reward_owner = 'presslight'


try:
    from .colight import CoLightAgent
except ModuleNotFoundError as error:
    if error.name not in {'torch_scatter', 'torch_geometric'}:
        raise
else:
    @Registry.register_model('paper_colight')
    class PaperCoLight(PaperRewardMixin, CoLightAgent):
        reward_owner = 'colight'
