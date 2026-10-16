import sys
import types

import numpy as np
import torch

# Unit tests do not construct a SUMO World.  Permit importing the agent stack
# on machines where the optional libsumo wheel is absent.
sys.modules.setdefault('libsumo', types.ModuleType('libsumo'))

from agent.dqn import DQNNet
from agent.shared_dqn import SharedDQNAgent
from arterial.control import build_action_mask


class FixedQ(torch.nn.Module):
    def forward(self, value, train=True):
        rows = value.shape[0]
        return torch.arange(8, dtype=torch.float32).repeat(rows, 1)


def test_shared_dqn_batched_output_is_local_eight_actions():
    network = DQNNet(16, 8)
    output = network(torch.zeros((6, 16)))
    assert output.shape == (6, 8)


def test_shared_get_action_applies_each_intersection_mask():
    agent = SharedDQNAgent.__new__(SharedDQNAgent)
    agent.phase = False
    agent.sub_agents = 3
    agent.action_masks = build_action_mask([8, 4, 6])
    agent.model = FixedQ()
    agent.epsilon = 0.0
    agent.exploration_actions = 0
    agent.greedy_actions = 0
    actions = agent.get_action(np.zeros((3, 5)), np.zeros(3), test=True)
    assert actions.tolist() == [7, 3, 5]
