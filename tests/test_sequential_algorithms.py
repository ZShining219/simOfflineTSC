import unittest

import torch
from torch import nn

from sequential.agent import (
    DuelingDQNNet,
    SequentialDQNAgent,
    build_q_network,
    normalize_algorithm_id,
)
from sequential.initial_state import _dueling_state_dict


class FixedQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer('values', torch.as_tensor(values, dtype=torch.float32))

    def forward(self, state):
        return self.values.expand(state.shape[0], -1)


class SequentialAlgorithmTest(unittest.TestCase):
    def test_algorithm_aliases_and_network_contract(self):
        self.assertEqual(normalize_algorithm_id('dqn'), 'independent_dqn')
        self.assertEqual(normalize_algorithm_id('ddqn'), 'double_dqn')
        self.assertIsInstance(build_q_network('dueling_ddqn', 16, 8), DuelingDQNNet)
        self.assertEqual(tuple(build_q_network('double_dqn', 16, 8)(torch.zeros(2, 16)).shape), (2, 8))

    def test_double_dqn_uses_online_argmax_and_target_value(self):
        agent = SequentialDQNAgent.__new__(SequentialDQNAgent)
        agent.target_operator = 'double_dqn'
        agent.model = FixedQ([[1.0, 5.0, 2.0]])
        agent.target_model = FixedQ([[10.0, 20.0, 30.0]])
        result = agent._bootstrap_values(torch.zeros(2, 4))
        self.assertTrue(torch.equal(result, torch.tensor([20.0, 20.0])))

    def test_plain_dqn_keeps_target_network_max(self):
        agent = SequentialDQNAgent.__new__(SequentialDQNAgent)
        agent.target_operator = 'plain_dqn'
        agent.model = FixedQ([[1.0, 5.0, 2.0]])
        agent.target_model = FixedQ([[10.0, 20.0, 30.0]])
        result = agent._bootstrap_values(torch.zeros(2, 4))
        self.assertTrue(torch.equal(result, torch.tensor([30.0, 30.0])))

    def test_dueling_conversion_preserves_frozen_dqn_values(self):
        source = {
            'dense_1.weight': torch.randn(20, 16),
            'dense_1.bias': torch.randn(20),
            'dense_2.weight': torch.randn(20, 20),
            'dense_2.bias': torch.randn(20),
            'dense_3.weight': torch.randn(8, 20),
            'dense_3.bias': torch.randn(8),
        }
        converted = _dueling_state_dict(source)
        dqn = build_q_network('independent_dqn', 16, 8)
        dqn.load_state_dict(source)
        dueling = build_q_network('dueling_double_dqn', 16, 8)
        dueling.load_state_dict(converted)
        states = torch.randn(7, 16)
        self.assertTrue(torch.allclose(dqn(states), dueling(states), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
