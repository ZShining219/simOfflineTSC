"""Phase input must reach inference and both sides of replay TD updates."""
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent.colight import CoLightAgent, ColightNet


def make_agent(phase=True, one_hot=True):
    torch.manual_seed(7)
    agent = object.__new__(CoLightAgent)
    agent.phase, agent.one_hot = phase, one_hot
    agent.sub_agents, agent.lane_ob_length = 2, 3
    agent.phase_lengths = np.array([3, 2])
    agent.action_space = SimpleNamespace(n=3)
    agent.ob_length = 3 + (3 if one_hot else 1) if phase else 3
    agent.vehicle_max, agent.epsilon, agent.get_attention = 2, 0., False
    agent.ob_generator = [
        (0, SimpleNamespace(generate=lambda: np.array([2., 4.]))),
        (1, SimpleNamespace(generate=lambda: np.array([6., 8., 10.]))),
    ]
    agent.edge_idx = torch.tensor([[0, 1], [1, 0]])
    agent.replay_buffer = deque(maxlen=10)
    agent.model = ColightNet(agent.ob_length, 3, agent.phase_lengths, batch_size=2,
                            phase=phase, one_hot=one_hot, NODE_EMB_DIM=[8],
                            N_LAYERS=1, INPUT_DIM=[8], OUTPUT_DIM=[8],
                            NODE_LAYER_DIMS_EACH_HEAD=[4], NUM_HEADS=[2], OUTPUT_LAYERS=[])
    return agent


@pytest.mark.parametrize('one_hot', [True, False], ids=['one_hot', 'index'])
def test_phase_reaches_policy_and_replay_without_live_lookup(one_hot):
    agent = make_agent(one_hot=one_hot)
    observed = agent.get_ob()
    np.testing.assert_array_equal(observed, [[1, 2, 0], [3, 4, 5]])
    phases, next_phases = np.array([2, 1]), np.array([0, 0])
    expected_phase = np.array([[0, 0, 1], [0, 1, 0]]) if one_hot else np.array([[2], [1]])
    expected = np.concatenate([observed, expected_phase], axis=1).astype(np.float32)
    # The first real neural layer sees the assembled input at policy inference.
    captured = []
    handle = agent.model.embedding_MLP.embedding_node[0].register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone()))
    agent.get_action(observed, phases, test=True)
    torch.testing.assert_close(captured[-1], torch.tensor(expected))
    agent.get_action(observed, next_phases, test=True)
    assert not torch.equal(captured[-1], captured[-2])
    handle.remove()
    agent.remember(observed, phases, [1, 0], None, [-1., -2.], observed + 1,
                   next_phases, False, 'first')
    phases[:] = 0
    next_phases[:] = 1
    agent.remember(observed + 2, phases, [0, 1], None, [-2., -1.], observed + 3,
                   next_phases, False, 'second')
    agent.get_phase = lambda: pytest.fail('Replay must not consult live phases')
    before = [row[1][1].copy() for row in agent.replay_buffer]
    current, following, _, _ = agent._batchwise(list(agent.replay_buffer))
    torch.testing.assert_close(current.x[:2], torch.tensor(expected))
    zero_phase = np.array([[1, 0, 0], [1, 0, 0]]) if one_hot else np.zeros((2, 1))
    torch.testing.assert_close(following.x[:2], torch.tensor(
        np.concatenate([observed + 1, zero_phase], axis=1), dtype=torch.float32))
    other = np.array([[1, 0, 0], [1, 0, 0]]) if one_hot else np.zeros((2, 1))
    torch.testing.assert_close(current.x[2:], torch.tensor(
        np.concatenate([observed + 2, other], axis=1), dtype=torch.float32))
    for row, previous in zip(agent.replay_buffer, before):
        np.testing.assert_array_equal(row[1][1], previous)
    q = agent.model(current.x, current.edge_index)
    q.sum().backward()
    phase_grad = agent.model.embedding_MLP.embedding_node[0].weight.grad[:, 3:]
    assert torch.isfinite(phase_grad).all() and torch.count_nonzero(phase_grad) > 0


@pytest.mark.parametrize('bad', [[-1, 0], [3, 0], [0, 2], [0., 1.], [True, False], [0], [[0, 1]]])
def test_invalid_phase_is_rejected(bad):
    agent = make_agent()
    with pytest.raises(ValueError, match='valid integer index'):
        agent.get_action(agent.get_ob(), np.array(bad), test=True)


@pytest.mark.parametrize('phase', [False, True])
def test_input_version_checkpoint_compatibility(phase):
    agent = make_agent(phase=phase)
    saved = agent.model.state_dict()
    restored = make_agent(phase=phase).model
    restored.load_state_dict(saved)
    if phase:
        legacy = {k: v for k, v in saved.items() if k != '_phase_input_version'}
        with pytest.raises(RuntimeError, match='phase input version mismatch'):
            restored.load_state_dict(legacy)
        corrupt = dict(saved, _phase_input_version=torch.tensor(2))
        with pytest.raises(RuntimeError, match='phase input version mismatch'):
            restored.load_state_dict(corrupt)
    else:
        assert '_phase_input_version' not in saved
        np.testing.assert_array_equal(agent._network_input(agent.get_ob(), None), agent.get_ob())
        assert np.array_equal(agent.get_action(agent.get_ob(), [0, 0], test=True),
                              agent.get_action(agent.get_ob(), [2, 1], test=True))
