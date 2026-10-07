import numpy as np
import torch
from types import SimpleNamespace

from agent.colight import ColightNet
from agent.scene_attention import MPLightSGAAdapter, SceneGuidedAttention
from agent.mplight import FRAP


def test_sga_attention_is_normalized_and_differentiable():
    torch.manual_seed(7)
    module = SceneGuidedAttention(hidden_dim=8, attention_dim=4)
    traffic = torch.randn(2, 5, 8, requires_grad=True)
    scene = torch.randn(2, 8, requires_grad=True)
    output, alpha = module(traffic, scene)
    assert output.shape == (2, 5, 8)
    assert alpha.shape == (2, 5)
    np.testing.assert_allclose(alpha.detach().sum(1).numpy(), np.ones(2), atol=1e-6)
    (output.square().mean() + alpha[:, 0].mean()).backward()
    assert module.scene_query.weight.grad is not None
    assert float(module.scene_query.weight.grad.norm()) > 0


def test_colight_sga_forward_keeps_legal_action_shape():
    torch.manual_seed(7)
    network = ColightNet(
        10, 8, [8] * 16, batch_size=2, phase=False,
        N_LAYERS=1, INPUT_DIM=[128], OUTPUT_DIM=[128], NODE_EMB_DIM=[128],
        NUM_HEADS=[2], NODE_LAYER_DIMS_EACH_HEAD=[16], OUTPUT_LAYERS=[],
        sga_enabled=True, sga_attention_dim=32,
    )
    edge = torch.tensor([
        list(range(16)), list(range(1, 16)) + [0],
    ], dtype=torch.long)
    observations = torch.randn(32, 10)
    scene = network.encode_scene([
        {'event_type': 'unknown'}, {'event_type': 'lane_blockage'},
    ])
    values = network(observations, edge, train=True, scene_embedding=scene)
    assert values.shape == (32, 8)
    assert network.last_scene_attention.shape == (2, 16)
    np.testing.assert_allclose(
        network.last_scene_attention.detach().sum(1).numpy(), np.ones(2), atol=1e-6)


def test_mplight_adapter_responds_to_scene_and_trains():
    torch.manual_seed(7)
    adapter = MPLightSGAAdapter(state_dim=4, num_actions=8, hidden_dim=16,
                                attention_dim=8)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.02)
    states = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    normal = torch.zeros(2, 16)
    event = torch.zeros(2, 16)
    event[:, 0] = 2.0
    target_normal = torch.full((2, 4), 0.25)
    target_event = torch.zeros(2, 4)
    target_event[:, 2] = 1.0
    for _ in range(120):
        _, alpha_normal = adapter(states, normal)
        _, alpha_event = adapter(states, event)
        loss = ((alpha_normal - target_normal) ** 2).mean()
        loss = loss + ((alpha_event - target_event) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    _, alpha_normal = adapter(states, normal)
    _, alpha_event = adapter(states, event)
    assert float((alpha_normal - alpha_event).abs().sum()) > 0.5
    assert int(alpha_event[0].argmax()) == 2
    assert any(p.grad is not None and float(p.grad.norm()) > 0
               for p in adapter.parameters())


def test_mplight_frap_sga_forward_uses_augmented_public_scene():
    params = SimpleNamespace(param={
        'demand_shape': 1, 'one_hot': False, 'sga_enabled': True,
        'scene_feature_dim': 16, 'sga_attention_dim': 8,
        'sga_temperature': 1.0, 'sga_dropout': 0.0,
        'sga_residual_scale': 1.0,
    })
    phase_pairs = [[4, 10], [1, 7], [5, 11], [2, 8],
                   [10, 11], [4, 5], [7, 8], [1, 2]]
    competition = torch.zeros(8, 7, dtype=torch.long)
    network = FRAP(params, 8, phase_pairs, competition)
    states = torch.randn(4, 1 + 16 + 12)
    states[:, 0] = 0
    values = network(states)
    assert values.params[0].shape == (4, 8)
    assert network.last_sga_attention.shape == (4, 12)
    np.testing.assert_allclose(
        network.last_sga_attention.detach().sum(1).numpy(), np.ones(4), atol=1e-6)
