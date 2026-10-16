"""Unit checks for the formal same-backbone concat comparator."""

import numpy as np
import torch

from agent.colight import ColightNet
from agent.scene_attention import EventSceneRepresentation


def _network():
    return ColightNet(
        10, 8, [8] * 16, batch_size=2, phase=False,
        N_LAYERS=1, INPUT_DIM=[128], OUTPUT_DIM=[128], NODE_EMB_DIM=[128],
        NUM_HEADS=[2], NODE_LAYER_DIMS_EACH_HEAD=[16], OUTPUT_LAYERS=[],
        sga_enabled=True, sga_input_mode='concat', sga_attention_dim=32)


def _representation(event_count=2):
    mask = torch.zeros(16, event_count, dtype=torch.bool)
    relation = torch.zeros(16, event_count, 3, dtype=torch.bool)
    for index in range(event_count):
        mask[index, index] = True
        relation[index, index, 0] = True
    return EventSceneRepresentation(
        torch.randn(event_count, 128), tuple(f'e{idx}' for idx in range(event_count)),
        mask, relation)


def test_concat_forward_is_trainable_and_legal():
    torch.manual_seed(7)
    network = _network()
    edge = torch.tensor([list(range(16)), list(range(1, 16)) + [0]], dtype=torch.long)
    observations = torch.randn(16, 10, requires_grad=True)
    output = network(observations, edge, train=True, scene_embedding=_representation())
    assert output.shape == (16, 8)
    output.square().mean().backward()
    assert network.concat_scene_fusion[0].weight.grad is not None
    assert float(network.concat_scene_fusion[0].weight.grad.norm()) > 0


def test_concat_empty_scene_is_zero_and_finite():
    torch.manual_seed(7)
    network = _network()
    edge = torch.tensor([list(range(16)), list(range(1, 16)) + [0]], dtype=torch.long)
    observations = torch.randn(16, 10)
    empty = EventSceneRepresentation(
        torch.zeros(0, 128), (), torch.zeros(16, 0, dtype=torch.bool),
        torch.zeros(16, 0, 3, dtype=torch.bool))
    output = network(observations, edge, train=False, scene_embedding=empty)
    assert output.shape == (16, 8)
    assert np.isfinite(output.detach().numpy()).all()
    assert float(network.last_concat_scene.abs().sum()) == 0.0
