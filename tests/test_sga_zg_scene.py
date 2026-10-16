"""Mechanism tests for the frozen Zscene/Gscene SGA contract."""

import numpy as np
import torch

from agent.scene_attention import (
    MultiEventSceneGuidedAttention,
    SceneCondition,
    SceneContextEncoder,
    StructuredSceneMeta,
)
from utils.text_grounding import Gnode


class DummyFrozenText(torch.nn.Module):
    output_dim = 4

    def forward(self, texts):
        rows = []
        for text in texts:
            rows.append(torch.tensor([len(text), 1.0, 2.0, 3.0]))
        return torch.stack(rows)


def gnode():
    return Gnode(
        node_mask=np.array([True, True, False]),
        node_report_mask=np.array([[True, False], [False, True], [False, False]]),
        feature_report_mask=np.zeros((3, 2, 2), dtype=bool),
        valid_feature_mask=np.ones((3, 2), dtype=bool),
        lane_report_mask=np.zeros((2, 2), dtype=bool),
        report_ids=('event-a', 'event-b'), scope='local')


def test_event_fusion_keeps_location_out_of_zscene():
    encoder = SceneContextEncoder(
        text_encoder=DummyFrozenText(), text_dim=4,
        text_hidden_dim=8, meta_dim=4, fusion_hidden_dim=128)
    base = StructuredSceneMeta(event_type='lane_blockage', location='unknown')
    located = StructuredSceneMeta(event_type='lane_blockage', location='junction-2-1')
    first = encoder.encode_events(('event text',), (base,), Gnode(
        np.array([True]), np.array([[True]]), np.zeros((1, 1, 1), bool),
        np.ones((1, 1), bool), np.zeros((1, 1), bool), ('event-a',), 'local'))
    second = encoder.encode_events(('event text',), (located,), Gnode(
        np.array([True]), np.array([[True]]), np.zeros((1, 1, 1), bool),
        np.ones((1, 1), bool), np.zeros((1, 1), bool), ('event-a',), 'local'))
    torch.testing.assert_close(first.z_events, second.z_events)


def test_multi_event_attention_is_permutation_invariant():
    torch.manual_seed(3)
    module = MultiEventSceneGuidedAttention(hidden_dim=8, attention_dim=4)
    traffic = torch.randn(1, 3, 8)
    events = torch.randn(1, 2, 8)
    mask = torch.tensor([[True, True]])
    grounding = torch.tensor([[[True, False], [False, True], [False, False]]])
    first = module(traffic, SceneCondition(events, mask, grounding))
    second = module(traffic, SceneCondition(
        events[:, [1, 0]], mask[:, [1, 0]], grounding[:, :, [1, 0]]))
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[2], second[2])
    torch.testing.assert_close(first[3], second[3])


def test_no_event_is_exact_residual_identity():
    torch.manual_seed(4)
    module = MultiEventSceneGuidedAttention(hidden_dim=8, attention_dim=4)
    traffic = torch.randn(2, 3, 8)
    empty = SceneCondition(torch.zeros(2, 0, 8), torch.zeros(2, 0, dtype=torch.bool),
                           torch.zeros(2, 3, 0, dtype=torch.bool))
    output, attention, node_gate, feature_gate = module(traffic, empty)
    torch.testing.assert_close(output, traffic)
    assert attention.shape == (2, 3, 0)
    assert torch.count_nonzero(node_gate) == 0
    torch.testing.assert_close(feature_gate, torch.ones_like(feature_gate))


def test_single_event_grounding_changes_independent_relevance():
    module = MultiEventSceneGuidedAttention(hidden_dim=8, attention_dim=4,
                                            grounding_bias=4.0)
    traffic = torch.zeros(1, 2, 8)
    events = torch.zeros(1, 1, 8)
    condition = SceneCondition(
        events, torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([[[True], [False]]]))
    _, relevance, _, _ = module(traffic, condition)
    assert relevance.shape == (1, 2, 1)
    assert float(relevance[0, 0, 0]) > float(relevance[0, 1, 0])
    assert float(relevance[0, 0, 0]) < 1.0
