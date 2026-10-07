"""Mechanism tests for the SGA-FLX zero-init cross-attention injection.

The module must (a) be an exact identity at init even with events present,
(b) hand its zero-initialized output head a nonzero gradient on the very
first backward pass, and (c) respect the same event-mask/grounding contract
as MultiEventSceneGuidedAttention.
"""

import numpy as np
import torch

from agent.scene_attention import SceneCondition, SceneCrossAttentionResidual


def condition(batch=2, nodes=3, events=2, dim=8):
    torch.manual_seed(0)
    return SceneCondition(
        torch.randn(batch, events, dim),
        torch.ones(batch, events, dtype=torch.bool),
        torch.zeros(batch, nodes, events, dtype=torch.bool),
    ), torch.randn(batch, nodes, dim)


def test_zero_init_output_is_exact_identity_with_events():
    module = SceneCrossAttentionResidual(hidden_dim=8, attention_dim=8, heads=2)
    cond, traffic = condition()
    output, attention, node_norm, delta = module(traffic, cond)
    torch.testing.assert_close(output, traffic)
    torch.testing.assert_close(delta, torch.zeros_like(delta))
    assert attention.shape == (2, 3, 2)
    assert module.branch_norm() == 0.0


def test_empty_events_is_identity_with_empty_diagnostics():
    module = SceneCrossAttentionResidual(hidden_dim=8, attention_dim=8, heads=2)
    traffic = torch.randn(2, 3, 8)
    empty = SceneCondition(torch.zeros(2, 0, 8),
                           torch.zeros(2, 0, dtype=torch.bool),
                           torch.zeros(2, 3, 0, dtype=torch.bool))
    output, attention, node_norm, delta = module(traffic, empty)
    torch.testing.assert_close(output, traffic)
    assert attention.shape == (2, 3, 0)
    torch.testing.assert_close(node_norm, torch.zeros_like(node_norm))


def test_output_head_gets_gradient_at_init_inner_branch_waits():
    """ControlNet/LoRA contract: zero head still receives gradient."""
    module = SceneCrossAttentionResidual(hidden_dim=8, attention_dim=8, heads=2)
    cond, traffic = condition()
    output, *_ = module(traffic, cond)
    output.square().mean().backward()
    assert module.out.weight.grad is not None
    assert float(module.out.weight.grad.norm()) > 0
    # The inner branch is disconnected only until the head moves.
    for name in ('query', 'key', 'value', 'relation_projection'):
        grad = getattr(module, name).weight.grad
        assert grad is None or float(grad.norm()) == 0


def test_branch_opens_after_optimizer_step():
    module = SceneCrossAttentionResidual(hidden_dim=8, attention_dim=8, heads=2)
    cond, traffic = condition()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.5)
    output, *_ = module(traffic, cond)
    torch.testing.assert_close(output, traffic)
    output.square().mean().backward()
    optimizer.step()
    assert module.branch_norm() > 0
    output2, *_ = module(traffic, cond)
    assert not torch.allclose(output2, traffic)


def test_invalid_event_rows_get_zero_attention():
    module = SceneCrossAttentionResidual(hidden_dim=8, attention_dim=8, heads=2)
    traffic = torch.randn(2, 3, 8)
    events = torch.randn(2, 2, 8)
    # Batch row 1 has no valid event at all.
    mask = torch.tensor([[True, True], [False, False]])
    grounding = torch.zeros(2, 3, 2, dtype=torch.bool)
    _, attention, _, _ = module(traffic, SceneCondition(events, mask, grounding))
    torch.testing.assert_close(attention[1], torch.zeros_like(attention[1]))
    np.testing.assert_allclose(attention[0].sum(-1).numpy(),
                               np.ones(3), atol=1e-6)


def test_grounding_bias_concentrates_attention():
    """With two competing events, a grounded node favors its grounded event."""
    module = SceneCrossAttentionResidual(hidden_dim=8, attention_dim=8, heads=2,
                                         grounding_bias=6.0)
    traffic = torch.zeros(1, 2, 8)
    events = torch.zeros(1, 2, 8)
    grounding = torch.tensor([[[True, False], [False, False]]])
    _, attention, _, _ = module(
        traffic, SceneCondition(events, torch.ones(1, 2, dtype=torch.bool), grounding))
    # Node 0 is grounded to event 0; node 1 is ungrounded and stays uniform.
    assert float(attention[0, 0, 0]) > float(attention[0, 0, 1])
    np.testing.assert_allclose(attention[0, 1].numpy(), [0.5, 0.5], atol=1e-6)


def test_permutation_invariant_and_diagnostics_populated():
    torch.manual_seed(3)
    module = SceneCrossAttentionResidual(hidden_dim=8, attention_dim=8, heads=2)
    traffic = torch.randn(1, 3, 8)
    events = torch.randn(1, 2, 8)
    mask = torch.ones(1, 2, dtype=torch.bool)
    grounding = torch.tensor([[[True, False], [False, True], [False, False]]])
    first = module(traffic, SceneCondition(events, mask, grounding))
    second = module(traffic, SceneCondition(
        events[:, [1, 0]], mask[:, [1, 0]], grounding[:, :, [1, 0]]))
    torch.testing.assert_close(first[0], second[0])
    # The attention tensor itself is indexed by event: compare under the
    # inverse permutation rather than elementwise.
    torch.testing.assert_close(first[1], second[1][:, :, [1, 0]])
    for attr in ('last_semantic_score', 'last_direct_bias',
                 'last_relation_bias', 'last_relevance', 'last_delta'):
        assert getattr(module, attr) is not None
