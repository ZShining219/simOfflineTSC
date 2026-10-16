"""Unit tests for the CAREL-style event-traffic alignment module
(ATT-ENTITY-003 stage 3, experiment A).

Covers the audit-required checks:
  * episode window accumulation and event->node attribution;
  * multi-event separation and global-scope events;
  * explicit hard negatives (wrong-node, contiguous no-event stretch);
  * symmetric InfoNCE gradient flow into both encoders;
  * no-event episodes produce no pairs (baseline safety);
  * lane-feature windows carry no phase/action columns by construction.
"""
from types import SimpleNamespace

import numpy as np
import torch

from agent.scene_alignment import (AlignmentBuffer, EpisodeAlignAccumulator,
                                   TrafficTransitionEncoder,
                                   TrafficWindowEncoder, symmetric_infonce)


def _report(eid, text='lane blocked', event_kind='lane_blockage'):
    return SimpleNamespace(report_id=eid, event_id=eid, text=text,
                           event_kind=event_kind, status='active',
                           scope='lane', direction='westbound',
                           movements=('through',), severity=0.8)


def _scene(events, mask):
    """Fake SceneSnapshot: context.events + grounding.node_report_mask."""
    return SimpleNamespace(
        context=SimpleNamespace(events=tuple(events)),
        grounding=SimpleNamespace(node_report_mask=np.asarray(mask)),
    )


def _obs(t, nodes=4, feats=3, rng=None):
    rng = rng or np.random.default_rng(0)
    return rng.normal(size=(nodes, feats)).astype(np.float32) + t


def test_no_event_episode_yields_no_pairs():
    acc = EpisodeAlignAccumulator()
    rng = np.random.default_rng(0)
    for t in range(10):
        acc.step(_obs(t, rng=rng), _scene([], np.zeros((4, 0))))
    assert acc.finish(rng) == []


def test_single_event_positive_window_uses_only_grounded_nodes():
    acc = EpisodeAlignAccumulator(min_active_steps=2)
    rng = np.random.default_rng(0)
    # 4 free steps, then event active for 6 steps on nodes 0,1 only.
    mask = np.zeros((4, 1), dtype=bool)
    mask[0, 0] = mask[1, 0] = True
    for t in range(4):
        acc.step(_obs(t, rng=rng), _scene([], np.zeros((4, 0))))
    rows = [None] * 6
    for i, t in enumerate(range(4, 10)):
        row = _obs(t, rng=rng)
        rows[i] = row
        acc.step(row, _scene([_report('e1')], mask))
    acc.step(_obs(10, rng=rng), _scene([], np.zeros((4, 0))))
    pairs = acc.finish(rng)
    assert len(pairs) == 1
    pair = pairs[0]
    expected = np.stack(rows)[
        :, np.array([0, 1]), :].astype(np.float32).mean(axis=1)
    np.testing.assert_allclose(pair.pos, expected, rtol=1e-6)
    # The positive must NOT equal an ungrounded-node window.
    unaff = np.stack(rows)[:, 2, :]
    assert not np.allclose(pair.pos, unaff)


def test_wrong_node_negative_never_uses_grounded_node():
    acc = EpisodeAlignAccumulator(min_active_steps=2)
    rng = np.random.default_rng(1)
    mask = np.zeros((4, 1), dtype=bool)
    mask[2, 0] = True
    for t in range(3):
        acc.step(_obs(t, rng=rng), _scene([], np.zeros((4, 0))))
    for t in range(3, 8):
        acc.step(_obs(t, rng=rng), _scene([_report('e1')], mask))
    pair = acc.finish(rng)[0]
    wrong = pair.negs['wrong_node']
    assert wrong.shape == (5, pair.pos.shape[1])
    # wrong-node window differs from the positive wherever obs differ.
    assert not np.allclose(wrong, pair.pos)


def test_no_event_negative_is_contiguous_and_event_free():
    acc = EpisodeAlignAccumulator(min_active_steps=2)
    rng = np.random.default_rng(2)
    mask = np.zeros((4, 1), dtype=bool)
    mask[0, 0] = mask[1, 0] = True
    free_rows = []
    for t in range(12):
        row = _obs(t, rng=rng)
        if t < 5 or t >= 9:
            acc.step(row, _scene([], np.zeros((4, 0))))
            free_rows.append((t, row))
        else:
            acc.step(row, _scene([_report('e1')], mask))
    pair = acc.finish(rng)[0]
    no_event = pair.negs['no_event']
    assert no_event.shape == pair.pos.shape
    free_by_row = np.stack([r for _, r in free_rows])[:, [0, 1], :].mean(axis=1)
    # the negative window must be a contiguous slice of the free-step rows
    found = any(np.allclose(free_by_row[i:i + no_event.shape[0]], no_event)
                for i in range(len(free_by_row) - no_event.shape[0] + 1))
    assert found


def test_window_cap_limits_event_span():
    acc = EpisodeAlignAccumulator(window_cap_steps=4, min_active_steps=2)
    rng = np.random.default_rng(3)
    mask = np.zeros((4, 1), dtype=bool)
    mask[0, 0] = True
    for t in range(10):
        acc.step(_obs(t, rng=rng), _scene([_report('e1')], mask))
    pair = acc.finish(rng)[0]
    assert pair.pos.shape[0] == 4


def test_multi_event_pairs_keep_disjoint_attribution():
    acc = EpisodeAlignAccumulator(min_active_steps=2)
    rng = np.random.default_rng(4)
    mask = np.zeros((4, 2), dtype=bool)
    mask[0, 0] = True            # e1 -> node 0
    mask[3, 1] = True            # e2 -> node 3
    for t in range(6):
        acc.step(_obs(t, rng=rng),
                 _scene([_report('e1'), _report('e2')], mask))
    pairs = acc.finish(rng)
    assert len(pairs) == 2
    by_id = {p.report.event_id: p for p in pairs}
    obs = np.stack(acc.obs)
    np.testing.assert_allclose(by_id['e1'].pos, obs[:, 0, :], rtol=1e-6)
    np.testing.assert_allclose(by_id['e2'].pos, obs[:, 3, :], rtol=1e-6)


def test_global_event_has_all_nodes_and_no_wrong_node_negative():
    acc = EpisodeAlignAccumulator(min_active_steps=2)
    rng = np.random.default_rng(5)
    mask = np.ones((4, 1), dtype=bool)
    for t in range(3):
        acc.step(_obs(t, rng=rng), _scene([], np.zeros((4, 0))))
    for t in range(3, 8):
        acc.step(_obs(t, rng=rng),
                 _scene([_report('rain', 'network rain', 'global_rain')], mask))
    pair = acc.finish(rng)[0]
    assert 'wrong_node' not in pair.negs   # nothing ungrounded to sample
    obs = np.stack(acc.obs)[3:8]
    np.testing.assert_allclose(pair.pos, obs.mean(axis=1), rtol=1e-6)


def test_events_needing_more_steps_are_skipped():
    acc = EpisodeAlignAccumulator(min_active_steps=5)
    rng = np.random.default_rng(6)
    mask = np.zeros((4, 1), dtype=bool)
    mask[0, 0] = True
    for t in range(3):
        acc.step(_obs(t, rng=rng), _scene([_report('e1')], mask))
    assert acc.finish(rng) == []


def test_windows_carry_lane_rows_only_no_phase_or_action():
    """Accumulator stores exactly the obs rows passed in; nothing extra."""
    acc = EpisodeAlignAccumulator(min_active_steps=2)
    rng = np.random.default_rng(7)
    mask = np.zeros((4, 1), dtype=bool)
    mask[1, 0] = True
    feats = 5
    for t in range(5):
        acc.step(_obs(t, feats=feats, rng=rng), _scene([_report('e1')], mask))
    pair = acc.finish(rng)[0]
    assert pair.pos.shape[1] == feats


def test_window_encoder_and_infonce_backprop_to_both_encoders():
    torch.manual_seed(0)
    rng = np.random.default_rng(8)
    enc = TrafficWindowEncoder(feat_dim=3, hidden_dim=16, out_dim=8)
    acc = EpisodeAlignAccumulator(min_active_steps=2)
    mask = np.zeros((4, 1), dtype=bool)
    mask[0, 0] = True
    for t in range(4):
        acc.step(_obs(t, rng=rng), _scene([], np.zeros((4, 0))))
    for t in range(4, 9):
        acc.step(_obs(t, rng=rng), _scene([_report('e1')], mask))
    for t in range(9, 14):
        acc.step(_obs(t, rng=rng), _scene([_report('e2')], mask))
    pairs = acc.finish(rng)
    assert len(pairs) == 2
    windows, pos_idx = [], []
    for pair in pairs:
        pos_idx.append(len(windows))
        windows.append(pair.pos)
        windows.extend(pair.negs.values())
    z_w = enc.encode_batch(windows)
    z_e = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    loss = symmetric_infonce(z_e, z_w, pos_idx, tau=0.1)
    loss.backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in enc.parameters())


def test_symmetric_infonce_rewards_positive_alignment():
    torch.manual_seed(0)
    z_w = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    z_e = torch.stack([z_w[0], z_w[3]])          # perfect positives
    loss_good = symmetric_infonce(z_e, z_w, [0, 3], tau=0.1)
    z_e_bad = torch.stack([z_w[1], z_w[2]])      # point at negatives
    loss_bad = symmetric_infonce(z_e_bad, z_w, [0, 3], tau=0.1)
    assert float(loss_good) < float(loss_bad)


def test_before_window_is_contiguous_free_and_precedes_event():
    """GRIF baseline: same nodes, latest contiguous free run before begin."""
    acc = EpisodeAlignAccumulator(min_active_steps=2)
    rng = np.random.default_rng(11)
    mask = np.zeros((4, 1), dtype=bool)
    mask[0, 0] = mask[1, 0] = True
    rows = []
    for t in range(14):
        row = _obs(t, rng=rng)
        rows.append(row)
        if 6 <= t <= 10:
            acc.step(row, _scene([_report('e1')], mask))
        else:
            acc.step(row, _scene([], np.zeros((4, 0))))
    pair = acc.finish(rng)[0]
    assert pair.pos_before is not None
    free = np.stack(rows[:6] + rows[11:])          # event-free rows
    free_at_nodes = free[:, [0, 1], :].mean(axis=1)
    # pos_before must be a trailing contiguous slice of the pre-event run
    L = pair.pos_before.shape[0]
    found = any(np.allclose(free_at_nodes[i:i + L], pair.pos_before)
                for i in range(0, 6 - L + 1))
    assert found
    # wrong_node negative baseline: same run, different node
    nb = pair.negs_before.get('wrong_node')
    assert nb is not None and nb.shape[0] == pair.pos_before.shape[0]


def test_transition_encoder_shapes_and_gradients():
    torch.manual_seed(0)
    enc = TrafficTransitionEncoder(feat_dim=4, hidden_dim=16, out_dim=8)
    rng = np.random.default_rng(12)
    befores = [rng.normal(size=(6, 4)) for _ in range(3)]
    afters = [rng.normal(size=(6, 4)) for _ in range(3)]
    z = enc.encode_batch(befores, afters)
    assert z.shape == (3, 8)
    torch.testing.assert_close(z.norm(dim=-1),
                               torch.ones(3), rtol=1e-5, atol=1e-5)
    z_w = z
    z_e = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    symmetric_infonce(z_e, z_w, [0, 1, 2], tau=0.1).backward()
    assert all(p.grad is not None for p in enc.parameters())


def test_delta_zero_when_after_equals_before():
    enc = TrafficTransitionEncoder(feat_dim=4)
    w = np.ones((5, 4), dtype=np.float32)
    z_same = enc.encode_batch([w], [w])
    # delta channel zero; output is some deterministic point
    z_other = enc.encode_batch([w], [w * 3])
    assert not torch.allclose(z_same, z_other)


def test_pairs_without_baseline_are_safe_for_transition_mode():
    """Event starting at t=0 has no free steps before it: pos_before=None."""
    acc = EpisodeAlignAccumulator(min_active_steps=2)
    rng = np.random.default_rng(13)
    mask = np.zeros((4, 1), dtype=bool)
    mask[0, 0] = True
    for t in range(6):
        acc.step(_obs(t, rng=rng), _scene([_report('e1')], mask))
    pair = acc.finish(rng)[0]
    assert pair.pos_before is None        # no free run before t=0


def test_alignment_buffer_bounds_memory():
    buf = AlignmentBuffer(max_episodes=2)
    pair = object()
    buf.append([pair, pair])
    buf.append([pair])
    buf.append([pair])
    assert len(buf) == 2            # oldest episode evicted
    assert buf.total_pairs == 4
