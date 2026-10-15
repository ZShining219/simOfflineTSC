import unittest

import numpy as np

from hitl.pipeline.segments import (
    label_pair, segment_episode, service_gap_series, teacher_series)


def _ctx(n_lanes=4, action_dim=4):
    return {
        'in_lanes': [f'lane{i}' for i in range(n_lanes)],
        'lane_capacity': [38.0] * n_lanes,
        'phase_served_lanes': [[0, 1], [2], [3], []],
        'action_interval': 10.0,
        'action_dim': action_dim,
        'T_u': 60.0,
    }


def _ep(T=12, n_lanes=4, action_dim=4):
    rng = np.random.RandomState(0)
    state = rng.rand(T, 1, n_lanes).astype(np.float32) * 5
    phase = (np.arange(T) // 3) % action_dim
    return {
        'state': state,
        'next_state': np.roll(state, -1, axis=0),
        'current_phase': phase.astype(np.int8),
        'next_phase': np.roll(phase, -1).astype(np.int8),
        'action': phase.astype(np.int64),
        'queue': state.sum(axis=(1, 2)),
        'reward': -np.ones(T, dtype=np.float32),
        'truncated': np.zeros(T, dtype=bool),
        'terminated': np.zeros(T, dtype=bool),
        'decision_step': np.arange(T),
        'episode_id': np.full(T, 0),
    }


class SegmentsTest(unittest.TestCase):
    def test_gap_series_resets_on_served_and_empty(self):
        occ = np.tile([2, 0, 3, 1], (6, 1)).astype(np.float64)
        gaps = service_gap_series(
            np.zeros(6, dtype=int), [[0, 1], [2]], 10.0, occ)
        # phase 0 serves lanes {0,1}: lane0 resets, lane1 empty resets,
        # lanes 2,3 accumulate
        np.testing.assert_array_equal(gaps[0], [0, 0, 10, 10])
        np.testing.assert_array_equal(gaps[2], [0, 0, 30, 30])

    def test_teacher_score_components(self):
        ep = _ep()
        ctx = _ctx()
        h, Q, U, q_ratio, gaps = teacher_series(ep, ctx)
        self.assertEqual(len(h), len(ep['decision_step']))
        # h = -0.5Q - 0.5U elementwise
        np.testing.assert_allclose(h, -0.5 * Q - 0.5 * U)
        self.assertTrue(np.all(h <= 0))

    def test_segment_episode_full_windows_only(self):
        ep = _ep(T=14)
        segs = segment_episode(ep, _ctx(), window=6)
        self.assertEqual(2, len(segs))  # 14 -> 2 full windows
        self.assertEqual((0, 6), (segs[0]['start'], segs[0]['end']))
        self.assertTrue(all(s['valid'] for s in segs))

    def test_truncated_window_marked_invalid(self):
        ep = _ep(T=14)
        ep['truncated'][3] = True
        segs = segment_episode(ep, _ctx(), window=6)
        self.assertFalse(segs[0]['valid'])
        self.assertTrue(segs[1]['valid'])

    def test_label_pair_tie(self):
        y, d = label_pair(-1.00, -1.02, tie_tol=0.05)
        self.assertEqual(0.5, y)
        y, _ = label_pair(-0.5, -1.5, tie_tol=0.05)
        self.assertEqual(1.0, y)


if __name__ == '__main__':
    unittest.main()
