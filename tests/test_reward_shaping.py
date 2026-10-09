import unittest

import numpy as np

from sequential.reward_shaping import (
    RewardShaper, ServiceGapTracker, lane_capacities_from_lengths)


class ServiceGapTrackerTest(unittest.TestCase):
    """Spec rule: u_l += dt only while the lane has demand and is not
    served; resets to 0 when served or empty."""

    def test_gap_grows_only_on_unserved_demand_lanes(self):
        # phase 0 serves lanes {0,1}; phase 1 serves {2}
        served = [[0, 1], [2]]
        tracker = ServiceGapTracker(served, n_lanes=4, dt=10.0)
        occupancy = np.array([1, 0, 5, 3], dtype=np.float64)
        gaps = tracker.advance(0, occupancy)          # serves 0,1
        np.testing.assert_array_equal(gaps, [0, 0, 10, 10])
        gaps = tracker.advance(0, occupancy)          # still serving 0,1
        np.testing.assert_array_equal(gaps, [0, 0, 20, 20])
        gaps = tracker.advance(1, occupancy)          # switch: serves 2
        np.testing.assert_array_equal(gaps, [10, 0, 0, 30])

    def test_empty_lane_gap_resets(self):
        served = [[0]]
        tracker = ServiceGapTracker(served, n_lanes=2, dt=10.0)
        tracker.advance(1, np.array([3, 4]))
        tracker.advance(1, np.array([3, 4]))
        gaps = tracker.advance(1, np.array([0, 4]))   # lane0 emptied
        np.testing.assert_array_equal(gaps, [0, 30])

    def test_reset_clears(self):
        tracker = ServiceGapTracker([[0]], 2, 10.0)
        tracker.advance(1, np.array([1, 1]))
        tracker.reset()
        np.testing.assert_array_equal(tracker.gaps, [0, 0])


class RewardShaperTest(unittest.TestCase):
    def _shaper(self, mode='b1', lam=0.1, calibration=None, **kw):
        capacities = [38.0] * 4
        served = [[0, 1], [2], [3], []]
        calib = calibration or {'h': {'mu': -0.5, 'sigma': 0.25}}
        kw.setdefault('model_path', None)
        return RewardShaper(
            mode, capacities, served, action_interval=10.0,
            action_dim=4, lam=lam, calibration=calib, **kw)

    def test_b1_shapes_reward_and_keeps_r_base(self):
        shaper = self._shaper()
        obs = np.ones(12)     # 4 lanes + 4 onehot slots (mock dim)
        nxt = np.ones(12) * 2
        occ = np.array([2, 1, 0, 3])
        r_train, aux = shaper.shape(
            -1.0, obs, action=0, next_observation=nxt,
            queue_per_lane=np.array([4, 4, 4, 4]),
            occupancy=occ, phase_after_action=0)
        self.assertEqual(-1.0, aux['r_base'])
        self.assertAlmostEqual(
            r_train, -1.0 + 0.1 * aux['z'], places=6)
        self.assertIn('h', aux)
        # h = -0.5*mean(4/38) - 0.5*max(u/T_u): lanes 2,3 unserved w/ demand
        # after one interval u=10 -> U=10/60
        h_expected = -0.5 * (4 / 38.0) - 0.5 * (10.0 / 60.0)
        self.assertAlmostEqual(aux['h'], h_expected, places=6)

    def test_collapsed_calibration_falls_back_to_r0(self):
        shaper = self._shaper(calibration={
            'h': {'mu': 0.0, 'sigma': 0.0, 'collapsed': True}})
        r_train, aux = shaper.shape(
            -2.0, np.zeros(12), 1, np.zeros(12),
            np.zeros(4), np.ones(4), 1)
        self.assertEqual(-2.0, r_train)
        self.assertEqual('collapsed', aux['fallback'])

    def test_none_mode_is_identity(self):
        shaper = self._shaper(mode='none')
        r_train, aux = shaper.shape(
            -3.0, np.zeros(12), 0, np.zeros(12),
            np.ones(4), np.ones(4), 0)
        self.assertEqual(-3.0, r_train)
        self.assertEqual(-3.0, aux['r_base'])

    def test_missing_calibration_raises(self):
        shaper = self._shaper(calibration={'h': None})
        with self.assertRaises(ValueError):
            shaper.shape(-1.0, np.zeros(12), 0, np.zeros(12),
                         np.ones(4), np.ones(4), 0)

    def test_lane_capacity_floor(self):
        caps = lane_capacities_from_lengths([289.6, 10.0, 0.0])
        self.assertEqual([38.0, 1.0, 1.0], caps)


if __name__ == '__main__':
    unittest.main()
