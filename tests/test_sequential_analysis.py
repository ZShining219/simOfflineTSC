import json
import os
import tempfile
import unittest

import numpy as np

from sequential.analysis import (
    exact_sign_flip_test, holm_adjust, normalized_auc, paired_bootstrap, raw_auc,
    replay_uniformity_envelope, _select_analysis_children,
)
from sequential.core import canonical_transition_digest
from sequential.validation import validate_trajectory_marker
from tools.experiment_plotting.sequential import _flatten


class SequentialAnalysisTests(unittest.TestCase):
    def test_replay_plotting_uses_per_episode_sample_deltas(self):
        run = {
            'logical_run_id': 'run', 'order_id': 'O1', 'training_seed': 0,
            'policy': 'fifo', 'primary_normalized_auc': 1.0,
            'secondary_metrics': {},
            'stages': [{'stage_index': 2, 'network': 'current'}],
            'adaptation_curves': [], 'final_retention_normalized': {},
            'stage_end_forgetting_travel_time': {},
            'replay_diagnostics': [
                {
                    'stage_index': 2, 'local_episode': 1,
                    'current_network': 'current',
                    'replay_ratio_by_scene': {'current': 0.25},
                    'samples_drawn_by_scene': {'old': 80, 'current': 20},
                    'current_sample_fraction': 0.2,
                },
                {
                    'stage_index': 2, 'local_episode': 2,
                    'current_network': 'current',
                    'replay_ratio_by_scene': {'current': 0.5},
                    'samples_drawn_by_scene': {'old': 90, 'current': 110},
                    'current_sample_fraction': 0.55,
                },
            ],
        }
        replay = _flatten({'runs': [run]})[-1]
        self.assertEqual(replay[0]['episode_current_sample_fraction'], 0.2)
        self.assertEqual(replay[1]['episode_current_sample_fraction'], 0.9)
        self.assertEqual(replay[1]['current_buffer_fraction'], 0.5)

    def test_selected_orders_require_a_complete_policy_seed_matrix(self):
        plan = {
            'orders': {'O1': [], 'O2': []}, 'training_seeds': [0, 1],
            'policies': ['clear', 'fifo'],
            'children': [
                {'order_id': order, 'training_seed': seed, 'policy': policy}
                for order in ('O1', 'O2') for seed in (0, 1)
                for policy in ('clear', 'fifo')
            ],
        }
        selected, orders = _select_analysis_children(plan, ['O2'])
        self.assertEqual(orders, ['O2'])
        self.assertEqual(len(selected), 4)
        plan['children'].pop()
        with self.assertRaisesRegex(ValueError, 'matrix is incomplete'):
            _select_analysis_children(plan, ['O2'])

    def test_normalized_auc_uses_local_zero_through_horizon(self):
        self.assertEqual(normalized_auc([10.0, 10.0, 10.0], 10.0, 2), 1.0)
        self.assertEqual(normalized_auc([10.0, 20.0, 30.0], 10.0, 2), 2.0)
        self.assertEqual(raw_auc([10.0, 20.0, 30.0], 2), 20.0)

    def test_exact_sign_flip_and_holm(self):
        result = exact_sign_flip_test([1.0, 1.0])
        self.assertEqual(result['enumerations'], 4)
        self.assertEqual(result['p_value'], 0.5)
        self.assertEqual(holm_adjust([0.01, 0.04]), [0.02, 0.04])

    def test_exact_sign_flip_enumerates_all_twenty_pairs(self):
        result = exact_sign_flip_test(np.ones(20))
        self.assertEqual(result['enumerations'], 2 ** 20)
        self.assertEqual(result['p_value'], 2 / (2 ** 20))

    def test_paired_bootstrap_is_fixed_seed_reproducible(self):
        first = paired_bootstrap([1.0, 2.0, 3.0], 42, resamples=1000)
        second = paired_bootstrap([1.0, 2.0, 3.0], 42, resamples=1000)
        self.assertEqual(first, second)

    def test_uniformity_envelope_is_simultaneous_and_reproducible(self):
        windows = [{
            'replay_size': 100,
            'population_count_by_scene': {'a': 50, 'b': 50},
            'sample_count_by_scene': {'a': 5, 'b': 5},
            'sample_size': 10,
        } for _ in range(8)]
        first = replay_uniformity_envelope(windows, 7, resamples=1000)
        second = replay_uniformity_envelope(windows, 7, resamples=1000)
        self.assertEqual(first, second)
        self.assertTrue(first['available'])
        self.assertTrue(first['valid'])
        self.assertEqual(first['window_count'], 8)

    def test_trajectory_marker_recomputes_training_semantics(self):
        transition = {
            'stage_index': 2, 'local_episode': 1, 'global_episode': 401,
            'decision_index': 1, 'global_decision_step': 144001,
            'state': np.zeros((1, 2), dtype=np.float32),
            'phase': np.asarray([0], dtype=np.int8),
            'action': np.asarray([1], dtype=np.int64),
            'reward': np.array(-1.5, dtype=np.float64),
            'next_state': np.ones((1, 2), dtype=np.float32),
            'next_phase': np.asarray([1], dtype=np.int8),
            'terminated': False, 'truncated': False,
        }
        with tempfile.TemporaryDirectory() as directory:
            shard = os.path.join(directory, 'episode.npz')
            np.savez_compressed(
                shard, **{key: np.asarray([value]) for key, value in transition.items()},
                transition_id=np.asarray(['id-1']),
            )
            from sequential.io import sha256_file
            marker = {
                'schema_version': 1, 'identity': {}, 'transition_count': 1,
                'trajectory_path': shard, 'trajectory_sha256': sha256_file(shard),
                'canonical_transition_digests': [
                    canonical_transition_digest(transition)
                ],
            }
            marker_path = os.path.join(directory, 'marker.json')
            with open(marker_path, 'w', encoding='utf-8') as handle:
                json.dump(marker, handle)
            self.assertEqual(
                validate_trajectory_marker(marker_path),
                marker['canonical_transition_digests'],
            )


if __name__ == '__main__':
    unittest.main()
