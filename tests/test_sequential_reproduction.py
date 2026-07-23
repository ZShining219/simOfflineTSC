import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from sequential.io import sha256_file
from sequential.reproduction import (
    compare_plan1_reproduction, plan1_trajectory_digest,
)


class SequentialReproductionTests(unittest.TestCase):
    def test_plan1_trajectory_digest_covers_all_semantic_fields(self):
        with tempfile.TemporaryDirectory() as run_path:
            root = os.path.join(run_path, 'trajectory')
            episodes = os.path.join(root, 'episodes')
            os.makedirs(episodes)
            index_path = os.path.join(root, 'index.jsonl')
            with open(index_path, 'w', encoding='utf-8') as index:
                for episode in range(1, 401):
                    relative = f'episodes/episode_{episode:04d}.npz'
                    path = os.path.join(root, relative)
                    np.savez_compressed(
                        path,
                        episode_id=np.asarray([episode]),
                        decision_step=np.asarray([1]),
                        global_step=np.asarray([episode]),
                        state=np.zeros((1, 1, 1, 2), dtype=np.float32),
                        current_phase=np.zeros((1, 1, 1), dtype=np.int8),
                        action=np.zeros((1, 1, 1), dtype=np.int64),
                        reward=np.zeros((1, 1), dtype=np.float64),
                        next_state=np.ones((1, 1, 1, 2), dtype=np.float32),
                        next_phase=np.ones((1, 1, 1), dtype=np.int8),
                        terminated=np.asarray([False]),
                        truncated=np.asarray([True]),
                    )
                    index.write(json.dumps({
                        'episode_id': episode, 'file': relative,
                        'transition_count': 1, 'sha256': sha256_file(path),
                    }) + '\n')
            result = plan1_trajectory_digest(run_path)
            self.assertEqual(result['episode_count'], 400)
            self.assertEqual(result['transition_count'], 400)
            self.assertEqual(len(result['episode_digests']), 400)

    def test_comparison_requires_every_checkpoint_and_trajectory_digest(self):
        signature = {
            'trajectory': {
                'canonical_trajectory_digest': 'trajectory',
                'episode_digests': ['episode'],
            },
            'online_parameter_digest': 'online',
            'target_parameter_digest': 'target',
            'optimizer_state_digest': 'optimizer',
            'replay_content_digest': 'replay-content',
            'replay_metadata_digest': 'replay-metadata',
            'rng_state_digest': 'rng', 'epsilon': 0.01,
            'global_decision_step': 144000, 'gradient_updates': 143000,
            'target_updates': 14300, 'next_target_sync_update': 143009,
        }
        source = {'run_path': '/source', 'network': 'network', 'training_seed': 0,
                  **signature}
        reproduction = {'run_path': '/reproduction', 'network': 'network',
                        'training_seed': 0, **signature}
        with mock.patch(
            'sequential.reproduction.plan1_reproduction_signature',
            side_effect=[source, reproduction],
        ):
            self.assertTrue(compare_plan1_reproduction(
                '/source', '/reproduction', 'network'
            )['valid'])
        reproduction = dict(reproduction)
        reproduction['rng_state_digest'] = 'different'
        with mock.patch(
            'sequential.reproduction.plan1_reproduction_signature',
            side_effect=[source, reproduction],
        ):
            report = compare_plan1_reproduction(
                '/source', '/reproduction', 'network'
            )
            self.assertFalse(report['valid'])
            self.assertFalse(report['checks']['rng_state_digest'])


if __name__ == '__main__':
    unittest.main()
