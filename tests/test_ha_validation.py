import os
import tempfile
import unittest
from unittest import mock

from sequential.io import atomic_json
from sequential.validation import (
    _resolve_attempt_chain_artifact, _validate_ha_visibility_networks,
    validate_ha_attempt,
)


class HAValidationTest(unittest.TestCase):
    def test_recovery_artifact_resolves_from_immutable_predecessor_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            logical_root = os.path.join(directory, 'logical')
            first = os.path.join(logical_root, 'attempts', 'attempt_1')
            second = os.path.join(logical_root, 'attempts', 'attempt_2')
            os.makedirs(os.path.join(first, 'archive'))
            os.makedirs(second)
            artifact = os.path.join(first, 'archive', 'stage_01.json')
            atomic_json(artifact, {'stage': 1})
            atomic_json(os.path.join(logical_root, 'logical_run_manifest.json'), {
                'attempts': [
                    {'attempt_id': 'attempt_1', 'attempt_dir': first},
                    {'attempt_id': 'attempt_2', 'attempt_dir': second},
                ],
                'effective_attempt': 'attempt_2',
            })
            self.assertEqual(
                artifact,
                _resolve_attempt_chain_artifact(
                    second, 'archive', 'stage_01.json',
                ),
            )

    def test_p1f_visibility_accepts_archive_order_but_p1c_requires_prefix_order(self):
        expected = ['n1', 'n4', 'n2', 'n3']
        archive_order = ['n2', 'n1', 'n4', 'n3']
        self.assertTrue(_validate_ha_visibility_networks(
            'P1F', archive_order, expected,
        ))
        self.assertFalse(_validate_ha_visibility_networks(
            'P1F', ['n1', 'n4', 'n2', 'n2'], expected,
        ))
        self.assertFalse(_validate_ha_visibility_networks(
            'P1C', archive_order, expected,
        ))

    def test_p1c_current_scene_offline_sample_is_rejected(self):
        child = {
            'protocol_id': 'ha_sodqn_b100_v1',
            'archive_mode': 'P1C', 'offline_ratio': 0.5,
            'training_seed': 0, 'networks': ['past', 'current'],
            'stage_episodes': [1, 1],
            'archive_root_manifest': '/archive.json',
        }
        diagnostics = {
            '/diag1': {
                'stage_index': 1, 'local_episode': 1,
                'visible_archive': None, 'sampling_windows': [],
                'offline_samples_by_behavior_seed': {},
                'offline_samples_by_episode': {},
            },
            '/diag2': {
                'stage_index': 2, 'local_episode': 1,
                'visible_archive': {
                    'visibility': {'visible_networks': ['past']},
                },
                'sampling_windows': [{
                    'online_count': 32, 'offline_count': 32,
                    'actual_offline_ratio': 0.5,
                    'offline_sample_count_by_scene': {'current': 32},
                    'loss_online': 1.0, 'loss_offline': 1.0,
                    'loss_total': 1.0,
                }],
                'offline_samples_by_behavior_seed': {'1': 32},
                'offline_samples_by_episode': {'1': 32},
            },
        }
        validated = {
            'child_manifest': child,
            'state': {'completed_operations': {
                'stage_1:episode_1:diagnostic': {'artifact': '/diag1'},
                'stage_2:episode_1:diagnostic': {'artifact': '/diag2'},
            }},
            'logical_episode_view': {'child_episode_references': [
                {'transition_count': 360}, {'transition_count': 360},
            ]},
            'final_checkpoint': {'agent_state': {'counters': {
                'global_decision_step': 720,
                'gradient_updates': 0, 'target_updates': 0,
            }}},
        }

        def read(path):
            if path == '/archive.json':
                return {
                    'behavior_training_seeds': [0, 1, 2, 3, 4],
                    'behavior_seed_rule': {'exclude_matching_training_seed': True},
                }
            return diagnostics[path]

        with mock.patch(
            'sequential.validation.validate_attempt', return_value=validated,
        ), mock.patch('sequential.validation.read_json', side_effect=read):
            with self.assertRaisesRegex(ValueError, 'current/future leakage'):
                validate_ha_attempt('/run')


if __name__ == '__main__':
    unittest.main()
