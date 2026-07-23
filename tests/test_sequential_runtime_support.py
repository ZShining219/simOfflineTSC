import os
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from sequential.core import (
    ReplayMetadata, ReplayRecord, SequentialReplay, TrainingPayload,
)
from sequential.diagnostics import ReplayDiagnostics
from sequential.io import atomic_json, read_json
from sequential.manifest import build_pilot_plan
from sequential.runtime import SequentialChildRunner


class SequentialRuntimeSupportTests(unittest.TestCase):
    @staticmethod
    def _record(index, network, stage):
        payload = TrainingPayload.from_legacy((
            np.zeros((1, 8), dtype=np.float32), np.array([0]),
            np.array([0]), np.array(0.0),
            np.ones((1, 8), dtype=np.float32), np.array([1]),
        ))
        return ReplayRecord(payload, ReplayMetadata(
            f't{index}', network, stage, 1, index, index,
        ))

    def test_replay_diagnostics_aggregate_without_full_batch_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = ReplayDiagnostics(directory, trace_samples=False)
            old = 'sumohz1x1_config3'
            current = 'sumohz1x1_config2'
            records = [self._record(index, old, 1) for index in range(50)]
            records += [self._record(50 + index, current, 2) for index in range(50)]
            replay = SequentialReplay(5000, records)
            agent = SimpleNamespace(
                replay=replay, current_network=current,
                counters=SimpleNamespace(
                    global_decision_step=100,
                    gradient_updates=143001, target_updates=14300,
                ),
            )
            for record in records[-10:]:
                diagnostics.record_transition(record.metadata)
            diagnostics.record_update({
                'sample_sources': [old] * 32 + [current] * 32,
                'sample_ages': list(range(64)),
                'sample_transition_ids': [f't{index}' for index in range(64)],
            }, 143001, replay_composition=replay.composition())
            path, result = diagnostics.episode_record(agent, 2, 1, 401)
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(result['replay_count_by_scene'], {old: 50, current: 50})
            self.assertEqual(result['current_sample_fraction'], 0.5)
            self.assertEqual(result['historical_sample_fraction'], 0.5)
            self.assertEqual(result['sample_age_p50'], 31.5)
            self.assertEqual(result['sparse_full_batches'], [])
            self.assertEqual(len(result['sampling_windows']), 1)
            self.assertEqual(
                result['sampling_windows'][0]['population_count_by_scene'],
                {old: 50, current: 50},
            )
            self.assertIn('0.5', result['replacement_thresholds'])

    def test_sparse_and_explicit_replay_trace_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            update = {
                'sample_sources': ['scene'] * 64,
                'sample_ages': [0] * 64,
                'sample_transition_ids': [str(index) for index in range(64)],
            }
            sparse = ReplayDiagnostics(os.path.join(directory, 'sparse'))
            sparse.record_update(update, 1000)
            self.assertEqual(len(sparse.full_batches), 1)
            traced = ReplayDiagnostics(
                os.path.join(directory, 'traced'), trace_samples=True
            )
            traced.record_update(update, 1001)
            self.assertEqual(len(traced.full_batches), 1)

    def test_pilot_manifest_is_exact_six_child_o1_seed0_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            parents = [{
                'order_id': 'O1', 'training_seed': 0,
                'import_manifest_path': '/parent.json',
                'checkpoint_path': '/parent.pt',
                'checkpoint_file_sha256': 'a' * 64,
                'digests': {'online_parameter_digest': 'digest'},
            }]
            catalog = os.path.join(directory, 'catalog.json')
            output = os.path.join(directory, 'pilot.json')
            atomic_json(catalog, {'parents': parents})
            plan = build_pilot_plan(catalog, output, later_stage_episodes=15)
            self.assertEqual(plan['child_count'], 6)
            self.assertEqual(
                {child['variant'] for child in plan['children']},
                {'control', 'fault'},
            )
            self.assertEqual(
                {child['policy'] for child in plan['children']},
                {'clear', 'fifo', 'fifo_matched_wait'},
            )
            self.assertTrue(all(
                child['stage_episodes'] == [400, 15, 15, 15]
                for child in plan['children']
            ))
            self.assertTrue(plan['launch_authorized'])

    def test_matched_fault_persists_exact_step_180_trigger(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = SequentialChildRunner.__new__(SequentialChildRunner)
            runner.child = {
                'variant': 'fault',
                'fault_point': 'stage3_episode4_simulation_step180',
            }
            runner.attempt_dir = directory
            runner.journal = mock.Mock()
            progress = {
                'stage_index': 3, 'local_episode': 4, 'global_episode': 419,
                'decision_index': 18, 'simulation_step': 180,
                'transition_id': 'transition-18',
            }
            with mock.patch(
                'sequential.runtime.time.monotonic', side_effect=[0.0, 31.0]
            ), mock.patch('sequential.runtime.time.sleep'):
                runner._decision_progress_hook(progress)
            persisted = read_json(os.path.join(directory, 'training_progress.json'))
            self.assertEqual(persisted['simulation_step'], 180)
            runner.journal.record_event.assert_called_once()
            self.assertEqual(
                runner.journal.record_event.call_args.args[0], 'FAULT_TRIGGER_READY'
            )


if __name__ == '__main__':
    unittest.main()
