import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

import torch

from agent.dqn import DQNNet
from sequential.core import canonical_digest
from sequential.evaluator import IndependentEvaluator, save_online_snapshot
from sequential.io import atomic_json, read_json


def fixture_evaluation_worker(request_path):
    request = read_json(request_path)
    attempt_dir = request['attempt_dir']
    attempt = int(os.path.basename(attempt_dir).split('_')[1])
    if request['evaluation_network'] == 'fail_twice' and attempt < 3:
        atomic_json(os.path.join(attempt_dir, 'error.json'), {
            'error_type': 'InjectedFailure', 'attempt': attempt,
        })
        raise RuntimeError('injected evaluator failure')
    if request['evaluation_network'] == 'always_fail':
        atomic_json(os.path.join(attempt_dir, 'error.json'), {
            'error_type': 'InjectedFailure', 'attempt': attempt,
        })
        raise RuntimeError('injected evaluator failure')
    summary_path = os.path.join(attempt_dir, 'summary.json')
    decisions_path = os.path.join(attempt_dir, 'decisions.jsonl')
    atomic_json(summary_path, {
        'travel_time': 1.0, 'attempt': attempt,
        'checkpoint_digest': request['checkpoint_digest'],
    })
    atomic_json(decisions_path, {'actions': [0]})
    atomic_json(os.path.join(attempt_dir, 'success.json'), {
        'valid': True, 'summary_path': summary_path,
        'decisions_path': decisions_path,
    })


def fixture_success_worker(request_path):
    request = read_json(request_path)
    attempt_dir = request['attempt_dir']
    attempt = int(os.path.basename(attempt_dir).split('_')[1])
    summary_path = os.path.join(attempt_dir, 'summary.json')
    decisions_path = os.path.join(attempt_dir, 'decisions.jsonl')
    atomic_json(summary_path, {
        'travel_time': 1.0, 'attempt': attempt,
        'checkpoint_digest': request['checkpoint_digest'],
    })
    atomic_json(decisions_path, {'actions': [0]})
    atomic_json(os.path.join(attempt_dir, 'success.json'), {
        'valid': True, 'summary_path': summary_path,
        'decisions_path': decisions_path,
    })


def fixture_slow_success_worker(request_path):
    time.sleep(0.25)
    fixture_success_worker(request_path)


class DummySnapshotAgent:
    def __init__(self):
        self.model = DQNNet(16, 8)
        self.ob_length = 16
        self.action_space = SimpleNamespace(n=8)
        self.phase = True
        self.one_hot = True
        self.optimizer = object()
        self.replay = object()
        self.epsilon = 0.42


class SequentialEvaluatorTests(unittest.TestCase):
    def _snapshot(self, directory):
        agent = DummySnapshotAgent()
        path = os.path.join(directory, 'online.pt')
        before = canonical_digest(agent.model.state_dict())
        payload = save_online_snapshot(agent, path, {'stage_index': 1})
        after = canonical_digest(agent.model.state_dict())
        self.assertEqual(before, after)
        self.assertEqual(payload['checkpoint_type'], 'online_only')
        self.assertNotIn('optimizer', payload)
        self.assertNotIn('replay', payload)
        self.assertNotIn('epsilon', payload)
        return path

    @staticmethod
    def _protocol():
        return {
            'simulator_config': 'unused-test-config.cfg',
            'interface': 'libsumo', 'steps': 3600, 'action_interval': 10,
            'sumo_seed_mode': 'fixed_default',
        }

    @staticmethod
    def _identity(local_episode):
        return {
            'stage_index': 2,
            'training_network': 'sumohz1x1_config2',
            'evaluation_network': 'fail_twice',
            'local_episode': local_episode,
            'global_episode': 400 + local_episode,
        }

    def test_snapshot_is_immutable_and_online_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._snapshot(directory)
            with self.assertRaises(FileExistsError):
                save_online_snapshot(DummySnapshotAgent(), path, {'stage_index': 2})

    def test_spawn_retry_and_physical_cell_alias_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._snapshot(directory)
            evaluator = IndependentEvaluator(
                os.path.join(directory, 'evaluation'), retries=3,
                timeout_seconds=30, worker_target=fixture_evaluation_worker,
            )
            first = evaluator.evaluate(
                snapshot, 'fail_twice', self._protocol(), self._identity(0),
            )
            self.assertFalse(first['reused'])
            self.assertEqual(first['physical']['successful_attempt'], 3)
            physical_dir = os.path.dirname(
                first['physical']['summary_path']
            )
            root = os.path.dirname(physical_dir)
            self.assertTrue(os.path.isfile(os.path.join(root, 'attempt_1', 'error.json')))
            self.assertTrue(os.path.isfile(os.path.join(root, 'attempt_2', 'error.json')))
            second = evaluator.evaluate(
                snapshot, 'fail_twice', self._protocol(), self._identity(1),
            )
            self.assertTrue(second['reused'])
            self.assertEqual(
                first['physical']['physical_key'], second['physical']['physical_key']
            )
            self.assertNotEqual(first['alias_path'], second['alias_path'])
            self.assertFalse(os.path.exists(os.path.join(root, 'attempt_4')))

    def test_three_failed_attempts_leave_errors_without_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._snapshot(directory)
            evaluator = IndependentEvaluator(
                os.path.join(directory, 'evaluation'), retries=3,
                timeout_seconds=30, worker_target=fixture_evaluation_worker,
            )
            identity = self._identity(0)
            identity['evaluation_network'] = 'always_fail'
            with self.assertRaisesRegex(RuntimeError, 'three attempts'):
                evaluator.evaluate(
                    snapshot, 'always_fail', self._protocol(), identity,
                )
            physical_root = os.path.join(directory, 'evaluation', 'physical')
            cells = os.listdir(physical_root)
            self.assertEqual(len(cells), 1)
            cell = os.path.join(physical_root, cells[0])
            self.assertFalse(os.path.exists(os.path.join(cell, 'committed.json')))
            self.assertEqual(
                sorted(
                    name for name in os.listdir(cell)
                    if name.startswith('attempt_')
                ),
                ['attempt_1', 'attempt_2', 'attempt_3'],
            )

            resumed = IndependentEvaluator(
                os.path.join(directory, 'evaluation'), retries=3,
                timeout_seconds=30, worker_target=fixture_success_worker,
            ).evaluate(snapshot, 'always_fail', self._protocol(), identity)
            self.assertEqual(resumed['physical']['successful_attempt'], 4)
            self.assertTrue(os.path.isfile(os.path.join(cell, 'committed.json')))
            self.assertFalse(os.path.exists(os.path.join(cell, 'attempt_5')))

    def test_concurrent_identical_physical_cell_is_evaluated_once(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._snapshot(directory)
            output_root = os.path.join(directory, 'evaluation')
            results = []
            errors = []
            barrier = threading.Barrier(2)

            def evaluate(local_episode):
                try:
                    barrier.wait()
                    identity = self._identity(local_episode)
                    identity['evaluation_network'] = 'shared'
                    results.append(IndependentEvaluator(
                        output_root, retries=3, timeout_seconds=30,
                        worker_target=fixture_slow_success_worker,
                    ).evaluate(snapshot, 'shared', self._protocol(), identity))
                except BaseException as error:
                    errors.append(error)

            threads = [
                threading.Thread(target=evaluate, args=(episode,))
                for episode in (0, 1)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(60)

            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            self.assertEqual(sorted(result['reused'] for result in results), [False, True])
            physical_root = os.path.join(output_root, 'physical')
            cells = os.listdir(physical_root)
            self.assertEqual(len(cells), 1)
            cell = os.path.join(physical_root, cells[0])
            self.assertTrue(os.path.isfile(os.path.join(cell, 'committed.json')))
            self.assertTrue(os.path.isdir(os.path.join(cell, 'attempt_1')))
            self.assertFalse(os.path.exists(os.path.join(cell, 'attempt_2')))


if __name__ == '__main__':
    unittest.main()
