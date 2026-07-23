import copy
import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from trainer.offline_tsc_trainer import (
    OfflineMetricLogger, OfflineTSCTrainer, isolated_random_seed,
    should_update_target,
)
from world.world_sumo import World


class FakeDataset:
    def __init__(self):
        self.state = random.Random(19).getstate()

    def rng_state(self):
        return self.state

    def set_rng_state(self, state):
        self.state = state


class TinyAgent:
    offline_algorithm = 'cql_dqn'

    def __init__(self):
        self.model = nn.Linear(2, 3)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=.01)
        self.cql_alpha = 1.0
        self.backend = {'name': 'native'}

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())


def make_checkpoint_trainer(root, update):
    trainer = OfflineTSCTrainer.__new__(OfflineTSCTrainer)
    trainer.output_path = str(root)
    trainer.agents = [TinyAgent()]
    trainer.offline_dataset = FakeDataset()
    trainer.structured_metrics = OfflineMetricLogger(str(root))
    trainer.dataset_manifest_sha256 = 'd' * 64
    trainer.resume_fingerprint = 'r' * 64
    trainer.current_update = update
    trainer.gradient_updates = update
    trainer.target_updates = update // 10
    trainer.evaluation_results = {}
    trainer.logical_run_id = 'logical'
    trainer.total_updates = 20
    trainer.final_evaluation_completed = False
    return trainer


class MetricAndResumeGateTest(unittest.TestCase):
    def test_initial_target_is_exactly_synchronized(self):
        agent = TinyAgent()
        for online, target in zip(
            agent.model.parameters(), agent.target_model.parameters()
        ):
            self.assertTrue(torch.equal(online, target))

    def _record(self, update, record_type='TRAIN'):
        return {
            field: None for field in OfflineMetricLogger.REQUIRED_FIELDS
        } | {
            'schema_version': 1, 'record_type': record_type,
            'training_update': update, 'logical_run_id': 'logical',
            'physical_run_id': 'physical',
        }

    def test_jsonl_rejects_duplicate_non_monotonic_and_partial_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = OfflineMetricLogger(temporary)
            logger.append(self._record(1))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                logger.append(self._record(1))
            logger.append(self._record(2, 'EVALUATION'))
            with self.assertRaisesRegex(ValueError, 'monotonic'):
                logger.append(self._record(0))
            with open(logger.path, 'a', encoding='utf-8') as handle:
                handle.write('{"schema_version":1}')
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                logger.validate()

    def test_resume_9_and_10_preserves_real_target_parameter_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for update in (9, 10):
                source = make_checkpoint_trainer(root / f'source_{update}', update)
                with torch.no_grad():
                    for parameter in source.agents[0].model.parameters():
                        parameter.fill_(float(update))
                    for parameter in source.agents[0].target_model.parameters():
                        parameter.fill_(0.0 if update == 9 else 10.0)
                checkpoint = source.save_checkpoint('resumable', update)
                payload = torch.load(checkpoint, map_location='cpu')
                self.assertEqual(1.0, payload['cql_alpha'])

                resumed = make_checkpoint_trainer(root / f'resumed_{update}', 0)
                resumed.load_resumable_checkpoint(checkpoint)
                self.assertEqual(update, resumed.current_update)
                next_update = update + 1
                with torch.no_grad():
                    for parameter in resumed.agents[0].model.parameters():
                        parameter.fill_(float(next_update))
                if should_update_target(next_update, 10):
                    resumed.agents[0].update_target_network()
                expected = 10.0
                for parameter in resumed.agents[0].target_model.parameters():
                    self.assertTrue(torch.equal(
                        parameter, torch.full_like(parameter, expected)
                    ))

    def test_update_zero_completed_evaluation_is_idempotent(self):
        trainer = OfflineTSCTrainer.__new__(OfflineTSCTrainer)
        expected = {'training_update': 0, 'status': 'complete'}
        trainer.evaluation_results = {0: expected}
        self.assertIs(expected, trainer._evaluate(0))

    def test_evaluation_rng_is_explicit_and_training_rng_is_restored(self):
        random.seed(1)
        np.random.seed(2)
        torch.manual_seed(3)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with isolated_random_seed(100):
            first = (random.random(), np.random.rand(), torch.rand(1))
        after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1][1], after[1][1])
        self.assertTrue(torch.equal(before[2], after[2]))
        with isolated_random_seed(100):
            second = (random.random(), np.random.rand(), torch.rand(1))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))


class FakeProcess:
    def __init__(self):
        self.killed = False
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self):
        return 0 if self.killed else None

    def wait(self, timeout=None):
        if not self.killed:
            raise subprocess.TimeoutExpired('sumo', timeout)
        return 0

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1
        self.killed = True


class FakeConnection:
    def __init__(self, process):
        self._process = process
        self.close_calls = 0

    def close(self, wait=True):
        self.close_calls += 1


class SumoLifecycleGateTest(unittest.TestCase):
    def test_close_is_idempotent_and_force_reaps_owned_process(self):
        process = FakeProcess()
        connection = FakeConnection(process)
        world = World.__new__(World)
        world.interface_flag = False
        world._connection_open = True
        world._sumo_stdout_handle = None
        world.eng = connection
        first = world.close()
        self.assertEqual(1, connection.close_calls)
        self.assertEqual(1, process.terminate_calls)
        self.assertEqual(1, process.kill_calls)
        self.assertTrue(first['killed'])
        self.assertFalse(first['process_alive'])
        second = world.close()
        self.assertEqual(1, connection.close_calls)
        self.assertFalse(second['connection_was_open'])


if __name__ == '__main__':
    unittest.main()
