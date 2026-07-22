import unittest
import random
from collections import deque

import numpy as np
import torch

from trainer.tsc_trainer import EvaluationIsolationGuard


class DummyDataset:
    def __len__(self):
        return 4


class DummyAgent:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.target_model = torch.nn.Linear(2, 2)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.RMSprop(self.model.parameters())
        self.epsilon = 0.1
        self.replay_buffer = deque([1, 2, 3])

    def remember(self, *args):
        self.replay_buffer.append(args)


class DummyTrainer:
    def __init__(self):
        self.agents = [DummyAgent()]
        self.dataset = DummyDataset()
        self.gradient_updates = 2
        self.global_decision_step = 9
        self.evaluation_isolation_checks = []
        self.trajectory_writer = type('Writer', (), {'total_count': 5})()


class EvaluationIsolationTest(unittest.TestCase):
    def test_unchanged_state_passes_and_records_audit(self):
        trainer = DummyTrainer()
        with EvaluationIsolationGuard(trainer, 'EVALUATION'):
            pass
        self.assertEqual(0, trainer.evaluation_isolation_checks[0]['remember_calls'])
        self.assertEqual([3], trainer.evaluation_isolation_checks[0]['replay_lengths'])
        self.assertEqual(5, trainer.evaluation_isolation_checks[0]['trajectory_writes'])
        self.assertTrue(trainer.evaluation_isolation_checks[0]['rng_unchanged'])

    def test_remember_call_fails_immediately(self):
        trainer = DummyTrainer()
        with self.assertRaisesRegex(RuntimeError, 'attempted to call'):
            with EvaluationIsolationGuard(trainer, 'EVALUATION'):
                trainer.agents[0].remember('forbidden')

    def test_model_and_counter_mutations_fail(self):
        trainer = DummyTrainer()
        with self.assertRaisesRegex(RuntimeError, 'mutated protected'):
            with EvaluationIsolationGuard(trainer, 'EVALUATION'):
                trainer.gradient_updates += 1
        trainer = DummyTrainer()
        with self.assertRaisesRegex(RuntimeError, 'mutated protected'):
            with EvaluationIsolationGuard(trainer, 'FINAL_EVALUATION'):
                with torch.no_grad():
                    trainer.agents[0].model.weight.add_(1)
        trainer = DummyTrainer()
        with self.assertRaisesRegex(RuntimeError, 'mutated protected'):
            with EvaluationIsolationGuard(trainer, 'EVALUATION'):
                trainer.trajectory_writer.total_count += 1

    def test_rng_mutations_fail(self):
        mutations = (
            random.random,
            np.random.random,
            lambda: torch.rand(1),
        )
        for mutation in mutations:
            trainer = DummyTrainer()
            with self.assertRaisesRegex(RuntimeError, 'mutated protected'):
                with EvaluationIsolationGuard(trainer, 'EVALUATION'):
                    mutation()


if __name__ == '__main__':
    unittest.main()
