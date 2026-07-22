import os
import tempfile
import unittest
from collections import deque

import torch

from trainer.tsc_trainer import TSCTrainer
from utils.logger import hash_torch_state_dict


class DummyAgent:
    def __init__(self):
        self.rank = 0
        self.model = torch.nn.Linear(2, 2)
        self.target_model = torch.nn.Linear(2, 2)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.01)
        self.epsilon = 0.2
        self.replay_buffer = deque([('key', ('value',))], maxlen=5)

    def train(self):
        self.optimizer.zero_grad()
        loss = self.model(torch.ones(1, 2)).pow(2).mean()
        loss.backward()
        self.optimizer.step()
        return loss.detach().numpy()


def dummy_trainer(directory):
    trainer = object.__new__(TSCTrainer)
    trainer.agents = [DummyAgent()]
    trainer.output_path = directory
    trainer.config_hash = 'a' * 64
    trainer.global_decision_step = 7
    trainer.gradient_updates = 3
    trainer.target_updates = 1
    trainer.epoch = 2
    trainer.step = 9
    return trainer


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_evaluation_checkpoint_contains_online_only(self):
        trainer = dummy_trainer(self.temporary_directory.name)
        expected = hash_torch_state_dict(trainer.agents[0].model.state_dict())
        path = trainer.save_checkpoint('evaluation', 4)
        payload = trainer.load_checkpoint_payload(path, 'evaluation')
        self.assertEqual(
            expected, hash_torch_state_dict(payload['agents'][0]['online_model_state_dict'])
        )
        self.assertNotIn('target_model_state_dict', payload['agents'][0])
        self.assertNotIn('optimizer_state_dict', payload['agents'][0])
        with torch.no_grad():
            trainer.agents[0].model.weight.add_(5)
        trainer.load_evaluation_checkpoint(path)
        self.assertEqual(expected, hash_torch_state_dict(trainer.agents[0].model.state_dict()))

    def test_resumable_round_trip_restores_training_state(self):
        trainer = dummy_trainer(self.temporary_directory.name)
        path = trainer.save_checkpoint('resumable', 4)
        expected_hash = hash_torch_state_dict(trainer.agents[0].model.state_dict())
        with torch.no_grad():
            trainer.agents[0].model.weight.add_(10)
        trainer.agents[0].epsilon = 0.9
        trainer.agents[0].replay_buffer.clear()
        trainer.gradient_updates = 99
        trainer.load_resumable_checkpoint(path)
        self.assertEqual(expected_hash, hash_torch_state_dict(trainer.agents[0].model.state_dict()))
        self.assertEqual(0.2, trainer.agents[0].epsilon)
        self.assertEqual(1, len(trainer.agents[0].replay_buffer))
        self.assertEqual(3, trainer.gradient_updates)
        old_hash = hash_torch_state_dict(trainer.agents[0].model.state_dict())
        trainer.optimizer_update_from_replay()
        self.assertEqual(4, trainer.gradient_updates)
        self.assertNotEqual(old_hash, hash_torch_state_dict(trainer.agents[0].model.state_dict()))

    def test_invalid_type_missing_field_corrupt_and_hash_mismatch_fail(self):
        trainer = dummy_trainer(self.temporary_directory.name)
        with self.assertRaisesRegex(ValueError, 'Invalid checkpoint_type'):
            trainer.save_checkpoint('legacy', 0)
        with self.assertRaisesRegex(ValueError, 'missing required'):
            trainer.validate_checkpoint_payload({'checkpoint_type': 'evaluation'})
        corrupt = os.path.join(self.temporary_directory.name, 'corrupt.pt')
        with open(corrupt, 'wb') as handle:
            handle.write(b'not a checkpoint')
        with self.assertRaisesRegex(IOError, 'Cannot load'):
            trainer.load_checkpoint_payload(corrupt)
        path = trainer.save_checkpoint('resumable', 1)
        trainer.config_hash = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'config_hash'):
            trainer.load_resumable_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
