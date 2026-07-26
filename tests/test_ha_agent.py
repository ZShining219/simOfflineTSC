import random
import unittest

import numpy as np
import torch

from sequential.core import (
    ReplayMetadata, ReplayRecord, SequentialReplay, TargetUpdateScheduler,
    TrainingPayload,
)
from sequential.ha_agent import HASODQNAgent
from sequential.historical_archive import HistoricalBatch


class Space:
    n = 8


class Visible:
    def __len__(self):
        return 100


class OfflineSampler:
    def __init__(self):
        self.counts = []

    def sample(self, count):
        self.counts.append(count)
        return HistoricalBatch(
            observations=np.ones((count, 16), dtype=np.float32),
            actions=np.arange(count, dtype=np.int64) % 8,
            rewards=np.full(count, -2, dtype=np.float32),
            next_observations=np.full((count, 16), 2, dtype=np.float32),
            terminated=np.zeros(count, dtype=np.bool_),
            truncated=np.zeros(count, dtype=np.bool_),
            source_networks=np.asarray(['past'] * count, dtype=object),
            local_indices=np.arange(count),
            transition_ids=np.asarray([f'offline:{i}' for i in range(count)], dtype=object),
            behavior_training_seeds=np.zeros(count, dtype=np.int64),
            episode_ids=np.ones(count, dtype=np.int64),
            decision_steps=np.arange(1, count + 1),
            run_ids=np.asarray(['run'] * count, dtype=object),
            shard_sha256=np.asarray(['a' * 64] * count, dtype=object),
        )


def agent(ratio=0.0):
    result = HASODQNAgent.__new__(HASODQNAgent)
    result.model = torch.nn.Linear(16, 8)
    result.target_model = torch.nn.Linear(16, 8)
    result.target_model.load_state_dict(result.model.state_dict())
    result.optimizer = torch.optim.SGD(result.model.parameters(), lr=0.0)
    result.criterion = torch.nn.MSELoss(reduction='mean')
    result.gamma = 0.95
    result.grad_clip = 5.0
    result.epsilon = 0.1
    result.epsilon_min = 0.01
    result.epsilon_decay = 0.995
    result.batch_size = 64
    result.learning_start = 1000
    result.phase = True
    result.one_hot = True
    result.action_space = Space()
    result.policy = 'ha_sodqn'
    result.archive_mode = 'NONE' if ratio == 0 else 'P1C'
    result.training_seed = 0
    result.ordered_networks = ('current', 'past')
    result.method = 'CONT' if ratio == 0 else 'DHOA'
    result.offline_ratio = ratio
    result.owp_capacity = 5000
    result.alignment_warmup_episodes = 10
    result.last_local_episode = 11
    result.visible_archive = None if ratio == 0 else Visible()
    result.historical_sampler = None if ratio == 0 else OfflineSampler()
    result.online_rng = random.Random(11)
    result.hoa_rng = random.Random(12)
    result.alignment_observations = []
    result.current_stage_index = 1
    result.current_network = 'current'
    result.counters = type('Counters', (), {
        'global_decision_step': 1100, 'gradient_updates': 0, 'target_updates': 0,
    })()
    result.target_scheduler = TargetUpdateScheduler(10, 1)
    records = []
    for index in range(1001):
        phase = np.asarray([index % 8], dtype=np.int8)
        records.append(ReplayRecord(
            TrainingPayload(
                np.full((1, 8), index % 5, dtype=np.float32), phase,
                np.asarray([index % 8]), np.asarray([-1.0]),
                np.full((1, 8), (index + 1) % 5, dtype=np.float32),
                np.asarray([(index + 1) % 8], dtype=np.int8),
            ),
            ReplayMetadata(
                f'online:{index}', 'current', 1, 11, index + 1, index + 1,
            ),
        ))
    result.replay = SequentialReplay(5000, records)
    return result


class HASODQNAgentTest(unittest.TestCase):
    def test_empty_archive_fallback_uses_full_online_batch(self):
        subject = agent(0.0)
        update = subject.successful_gradient_update()
        self.assertEqual(64, update['online_count'])
        self.assertEqual(0, update['offline_count'])
        self.assertEqual(0.0, update['actual_offline_ratio'])
        self.assertIsNone(update['loss_offline'])

    def test_mixed_quota_and_weighted_branch_loss(self):
        for ratio, offline_count in ((0.25, 16), (0.5, 32), (0.75, 48)):
            subject = agent(ratio)
            update = subject.successful_gradient_update()
            self.assertEqual(64 - offline_count, update['online_count'])
            self.assertEqual(offline_count, update['offline_count'])
            self.assertEqual([offline_count], subject.historical_sampler.counts)
            self.assertAlmostEqual(
                update['loss'],
                (1 - ratio) * update['loss_online'] + ratio * update['loss_offline'],
                places=6,
            )

    def test_replay_sampling_does_not_consume_action_numpy_rng(self):
        subject = agent(0.5)
        before = np.random.get_state()
        subject.successful_gradient_update()
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_uniform_warmup_disables_offline_branch(self):
        subject = agent(0.5)
        subject.last_local_episode = 10
        update = subject.successful_gradient_update()
        self.assertEqual(0, update['offline_count'])
        self.assertEqual(64, update['online_count'])

    def test_episode_zero_path_restores_plan1_random_action_warmup(self):
        subject = agent(0.0)
        subject.counters.global_decision_step = 1000
        self.assertTrue(subject.should_use_random_warmup_action())
        subject.counters.global_decision_step = 1001
        self.assertFalse(subject.should_use_random_warmup_action())

    def test_cont_checkpoint_round_trip_restores_private_rng_and_orb(self):
        subject = agent(0.0)
        state = subject.full_state_dict()
        expected_online = subject.online_rng.random()
        expected_hoa = subject.hoa_rng.random()
        restored = agent(0.0)
        restored.load_full_state_dict(state)
        self.assertEqual(1001, len(restored.replay.records))
        self.assertEqual(expected_online, restored.online_rng.random())
        self.assertEqual(expected_hoa, restored.hoa_rng.random())


if __name__ == '__main__':
    unittest.main()
