import json
import os
import tempfile
import unittest
from collections import deque

import numpy as np

from agent.dqn import DQNAgent
from utils.trajectory import EpisodeTrajectoryWriter


def append_episode(writer, episode_id, first_global_step, count=3):
    writer.start_episode(episode_id)
    for decision in range(1, count + 1):
        state_value = (episode_id - 1) * 10 + decision - 1
        writer.append(
            episode_id=episode_id,
            decision_step=decision,
            global_step=first_global_step + decision - 1,
            state=np.asarray([[[state_value, state_value + 0.5]]], dtype=np.float32),
            current_phase=np.asarray([[decision - 1]], dtype=np.int64),
            action=np.asarray([[decision % 2]], dtype=np.int64),
            reward=np.asarray([[-float(decision)]], dtype=np.float32),
            next_state=np.asarray(
                [[[state_value + 1, state_value + 1.5]]], dtype=np.float32
            ),
            next_phase=np.asarray([[decision]], dtype=np.int64),
            terminated=False,
            truncated=decision == count,
            epsilon=0.1,
            behavior_mode='random_warmup',
            queue=float(decision),
            approximate_delay=0.2,
            real_delay=0.3,
            throughput=decision,
            waiting_time=0.4,
        )
    return writer.finish_episode()


class Plan1TrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.writer = EpisodeTrajectoryWriter(
            output_path=self.temporary_directory.name,
            network='sumohz1x1',
            behavior_training_seed=0,
            config_hash='a' * 64,
            simulation_steps=30,
            action_interval=10,
            action_dim=2,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_episode_shards_are_atomic_indexed_and_validated(self):
        first = append_episode(self.writer, 1, 1)
        second = append_episode(self.writer, 2, 4)
        self.assertTrue(os.path.isfile(first))
        self.assertTrue(os.path.isfile(second))
        result = self.writer.validate(expected_episodes=2)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(6, result['transition_count'])
        self.assertEqual(0, result['evaluation_transition_count'])
        with open(self.writer.index_path, encoding='utf-8') as handle:
            entries = [json.loads(line) for line in handle]
        self.assertEqual([1, 2], [entry['episode_id'] for entry in entries])
        with np.load(first, allow_pickle=False) as data:
            self.assertEqual('sumohz1x1', data['network'].item())
            self.assertEqual((3, 1, 1, 2), data['state'].shape)
            self.assertEqual('random_warmup', data['behavior_mode'][0])

    def test_validation_reports_corrupt_semantics_without_overwriting_shard(self):
        path = append_episode(self.writer, 1, 1)
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays['action'] = arrays['action'].copy()
        arrays['action'][1] = 9
        with open(path, 'wb') as handle:
            np.savez_compressed(handle, **arrays)
        result = self.writer.validate(expected_episodes=1)
        self.assertFalse(result['valid'])
        self.assertTrue(any('hash mismatch' in item for item in result['errors']))
        with self.assertRaises(FileExistsError):
            append_episode(self.writer, 1, 1)

    def test_online_replay_tuple_is_not_expanded(self):
        agent = DQNAgent.__new__(DQNAgent)
        agent.replay_buffer = deque(maxlen=5)
        agent.remember(
            'obs', 'phase', 'action', None, 'reward', 'next_obs',
            'next_phase', False, 'key',
        )
        key, payload = agent.replay_buffer[0]
        self.assertEqual('key', key)
        self.assertEqual(6, len(payload))
        self.assertEqual(
            ('obs', 'phase', 'action', 'reward', 'next_obs', 'next_phase'),
            payload,
        )


if __name__ == '__main__':
    unittest.main()
