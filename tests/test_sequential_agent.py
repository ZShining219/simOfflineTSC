import copy
import os
import random
import tempfile
import unittest

import numpy as np
import torch

from sequential.agent import SequentialDQNAgent, SequentialCounters
from sequential.config import load_sequential_config
from sequential.core import (
    ReplayMetadata, ReplayRecord, SequentialReplay, TargetUpdateScheduler,
    TrainingPayload, capture_rng_state, online_parameter_digest,
    optimizer_state_digest, replay_content_digest, replay_metadata_digest,
    rng_state_digest, target_parameter_digest,
)
from sequential.parents import convert_parent_replay
from sequential.trainer import SequentialStageTrainer


class FakeIntersection:
    def __init__(self, identity='intersection_1_1'):
        self.id = identity


class FakeGenerator:
    def __init__(self, world, intersection, value):
        self.world = world
        self.I = intersection
        self.value = value

    def generate(self):
        return np.array(self.value, copy=True)


class FakeWorld:
    def __init__(self, name):
        self.name = name
        self.intersection = FakeIntersection()
        self.steps = 0

    def reset(self):
        self.steps = 0

    def step(self, action):
        self.steps += 1


def fake_binding(world, rank):
    inter = world.intersection
    phase_mapping = tuple(f'phase_{index}' for index in range(8))
    lanes = tuple(f'lane_{index}' for index in range(8))
    return {
        'world': world,
        'inter': inter,
        'ob_generator': FakeGenerator(world, inter, np.zeros(8, dtype=np.float32)),
        'phase_generator': FakeGenerator(world, inter, [0]),
        'reward_generator': FakeGenerator(world, inter, [-1.0]),
        'queue_generator': FakeGenerator(world, inter, np.ones(8)),
        'delay_generator': FakeGenerator(world, inter, [0.25]),
        'signature': {
            'intersection_id': inter.id,
            'incoming_lane_mapping': lanes,
            'phase_action_mapping': phase_mapping,
            'state_dim': 8,
            'action_dim': 8,
        },
    }


class SequentialAgentTests(unittest.TestCase):
    def setUp(self):
        config = load_sequential_config()
        self.model_config = config['model']
        self.trainer_config = config['trainer']
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)

    @staticmethod
    def _legacy_payload(value):
        return (
            np.full((1, 8), value, dtype=np.float32),
            np.array([value % 8], dtype=np.int8),
            np.array([value % 8], dtype=np.int64),
            np.array(float(value % 3), dtype=np.float32),
            np.full((1, 8), value + 1, dtype=np.float32),
            np.array([(value + 1) % 8], dtype=np.int8),
        )

    def _agent(self, name='world_a'):
        return SequentialDQNAgent(
            FakeWorld(name), 0, self.model_config, self.trainer_config,
            binding_factory=fake_binding,
        )

    def _record(self, index, network='sumohz1x1_config3', stage=1):
        payload = TrainingPayload.from_legacy(self._legacy_payload(index))
        return ReplayRecord(payload, ReplayMetadata(
            str(index), network, stage, 1, index, index,
        ))

    def _parent_fixture(self, directory):
        source = self._agent()
        items = [
            (f'399_{index}_intersection_1_1', self._legacy_payload(index))
            for index in range(1, 66)
        ]
        source.replay = convert_parent_replay(
            items, 'sumohz1x1_config3', 144000, 5000,
        )
        source.epsilon = 0.00996820918179748
        source.counters = SequentialCounters(144000, 143000, 14300)
        source.target_scheduler = TargetUpdateScheduler.from_plan1_parent(
            143000, 14300, 10,
        )
        rng = capture_rng_state()
        checkpoint = {
            'agents': [{
                'online_model_state_dict': copy.deepcopy(source.model.state_dict()),
                'target_model_state_dict': copy.deepcopy(source.target_model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(source.optimizer.state_dict()),
                'epsilon': source.epsilon,
                'replay_state': {'capacity': 5000, 'items': items},
            }],
            'training_counters': {
                'global_decision_step': 144000,
                'gradient_updates': 143000,
                'target_updates': 14300,
            },
            **rng,
        }
        path = os.path.join(directory, 'parent.pt')
        torch.save(checkpoint, path)
        replay = convert_parent_replay(items, 'sumohz1x1_config3', 144000, 5000)
        manifest = {
            'checkpoint_path': path,
            'network': 'sumohz1x1_config3',
            'epsilon': source.epsilon,
            'next_target_sync_update': 143009,
            'digests': {
                'online_parameter_digest': online_parameter_digest(
                    source.model.state_dict()
                ),
                'target_parameter_digest': target_parameter_digest(
                    source.target_model.state_dict()
                ),
                'optimizer_state_digest': optimizer_state_digest(
                    source.optimizer.state_dict()
                ),
                'rng_state_digest': rng_state_digest(
                    rng['python_random_state'], rng['numpy_random_state'],
                    rng['torch_cpu_rng_state'], rng['torch_cuda_rng_states'],
                ),
                'replay_content_digest': replay_content_digest(replay.records),
                'replay_metadata_digest': replay_metadata_digest(replay.records),
            },
        }
        return manifest

    def test_three_policy_forks_are_identical_before_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._parent_fixture(directory)
            agents = [
                SequentialDQNAgent.from_parent(
                    FakeWorld(policy), 0, self.model_config, self.trainer_config,
                    manifest, binding_factory=fake_binding,
                )
                for policy in ('clear', 'fifo', 'fifo_matched_wait')
            ]
            digests = [agent.training_state_digests() for agent in agents]
            keys = set(digests[0]) - {'environment_signature_digest'}
            for key in keys:
                self.assertEqual(digests[0][key], digests[1][key], key)
                self.assertEqual(digests[0][key], digests[2][key], key)

    def test_policy_application_changes_only_clear_replay(self):
        agents = [self._agent(policy) for policy in ('clear', 'fifo', 'fifo_matched_wait')]
        records = [self._record(index) for index in range(70)]
        for agent in agents:
            agent.replay = SequentialReplay(5000, records)
        agents[0].begin_stage(2, 'sumohz1x1_config2', 'clear')
        agents[1].begin_stage(2, 'sumohz1x1_config2', 'fifo')
        agents[2].begin_stage(2, 'sumohz1x1_config2', 'fifo_matched_wait')
        self.assertEqual(len(agents[0].replay.records), 0)
        self.assertEqual(list(agents[1].replay.records), records)
        self.assertEqual(list(agents[2].replay.records), records)

    def test_rebind_preserves_training_state_and_drops_old_references(self):
        agent = self._agent('old')
        agent.replay = SequentialReplay(5000, [self._record(index) for index in range(70)])
        old_world = agent.world
        old_inter = agent.inter
        old_generators = (
            agent.ob_generator, agent.phase_generator, agent.reward_generator,
            agent.queue, agent.delay,
        )
        before = agent.training_state_digests()
        audit = agent.rebind_environment(FakeWorld('new'))
        after = agent.training_state_digests()
        for key in set(before) - {'environment_signature_digest'}:
            self.assertEqual(before[key], after[key], key)
        self.assertIsNot(agent.world, old_world)
        self.assertIsNot(agent.inter, old_inter)
        for generator in old_generators:
            self.assertNotIn(generator, (
                agent.ob_generator, agent.phase_generator,
                agent.reward_generator, agent.queue, agent.delay,
            ))
        self.assertTrue(audit['training_state_preserved'])
        self.assertIsNone(agent.last_observation)

    def test_clear_first_update_occurs_on_1001st_insertion(self):
        agent = self._agent()
        agent.begin_stage(2, 'sumohz1x1_config2', 'clear')
        initial_epsilon = agent.epsilon
        initial_target = target_parameter_digest(agent.target_model.state_dict())
        for index in range(1, 1001):
            payload = self._legacy_payload(index)
            agent.remember(
                payload[0], payload[1], payload[2], payload[3], payload[4], payload[5],
                1 + (index - 1) // 360, index, truncated=False,
            )
            self.assertIsNone(agent.successful_gradient_update())
        self.assertFalse(agent.should_use_random_warmup_action())
        self.assertEqual(agent.epsilon, initial_epsilon)
        self.assertEqual(agent.counters.gradient_updates, 0)
        self.assertEqual(agent.counters.target_updates, 0)
        self.assertEqual(
            target_parameter_digest(agent.target_model.state_dict()), initial_target
        )
        payload = self._legacy_payload(1001)
        agent.remember(
            payload[0], payload[1], payload[2], payload[3], payload[4], payload[5],
            3, 281,
        )
        result = agent.successful_gradient_update()
        self.assertIsNotNone(result)
        self.assertEqual(agent.counters.gradient_updates, 1)
        self.assertLess(agent.epsilon, initial_epsilon)

    def test_matched_wait_uses_stage_insertions_not_total_size(self):
        agent = self._agent()
        agent.replay = SequentialReplay(5000, [self._record(index) for index in range(5000)])
        agent.begin_stage(2, 'sumohz1x1_config2', 'fifo_matched_wait')
        self.assertFalse(agent.is_update_ready())
        for index in range(1, 1001):
            agent.replay.append(self._record(5000 + index, stage=2))
        self.assertFalse(agent.is_update_ready())
        agent.replay.append(self._record(6001, stage=2))
        self.assertTrue(agent.is_update_ready())

    def test_fifo_preserves_order_then_evicts_strictly_oldest(self):
        agent = self._agent()
        records = [self._record(index) for index in range(5000)]
        agent.replay = SequentialReplay(5000, records)
        original_content = replay_content_digest(agent.replay.records)
        original_metadata = replay_metadata_digest(agent.replay.records)
        agent.begin_stage(2, 'sumohz1x1_config2', 'fifo')
        self.assertEqual(replay_content_digest(agent.replay.records), original_content)
        self.assertEqual(replay_metadata_digest(agent.replay.records), original_metadata)
        agent.replay.append(self._record(5000, 'sumohz1x1_config2', 2))
        self.assertEqual(agent.replay.records[0].metadata.transition_id, '1')
        self.assertEqual(agent.replay.records[-1].metadata.transition_id, '5000')

    def test_stage_trainer_uses_agent_switch_api_and_writes_one_transition_per_decision(self):
        agent = self._agent()
        config = dict(self.trainer_config)
        config.update({'steps': 20, 'action_interval': 10})
        trainer = SequentialStageTrainer(agent, agent.world, config)
        new_world = FakeWorld('stage2')
        audit = trainer.rebind_environment(new_world)
        trainer.apply_replay_policy(2, 'sumohz1x1_config2', 'clear')
        result = trainer.train_episode(local_episode=1, global_episode=401)
        self.assertTrue(audit['training_state_preserved'])
        self.assertIs(agent.world, new_world)
        self.assertIs(trainer.world, new_world)
        self.assertEqual(result['simulation_steps'], 20)
        self.assertEqual(result['decision_steps'], 2)
        self.assertEqual(len(result['transitions']), 2)
        self.assertEqual(agent.counters.global_decision_step, 2)
        self.assertEqual(agent.replay.stage_insertions, 2)
        self.assertTrue(result['transitions'][-1]['truncated'])

    def test_full_checkpoint_restore_preserves_fifo_metadata_and_scheduler(self):
        agent = self._agent()
        agent.replay = SequentialReplay(5000, [self._record(index) for index in range(100)])
        agent.replay.stage_insertions = 17
        agent.policy = 'fifo'
        agent.current_stage_index = 2
        agent.current_network = 'sumohz1x1_config2'
        agent.counters = SequentialCounters(144017, 143017, 14301)
        agent.target_scheduler = TargetUpdateScheduler(10, 143019)
        before = agent.training_state_digests()
        state = agent.full_state_dict()
        agent.replay.append(self._record(100, stage=2))
        agent.epsilon = 0.5
        agent.counters.gradient_updates += 1
        agent.target_scheduler.next_update += 10
        agent.load_full_state_dict(state)
        after = agent.training_state_digests()
        self.assertEqual(before, after)
        self.assertEqual(agent.replay.stage_insertions, 17)
        self.assertEqual(
            [record.metadata.transition_id for record in agent.replay.records],
            [str(index) for index in range(100)],
        )


if __name__ == '__main__':
    unittest.main()
