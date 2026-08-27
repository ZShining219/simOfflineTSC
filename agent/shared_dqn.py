"""Parameter-shared decentralized DQN for one or more intersections."""

from __future__ import annotations

import os
import random
import dataclasses
from datetime import datetime, timezone
from collections import Counter, deque

import gym
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from arterial.control import build_action_mask, stable_intersection_order
from arterial.experiment import ARTERIAL_SCENES
from arterial.replay import (HistoricalPool, LocalTransition, mixed_batch,
                             split_local_transitions)
from common.registry import Registry
from generator import IntersectionPhaseGenerator, LaneVehicleGenerator
from .base import BaseAgent
from .dqn import DQNNet
from . import utils


@Registry.register_model('shared_dqn')
class SharedDQNAgent(BaseAgent):
    """One DQN/target/optimizer shared by all controlled intersections."""

    def __init__(self, world, rank=0):
        super().__init__(world)
        if rank != 0:
            raise ValueError('SharedDQNAgent must be instantiated once at rank 0')
        trainer = Registry.mapping['trainer_mapping']['setting'].param
        model = Registry.mapping['model_mapping']['setting'].param
        self.rank = 0
        self.intersection_ids = stable_intersection_order(world.intersection_ids)
        self.id = ','.join(self.intersection_ids)
        self.sub_agents = len(self.intersection_ids)
        expected_intersections = model.get('expected_num_intersections')
        if (expected_intersections is not None
                and int(expected_intersections) != self.sub_agents):
            raise ValueError(
                f'Configured num_intersections={expected_intersections} but '
                f'world discovered {self.sub_agents}')
        self.intersections = [world.id2intersection[x] for x in self.intersection_ids]
        self.action_dims = tuple(len(x.phases) for x in self.intersections)
        self.action_dim = max(self.action_dims)
        self.action_masks = build_action_mask(self.action_dims)
        self.action_space = gym.spaces.Discrete(self.action_dim)
        self.phase = bool(model.get('phase', True))
        self.one_hot = bool(model.get('one_hot', True))
        self.state_variant = model.get('state_variant', 'base')
        if self.state_variant not in {'base', 'in_out'}:
            raise ValueError(f'Unsupported state_variant: {self.state_variant}')
        self.use_position_encoding = bool(model.get('use_position_encoding', False))
        self.normalize_lane_counts = bool(model.get('normalize_lane_counts', False))
        self.lane_capacity = float(model.get('lane_capacity', 1.0))
        if self.normalize_lane_counts and self.lane_capacity <= 0:
            raise ValueError('lane_capacity must be positive')
        self.reward_variant = model.get('reward_variant', 'original')
        if self.reward_variant not in {'original', 'pressure'}:
            raise ValueError(f'Unsupported reward_variant: {self.reward_variant}')
        self.gamma = float(model['gamma'])
        self.grad_clip = float(model['grad_clip'])
        self.epsilon = float(model['epsilon'])
        self.initial_epsilon = self.epsilon
        self.epsilon_decay = float(model['epsilon_decay'])
        self.epsilon_min = float(model['epsilon_min'])
        self.learning_rate = float(model['learning_rate'])
        self.batch_size = int(model['batch_size'])
        self.buffer_size = int(trainer['buffer_size'])
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.replay_total_collected = 0
        self.replay_sample_total = 0
        self.replay_sample_counts = Counter()
        self.offline_pool = HistoricalPool()
        self.offline_ratio = float(model.get('offline_ratio', 0.0))
        self.offline_sampling_strategy = model.get(
            'offline_sampling_strategy', 'scene_intersection_balanced')
        self.visible_history_scenes = tuple()
        command = Registry.mapping['command_mapping']['setting'].param
        self.training_seed = int(command['seed'])
        self.scene_id = str(model.get('scene_id', command['network']))
        if (command['network'].startswith('sumoarterial1x6_')
                and self.scene_id not in ARTERIAL_SCENES):
            raise ValueError(
                f'Invalid arterial scene_id {self.scene_id!r}; quote YAML scene IDs')
        self.roadnet_id = str(model.get('roadnet_id', command['network']))
        self.training_stage = int(model.get('training_stage', 0))
        self.policy_version = 0
        self.exploration_actions = 0
        self.greedy_actions = 0
        self.last_train_diagnostics = None
        self.collect_history = bool(model.get('collect_history', False))
        self.archive_pending = []
        self._create_generators()
        self.ob_length = self._infer_ob_length()
        self.model = DQNNet(self.ob_length, self.action_dim)
        self.target_model = DQNNet(self.ob_length, self.action_dim)
        self.update_target_network()
        self.optimizer = optim.RMSprop(
            self.model.parameters(), lr=self.learning_rate,
            alpha=0.9, centered=False, eps=1e-7)
        history_paths = model.get('history_paths', [])
        scene_order = tuple(model.get('scene_order', [self.scene_id]))
        history_mode = model.get('history_access_mode', 'causal')
        if scene_order:
            self.visible_history_scenes = HistoricalPool.visible_scenes(
                scene_order, self.training_stage, history_mode)
        sampling_index = model.get('hoa_sampling_index_path')
        if history_paths:
            self.offline_pool = (
                HistoricalPool.load_sampling_index(sampling_index, history_paths)
                if sampling_index else HistoricalPool.load_many(history_paths))
            expected_state = {
                'variant': self.state_variant, 'dimension': self.ob_length,
                'phase': self.phase, 'phase_one_hot': self.one_hot,
                'position_encoding': self.use_position_encoding,
                'normalized_lane_counts': self.normalize_lane_counts,
            }
            for manifest in self.offline_pool.manifests:
                if tuple(manifest['intersection_ids']) != self.intersection_ids:
                    raise ValueError('History intersection IDs/order do not match world')
                if manifest['state_schema'] != expected_state:
                    raise ValueError('History state schema does not match shared DQN')
                action = manifest['action_schema']
                if (tuple(action['dimensions']) != self.action_dims or
                        int(action['shared_network_output_dimension']) != self.action_dim):
                    raise ValueError('History action schema does not match shared DQN')
                if manifest['reward_schema']['variant'] != self.reward_variant:
                    raise ValueError('History reward schema does not match shared DQN')
            if not self.visible_history_scenes and self.offline_ratio:
                self.offline_ratio = 0.0
        elif self.offline_ratio and self.visible_history_scenes:
            raise ValueError(
                'offline_ratio requires history_paths for the visible history set')

    @staticmethod
    def _lanes(roads, intersection):
        return [lane for road in roads for lane in intersection.road_lane_mapping[road]]

    def _create_generators(self):
        self.in_generators = [LaneVehicleGenerator(
            self.world, inter, ['lane_count'], in_only=True, average=None)
            for inter in self.intersections]
        self.phase_generators = [IntersectionPhaseGenerator(
            self.world, inter, ['phase'], targets=['cur_phase'], negative=False)
            for inter in self.intersections]
        self.original_reward_generators = [LaneVehicleGenerator(
            self.world, inter, ['lane_waiting_count'], in_only=True,
            average='all', negative=True) for inter in self.intersections]
        self.queue_generators = [LaneVehicleGenerator(
            self.world, inter, ['lane_waiting_count'], in_only=True,
            negative=False) for inter in self.intersections]
        self.delay_generators = [LaneVehicleGenerator(
            self.world, inter, ['lane_delay'], in_only=True,
            average='all', negative=False) for inter in self.intersections]
        self.in_lanes = [self._lanes(inter.in_roads, inter)
                         for inter in self.intersections]
        self.out_lanes = [self._lanes(inter.out_roads, inter)
                          for inter in self.intersections]

    def _lane_values(self, lanes):
        values = self.world.get_info('lane_count')
        divisor = self.lane_capacity if self.normalize_lane_counts else 1.0
        return np.asarray([values[lane] / divisor for lane in lanes], dtype=np.float32)

    def _local_ob(self, index):
        incoming = np.asarray(self.in_generators[index].generate(), dtype=np.float32)
        if self.normalize_lane_counts:
            incoming = incoming / self.lane_capacity
        chunks = [incoming]
        if self.state_variant == 'in_out':
            chunks.append(self._lane_values(self.out_lanes[index]))
        if self.use_position_encoding:
            denominator = max(1, self.sub_agents - 1)
            chunks.append(np.asarray([
                index / denominator, float(index == 0),
                float(index == self.sub_agents - 1),
            ], dtype=np.float32))
        return np.concatenate(chunks)

    def _infer_ob_length(self):
        lengths = []
        for index in range(self.sub_agents):
            length = int(self.in_generators[index].ob_length)
            if self.state_variant == 'in_out':
                length += len(self.out_lanes[index])
            if self.use_position_encoding:
                length += 3
            lengths.append(length)
        if len(set(lengths)) != 1:
            raise ValueError(
                'Parameter sharing requires equal local state dimensions; '
                f'got {dict(zip(self.intersection_ids, lengths))}')
        return lengths[0] + (self.action_dim if self.phase and self.one_hot
                             else int(self.phase))

    def reset(self):
        self.intersections = [self.world.id2intersection[x] for x in self.intersection_ids]
        self._create_generators()

    def get_ob(self):
        return np.stack([self._local_ob(i) for i in range(self.sub_agents)])

    def get_phase(self):
        return np.asarray([
            getattr(inter, 'virtual_phase', self.phase_generators[i].generate()[0])
            for i, inter in enumerate(self.intersections)
        ], dtype=np.int64)

    def get_reward(self):
        if self.reward_variant == 'pressure':
            pressures = self.world.get_pressure()
            return -np.asarray([pressures[x] for x in self.intersection_ids],
                               dtype=np.float32)
        return np.asarray([
            float(np.mean(generator.generate())) * 12.0
            for generator in self.original_reward_generators
        ], dtype=np.float32)

    def get_queue(self):
        return np.asarray([np.sum(x.generate()) for x in self.queue_generators],
                          dtype=np.float32)

    def get_delay(self):
        return np.asarray([np.sum(x.generate()) for x in self.delay_generators],
                          dtype=np.float32)

    def get_pressure(self):
        pressures = self.world.get_pressure()
        return np.asarray([pressures[x] for x in self.intersection_ids],
                          dtype=np.float32)

    def _features(self, ob, phase):
        ob = np.asarray(ob, dtype=np.float32)
        if not self.phase:
            return ob
        phase = np.asarray(phase, dtype=np.int64).reshape(-1)
        if self.one_hot:
            return np.concatenate([ob, utils.idx2onehot(phase, self.action_dim)], axis=1)
        return np.concatenate([ob, phase[:, None]], axis=1)

    def get_action(self, ob, phase, test=False, action_mask=None):
        mask = self.action_masks if action_mask is None else np.asarray(action_mask, dtype=bool)
        features = torch.as_tensor(self._features(ob, phase), dtype=torch.float32)
        q_values = self.model(features, train=False).cpu().numpy()
        q_values[~mask] = -np.inf
        greedy = np.argmax(q_values, axis=1)
        if test:
            return greedy
        actions = greedy.copy()
        for index in range(self.sub_agents):
            if np.random.rand() <= self.epsilon:
                actions[index] = np.random.choice(np.flatnonzero(mask[index]))
                self.exploration_actions += 1
            else:
                self.greedy_actions += 1
        return actions

    def sample(self):
        return np.asarray([np.random.randint(0, dim) for dim in self.action_dims])

    def get_action_prob(self, ob, phase):
        return None

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards,
                 obs, cur_phase, done, key, *, episode_id=None,
                 decision_step=None, terminated=None, truncated=False):
        del actions_prob
        parts = str(key).split('_')
        episode_id = int(episode_id if episode_id is not None else parts[0])
        decision_step = int(decision_step if decision_step is not None else parts[1])
        terminated = bool(done if terminated is None else terminated)
        transitions = split_local_transitions(
            self.intersection_ids, last_obs, last_phase, actions, rewards,
            obs, cur_phase, terminated, truncated, self.scene_id,
            episode_id, decision_step, self.training_stage, self.policy_version,
            collector_training_seed=self.training_seed)
        for transition in transitions:
            self.replay_buffer.append(transition)
            self.replay_total_collected += 1
            if self.collect_history:
                self.archive_pending.append(dataclasses.replace(
                    transition, metadata=dataclasses.replace(
                        transition.metadata, source='offline_history')))

    def configure_history(self, pool, visible_scenes, offline_ratio=None,
                          sampling_strategy=None):
        self.offline_pool = pool
        self.visible_history_scenes = tuple(visible_scenes)
        if offline_ratio is not None:
            self.offline_ratio = float(offline_ratio)
        if sampling_strategy is not None:
            self.offline_sampling_strategy = sampling_strategy

    def export_history(self, directory, roadnet_id, training_seed,
                       num_episodes, num_decision_steps, source_policy='shared_dqn'):
        records = [dataclasses.replace(
            item, metadata=dataclasses.replace(
                item.metadata, source='offline_history'))
            for item in self.replay_buffer]
        pool = HistoricalPool(records)
        manifest = {
            'roadnet_id': str(roadnet_id),
            'scene_id': self.scene_id,
            'intersection_ids': list(self.intersection_ids),
            'state_schema': {
                'variant': self.state_variant,
                'dimension': self.ob_length,
                'phase': self.phase,
                'phase_one_hot': self.one_hot,
                'position_encoding': self.use_position_encoding,
                'normalized_lane_counts': self.normalize_lane_counts,
            },
            'action_schema': {
                'kind': 'local_discrete', 'dimensions': list(self.action_dims),
                'shared_network_output_dimension': self.action_dim,
            },
            'reward_schema': {
                'variant': self.reward_variant,
                'original': 'negative mean incoming lane waiting count times 12',
                'pressure': 'negative(incoming vehicle count - outgoing vehicle count)',
                'network_aggregation': 'mean for reporting; local rewards for DQN updates',
            },
            'num_episodes': int(num_episodes),
            'num_decision_steps': int(num_decision_steps),
            'num_transitions': len(records),
            'training_seed': int(training_seed),
            'source_policy': str(source_policy),
            'collection_stage': int(self.training_stage),
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        pool.save(directory, manifest)
        return manifest

    def drain_archive_records(self):
        records, self.archive_pending = self.archive_pending, []
        return records

    def history_manifest(self, roadnet_id, training_seed, num_episodes,
                         num_decision_steps, source_policy='shared_dqn'):
        return {
            'roadnet_id': str(roadnet_id), 'scene_id': self.scene_id,
            'intersection_ids': list(self.intersection_ids),
            'state_schema': {
                'variant': self.state_variant, 'dimension': self.ob_length,
                'phase': self.phase, 'phase_one_hot': self.one_hot,
                'position_encoding': self.use_position_encoding,
                'normalized_lane_counts': self.normalize_lane_counts,
            },
            'action_schema': {
                'kind': 'local_discrete', 'dimensions': list(self.action_dims),
                'shared_network_output_dimension': self.action_dim,
            },
            'reward_schema': {
                'variant': self.reward_variant,
                'original': 'negative mean incoming lane waiting count times 12',
                'pressure': 'negative(incoming vehicle count - outgoing vehicle count)',
                'network_aggregation': 'mean for reporting; local rewards for DQN updates',
            },
            'num_episodes': int(num_episodes),
            'num_decision_steps': int(num_decision_steps),
            'training_seed': int(training_seed),
            'source_policy': str(source_policy),
            'collection_stage': int(self.training_stage),
            'created_at': datetime.now(timezone.utc).isoformat(),
        }

    def _batch(self, tagged):
        records = [item[0] for item in tagged]
        states = np.stack([x.state for x in records])
        phases = np.concatenate([x.phase for x in records])
        next_states = np.stack([x.next_state for x in records])
        next_phases = np.concatenate([x.next_phase for x in records])
        masks = np.stack([
            self.action_masks[self.intersection_ids.index(
                x.metadata.intersection_id)] for x in records
        ])
        return (
            torch.as_tensor(self._features(states, phases), dtype=torch.float32),
            torch.as_tensor(self._features(next_states, next_phases), dtype=torch.float32),
            torch.as_tensor([x.action for x in records], dtype=torch.long),
            torch.as_tensor([x.reward for x in records], dtype=torch.float32),
            torch.as_tensor([x.terminated or x.truncated for x in records], dtype=torch.float32),
            torch.as_tensor(masks, dtype=torch.bool),
        )

    def is_training_ready(self):
        offline_count = int(round(self.batch_size * self.offline_ratio))
        online_count = self.batch_size - offline_count
        if len(self.replay_buffer) < online_count:
            return False
        if offline_count:
            return bool(self.visible_history_scenes and self.offline_pool.records)
        return True

    def train(self):
        class OnlineView:
            def __init__(self, records): self.records = records
            def __len__(self): return len(self.records)
            def sample(self, count, rng=random):
                return rng.sample(list(self.records), count)
        tagged = mixed_batch(
            OnlineView(self.replay_buffer), self.offline_pool, self.batch_size,
            self.offline_ratio, self.visible_history_scenes,
            strategy=self.offline_sampling_strategy, rng=random)
        state, next_state, actions, rewards, dones, action_masks = self._batch(tagged)
        q_values = self.model(state, train=True)
        chosen_q = q_values.gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_values = self.target_model(next_state, train=False)
            next_values = next_values.masked_fill(~action_masks, -torch.inf)
            next_q = next_values.max(dim=1).values
            target_q = rewards + self.gamma * (1.0 - dones) * next_q
        td_error = chosen_q - target_q
        per_item_loss = td_error.square()
        total_loss = per_item_loss.mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        gradient_norm = float(clip_grad_norm_(self.model.parameters(), self.grad_clip))
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        sources = [source for _, source in tagged]
        online_mask = torch.as_tensor([x == 'online_current' for x in sources])
        offline_mask = ~online_mask
        def mean_or_none(values, mask):
            return None if not bool(mask.any()) else float(values[mask].mean().detach())
        offline_records = [record for record, source in tagged
                           if source == 'offline_history']
        self.replay_sample_total += len(tagged)
        self.replay_sample_counts.update(
            record.metadata.transition_id for record, _ in tagged)
        self.last_train_diagnostics = {
            'total_td_loss': float(total_loss.detach()),
            'online_td_loss': mean_or_none(per_item_loss, online_mask),
            'offline_td_loss': mean_or_none(per_item_loss, offline_mask),
            'online_mean_abs_td_error': mean_or_none(td_error.abs(), online_mask),
            'offline_mean_abs_td_error': mean_or_none(td_error.abs(), offline_mask),
            'online_mean_q': mean_or_none(chosen_q, online_mask),
            'offline_mean_q': mean_or_none(chosen_q, offline_mask),
            'target_q_mean': float(target_q.mean()),
            'gradient_norm': gradient_norm,
            'online_sample_count': int(online_mask.sum()),
            'offline_sample_count': int(offline_mask.sum()),
            'source_scene_distribution': dict(Counter(
                x.metadata.scene_id for x in offline_records)),
            'source_intersection_distribution': dict(Counter(
                x.metadata.intersection_id for x in offline_records)),
            'source_collector_seed_distribution': dict(Counter(
                str(x.metadata.collector_training_seed) for x in offline_records)),
        }
        self.policy_version += 1
        return total_loss.detach().cpu().numpy()

    def replay_utilization(self):
        values = sorted(self.replay_sample_counts.values())
        collected = self.replay_total_collected
        sampled_unique = sum(
            transition_id.startswith(f'{self.scene_id}:')
            for transition_id in self.replay_sample_counts)
        def percentile(q):
            return 0.0 if not values else float(np.quantile(values, q))
        return {
            'collected_transitions': collected,
            'sampled_transitions': self.replay_sample_total,
            'unique_sampled_transitions': sampled_unique,
            'unique_coverage': 0.0 if not collected else sampled_unique / collected,
            'sample_count_mean': (0.0 if not collected
                                  else self.replay_sample_total / collected),
            'sample_count_median': percentile(0.5),
            'sample_count_p95': percentile(0.95),
            'sample_count_max': float(max(values, default=0)),
            'sampled_transition_mean_age': 0.0,
        }

    def replay_utilization_state(self):
        return {
            'total_collected': self.replay_total_collected,
            'sample_total': self.replay_sample_total,
            'sample_counts': dict(self.replay_sample_counts),
        }

    def load_replay_utilization_state(self, state):
        self.replay_total_collected = int(state['total_collected'])
        self.replay_sample_total = int(state['sample_total'])
        self.replay_sample_counts = Counter(state['sample_counts'])

    def initialize_replay_utilization_from_buffer(self, total_collected):
        self.replay_total_collected = max(
            int(total_collected) * self.sub_agents, len(self.replay_buffer))

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def reset_epsilon(self, mode):
        if mode == 'reset_schedule':
            self.epsilon = self.initial_epsilon
        elif mode == 'fixed_low':
            self.epsilon = self.epsilon_min
        elif mode != 'continue_schedule':
            raise ValueError(f'Unknown epsilon mode: {mode}')

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        os.makedirs(path, exist_ok=True)
        torch.save(self.target_model.state_dict(), os.path.join(path, f'{e}_0.pt'))

    def load_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                            'model', f'{e}_0.pt')
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state)
        self.target_model.load_state_dict(state)
