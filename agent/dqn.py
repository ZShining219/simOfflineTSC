from . import RLAgent
from common.registry import Registry
from agent import utils
import dataclasses
import numpy as np
import os
import random
from collections import Counter, deque
from datetime import datetime, timezone
import gym

from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from utils.scene_context import pack_scene_transition, unpack_scene_transition
from arterial.replay import (HistoricalPool, LocalTransition, mixed_batch,
                             split_local_transitions)


@Registry.register_model('dqn')
class DQNAgent(RLAgent):
    '''
    DQNAgent determines each intersection's action with its own intersection information.
    '''
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)
        # Low-overhead cumulative replay diagnostics.  Only counters and a
        # histogram are retained; sampled mini-batches are never logged.
        self.replay_total_collected = 0
        self.replay_sample_total = 0
        self.replay_sample_counts = {}
        self.replay_sample_count_histogram = {}
        self.replay_insert_step = {}
        self.replay_sample_age_sum = 0.0

        self.world = world
        self.sub_agents = 1
        self.rank = rank

        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']

        # get generator for each DQNAgent
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.inter_id = inter_id
        self.inter = inter_obj
        self.ob_generator = LaneVehicleGenerator(self.world,  self.inter, ['lane_count'], in_only=True, average=None)

        self.phase_generator = IntersectionPhaseGenerator(world,  self.inter, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world,  self.inter, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)
        self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[inter_id].phases))

        if self.phase:
            if self.one_hot:
                self.ob_length = self.ob_generator.ob_length + len(self.world.id2intersection[inter_id].phases)
            else:
                self.ob_length = self.ob_generator.ob_length + 1
        else:
            self.ob_length = self.ob_generator.ob_length

        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.initial_epsilon = self.epsilon
        self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']
        self.double_dqn = bool(
            Registry.mapping['model_mapping']['setting'].param.get('double_dqn', False))
        self.dueling = bool(
            Registry.mapping['model_mapping']['setting'].param.get('dueling', False))

        model_param = Registry.mapping['model_mapping']['setting'].param
        command_param = Registry.mapping['command_mapping']['setting'].param
        self.offline_pool = HistoricalPool()
        self.offline_ratio = float(model_param.get('offline_ratio', 0.0))
        self.offline_sampling_strategy = model_param.get(
            'offline_sampling_strategy', 'scene_intersection_balanced')
        self.visible_history_scenes = tuple()
        self.scene_id = str(model_param.get('scene_id', command_param['network']))
        self.roadnet_id = str(model_param.get('roadnet_id', command_param['network']))
        self.training_stage = int(model_param.get('training_stage', 0))
        self.training_seed = int(command_param['seed'])
        self.policy_version = 0
        self.collect_history = bool(model_param.get('collect_history', False))
        self.archive_pending = []
        self.last_train_diagnostics = None
        history_paths = model_param.get('history_paths', [])
        scene_order = tuple(model_param.get('scene_order', [self.scene_id]))
        history_mode = model_param.get('history_access_mode', 'causal')
        if scene_order:
            self.visible_history_scenes = HistoricalPool.visible_scenes(
                scene_order, self.training_stage, history_mode)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=self.learning_rate,
                                       alpha=0.9, centered=False, eps=1e-7)

        if history_paths:
            self.offline_pool = self._materialize_history(history_paths)
            self._validate_history_manifests()
            if not self.visible_history_scenes and self.offline_ratio:
                self.offline_ratio = 0.0
        elif self.offline_ratio and self.visible_history_scenes:
            raise ValueError(
                'offline_ratio requires history_paths for the visible history set')

        self.anchor_lambda_policy = float(
            model_param.get('anchor_lambda_policy', 0.0))
        self.anchor_lambda_value = float(
            model_param.get('anchor_lambda_value', 0.0))
        self.anchor_temperature = float(
            model_param.get('anchor_temperature', 1.0))
        self.anchor_teachers = {}
        if self.anchor_lambda_policy or self.anchor_lambda_value:
            teacher_paths = model_param.get('teacher_paths') or {}
            missing = [scene for scene in self.visible_history_scenes
                       if scene not in teacher_paths]
            if self.visible_history_scenes and missing:
                raise ValueError(
                    'Anchor losses require teacher_paths covering every '
                    f'visible history scene; missing: {missing}')
            self.anchor_teachers = self._load_anchor_teachers(teacher_paths)

    def _load_anchor_teachers(self, teacher_paths):
        """Freeze stage-end online networks as per-scene anchor teachers."""
        teachers = {}
        rank = int(getattr(self, 'rank', 0))
        for scene_id, path in teacher_paths.items():
            payload = torch.load(str(path), map_location='cpu')
            agents = payload.get('agents')
            if agents:
                state = agents[rank].get('online_model_state_dict')
            else:
                state = payload.get('online_model_state_dict')
            if state is None:
                raise ValueError(
                    'Anchor teacher checkpoint lacks online_model_state_dict: '
                    f'{path}')
            net = self._build_model()
            net.load_state_dict(state)
            net.eval()
            for param in net.parameters():
                param.requires_grad_(False)
            teachers[str(scene_id)] = net
        return teachers

    def _anchor_loss(self, b_t, q_pred, actions, anchor_groups):
        """Output-anchoring losses on replayed historical states.

        The TD term only constrains the executed action of each replayed
        transition, so the remaining logits drift through shared weights.
        For each source scene, the frozen stage-end teacher anchors the
        softmax action distribution (policy anchor) and optionally the Q
        values of the non-executed actions (value anchor, masked so it does
        not fight the TD residual on the sampled action).
        """
        policy_sum = q_pred.new_zeros(())
        value_sum = q_pred.new_zeros(())
        total = 0
        for scene_id, rows in anchor_groups.items():
            teacher = self.anchor_teachers.get(str(scene_id))
            if teacher is None:
                raise RuntimeError(
                    'Anchor loss enabled but no teacher for scene '
                    f'{scene_id}')
            idx = torch.as_tensor(rows, dtype=torch.long)
            n = idx.numel()
            total += n
            q_cur = q_pred.index_select(0, idx)
            with torch.no_grad():
                q_ref = teacher(b_t.index_select(0, idx), train=False)
            if self.anchor_lambda_policy:
                temperature = self.anchor_temperature
                logp = F.log_softmax(q_cur / temperature, dim=1)
                p_ref = F.softmax(q_ref / temperature, dim=1)
                policy_sum = policy_sum + n * F.kl_div(
                    logp, p_ref, reduction='batchmean')
            if self.anchor_lambda_value:
                keep = torch.ones_like(q_cur, dtype=torch.bool)
                keep.scatter_(
                    1, actions.index_select(0, idx).view(n, 1), False)
                value_sum = value_sum + n * F.smooth_l1_loss(
                    q_cur.masked_select(keep).view(n, -1),
                    q_ref.masked_select(keep).view(n, -1))
        scale = max(total, 1)
        return policy_sum / scale, value_sum / scale

    @staticmethod
    def _materialize_history(history_paths):
        """Load all archive shards eagerly.

        LazyHistoricalPool resolves each sampled record by loading a random
        shard file, which is right for multi-intersection archives but costs
        a torch.load per draw here: hz1x1 keeps one archive per scene with
        ~36k single-intersection records, so materializing once is trivial
        in memory and removes per-step disk I/O from the update path.
        """
        from pathlib import Path as _Path
        lazy = HistoricalPool.load_many(history_paths)
        records = []
        for directory, manifest in zip(history_paths, lazy.manifests):
            directory = _Path(directory)
            for name in manifest.get('transition_files', ['transitions.pt']):
                records.extend(
                    torch.load(directory / name, weights_only=False))
        pool = HistoricalPool(records)
        pool.manifests = lazy.manifests
        return pool

    def _validate_history_manifests(self):
        # Cross-scene archives describe different physical junctions, so the
        # junction identity itself is intentionally not compared.  Structural
        # equivalence (one controlled intersection, same observation/action/
        # reward schema) is what makes offline transitions consumable here.
        expected_state = {
            'variant': 'base', 'dimension': self.ob_length,
            'phase': self.phase, 'phase_one_hot': self.one_hot,
            'position_encoding': False, 'normalized_lane_counts': False,
        }
        for manifest in self.offline_pool.manifests:
            if len(manifest['intersection_ids']) != self.sub_agents:
                raise ValueError('History intersection count does not match agent')
            if manifest['state_schema'] != expected_state:
                raise ValueError('History state schema does not match DQN agent')
            action = manifest['action_schema']
            if (tuple(action['dimensions']) != (self.action_space.n,) or
                    int(action['shared_network_output_dimension']) != self.action_space.n):
                raise ValueError('History action schema does not match DQN agent')
            if manifest['reward_schema']['variant'] != 'original':
                raise ValueError('History reward schema does not match DQN agent')

    def __repr__(self):
        return self.model.__repr__()

    def reset(self):
        '''
        reset
        Reset information, including ob_generator, phase_generator, queue, delay, etc.

        :param: None
        :return: None
        '''
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.ob_generator = LaneVehicleGenerator(self.world, inter_obj, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, inter_obj, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)
        self.queue = LaneVehicleGenerator(self.world, inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)
        self.delay = LaneVehicleGenerator(self.world, inter_obj,
                                                     ["lane_delay"], in_only=True, average="all",
                                                     negative=False)

    def get_ob(self):
        '''
        get_ob
        Get observation from environment.

        :param: None
        :return x_obs: observation generated by ob_generator
        '''
        x_obs = []
        x_obs.append(self.ob_generator.generate())
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def get_reward(self):
        '''
        get_reward
        Get reward from environment.

        :param: None
        :return rewards: rewards generated by reward_generator
        '''
        rewards = []
        rewards.append(self.reward_generator.generate())
        rewards = np.squeeze(np.array(rewards)) * 12
        return rewards

    def get_phase(self):
        '''
        get_phase
        Get current phase of intersection(s) from environment.

        :param: None
        :return phase: current phase generated by phase_generator
        '''
        phase = []
        phase.append(self.phase_generator.generate())
        # phase = np.concatenate(phase, dtype=np.int8)
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase

    def get_action(self, ob, phase, test=False):
        '''
        get_action
        Generate action.

        :param ob: observation
        :param phase: current phase
        :param test: boolean, decide whether is test process
        :return action: action that has the highest score
        '''
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        if self.phase:
            if self.one_hot:
                feature = np.concatenate([ob, utils.idx2onehot(phase, self.action_space.n)], axis=1)
            else:
                feature = np.concatenate([ob, phase], axis=1)
        else:
            feature = ob
        observation = torch.tensor(feature, dtype=torch.float32)
        # TODO: no need to calculate gradient when interacting with environment
        actions = self.model(observation, train=False)
        actions = actions.clone().detach().numpy()
        return np.argmax(actions, axis=1)

    def sample(self):
        '''
        sample
        Sample action randomly.

        :param: None
        :return: action generated randomly.
        '''
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def _build_model(self):
        '''
        _build_model
        Build a DQN model.

        :param: None
        :return model: DQN model
        '''
        if getattr(self, 'dueling', False):
            model = DuelingDQNNet(self.ob_length, self.action_space.n)
        else:
            model = DQNNet(self.ob_length, self.action_space.n)
        return model

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs,
                 cur_phase, done, key, *, scene_t=None, scene_t1=None,
                 truncated=False):
        '''
        remember
        Put current step information into replay buffer for training agent later.

        :param last_obs: last step observation
        :param last_phase: last step phase
        :param actions: actions executed by intersections
        :param actions_prob: the probability that the intersections execute the actions
        :param rewards: current step rewards
        :param obs: current step observation
        :param cur_phase: current step phase
        :param done: boolean, decide whether the process is done
        :param key: key to store this record, e.g., episode_step_agentid
        :param scene_t/scene_t1: optional replay SceneReplayState snapshots;
            when supplied, the ten-field scene contract is stored.
        :param truncated: optional horizon truncation flag stored with scene replay.
        :return: None
        '''
        self._ensure_replay_utilization_counters()
        if self.replay_buffer.maxlen and len(self.replay_buffer) == self.replay_buffer.maxlen:
            evicted_key = self.replay_buffer[0][0]
            self.replay_insert_step.pop(evicted_key, None)
        self.replay_total_collected += 1
        self.replay_insert_step[key] = self.replay_total_collected
        if scene_t is None and scene_t1 is None and not truncated:
            # Preserve the legacy payload for ordinary agents/checkpoints.
            payload = (last_obs, last_phase, actions, rewards, obs, cur_phase)
        else:
            payload = pack_scene_transition(
                last_obs, last_phase, scene_t, actions, rewards, obs,
                cur_phase, scene_t1, done, truncated)
        self.replay_buffer.append((key, payload))
        if self.collect_history:
            parts = str(key).split('_')
            record = split_local_transitions(
                [self.inter_id], last_obs, last_phase,
                np.atleast_1d(actions), np.atleast_1d(rewards), obs, cur_phase,
                bool(done), bool(truncated), self.scene_id,
                int(parts[0]), int(parts[1]), self.training_stage,
                self.policy_version,
                collector_training_seed=self.training_seed)[0]
            self.archive_pending.append(dataclasses.replace(
                record, metadata=dataclasses.replace(
                    record.metadata, source='offline_history')))

    def drain_archive_records(self):
        records, self.archive_pending = self.archive_pending, []
        return records

    def history_manifest(self, roadnet_id, training_seed, num_episodes,
                         num_decision_steps, source_policy='dqn'):
        return {
            'roadnet_id': str(roadnet_id), 'scene_id': self.scene_id,
            'intersection_ids': [self.inter_id],
            'state_schema': {
                'variant': 'base', 'dimension': self.ob_length,
                'phase': self.phase, 'phase_one_hot': self.one_hot,
                'position_encoding': False, 'normalized_lane_counts': False,
            },
            'action_schema': {
                'kind': 'local_discrete', 'dimensions': [self.action_space.n],
                'shared_network_output_dimension': self.action_space.n,
            },
            'reward_schema': {
                'variant': 'original',
                'original': 'negative mean incoming lane waiting count times 12',
            },
            'num_episodes': int(num_episodes),
            'num_decision_steps': int(num_decision_steps),
            'training_seed': int(training_seed),
            'source_policy': str(source_policy),
            'collection_stage': int(self.training_stage),
            'created_at': datetime.now(timezone.utc).isoformat(),
        }

    def _ensure_replay_utilization_counters(self):
        defaults = {
            'replay_total_collected': 0,
            'replay_sample_total': 0,
            'replay_sample_counts': {},
            'replay_sample_count_histogram': {},
            'replay_insert_step': {},
            'replay_sample_age_sum': 0.0,
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def _record_replay_samples(self, samples):
        self._ensure_replay_utilization_counters()
        for key, _ in samples:
            previous = self.replay_sample_counts.get(key, 0)
            if previous:
                remaining = self.replay_sample_count_histogram[previous] - 1
                if remaining:
                    self.replay_sample_count_histogram[previous] = remaining
                else:
                    del self.replay_sample_count_histogram[previous]
            current = previous + 1
            self.replay_sample_counts[key] = current
            self.replay_sample_count_histogram[current] = (
                self.replay_sample_count_histogram.get(current, 0) + 1
            )
            inserted_at = self.replay_insert_step.get(key)
            if inserted_at is not None:
                self.replay_sample_age_sum += self.replay_total_collected - inserted_at
        self.replay_sample_total += len(samples)

    def _sample_count_value_at(self, index):
        zero_count = self.replay_total_collected - len(self.replay_sample_counts)
        if index < zero_count:
            return 0.0
        offset = index - zero_count
        cumulative = 0
        for sample_count in sorted(self.replay_sample_count_histogram):
            cumulative += self.replay_sample_count_histogram[sample_count]
            if offset < cumulative:
                return float(sample_count)
        return 0.0

    def _sample_count_percentile(self, quantile):
        if self.replay_total_collected <= 0:
            return 0.0
        position = (self.replay_total_collected - 1) * quantile
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        lower_value = self._sample_count_value_at(lower)
        upper_value = self._sample_count_value_at(upper)
        return lower_value + (upper_value - lower_value) * (position - lower)

    def replay_utilization(self):
        collected = self.replay_total_collected
        sampled_unique = len(self.replay_sample_counts)
        return {
            'collected_transitions': collected,
            'sampled_transitions': self.replay_sample_total,
            'unique_sampled_transitions': sampled_unique,
            'unique_coverage': 0.0 if not collected else sampled_unique / collected,
            'sample_count_mean': (
                0.0 if not collected else self.replay_sample_total / collected
            ),
            'sample_count_median': self._sample_count_percentile(0.5),
            'sample_count_p95': self._sample_count_percentile(0.95),
            'sample_count_max': float(max(self.replay_sample_count_histogram, default=0)),
            'sampled_transition_mean_age': (
                0.0 if not self.replay_sample_total
                else self.replay_sample_age_sum / self.replay_sample_total
            ),
        }

    def replay_utilization_state(self):
        return {
            'total_collected': self.replay_total_collected,
            'sample_total': self.replay_sample_total,
            'sample_counts': dict(self.replay_sample_counts),
            'sample_count_histogram': dict(self.replay_sample_count_histogram),
            'insert_step': dict(self.replay_insert_step),
            'sample_age_sum': self.replay_sample_age_sum,
        }

    def load_replay_utilization_state(self, state):
        if not state:
            return
        self.replay_total_collected = int(state['total_collected'])
        self.replay_sample_total = int(state['sample_total'])
        self.replay_sample_counts = dict(state['sample_counts'])
        self.replay_sample_count_histogram = {
            int(key): int(value)
            for key, value in state['sample_count_histogram'].items()
        }
        self.replay_insert_step = dict(state['insert_step'])
        self.replay_sample_age_sum = float(state['sample_age_sum'])

    def initialize_replay_utilization_from_buffer(self, total_collected):
        """Compatibility fallback for checkpoints created before replay counters."""
        self.replay_total_collected = max(int(total_collected), len(self.replay_buffer))
        first_step = self.replay_total_collected - len(self.replay_buffer) + 1
        self.replay_insert_step = {
            item[0]: first_step + index
            for index, item in enumerate(self.replay_buffer)
        }

    def _batchwise(self, samples):
        '''
        _batchwise
        Reconstruct the samples into batch form(last state, current state, reward, action).

        :param samples: original samples record in replay buffer
        :return state_t, state_tp, rewards, actions: information with batch form
        '''
        payloads = [unpack_scene_transition(item[1]) for item in samples]
        obs_t = np.concatenate([item.obs_t for item in payloads])
        obs_tp = np.concatenate([item.obs_t1 for item in payloads])
        if self.phase:
            if self.one_hot:
                phase_t = np.concatenate([utils.idx2onehot(item.phase_t, self.action_space.n) for item in payloads])
                phase_tp = np.concatenate([utils.idx2onehot(item.phase_t1, self.action_space.n) for item in payloads])
            else:
                phase_t = np.concatenate([item.phase_t for item in payloads])
                phase_tp = np.concatenate([item.phase_t1 for item in payloads])
            feature_t = np.concatenate([obs_t, phase_t], axis=1)
            feature_tp = np.concatenate([obs_tp, phase_tp], axis=1)
        else:
            feature_t = obs_t
            feature_tp = obs_tp
        state_t = torch.tensor(feature_t, dtype=torch.float32)
        state_tp = torch.tensor(feature_tp, dtype=torch.float32)
        rewards = torch.tensor(np.array([item.reward for item in payloads]), dtype=torch.float32)  # TODO: BETTER WA
        actions = torch.tensor(np.array([item.action for item in payloads]), dtype=torch.long)
        return state_t, state_tp, rewards, actions

    def train(self):
        '''
        train
        Train the agent, optimize the action generated by agent.

        :param: None
        :return: value of loss
        '''
        # Early episodes can contain fewer transitions than one minibatch.
        # Return a zero loss until enough samples exist; this preserves the
        # normal update path once the replay buffer reaches batch_size.
        offline_count = int(round(self.batch_size * self.offline_ratio))
        online_needed = self.batch_size - offline_count
        if len(self.replay_buffer) < online_needed:
            return np.array(0.0, dtype=np.float32)
        anchor_groups = {}
        if offline_count:
            if not (self.visible_history_scenes and self.offline_pool.records):
                raise RuntimeError(
                    'Semi-offline training requires a non-empty visible '
                    'offline pool')
            tagged = mixed_batch(
                _OnlineReplayView(self.replay_buffer), self.offline_pool,
                self.batch_size, self.offline_ratio,
                self.visible_history_scenes,
                strategy=self.offline_sampling_strategy, rng=random)
            samples = []
            for index, (item, source) in enumerate(tagged):
                if source == 'offline_history':
                    samples.append(self._offline_to_sample(item))
                    if self.anchor_teachers:
                        anchor_groups.setdefault(
                            item.metadata.scene_id, []).append(index)
                else:
                    samples.append(item)
            self._record_replay_samples(
                [item for item, source in tagged if source == 'online_current'])
        else:
            samples = random.sample(self.replay_buffer, self.batch_size)
            self._record_replay_samples(samples)
        b_t, b_tp, rewards, actions = self._batchwise(samples)
        if self.double_dqn:
            next_actions = self.model(b_tp, train=False).argmax(dim=1, keepdim=True)
            out = self.target_model(b_tp, train=False).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * out
        else:
            out = self.target_model(b_tp, train=False)
            target = rewards + self.gamma * torch.max(out, dim=1)[0]
        target_f = self.model(b_t, train=False)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        pred = self.model(b_t, train=True)
        td_loss = self.criterion(pred, target_f)
        loss = td_loss
        policy_anchor = None
        value_anchor = None
        if anchor_groups:
            policy_anchor, value_anchor = self._anchor_loss(
                b_t, pred, actions, anchor_groups)
            if self.anchor_lambda_policy:
                loss = loss + self.anchor_lambda_policy * policy_anchor
            if self.anchor_lambda_value:
                loss = loss + self.anchor_lambda_value * value_anchor
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.policy_version += 1
        if self.anchor_teachers:
            self.last_train_diagnostics = {
                'td_loss': float(td_loss.detach()),
                'anchor_policy_loss': (
                    float(policy_anchor.detach())
                    if policy_anchor is not None else 0.0),
                'anchor_value_loss': (
                    float(value_anchor.detach())
                    if value_anchor is not None else 0.0),
                'offline_count': offline_count,
            }
        return loss.clone().detach().numpy()

    def is_training_ready(self):
        offline_count = int(round(self.batch_size * self.offline_ratio))
        online_needed = self.batch_size - offline_count
        if len(self.replay_buffer) < online_needed:
            return False
        if offline_count:
            return bool(self.visible_history_scenes and self.offline_pool.records)
        return True

    @staticmethod
    def _offline_to_sample(record):
        key = record.metadata.transition_id
        payload = (record.state.reshape(1, -1), record.phase,
                   np.asarray([record.action]), record.reward,
                   record.next_state.reshape(1, -1), record.next_phase)
        return (key, payload)

    def update_target_network(self):
        '''
        update_target_network
        Update params of target network.

        :param: None
        :return: None
        '''
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def reset_epsilon(self, mode):
        # Same semantics as shared_dqn.SharedDQNAgent.reset_epsilon so the
        # shared stage-checkpoint loader works for both agent bases.
        if mode == 'reset_schedule':
            self.epsilon = self.initial_epsilon
        elif mode == 'fixed_low':
            self.epsilon = self.epsilon_min
        elif mode != 'continue_schedule':
            raise ValueError(f'Unknown epsilon mode: {mode}')

    def load_model(self, e):
        '''
        load_model
        Load model params of an episode.

        :param e: specified episode
        :return: None
        '''
        model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                  'model', f'{e}_{self.rank}.pt')
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = self._build_model()
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        '''
        save_model
        Save model params of an episode.

        :param e: specified episode, used for file name
        :return: None
        '''
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.target_model.state_dict(), model_name)


class _OnlineReplayView:
    """Expose the (key, payload) deque through the mixed_batch sample API."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def sample(self, count, rng=random):
        return rng.sample(list(self.records), count)


class DQNNet(nn.Module):
    '''
    DQNNet consists of 3 dense layers.
    '''
    def __init__(self, input_dim, output_dim):
        super(DQNNet, self).__init__()
        self.activation_name = 'relu'
        self.dense_1 = nn.Linear(input_dim, 20)
        self.dense_2 = nn.Linear(20, 20)
        self.dense_3 = nn.Linear(20, output_dim)

    def _forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


class DuelingDQNNet(nn.Module):
    """Small dueling head with the same input/action contract as ``DQNNet``.

    Matches sequential/agent.py's DuelingDQNNet layout (feature trunk +
    value/advantage streams) so run.py-line arms can share the architecture.
    """

    def __init__(self, input_dim, output_dim):
        super(DuelingDQNNet, self).__init__()
        self.activation_name = 'relu'
        self.architecture_name = 'dueling_mlp'
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 20), nn.ReLU(),
            nn.Linear(20, 20), nn.ReLU(),
        )
        self.value = nn.Linear(20, 1)
        self.advantage = nn.Linear(20, output_dim)

    def forward(self, x, train=True):
        def compute():
            hidden = self.feature(x)
            value = self.value(hidden)
            advantage = self.advantage(hidden)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        if train:
            return compute()
        with torch.no_grad():
            return compute()
