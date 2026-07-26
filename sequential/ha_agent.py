"""HA-SODQN agent with persistent online FIFO and static Plan 1 archive."""

import copy
import random

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .agent import SequentialCounters, SequentialDQNAgent
from .core import SequentialReplay, TargetUpdateScheduler
from .historical_archive import HistoricalArchive, derive_rng_seed
from .owp import build_historical_sampler


class HASODQNAgent(SequentialDQNAgent):
    def __init__(self, world, rank, model_config, trainer_config,
                 binding_factory=None, archive_manifest=None,
                 ordered_networks=(), training_seed=0, archive_mode='none',
                 method='CONT', offline_ratio=0.0, owp_capacity=5000,
                 alignment_warmup_episodes=10, verify_archive_hashes=True):
        super().__init__(
            world, rank, model_config, trainer_config,
            binding_factory=binding_factory,
        )
        self.training_seed = int(training_seed)
        self.ordered_networks = tuple(ordered_networks)
        self.archive_mode = str(archive_mode).upper()
        self.method = str(method).upper()
        self.offline_ratio = float(offline_ratio)
        self.owp_capacity = int(owp_capacity)
        self.alignment_warmup_episodes = int(alignment_warmup_episodes)
        self.archive_manifest = archive_manifest
        self.verify_archive_hashes = bool(verify_archive_hashes)
        self._validate_ha_config()
        self.online_rng = random.Random(derive_rng_seed(training_seed, 'online_replay_sampler'))
        self.hoa_rng = random.Random(derive_rng_seed(training_seed, 'hoa_owp_sampler'))
        self.archive = None if self.archive_mode == 'NONE' else HistoricalArchive(
            archive_manifest, training_seed, verify_hashes=verify_archive_hashes,
        )
        self.visible_archive = None
        self.historical_sampler = None
        self.alignment_observations = []
        self.last_local_episode = 0
        self.replay = SequentialReplay(int(trainer_config['buffer_size']))

    def _validate_ha_config(self):
        if self.archive_mode not in {'NONE', 'P1C', 'P1F'}:
            raise ValueError('HA archive mode must be none/P1C/P1F')
        if self.archive_mode == 'NONE':
            if self.method != 'CONT' or self.offline_ratio != 0.0:
                raise ValueError('CONT requires archive_mode=none and ratio=0')
        else:
            if self.method not in {'DHOA', 'RAND', 'COV', 'CQ', 'CQA'}:
                raise ValueError('Unknown HA-SODQN historical method')
            if self.offline_ratio not in {0.25, 0.5, 0.75}:
                raise ValueError('HA-SODQN offline ratio must be 0.25/0.50/0.75')
            if not self.archive_manifest:
                raise ValueError('HA-SODQN requires an archive root manifest')
        if self.owp_capacity <= 0 or self.alignment_warmup_episodes < 0:
            raise ValueError('Invalid HA-SODQN working-pool configuration')
        if len(set(self.ordered_networks)) != len(self.ordered_networks):
            raise ValueError('HA-SODQN ordered networks contain duplicates')

    @classmethod
    def from_initial_state(cls, world, rank, model_config, trainer_config,
                           initial_state, **kwargs):
        agent = cls(world, rank, model_config, trainer_config, **kwargs)
        checkpoint = torch.load(initial_state['checkpoint_path'], map_location='cpu')
        if checkpoint.get('checkpoint_type') != 'resumable' or checkpoint.get('episode') != 0:
            raise ValueError('HA-SODQN must start from a Plan 1 episode-0 checkpoint')
        payload = checkpoint['agents'][rank]
        agent.model.load_state_dict(payload['online_model_state_dict'])
        agent.target_model.load_state_dict(payload['target_model_state_dict'])
        agent.optimizer.load_state_dict(payload['optimizer_state_dict'])
        agent.epsilon = float(payload['epsilon'])
        counters = checkpoint['training_counters']
        agent.counters = SequentialCounters(
            int(counters['global_decision_step']),
            int(counters['gradient_updates']), int(counters['target_updates']),
        )
        if any((agent.counters.global_decision_step,
                agent.counters.gradient_updates, agent.counters.target_updates)):
            raise ValueError('HA-SODQN initial-state counters must be zero')
        agent.target_scheduler = TargetUpdateScheduler.from_plan1_parent(
            0, 0, int(trainer_config['target_update_interval']),
        )
        random.setstate(checkpoint['python_random_state'])
        np.random.set_state(checkpoint['numpy_random_state'])
        torch.set_rng_state(checkpoint['torch_cpu_rng_state'])
        if torch.cuda.is_available() and checkpoint['torch_cuda_rng_states']:
            torch.cuda.set_rng_state_all(checkpoint['torch_cuda_rng_states'])
        current = agent.training_state_digests()
        for key in ('online_parameter_digest', 'target_parameter_digest',
                    'optimizer_state_digest', 'rng_state_digest'):
            expected = initial_state.get('digests', {}).get(key)
            if expected and current[key] != expected:
                raise ValueError(f'Initial-state digest mismatch: {key}')
        agent.replay = SequentialReplay(int(trainer_config['buffer_size']))
        return agent

    def _build_sampler(self):
        if self.visible_archive is None or not len(self.visible_archive):
            self.historical_sampler = None
            return None
        if self.method == 'CQA' and not self.alignment_observations:
            return None
        self.historical_sampler = build_historical_sampler(
            self.visible_archive, self.method, self.owp_capacity, self.hoa_rng,
            alignment_observations=(
                self.alignment_observations if self.method == 'CQA' else None
            ),
        )
        return self.historical_sampler

    def begin_stage(self, stage_index, network, policy='ha_sodqn'):
        if policy != 'ha_sodqn':
            raise ValueError('HASODQNAgent requires policy=ha_sodqn')
        stage_index = int(stage_index)
        if network != self.ordered_networks[stage_index - 1]:
            raise ValueError('HA-SODQN stage network differs from ordered manifest')
        self.policy = policy
        self.current_stage_index = stage_index
        self.current_network = network
        self.replay.stage_insertions = 0
        self.alignment_observations = []
        self.last_local_episode = 0
        if self.archive_mode == 'NONE':
            self.visible_archive = None
            self.historical_sampler = None
            visibility = {
                'archive_mode': 'NONE', 'stage_index': stage_index,
                'visible_networks': [], 'transition_count': 0,
            }
        else:
            self.visible_archive = self.archive.visibility(
                self.archive_mode, self.ordered_networks, stage_index,
            )
            visibility = dict(self.visible_archive.visibility)
            self.historical_sampler = None
            if self.method != 'CQA':
                self._build_sampler()
        return {
            'policy': policy,
            'orb_size': len(self.replay.records),
            'visibility': visibility,
            'owp_manifest': (
                None if self.historical_sampler is None
                else self.historical_sampler.manifest
            ),
        }

    def remember(self, state, phase, action, reward, next_state, next_phase,
                 local_episode, decision_index, terminated=False, truncated=False):
        metadata = super().remember(
            state, phase, action, reward, next_state, next_phase,
            local_episode, decision_index, terminated, truncated,
        )
        self.last_local_episode = int(local_episode)
        if self.last_local_episode <= self.alignment_warmup_episodes:
            phase_index = int(np.asarray(phase).reshape(-1)[0])
            phase_one_hot = np.zeros(self.action_space.n, dtype=np.float32)
            phase_one_hot[phase_index] = 1.0
            observation = np.concatenate((
                np.asarray(state, dtype=np.float32).reshape(-1), phase_one_hot,
            ))
            self.alignment_observations.append(observation)
        return metadata

    def is_update_ready(self):
        return (
            self.policy == 'ha_sodqn'
            and len(self.replay.records) >= self.batch_size
            and len(self.replay.records) > self.learning_start
        )

    def _branch_loss(self, state, next_state, rewards, actions):
        rewards = rewards.reshape(-1)
        actions = actions.reshape(-1)
        with torch.no_grad():
            target = rewards + self.gamma * torch.max(
                self.target_model(next_state), dim=1,
            )[0]
            target_full = self.model(state).detach().clone()
            for index, action in enumerate(actions):
                target_full[index][action] = target[index]
        prediction = self.model(state)
        return self.criterion(prediction, target_full), {
            'q_abs_max': float(torch.max(torch.abs(prediction)).detach().cpu()),
            'target_abs_max': float(torch.max(torch.abs(target_full)).detach().cpu()),
        }

    def _offline_tensors(self, batch):
        return (
            torch.as_tensor(batch.observations, dtype=torch.float32),
            torch.as_tensor(batch.next_observations, dtype=torch.float32),
            torch.as_tensor(batch.rewards, dtype=torch.float32),
            torch.as_tensor(batch.actions, dtype=torch.long),
        )

    def successful_gradient_update(self):
        if not self.is_update_ready():
            return None
        offline_enabled = (
            self.archive_mode != 'NONE'
            and self.last_local_episode > self.alignment_warmup_episodes
            and self.visible_archive is not None and len(self.visible_archive)
        )
        if offline_enabled and self.historical_sampler is None:
            self._build_sampler()
        offline_count = (
            int(round(self.batch_size * self.offline_ratio))
            if offline_enabled and self.historical_sampler is not None else 0
        )
        online_count = self.batch_size - offline_count
        online_records = self.online_rng.sample(list(self.replay.records), online_count)
        online_tensors = self._batchwise(online_records)
        loss_online, online_stability = self._branch_loss(*online_tensors)
        offline_batch = None
        loss_offline = None
        offline_stability = None
        if offline_count:
            offline_batch = self.historical_sampler.sample(offline_count)
            loss_offline, offline_stability = self._branch_loss(
                *self._offline_tensors(offline_batch)
            )
            loss = ((1.0 - self.offline_ratio) * loss_online
                    + self.offline_ratio * loss_offline)
        else:
            loss = loss_online
        if not torch.isfinite(loss):
            raise FloatingPointError('HA-SODQN loss is NaN/Inf')
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.counters.gradient_updates += 1
        target_synced = self.target_scheduler.after_successful_gradient(
            self.counters.gradient_updates
        )
        if target_synced:
            self.target_model.load_state_dict(self.model.state_dict())
            self.counters.target_updates += 1
        online_ids = [record.metadata.transition_id for record in online_records]
        offline_ids = [] if offline_batch is None else [
            str(value) for value in offline_batch.transition_ids
        ]
        offline_sources = [] if offline_batch is None else [
            str(value) for value in offline_batch.source_networks
        ]
        return {
            'loss': float(loss.detach().cpu()),
            'loss_online': float(loss_online.detach().cpu()),
            'loss_offline': (
                None if loss_offline is None else float(loss_offline.detach().cpu())
            ),
            'requested_offline_ratio': self.offline_ratio,
            'actual_offline_ratio': offline_count / self.batch_size,
            'online_count': online_count, 'offline_count': offline_count,
            'sample_transition_ids': online_ids + offline_ids,
            'online_transition_ids': online_ids,
            'offline_transition_ids': offline_ids,
            'sample_sources': [
                record.metadata.source_network for record in online_records
            ] + offline_sources,
            'sample_source_kinds': ['online'] * online_count + ['offline'] * offline_count,
            'sample_ages': [
                self.counters.global_decision_step - record.metadata.written_global_step
                for record in online_records
            ] + [0] * offline_count,
            'offline_behavior_training_seeds': (
                [] if offline_batch is None else
                [int(value) for value in offline_batch.behavior_training_seeds]
            ),
            'offline_episode_ids': (
                [] if offline_batch is None else
                [int(value) for value in offline_batch.episode_ids]
            ),
            'online_stability': online_stability,
            'offline_stability': offline_stability,
            'target_synced': target_synced,
        }

    def full_state_dict(self):
        state = super().full_state_dict()
        state['schema_version'] = 3
        state['ha_sodqn'] = {
            'training_seed': self.training_seed,
            'ordered_networks': self.ordered_networks,
            'archive_mode': self.archive_mode,
            'method': self.method,
            'offline_ratio': self.offline_ratio,
            'owp_capacity': self.owp_capacity,
            'alignment_warmup_episodes': self.alignment_warmup_episodes,
            'online_rng_state': self.online_rng.getstate(),
            'hoa_rng_state': self.hoa_rng.getstate(),
            'visible_archive_digest': (
                None if self.visible_archive is None else self.visible_archive.digest
            ),
            'historical_sampler_state': (
                None if self.historical_sampler is None
                else self.historical_sampler.state_dict()
            ),
            'alignment_observations': list(self.alignment_observations),
            'last_local_episode': self.last_local_episode,
        }
        return state

    def load_full_state_dict(self, state):
        if state.get('schema_version') != 3:
            raise ValueError('Unsupported HA-SODQN agent state schema')
        ha = state['ha_sodqn']
        expected = {
            'training_seed': self.training_seed,
            'ordered_networks': self.ordered_networks,
            'archive_mode': self.archive_mode,
            'method': self.method,
            'offline_ratio': self.offline_ratio,
            'owp_capacity': self.owp_capacity,
            'alignment_warmup_episodes': self.alignment_warmup_episodes,
        }
        actual = {key: ha[key] for key in expected}
        actual['ordered_networks'] = tuple(actual['ordered_networks'])
        if actual != expected:
            raise ValueError('HA-SODQN configuration changed on resume')
        legacy = copy.deepcopy(state)
        legacy['schema_version'] = 1
        legacy.pop('ha_sodqn')
        super().load_full_state_dict(legacy)
        self.online_rng.setstate(ha['online_rng_state'])
        self.hoa_rng.setstate(ha['hoa_rng_state'])
        self.alignment_observations = list(ha['alignment_observations'])
        self.last_local_episode = int(ha['last_local_episode'])
        if self.archive_mode != 'NONE':
            self.visible_archive = self.archive.visibility(
                self.archive_mode, self.ordered_networks, self.current_stage_index,
            )
            if self.visible_archive.digest != ha['visible_archive_digest']:
                raise ValueError('Visible archive changed on resume')
            sampler_state = ha.get('historical_sampler_state')
            if sampler_state is not None:
                self._build_sampler()
                self.historical_sampler.load_state_dict(sampler_state)
        return self
