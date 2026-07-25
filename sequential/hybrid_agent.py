"""Sequential DQN agent using the causal hybrid replay protocol."""

import copy
import random

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .agent import SequentialCounters, SequentialDQNAgent
from .core import (
    ReplayMetadata, ReplayRecord, TrainingPayload, SequentialReplay,
    TargetUpdateScheduler, capture_rng_state, online_parameter_digest,
    optimizer_state_digest, rng_state_digest, target_parameter_digest,
)
from .hybrid import HybridReplayPool


class HybridDQNAgent(SequentialDQNAgent):
    """CS-HR variant; parent replay is intentionally discarded at Stage 1."""

    def __init__(self, world, rank, model_config, trainer_config,
                 binding_factory=None, online_ratio=0.5,
                 historical_sampling='stage_balanced_episode_stratified'):
        super().__init__(world, rank, model_config, trainer_config,
                         binding_factory=binding_factory)
        self.hybrid_pool = HybridReplayPool(
            int(trainer_config['buffer_size']), float(online_ratio), random.Random(),
        )
        self.historical_sampling = historical_sampling
        # Keep the public replay attribute pointing at the active pool so the
        # trainer and diagnostics cannot accidentally inspect a stale parent.
        self.replay = self.hybrid_pool

    @classmethod
    def from_parent(cls, world, rank, model_config, trainer_config,
                    parent_manifest, binding_factory=None, online_ratio=0.5,
                    historical_sampling='stage_balanced_episode_stratified'):
        # Load only the frozen training state.  Parent replay is deliberately
        # excluded from CS-HR Stage 1 and therefore is never digest-checked.
        agent = cls(
            world, rank, model_config, trainer_config,
            binding_factory=binding_factory, online_ratio=online_ratio,
            historical_sampling=historical_sampling,
        )
        checkpoint = torch.load(parent_manifest['checkpoint_path'], map_location='cpu')
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
        agent.target_scheduler = TargetUpdateScheduler.from_plan1_parent(
            agent.counters.gradient_updates, agent.counters.target_updates,
            int(trainer_config['target_update_interval']),
        )
        random.setstate(checkpoint['python_random_state'])
        np.random.set_state(checkpoint['numpy_random_state'])
        torch.set_rng_state(checkpoint['torch_cpu_rng_state'])
        if torch.cuda.is_available() and checkpoint['torch_cuda_rng_states']:
            torch.cuda.set_rng_state_all(checkpoint['torch_cuda_rng_states'])
        current = agent.training_state_digests()
        expected = parent_manifest.get('digests', {})
        for key in ('online_parameter_digest', 'target_parameter_digest',
                    'optimizer_state_digest', 'rng_state_digest'):
            if expected.get(key) and current[key] != expected[key]:
                raise ValueError(f'Imported parent digest mismatch: {key}')
        # Replace the constructor's empty pool with a fresh causal pool while
        # preserving every loaded training counter.
        agent.hybrid_pool = HybridReplayPool(
            int(trainer_config['buffer_size']), float(online_ratio), random.Random(),
        )
        agent.hybrid_pool.rng.setstate(random.getstate())
        agent.historical_sampling = historical_sampling
        agent.replay = agent.hybrid_pool
        agent.current_stage_index = 1
        agent.current_network = None
        return agent

    def begin_stage(self, stage_index, network, policy='hybrid'):
        if policy != 'hybrid':
            raise ValueError('HybridDQNAgent requires policy=hybrid')
        stage_index = int(stage_index)
        if stage_index == 1:
            if self.hybrid_pool.stage_index != 1:
                raise ValueError('Stage 1 can only be entered once')
            self.hybrid_pool.begin_stage(1)
        else:
            if stage_index != self.hybrid_pool.stage_index + 1:
                raise ValueError('Hybrid stages must advance sequentially')
            self.hybrid_pool.freeze_stage(stage_index - 1)
            self.hybrid_pool.begin_stage(stage_index)
        self.current_stage_index = stage_index
        self.current_network = network
        self.policy = policy
        return {
            'policy': policy, 'replay_size': self.hybrid_pool.composition()['size'],
            'visible_historical_stages': list(self.hybrid_pool.visible_historical_stages()),
        }

    def is_update_ready(self):
        if self.policy != 'hybrid':
            raise RuntimeError('Hybrid replay policy has not been applied')
        # learning_start is satisfied by current-stage online data only.
        return (len(self.hybrid_pool.online) >= self.batch_size and
                len(self.hybrid_pool.online) > self.learning_start)

    def remember(self, state, phase, action, reward, next_state, next_phase,
                 local_episode, decision_index, terminated=False, truncated=False):
        next_global = self.counters.global_decision_step + 1
        payload = TrainingPayload(
            np.array(state, copy=True), np.array(phase, copy=True),
            np.array(action, copy=True), np.array(reward, copy=True),
            np.array(next_state, copy=True), np.array(next_phase, copy=True),
            bool(terminated), bool(truncated),
        )
        metadata = ReplayMetadata(
            transition_id=(
                f'{self.current_network}:stage{self.current_stage_index}:'
                f'global{next_global:09d}'
            ),
            source_network=self.current_network,
            stage_index=self.current_stage_index,
            local_episode=int(local_episode), decision_index=int(decision_index),
            written_global_step=next_global, source_kind='online',
            source_scene=self.current_network, written_stage=self.current_stage_index,
        )
        self.hybrid_pool.append_online(ReplayRecord(payload, metadata))
        self.counters.global_decision_step = next_global
        return metadata

    def full_state_dict(self):
        state = super().full_state_dict()
        state['schema_version'] = 2
        state['replay_state'] = self.hybrid_pool.state_dict()
        state['historical_sampling'] = self.historical_sampling
        return state

    def load_full_state_dict(self, state):
        if state.get('schema_version') != 2:
            return super().load_full_state_dict(state)
        # Restore model, optimizer and counters without restoring the legacy
        # replay object, then restore the causal pool and its private RNG.
        super().load_full_state_dict({**state, 'schema_version': 1,
                                      'replay_state': {'capacity': self.hybrid_pool.capacity,
                                                       'records': [],
                                                       'stage_insertions': 0}})
        self.hybrid_pool = HybridReplayPool.from_state_dict(state['replay_state'])
        self.replay = self.hybrid_pool
        self.historical_sampling = state.get(
            'historical_sampling', 'stage_balanced_episode_stratified',
        )
        return self

    def successful_gradient_update(self):
        if not self.is_update_ready():
            return None
        records = self.hybrid_pool.sample(
            self.batch_size, strategy=self.historical_sampling,
            current_global_step=self.counters.global_decision_step,
        )
        state, next_state, rewards, actions = self._batchwise(records)
        rewards = rewards.reshape(-1)
        actions = actions.reshape(-1)
        with torch.no_grad():
            target = rewards + self.gamma * torch.max(
                self.target_model(next_state), dim=1
            )[0]
            target_full = self.model(state).detach().clone()
            for index, action in enumerate(actions):
                target_full[index][action] = target[index]
        loss = self.criterion(self.model(state), target_full)
        self.optimizer.zero_grad(); loss.backward()
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
        return {
            'loss': float(loss.detach().cpu().item()),
            'sample_transition_ids': [r.metadata.transition_id for r in records],
            'sample_sources': [r.metadata.source_network for r in records],
            'sample_source_kinds': [r.metadata.source_kind for r in records],
            'sample_ages': [self.counters.global_decision_step - r.metadata.written_global_step for r in records],
            'replay_diagnostics': copy.deepcopy(self.hybrid_pool.last_diagnostics),
            'target_synced': target_synced,
        }
