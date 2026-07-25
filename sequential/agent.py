import copy
import random
from dataclasses import dataclass

import gym
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from agent import utils as agent_utils
from agent.dqn import DQNNet
from generator import IntersectionPhaseGenerator, LaneVehicleGenerator

from .core import (
    ReplayMetadata, ReplayRecord, SequentialReplay, TargetUpdateScheduler,
    TrainingPayload, capture_rng_state, canonical_digest,
    environment_signature_digest, online_parameter_digest,
    optimizer_state_digest, replay_content_digest, replay_metadata_digest,
    rng_state_digest, target_parameter_digest,
)
from .parents import ParentSource, convert_parent_replay, validate_parent_source


@dataclass
class SequentialCounters:
    global_decision_step: int
    gradient_updates: int
    target_updates: int


def _readonly_array(value):
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


class SequentialDQNAgent:
    """DQN implementation isolated from the repository's default Online path."""

    def __init__(self, world, rank, model_config, trainer_config,
                 binding_factory=None):
        self.rank = int(rank)
        self.sub_agents = 1
        self.model_config = copy.deepcopy(model_config)
        self.trainer_config = copy.deepcopy(trainer_config)
        self.phase = bool(model_config['phase'])
        self.one_hot = bool(model_config['one_hot'])
        self.gamma = float(model_config['gamma'])
        self.grad_clip = float(model_config['grad_clip'])
        self.epsilon_decay = float(model_config['epsilon_decay'])
        self.epsilon_min = float(model_config['epsilon_min'])
        self.batch_size = int(trainer_config['batch_size'])
        self.learning_start = int(trainer_config['learning_start'])
        self.policy = None
        self.current_stage_index = 1
        self.current_network = None
        self.last_observation = None
        self._binding_factory = binding_factory or self._default_binding_factory
        binding = self._make_binding(world)
        self._install_binding(binding)
        self.model = DQNNet(self.ob_length, self.action_space.n)
        self.target_model = DQNNet(self.ob_length, self.action_space.n)
        self.target_model.load_state_dict(self.model.state_dict())
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.RMSprop(
            self.model.parameters(), lr=float(model_config['learning_rate']),
            alpha=0.9, centered=False, eps=1e-7,
        )
        self.epsilon = float(model_config.get('epsilon', 1.0))
        self.replay = SequentialReplay(int(trainer_config['buffer_size']))
        self.counters = SequentialCounters(0, 0, 0)
        self.target_scheduler = TargetUpdateScheduler(
            int(trainer_config['target_update_interval']),
            int(trainer_config['target_update_interval']) - 1,
        )

    @staticmethod
    def _default_binding_factory(world, rank):
        inter_id = world.intersection_ids[rank]
        inter = world.id2intersection[inter_id]
        observation = LaneVehicleGenerator(
            world, inter, ['lane_count'], in_only=True, average=None,
        )
        phase = IntersectionPhaseGenerator(
            world, inter, ['phase'], targets=['cur_phase'], negative=False,
        )
        reward = LaneVehicleGenerator(
            world, inter, ['lane_waiting_count'], in_only=True,
            average='all', negative=True,
        )
        queue = LaneVehicleGenerator(
            world, inter, ['lane_waiting_count'], in_only=True,
            average=None, negative=False,
        )
        delay = LaneVehicleGenerator(
            world, inter, ['lane_delay'], in_only=True,
            average='all', negative=False,
        )
        lane_mapping = tuple(
            lane for lanes in observation.lanes for lane in lanes
        )
        phase_mapping = tuple(
            getattr(item, 'state', str(item)) for item in inter.green_phases
        )
        signature = {
            'intersection_id': inter_id,
            'incoming_lane_mapping': lane_mapping,
            'phase_action_mapping': phase_mapping,
            'state_dim': int(observation.ob_length),
            'action_dim': len(inter.phases),
        }
        return {
            'world': world, 'inter': inter,
            'ob_generator': observation, 'phase_generator': phase,
            'reward_generator': reward, 'queue_generator': queue,
            'delay_generator': delay, 'signature': signature,
        }

    def _make_binding(self, world):
        binding = self._binding_factory(world, self.rank)
        required = {
            'world', 'inter', 'ob_generator', 'phase_generator',
            'reward_generator', 'queue_generator', 'delay_generator', 'signature',
        }
        missing = sorted(required - set(binding))
        if missing:
            raise ValueError(f'Environment binding missing fields: {missing}')
        return binding

    def _install_binding(self, binding):
        self.world = binding['world']
        self.inter = binding['inter']
        self.id = self.inter.id
        self.ob_generator = binding['ob_generator']
        self.phase_generator = binding['phase_generator']
        self.reward_generator = binding['reward_generator']
        self.queue = binding['queue_generator']
        self.delay = binding['delay_generator']
        self.environment_signature = copy.deepcopy(binding['signature'])
        action_dim = int(self.environment_signature['action_dim'])
        state_dim = int(self.environment_signature['state_dim'])
        self.action_space = gym.spaces.Discrete(action_dim)
        self.ob_length = state_dim + action_dim if self.phase and self.one_hot else (
            state_dim + 1 if self.phase else state_dim
        )

    @classmethod
    def from_parent(cls, world, rank, model_config, trainer_config,
                    parent_manifest, binding_factory=None,
                    skip_replay_digest=False):
        agent = cls(
            world, rank, model_config, trainer_config,
            binding_factory=binding_factory,
        )
        checkpoint = torch.load(parent_manifest['checkpoint_path'], map_location='cpu')
        payload = checkpoint['agents'][rank]
        agent.model.load_state_dict(payload['online_model_state_dict'])
        agent.target_model.load_state_dict(payload['target_model_state_dict'])
        agent.optimizer.load_state_dict(payload['optimizer_state_dict'])
        agent.epsilon = float(payload['epsilon'])
        agent.replay = convert_parent_replay(
            payload['replay_state']['items'], parent_manifest['network'],
            checkpoint['training_counters']['global_decision_step'],
            payload['replay_state']['capacity'],
        )
        counters = checkpoint['training_counters']
        agent.counters = SequentialCounters(
            int(counters['global_decision_step']),
            int(counters['gradient_updates']), int(counters['target_updates']),
        )
        agent.target_scheduler = TargetUpdateScheduler.from_plan1_parent(
            agent.counters.gradient_updates, agent.counters.target_updates,
            int(trainer_config['target_update_interval']),
        )
        agent.current_network = parent_manifest['network']
        random.setstate(checkpoint['python_random_state'])
        np.random.set_state(checkpoint['numpy_random_state'])
        torch.set_rng_state(checkpoint['torch_cpu_rng_state'])
        if torch.cuda.is_available() and checkpoint['torch_cuda_rng_states']:
            torch.cuda.set_rng_state_all(checkpoint['torch_cuda_rng_states'])
        if skip_replay_digest:
            # Legacy Plan 1 replay metadata predates CS-HR provenance fields;
            # model/optimizer/RNG state is still checked by the caller.
            expected = dict(parent_manifest)
            expected['digests'] = dict(parent_manifest['digests'])
            expected['digests'].pop('replay_content_digest', None)
            expected['digests'].pop('replay_metadata_digest', None)
            agent.assert_matches_parent_manifest(expected)
        else:
            agent.assert_matches_parent_manifest(parent_manifest)
        return agent

    def assert_matches_parent_manifest(self, parent_manifest):
        current = self.training_state_digests()
        expected = parent_manifest['digests']
        for key in (
            'online_parameter_digest', 'target_parameter_digest',
            'optimizer_state_digest', 'rng_state_digest',
            'replay_content_digest', 'replay_metadata_digest',
        ):
            if key not in expected:
                continue
            if current[key] != expected[key]:
                raise ValueError(f'Imported parent digest mismatch: {key}')
        if self.epsilon != parent_manifest['epsilon']:
            raise ValueError('Imported parent epsilon mismatch')
        if self.target_scheduler.next_update != parent_manifest['next_target_sync_update']:
            raise ValueError('Imported target scheduler mismatch')

    def training_state_digests(self):
        rng = capture_rng_state()
        return {
            'online_parameter_digest': online_parameter_digest(self.model.state_dict()),
            'target_parameter_digest': target_parameter_digest(
                self.target_model.state_dict()
            ),
            'optimizer_state_digest': optimizer_state_digest(
                self.optimizer.state_dict()
            ),
            'rng_state_digest': rng_state_digest(
                rng['python_random_state'], rng['numpy_random_state'],
                rng['torch_cpu_rng_state'], rng['torch_cuda_rng_states'],
            ),
            'replay_content_digest': replay_content_digest(self.replay.records),
            'replay_metadata_digest': replay_metadata_digest(self.replay.records),
            'environment_signature_digest': environment_signature_digest(
                self.environment_signature
            ),
            'epsilon': float(self.epsilon),
            'global_decision_step': self.counters.global_decision_step,
            'gradient_updates': self.counters.gradient_updates,
            'target_updates': self.counters.target_updates,
            'next_target_sync_update': self.target_scheduler.next_update,
        }

    def rebind_environment(self, world, expected_signature=None):
        before = self.training_state_digests()
        old_identity = {
            'world_id': id(self.world), 'intersection_id': id(self.inter),
            'generator_ids': [
                id(self.ob_generator), id(self.phase_generator),
                id(self.reward_generator), id(self.queue), id(self.delay),
            ],
        }
        binding = self._make_binding(world)
        new_signature = binding['signature']
        if expected_signature is not None and new_signature != expected_signature:
            raise ValueError('New environment does not match frozen signature')
        if int(new_signature['state_dim']) + self.action_space.n != self.ob_length:
            raise ValueError('New environment state dimension is incompatible')
        if int(new_signature['action_dim']) != self.action_space.n:
            raise ValueError('New environment action dimension is incompatible')
        if (
            tuple(new_signature['phase_action_mapping'])
            != tuple(self.environment_signature['phase_action_mapping'])
        ):
            raise ValueError('New environment phase/action mapping is incompatible')
        if (
            tuple(new_signature['incoming_lane_mapping'])
            != tuple(self.environment_signature['incoming_lane_mapping'])
        ):
            raise ValueError('New environment lane mapping is incompatible')
        self._install_binding(binding)
        self.last_observation = None
        after = self.training_state_digests()
        preserved_keys = set(before) - {'environment_signature_digest'}
        changed = sorted(key for key in preserved_keys if before[key] != after[key])
        if changed:
            raise RuntimeError(f'Environment rebind changed training state: {changed}')
        return {
            'old': old_identity,
            'new': {
                'world_id': id(self.world), 'intersection_id': id(self.inter),
                'generator_ids': [
                    id(self.ob_generator), id(self.phase_generator),
                    id(self.reward_generator), id(self.queue), id(self.delay),
                ],
            },
            'state_dim': int(new_signature['state_dim']),
            'action_dim': int(new_signature['action_dim']),
            'phase_action_mapping_valid': True,
            'lane_mapping_valid': True,
            'training_state_preserved': True,
        }

    def begin_stage(self, stage_index, network, policy):
        before = self.training_state_digests()
        self.policy = policy
        self.current_stage_index = int(stage_index)
        self.current_network = network
        self.replay.begin_stage(policy)
        after = self.training_state_digests()
        allowed = {'replay_content_digest', 'replay_metadata_digest'}
        changed = {key for key in before if before[key] != after[key]}
        if policy == 'clear':
            if changed - allowed:
                raise RuntimeError(f'Clear policy changed forbidden state: {changed-allowed}')
        elif changed:
            raise RuntimeError(f'{policy} policy changed parent state: {changed}')
        return {'policy': policy, 'replay_size': len(self.replay.records)}

    def should_use_random_warmup_action(self):
        return False

    def is_update_ready(self):
        if self.policy is None:
            raise RuntimeError('Replay policy has not been applied')
        return self.replay.is_update_ready(
            self.policy, self.learning_start, self.batch_size,
        )

    def get_ob(self):
        observation = np.asarray([self.ob_generator.generate()], dtype=np.float32)
        self.last_observation = observation
        return observation

    def get_phase(self):
        return np.concatenate([self.phase_generator.generate()]).astype(np.int8)

    def get_reward(self):
        return np.squeeze(np.asarray([self.reward_generator.generate()])) * 12

    def get_queue(self):
        return float(np.sum(self.queue.generate()))

    def get_delay(self):
        return float(np.mean(self.delay.generate()))

    def sample(self):
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def get_action(self, observation, phase, test=False):
        if not test and np.random.rand() <= self.epsilon:
            return self.sample()
        if self.phase:
            if self.one_hot:
                phase_feature = agent_utils.idx2onehot(phase, self.action_space.n)
            else:
                phase_feature = phase
            feature = np.concatenate([observation, phase_feature], axis=1)
        else:
            feature = observation
        tensor = torch.as_tensor(feature, dtype=torch.float32)
        with torch.no_grad():
            return np.argmax(self.model(tensor).cpu().numpy(), axis=1)

    def remember(self, state, phase, action, reward, next_state, next_phase,
                 local_episode, decision_index, terminated=False, truncated=False):
        next_global = self.counters.global_decision_step + 1
        payload = TrainingPayload(
            _readonly_array(state), _readonly_array(phase),
            _readonly_array(action), _readonly_array(reward),
            _readonly_array(next_state), _readonly_array(next_phase),
            bool(terminated), bool(truncated),
        )
        metadata = ReplayMetadata(
            transition_id=(
                f'{self.current_network}:stage{self.current_stage_index}:'
                f'global{next_global:09d}'
            ),
            source_network=self.current_network,
            stage_index=self.current_stage_index,
            local_episode=int(local_episode),
            decision_index=int(decision_index),
            written_global_step=next_global,
        )
        self.replay.append(ReplayRecord(payload, metadata))
        self.counters.global_decision_step = next_global
        return metadata

    def _batchwise(self, records):
        state = np.concatenate([item.payload.state for item in records])
        next_state = np.concatenate([item.payload.next_state for item in records])
        if self.phase:
            if self.one_hot:
                phase = np.concatenate([
                    agent_utils.idx2onehot(item.payload.phase, self.action_space.n)
                    for item in records
                ])
                next_phase = np.concatenate([
                    agent_utils.idx2onehot(item.payload.next_phase, self.action_space.n)
                    for item in records
                ])
            else:
                phase = np.concatenate([item.payload.phase for item in records])
                next_phase = np.concatenate([
                    item.payload.next_phase for item in records
                ])
            state = np.concatenate([state, phase], axis=1)
            next_state = np.concatenate([next_state, next_phase], axis=1)
        rewards = torch.as_tensor(
            np.asarray([item.payload.reward for item in records]),
            dtype=torch.float32,
        )
        actions = torch.as_tensor(
            np.asarray([item.payload.action for item in records]), dtype=torch.long,
        )
        return (
            torch.as_tensor(state, dtype=torch.float32),
            torch.as_tensor(next_state, dtype=torch.float32), rewards, actions,
        )

    def successful_gradient_update(self):
        if not self.is_update_ready():
            return None
        records = self.replay.sample(self.batch_size, random)
        state, next_state, rewards, actions = self._batchwise(records)
        with torch.no_grad():
            target = rewards + self.gamma * torch.max(
                self.target_model(next_state), dim=1
            )[0]
            target_full = self.model(state).detach().clone()
            for index, action in enumerate(actions):
                target_full[index][action] = target[index]
        loss = self.criterion(self.model(state), target_full)
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
        return {
            'loss': float(loss.detach().cpu().item()),
            'sample_transition_ids': [r.metadata.transition_id for r in records],
            'sample_sources': [r.metadata.source_network for r in records],
            'sample_ages': [
                self.counters.global_decision_step - r.metadata.written_global_step
                for r in records
            ],
            'target_synced': target_synced,
        }

    def full_state_dict(self):
        return {
            'schema_version': 1,
            'online_model_state_dict': copy.deepcopy(self.model.state_dict()),
            'target_model_state_dict': copy.deepcopy(self.target_model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
            'epsilon': float(self.epsilon),
            'replay_state': self.replay.state_dict(),
            'counters': {
                'global_decision_step': self.counters.global_decision_step,
                'gradient_updates': self.counters.gradient_updates,
                'target_updates': self.counters.target_updates,
            },
            'target_scheduler': {
                'interval': self.target_scheduler.interval,
                'next_update': self.target_scheduler.next_update,
            },
            'policy': self.policy,
            'current_stage_index': self.current_stage_index,
            'current_network': self.current_network,
            'rng_state': capture_rng_state(),
        }

    def load_full_state_dict(self, state):
        if state.get('schema_version') != 1:
            raise ValueError('Unsupported Sequential agent state schema')
        self.model.load_state_dict(state['online_model_state_dict'])
        self.target_model.load_state_dict(state['target_model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.epsilon = float(state['epsilon'])
        self.replay = SequentialReplay.from_state_dict(state['replay_state'])
        counters = state['counters']
        self.counters = SequentialCounters(
            int(counters['global_decision_step']),
            int(counters['gradient_updates']), int(counters['target_updates']),
        )
        scheduler = state['target_scheduler']
        self.target_scheduler = TargetUpdateScheduler(
            int(scheduler['interval']), int(scheduler['next_update']),
        )
        self.policy = state['policy']
        self.current_stage_index = int(state['current_stage_index'])
        self.current_network = state['current_network']
        rng = state['rng_state']
        random.setstate(rng['python_random_state'])
        np.random.set_state(rng['numpy_random_state'])
        torch.set_rng_state(rng['torch_cpu_rng_state'])
        if torch.cuda.is_available() and rng['torch_cuda_rng_states']:
            torch.cuda.set_rng_state_all(rng['torch_cuda_rng_states'])
        self.last_observation = None
        return self
