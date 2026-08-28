"""Formal PFRL 0.4.0 PPO adapter for the Plan5 shared SUMO runtime."""
import collections
import copy
import importlib.metadata
import random

import gym
import numpy as np
import torch
from torch import nn

from agent import utils as agent_utils
from generator import IntersectionPhaseGenerator, LaneVehicleGenerator
from sequential.core import capture_rng_state, canonical_digest


def check_pfrl(required='0.4.0'):
    try:
        import pfrl
    except Exception as error:
        return {
            'available': False, 'version': None,
            'error': f'{type(error).__name__}: {error}',
        }
    try:
        version = importlib.metadata.version('pfrl')
    except importlib.metadata.PackageNotFoundError:
        version = getattr(pfrl, '__version__', None)
    return {
        'available': version == required, 'version': version,
        'error': None if version == required else 'version mismatch',
    }


class Plan5PPOAdapter:
    algorithm_id = 'PPO'
    sub_agents = 1

    def __init__(self, world, rank, learning_rate, entropy_coefficient,
                 training_seed=None):
        status = check_pfrl()
        if not status['available']:
            raise RuntimeError(
                'Plan5 formal PPO requires PFRL 0.4.0: ' + str(status)
            )
        if training_seed is not None:
            random.seed(training_seed)
            np.random.seed(training_seed)
            torch.manual_seed(training_seed)
        self.world = world
        self.rank = int(rank)
        self.device = torch.device('cpu')
        self.learning_rate = float(learning_rate)
        self.entropy_coefficient = float(entropy_coefficient)
        self.global_decision_step = 0
        self.rollout_updates = 0
        self.current_stage_index = 1
        self.current_network = None
        self._bind(world)
        self._build_core()

    def _bind(self, world):
        self.world = world
        inter_id = world.intersection_ids[self.rank]
        self.inter = world.id2intersection[inter_id]
        self.ob_generator = LaneVehicleGenerator(
            world, self.inter, ['lane_count'], in_only=True, average=None,
        )
        self.phase_generator = IntersectionPhaseGenerator(
            world, self.inter, ['phase'], targets=['cur_phase'], negative=False,
        )
        self.reward_generator = LaneVehicleGenerator(
            world, self.inter, ['lane_waiting_count'], in_only=True,
            average='all', negative=True,
        )
        self.queue = LaneVehicleGenerator(
            world, self.inter, ['lane_waiting_count'], in_only=True,
            average=None, negative=False,
        )
        self.delay = LaneVehicleGenerator(
            world, self.inter, ['lane_delay'], in_only=True,
            average='all', negative=False,
        )
        self.action_space = gym.spaces.Discrete(len(self.inter.phases))
        self.raw_lane_dim = int(self.ob_generator.ob_length)
        self.ob_length = self.raw_lane_dim + self.action_space.n
        if self.ob_length != 16 or self.action_space.n != 8:
            raise ValueError('Plan5 PPO requires raw16 and 8 discrete actions')

    @staticmethod
    def _lecun(layer):
        import pfrl
        pfrl.initializers.init_lecun_normal(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def _build_core(self):
        import pfrl
        from pfrl.nn import Branched
        from pfrl.policies import SoftmaxCategoricalHead
        trunk = nn.Sequential(
            self._lecun(nn.Linear(16, 64)), nn.ReLU(),
            self._lecun(nn.Linear(64, 64)), nn.ReLU(),
        )
        policy = nn.Sequential(
            self._lecun(nn.Linear(64, 8)), SoftmaxCategoricalHead(),
        )
        value = self._lecun(nn.Linear(64, 1))
        self.model = nn.Sequential(trunk, Branched(policy, value))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, eps=1e-5,
        )
        self.core = pfrl.agents.PPO(
            self.model, self.optimizer, obs_normalizer=None, gpu=None,
            gamma=0.95, lambd=0.95,
            phi=lambda x: np.asarray(x, dtype=np.float32),
            value_func_coef=1.0,
            entropy_coef=self.entropy_coefficient,
            update_interval=360, minibatch_size=90, epochs=4,
            clip_eps=0.2, clip_eps_vf=0.2,
            standardize_advantages=True, recurrent=False,
            act_deterministically=False, max_grad_norm=0.5,
        )

    def _feature(self, observation, phase):
        lane = np.asarray(observation, dtype=np.float32).reshape(
            -1, self.raw_lane_dim
        )
        one_hot = agent_utils.idx2onehot(
            np.asarray(phase).reshape(-1), self.action_space.n
        ).astype(np.float32)
        feature = np.concatenate([lane, one_hot], axis=1)
        if feature.shape[1] != 16:
            raise ValueError('Plan5 PPO feature is not raw16')
        return feature

    def get_ob(self):
        return np.asarray([self.ob_generator.generate()], dtype=np.float32)

    def get_phase(self):
        return np.concatenate([self.phase_generator.generate()]).astype(np.int8)

    def get_reward(self):
        return np.squeeze(np.asarray([self.reward_generator.generate()])) * 12

    def get_queue(self):
        return float(np.sum(self.queue.generate()))

    def get_delay(self):
        return float(np.mean(self.delay.generate()))

    def get_action(self, observation, phase, test=False):
        feature = self._feature(observation, phase)
        if test:
            with torch.no_grad():
                distribution, _ = self.model(torch.as_tensor(feature))
                return torch.argmax(distribution.probs, dim=1).cpu().numpy()
        action = self.core.batch_act([feature[0]])
        return np.asarray(action, dtype=np.int64)

    def observe(self, observation, phase, reward, *, terminated=False,
                truncated=False):
        feature = self._feature(observation, phase)
        before_updates = int(self.core.n_updates)
        self.core.batch_observe(
            [feature[0]], [float(np.asarray(reward).reshape(-1)[0])],
            [bool(terminated)], [bool(truncated)],
        )
        self.global_decision_step += 1
        delta = int(self.core.n_updates) - before_updates
        if truncated or terminated:
            if delta != 16:
                raise RuntimeError(
                    f'PPO episode update expected 16 minibatches, got {delta}'
                )
            self.rollout_updates += 1
            self.assert_rollout_empty()
        elif delta:
            raise RuntimeError('PPO updated before the 360-decision boundary')
        return {'minibatch_updates': delta}

    def assert_rollout_empty(self):
        core = self.core
        pending = (
            sum(len(episode) for episode in core.memory)
            + len(core.last_episode)
            + (0 if core.batch_last_episode is None else sum(
                len(episode) for episode in core.batch_last_episode
            ))
        )
        batch_pending = any(
            value is not None
            for collection in (core.batch_last_state, core.batch_last_action)
            if collection is not None
            for value in collection
        )
        if pending or core.last_state is not None or core.last_action is not None \
                or batch_pending:
            raise RuntimeError('PPO rollout/batch-last state is not empty')
        return True

    def begin_stage(self, stage_index, network):
        self.assert_rollout_empty()
        self.current_stage_index = int(stage_index)
        self.current_network = str(network)
        return {'rollout_empty': True, 'retained_optimizer': True}

    def rebind_environment(self, world):
        before_model = copy.deepcopy(self.model.state_dict())
        before_optimizer = copy.deepcopy(self.optimizer.state_dict())
        before_core = {
            'n_updates': self.core.n_updates,
            'memory': copy.deepcopy(self.core.memory),
            'last_episode': copy.deepcopy(self.core.last_episode),
            'batch_last_episode': copy.deepcopy(self.core.batch_last_episode),
        }
        self._bind(world)
        if any(
            not torch.equal(before_model[key], self.model.state_dict()[key])
            for key in before_model
        ):
            raise RuntimeError('PPO rebind changed model parameters')
        if canonical_digest(before_optimizer) != canonical_digest(
            self.optimizer.state_dict()
        ):
            raise RuntimeError('PPO rebind changed optimizer state')
        if canonical_digest(before_core) != canonical_digest({
            'n_updates': self.core.n_updates,
            'memory': self.core.memory,
            'last_episode': self.core.last_episode,
            'batch_last_episode': self.core.batch_last_episode,
        }):
            raise RuntimeError('PPO rebind changed algorithm state')
        return {'training_state_preserved': True}

    @staticmethod
    def _deque_state(value):
        return {'maxlen': value.maxlen, 'items': list(value)}

    @staticmethod
    def _load_deque(state):
        return collections.deque(state['items'], maxlen=state['maxlen'])

    def full_state_dict(self):
        core = self.core
        return {
            'schema_version': 1, 'algorithm_id': self.algorithm_id,
            'model_state_dict': copy.deepcopy(self.model.state_dict()),
            'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
            'memory': copy.deepcopy(core.memory),
            'last_episode': copy.deepcopy(core.last_episode),
            'last_state': copy.deepcopy(core.last_state),
            'last_action': copy.deepcopy(core.last_action),
            'batch_last_episode': copy.deepcopy(core.batch_last_episode),
            'batch_last_state': copy.deepcopy(core.batch_last_state),
            'batch_last_action': copy.deepcopy(core.batch_last_action),
            'n_updates': int(core.n_updates),
            'explained_variance': float(core.explained_variance),
            'statistics': {
                'value_record': self._deque_state(core.value_record),
                'entropy_record': self._deque_state(core.entropy_record),
                'value_loss_record': self._deque_state(core.value_loss_record),
                'policy_loss_record': self._deque_state(core.policy_loss_record),
            },
            'recurrent_state': {
                'train': copy.deepcopy(core.train_recurrent_states),
                'train_previous': copy.deepcopy(core.train_prev_recurrent_states),
                'test': copy.deepcopy(core.test_recurrent_states),
            },
            'counters': {
                'global_decision_step': int(self.global_decision_step),
                'rollout_updates': int(self.rollout_updates),
            },
            'stage': {
                'index': int(self.current_stage_index),
                'network': self.current_network,
            },
            'rng_state': capture_rng_state(),
        }

    def load_full_state_dict(self, state):
        if state.get('schema_version') != 1 \
                or state.get('algorithm_id') != self.algorithm_id:
            raise ValueError('Unsupported Plan5 PPO state identity')
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        core = self.core
        for name in (
            'memory', 'last_episode', 'last_state', 'last_action',
            'batch_last_episode', 'batch_last_state', 'batch_last_action',
        ):
            setattr(core, name, copy.deepcopy(state[name]))
        core.n_updates = int(state['n_updates'])
        core.explained_variance = float(state['explained_variance'])
        for name, value in state['statistics'].items():
            setattr(core, name, self._load_deque(value))
        core.train_recurrent_states = copy.deepcopy(
            state['recurrent_state']['train']
        )
        core.train_prev_recurrent_states = copy.deepcopy(
            state['recurrent_state']['train_previous']
        )
        core.test_recurrent_states = copy.deepcopy(
            state['recurrent_state']['test']
        )
        self.global_decision_step = int(
            state['counters']['global_decision_step']
        )
        self.rollout_updates = int(state['counters']['rollout_updates'])
        self.current_stage_index = int(state['stage']['index'])
        self.current_network = state['stage']['network']
        rng = state['rng_state']
        random.setstate(rng['python_random_state'])
        np.random.set_state(rng['numpy_random_state'])
        torch.set_rng_state(rng['torch_cpu_rng_state'])
        if torch.cuda.is_available() and rng['torch_cuda_rng_states']:
            torch.cuda.set_rng_state_all(rng['torch_cuda_rng_states'])
        return self

    def parameter_ownership(self):
        trunk = self.model[0]
        branch = self.model[1]
        shared = sum(parameter.numel() for parameter in trunk.parameters())
        actor = sum(
            parameter.numel()
            for parameter in branch.child_modules[0].parameters()
        )
        critic = sum(
            parameter.numel()
            for parameter in branch.child_modules[1].parameters()
        )
        return {
            'shared_parameters': shared, 'actor_only_parameters': actor,
            'critic_only_parameters': critic,
            'total_parameters': shared + actor + critic,
        }
