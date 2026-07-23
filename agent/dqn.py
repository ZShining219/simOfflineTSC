from . import RLAgent
from common.registry import Registry
from agent import utils
import numpy as np
import os
import random
from collections import deque
import gym

from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


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
        self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=self.learning_rate,
                                       alpha=0.9, centered=False, eps=1e-7)

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
        model = DQNNet(self.ob_length, self.action_space.n)
        return model

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
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
        :return: None
        '''
        self._ensure_replay_utilization_counters()
        if self.replay_buffer.maxlen and len(self.replay_buffer) == self.replay_buffer.maxlen:
            evicted_key = self.replay_buffer[0][0]
            self.replay_insert_step.pop(evicted_key, None)
        self.replay_total_collected += 1
        self.replay_insert_step[key] = self.replay_total_collected
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))

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
        obs_t = np.concatenate([item[1][0] for item in samples])
        obs_tp = np.concatenate([item[1][4] for item in samples])
        if self.phase:
            if self.one_hot:
                phase_t = np.concatenate([utils.idx2onehot(item[1][1], self.action_space.n) for item in samples])
                phase_tp = np.concatenate([utils.idx2onehot(item[1][5], self.action_space.n) for item in samples])
            else:
                phase_t = np.concatenate([item[1][1] for item in samples])
                phase_tp = np.concatenate([item[1][5] for item in samples])
            feature_t = np.concatenate([obs_t, phase_t], axis=1)
            feature_tp = np.concatenate([obs_tp, phase_tp], axis=1)
        else:
            feature_t = obs_t
            feature_tp = obs_tp
        state_t = torch.tensor(feature_t, dtype=torch.float32)
        state_tp = torch.tensor(feature_tp, dtype=torch.float32)
        rewards = torch.tensor(np.array([item[1][3] for item in samples]), dtype=torch.float32)  # TODO: BETTER WA
        actions = torch.tensor(np.array([item[1][2] for item in samples]), dtype=torch.long)
        return state_t, state_tp, rewards, actions

    def train(self):
        '''
        train
        Train the agent, optimize the action generated by agent.

        :param: None
        :return: value of loss
        '''
        samples = random.sample(self.replay_buffer, self.batch_size)
        self._record_replay_samples(samples)
        b_t, b_tp, rewards, actions = self._batchwise(samples)
        out = self.target_model(b_tp, train=False)
        target = rewards + self.gamma * torch.max(out, dim=1)[0]
        target_f = self.model(b_t, train=False)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        loss = self.criterion(self.model(b_t, train=True), target_f)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.clone().detach().numpy()

    def update_target_network(self):
        '''
        update_target_network
        Update params of target network.

        :param: None
        :return: None
        '''
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

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
