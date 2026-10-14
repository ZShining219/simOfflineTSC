from . import RLAgent
import dataclasses
import random
import numpy as np
from collections import deque
from datetime import datetime, timezone
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.registry import Registry
import gym
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator
from torch.nn.utils import clip_grad_norm_
from agent import utils
from arterial.replay import (HistoricalPool, LocalTransition, mixed_batch,
                             split_local_transitions)


@Registry.register_model('frap')
class FRAP_DQNAgent(RLAgent):
    '''
    FRAP_DQNAgent consists of FRAP and methods for training agents, communicating with environment, etc.
    '''
    def __init__(self, world, rank):
        super().__init__(world,world.intersection_ids[rank])
        self.dic_agent_conf = Registry.mapping['model_mapping']['setting']
        self.dic_traffic_env_conf = Registry.mapping['world_mapping']['setting']
        
        self.gamma = self.dic_agent_conf.param["gamma"]
        self.grad_clip = self.dic_agent_conf.param["grad_clip"]
        self.epsilon = self.dic_agent_conf.param["epsilon"]
        self.initial_epsilon = self.epsilon
        self.epsilon_min = self.dic_agent_conf.param["epsilon_min"]
        self.epsilon_decay = self.dic_agent_conf.param["epsilon_decay"]
        self.learning_rate = self.dic_agent_conf.param["learning_rate"]
        self.batch_size = self.dic_agent_conf.param["batch_size"]
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.world = world
        self.sub_agents = 1
        self.rank = rank

        self.phase = self.dic_agent_conf.param['phase']
        self.one_hot = self.dic_agent_conf.param['one_hot']

        # get generator for each Agent
        self.inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[self.inter_id]
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                 ["lane_count"], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj,
                                                          ['phase'], targets=['cur_phase'], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True, average="all",
                                                     negative=True)
        
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)
        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_delay"], in_only=True, average="all",
                                                     negative=False)

        map_name = self.dic_traffic_env_conf.param['network']

        # set valid action
        all_valid_acts = self.dic_traffic_env_conf.param['signal_config'][map_name]['valid_acts']
        if all_valid_acts is None:
            self.valid_acts = None
        else:
            if self.inter_id in all_valid_acts.keys():
                self.inter_name = self.inter_id
            else:
                if 'GS_' in self.inter_id:
                    self.inter_name = self.inter_id[3:]
                else:
                    self.inter_name = 'GS_' + self.inter_id
            self.valid_acts = all_valid_acts[self.inter_name]
        
        self.ob_order = None
        if 'lane_order' in self.dic_traffic_env_conf.param['signal_config'][map_name].keys():
            self.ob_order = self.dic_traffic_env_conf.param['signal_config'][map_name]['lane_order'][self.inter_name]
        
        # set phase_pairs
        self.phase_pairs = []
        all_phase_pairs = self.dic_traffic_env_conf.param['signal_config'][map_name]['phase_pairs']
        if self.valid_acts:
            for idx in self.valid_acts:
                self.phase_pairs.append([self.ob_order[x] for x in all_phase_pairs[idx]])
        else:
            self.phase_pairs = all_phase_pairs

        self.comp_mask = self.relation()
        self.num_phases = len(self.phase_pairs)
        self.num_actions = len(self.phase_pairs)
        

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, eps=1e-7)
        self.criterion = nn.MSELoss(reduction='mean')

        model_param = self.dic_agent_conf.param
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
            'variant': 'base', 'dimension': int(self.ob_generator.ob_length),
            'phase': self.phase, 'phase_one_hot': self.one_hot,
            'position_encoding': False, 'normalized_lane_counts': False,
        }
        for manifest in self.offline_pool.manifests:
            if len(manifest['intersection_ids']) != self.sub_agents:
                raise ValueError('History intersection count does not match agent')
            if manifest['state_schema'] != expected_state:
                raise ValueError('History state schema does not match FRAP agent')
            action = manifest['action_schema']
            if (tuple(action['dimensions']) != (self.action_space.n,) or
                    int(action['shared_network_output_dimension']) != self.action_space.n):
                raise ValueError('History action schema does not match FRAP agent')
            if manifest['reward_schema']['variant'] != 'original':
                raise ValueError('History reward schema does not match FRAP agent')

    def __repr__(self):
        return self.model.__repr__()

    def reset(self):
        '''
        reset
        Reset information, including ob_generator, phase_generator, queue, delay, etc.

        :param: None
        :return: None
        '''
        self.inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[self.inter_id]
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                 ["lane_count"], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj,
                                                          ['phase'], targets=['cur_phase'], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True, average="all",
                                                     negative=True)
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)
        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_delay"], in_only=True, average="all",
                                                     negative=False)
    
    def relation(self):
        '''
        relation
        Get the phase competition relation between traffic movements.

        :param: None
        :return comp_mask: matrix of phase competition relation
        '''
        comp_mask = []
        for i in range(len(self.phase_pairs)):
            zeros = np.zeros(len(self.phase_pairs) - 1, dtype=np.int64)
            cnt = 0
            for j in range(len(self.phase_pairs)):
                if i == j: continue
                pair_a = self.phase_pairs[i]
                pair_b = self.phase_pairs[j]
                if len(list(set(pair_a + pair_b))) == 3: zeros[cnt] = 1
                cnt += 1
            comp_mask.append(zeros)
        comp_mask = torch.from_numpy(np.asarray(comp_mask))
        return comp_mask 

    def _build_model(self):
        '''
        _build_model
        Build a FRAP agent.

        :param: None
        :return model: FRAP model
        '''
        model = FRAP(self.dic_agent_conf, self.num_actions, self.phase_pairs, self.comp_mask)
        return model

    def get_ob(self):
        '''
        get_ob
        Get observation from environment.

        :param: None
        :return x_obs: observation generated by ob_generator
        '''
        x_obs = []  # lane_nums
        tmp = self.ob_generator.generate()
        if self.ob_order != None:
            tt = []
            for i in range(12):
                # padding to 12 dims
                if i in self.ob_order.keys():
                    tt.append(tmp[self.ob_order[i]])
                else:
                    tt.append(0.)
            x_obs.append(np.array(tt))     
        else:
            x_obs.append(tmp)
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
        # TODO check whether to multiply 12
        rewards = np.squeeze(np.array(rewards))# * self.num_phases
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
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase

    def get_action(self, ob, phase, test=False):
        '''
        get_action
        Generate action.

        :param ob: observation, the shape is (1,12)
        :param phase: current phase, the shape is (1,)
        :param test: boolean, decide whether is test process
        :return: action that has the highest score
        '''
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        if self.phase:
            if self.one_hot:
                feature_p = utils.idx2onehot(phase, self.action_space.n)
                feature = np.concatenate([feature_p, ob], axis=1)
            else:
                feature = np.concatenate([phase.reshape(1,-1), ob], axis=1)
        else:
            feature = ob
        observation = torch.tensor(feature, dtype=torch.float32)
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
        # return np.random.randint(0, self.num_phases, self.sub_agents)
        return np.random.randint(0, self.action_space.n, self.sub_agents)

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
        # Same semantics as dqn.DQNAgent.reset_epsilon so the shared
        # stage-checkpoint loader works for FRAP arms too.
        if mode == 'reset_schedule':
            self.epsilon = self.initial_epsilon
        elif mode == 'fixed_low':
            self.epsilon = self.epsilon_min
        elif mode != 'continue_schedule':
            raise ValueError(f'Unknown epsilon mode: {mode}')

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
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))
        if self.collect_history:
            parts = str(key).split('_')
            record = split_local_transitions(
                [self.inter_id], last_obs, last_phase,
                np.atleast_1d(actions), np.atleast_1d(rewards), obs, cur_phase,
                bool(done), False, self.scene_id,
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
                         num_decision_steps, source_policy='frap'):
        return {
            'roadnet_id': str(roadnet_id), 'scene_id': self.scene_id,
            'intersection_ids': [self.inter_id],
            'state_schema': {
                'variant': 'base',
                'dimension': int(self.ob_generator.ob_length),
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

    def _batchwise(self, samples):
        '''
        _batchwise
        Reconstruct the samples into batch form(last state, current state, reward, action).

        :param samples: original samples record in replay buffer
        :return state_t, state_tp, rewards, actions: information with batch form
        '''
        # (batch_size,12)
        obs_t_all=[item[1][0] for item in samples] # last_obs(batch, 1, lane_num)
        obs_tp_all=[item[1][4] for item in samples] # cur_obs
        # obs_t = [utils.remove_right_lane(x) for x in obs_t_all]
        # obs_tp = [utils.remove_right_lane(x) for x in obs_tp_all]
        obs_t = obs_t_all
        obs_tp = obs_tp_all
        obs_t = np.concatenate(obs_t) # (batch,lane_num)
        obs_tp = np.concatenate(obs_tp) # (batch,lane_num)
        if self.phase:
            if self.one_hot:
                phase_t = np.concatenate([utils.idx2onehot(item[1][1], self.action_space.n, self.dic_phase_expansion) for item in samples])
                phase_tp = np.concatenate([utils.idx2onehot(item[1][5], self.action_space.n, self.dic_phase_expansion) for item in samples])
            else:
                phase_t = np.concatenate([item[1][1].reshape(1,-1) for item in samples]) # (batch, 1)
                phase_tp = np.concatenate([item[1][5].reshape(1,-1) for item in samples])
            feature_t = np.concatenate([phase_t, obs_t], axis=1) # (batch,ob_length)
            feature_tp = np.concatenate([phase_tp, obs_tp], axis=1)
        else:
            feature_t = obs_t
            feature_tp = obs_tp
        # (batch_size, ob_length)
        state_t = torch.tensor(feature_t, dtype=torch.float32)
        state_tp = torch.tensor(feature_tp, dtype=torch.float32)
        # rewards:(64)
        rewards = torch.tensor(np.array([item[1][3] for item in samples]), dtype=torch.float32)  # TODO: BETTER WA
        # actions:(64,1)
        actions = torch.tensor(np.array([item[1][2] for item in samples]), dtype=torch.long)
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
        else:
            samples = random.sample(self.replay_buffer, self.batch_size)
        b_t, b_tp, rewards, actions = self._batchwise(samples)
        out = self.target_model(b_tp, train=False) # (batch_size,num_actions)
        target = rewards + self.gamma * torch.max(out, dim=1)[0] # (batch_size)
        target_f = self.model(b_t, train=False) # (batch_size,num_actions)
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

    def load_model(self, e):
        '''
        load_model
        Load model params of an episode.

        :param e: specified episode
        :return: None
        '''
        model_name = os.path.join(
            Registry.mapping['logger_mapping']['path'].path, 'model', f'{e}_{self.rank}.pt')
        self.model = FRAP(self.dic_agent_conf, self.dic_phase_expansion, self.num_actions, self.phase_pairs, self.comp_mask)
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = FRAP(self.dic_agent_conf, self.dic_phase_expansion, self.num_actions, self.phase_pairs, self.comp_mask)
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        '''
        save_model
        Save model params of an episode.

        :param e: specified episode, used for file name
        :return: None
        '''
        path = os.path.join(
            Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.model.state_dict(), model_name)


class _OnlineReplayView:
    """Expose the (key, payload) deque through the mixed_batch sample API."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def sample(self, count, rng=random):
        return rng.sample(list(self.records), count)


class FRAP(nn.Module):
    '''
    FRAP captures the phase competition relation between traffic movements through a modified network structure.
    '''
    def __init__(self, dic_agent_conf, output_shape, phase_pairs, competition_mask):
        super(FRAP, self).__init__()
        self.oshape = output_shape
        self.phase_pairs = phase_pairs
        self.comp_mask = competition_mask
        self.demand_shape = dic_agent_conf.param['demand_shape']      # Allows more than just queue to be used
        self.one_hot = dic_agent_conf.param['one_hot']
        self.d_out = 4      # units in demand input layer
        self.p_out = 4      # size of phase embedding
        self.lane_embed_units = 16
        relation_embed_size = 4

        self.p = nn.Embedding(2, self.p_out)
        self.d = nn.Linear(self.demand_shape, self.d_out)

        self.lane_embedding = nn.Linear(self.p_out + self.d_out, self.lane_embed_units)

        self.lane_conv = nn.Conv2d(2*self.lane_embed_units, 20, kernel_size=(1, 1))

        self.relation_embedding = nn.Embedding(2, relation_embed_size)
        self.relation_conv = nn.Conv2d(relation_embed_size, 20, kernel_size=(1, 1))

        self.hidden_layer = nn.Conv2d(20, 20, kernel_size=(1, 1))
        self.before_merge = nn.Conv2d(20, 1, kernel_size=(1, 1))

    def _forward(self, states):
        '''
        states: [agents, ob_length]
        ob_length:concat[len(one_phase),len(intersection_lane)]
        '''
        # if lane_num=12,then num_movements=12, but turning right do not be used
        num_movements = int((states.size()[1]-1)/self.demand_shape) if not self.one_hot else int((states.size()[1]-len(self.phase_pairs))/self.demand_shape)
        batch_size = states.size()[0]
        acts = states[:, :1].to(torch.int64) if not self.one_hot else states[:, :len(self.phase_pairs)].to(torch.int64)
        states = states[:, 1:] if not self.one_hot else states[:, len(self.phase_pairs):]
        states = states.float()
        
        # Expand action index to mark demand input indices
        extended_acts = []
        if not self.one_hot:
            for i in range(batch_size):
                act_idx = acts[i]
                pair = self.phase_pairs[act_idx]
                zeros = torch.zeros(num_movements, dtype=torch.int64)
                zeros[pair[0]] = 1
                zeros[pair[1]] = 1
                extended_acts.append(zeros)
            extended_acts = torch.stack(extended_acts)
        else:
            extended_acts = acts
        phase_embeds = torch.sigmoid(self.p(extended_acts))

        phase_demands = []
        # if num_movements == 12:
        #     order_lane = [0,1,3,4,6,7,9,10] # remove turning_right phase
        # else:
        #     order_lane = [i for i in range(num_movements)]
        # for idx, i in enumerate(order_lane):
        for i in range(num_movements):
            # phase = phase_embeds[:, idx]  # size 4
            phase = phase_embeds[:, i]  # size 4
            demand = states[:, i:i+self.demand_shape]
            demand = torch.sigmoid(self.d(demand))    # size 4
            phase_demand = torch.cat((phase, demand), -1)
            phase_demand_embed = F.relu(self.lane_embedding(phase_demand))
            phase_demands.append(phase_demand_embed)
        phase_demands = torch.stack(phase_demands, 1)

        pairs = []
        for pair in self.phase_pairs:
            pairs.append(phase_demands[:, pair[0]] + phase_demands[:, pair[1]])

        rotated_phases = []
        for i in range(len(pairs)):
            for j in range(len(pairs)):
                if i != j: rotated_phases.append(torch.cat((pairs[i], pairs[j]), -1))
        rotated_phases = torch.stack(rotated_phases, 1)
        rotated_phases = torch.reshape(rotated_phases,
                                       (batch_size, self.oshape, self.oshape - 1, 2 * self.lane_embed_units))
        rotated_phases = rotated_phases.permute(0, 3, 1, 2)  # Move channels up
        rotated_phases = F.relu(self.lane_conv(rotated_phases))  # Conv-20x1x1  pair demand representation

        # Phase competition mask
        competition_mask = self.comp_mask.repeat((batch_size, 1, 1))
        relations = F.relu(self.relation_embedding(competition_mask))
        relations = relations.permute(0, 3, 1, 2)  # Move channels up
        relations = F.relu(self.relation_conv(relations))  # Pair demand representation

        # Phase pair competition
        combine_features = rotated_phases * relations
        combine_features = F.relu(self.hidden_layer(combine_features))  # Phase competition representation
        combine_features = self.before_merge(combine_features)  # Pairwise competition result

        # Phase score
        combine_features = torch.reshape(combine_features, (batch_size, self.oshape, self.oshape - 1))
        q_values = (lambda x: torch.sum(x, dim=2))(combine_features) # (b,8)
        return q_values
        

    def forward(self, states, train=True):
        if train:
            return self._forward(states)
        else:
            with torch.no_grad():
                return self._forward(states)
