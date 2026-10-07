from . import RLAgent
from common.registry import Registry
import numpy as np
import os
import json
import random
from collections import OrderedDict, deque
import gym

from generator.lane_vehicle import LaneVehicleGenerator
from generator.intersection_phase import IntersectionPhaseGenerator
import torch
from torch import nn
import torch.nn.functional as F
import torch_scatter
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from utils.scene_context import pack_scene_transition, unpack_scene_transition
from agent.scene_alignment import (AlignmentBuffer, EpisodeAlignAccumulator,
                                   TrafficTransitionEncoder,
                                   TrafficWindowEncoder, symmetric_infonce)
from agent.scene_attention import (EventSceneRepresentation, SceneCondition,
                                    SceneGuidedAttention,
                                    MultiEventSceneGuidedAttention,
                                    SceneCrossAttentionResidual,
                                    SceneContextEncoder, StructuredSceneEncoder,
                                    StructuredSceneMeta,
                                    batch_event_scene_representations)

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops


@Registry.register_model('colight')
class CoLightAgent(RLAgent):
    #  TODO: test multiprocessing effect on agents or need deep copy here
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        """
        multi-agents in one model-> modify self.action_space, self.reward_generator, self.ob_generator here
        """
        #  general setting of world and model structure
        # TODO: different phases matching
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.graph = Registry.mapping['world_mapping']['graph_setting'].graph
        self.world = world
        self.sub_agents = len(self.world.intersections)
        # TODO: support dynamic graph later
        self.edge_idx = torch.tensor(self.graph['sparse_adj'].T, dtype=torch.long)  # source -> target

        #  model parameters
        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']
        self.model_dict = Registry.mapping['model_mapping']['setting'].param

        #  get generator for CoLightAgent
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        #  get reward generator for CoLightAgent
        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        #  get queue generator for CoLightAgent
        queues = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"], 
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        #  get delay generator for CoLightAgent
        delays = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_delay"], 
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

        #  phase generator
        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, inter, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        # TODO: add irregular control of signals in the future
        self.phase_lengths = np.array([len(i.phases) for i in self.world.intersections])
        self.action_space = gym.spaces.Discrete(max(self.phase_lengths))
        self.lane_ob_length = max(ob[1].ob_length for ob in self.ob_generator)
        if self.phase:
            if self.one_hot:
                self.ob_length = self.lane_ob_length + self.action_space.n
            else:
                self.ob_length = self.lane_ob_length + 1
        else:
            self.ob_length = self.lane_ob_length

        self.get_attention = Registry.mapping['logger_mapping']['setting'].param['attention']
        # train parameters
        self.rank = rank
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

    def reset(self):
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        #  get reward generator for CoLightAgent
        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        #  phase generator
        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, inter, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        # queue metric
        queues = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"], 
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        # delay metric
        delays = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_delay"], 
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

    def get_ob(self):
        # Like the other DQN agents, expose traffic and phase separately.
        # Append the supplied phase only when constructing a network input.
        x_obs = []  # sub_agents * lane_nums,
        for i in range(len(self.ob_generator)):
            ob = self.ob_generator[i][1].generate()/ self.vehicle_max
            ob = np.pad(ob, (0, self.lane_ob_length - ob.shape[-1] ))
            x_obs.append(ob)
            
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def _network_input(self, ob, phase):
        """Build per-node features from a current or replay-saved snapshot.

        Never query the live World here: replay states and next states have
        their own phase vectors. Padding belongs to the lane block; phase
        features occupy a separate fixed-width block at its end.
        """
        lanes = np.asarray(ob, dtype=np.float32)
        if lanes.shape != (self.sub_agents, self.lane_ob_length):
            raise ValueError('CoLight expects one lane-feature row per intersection')
        if not self.phase:
            return lanes
        phases = np.asarray(phase)
        if phases.shape == (self.sub_agents, 1):
            phases = phases[:, 0]
        if (phases.shape != (self.sub_agents,)
                or not np.issubdtype(phases.dtype, np.integer)
                or np.any(phases < 0) or np.any(phases >= self.phase_lengths)):
            raise ValueError('CoLight phase must contain a valid integer index per intersection')
        if self.one_hot:
            encoded = np.eye(self.action_space.n, dtype=np.float32)[phases]
        else:
            encoded = phases.astype(np.float32)[:, None]
        return np.concatenate((lanes, encoded), axis=1)

    def get_reward(self):
        # TODO: test output
        rewards = []  # sub_agents
        for i in range(len(self.reward_generator)):
            rewards.append(self.reward_generator[i][1].generate())
        rewards = np.squeeze(np.array(rewards, dtype=np.float32)) * 12
        return rewards

    def get_phase(self):
        # TODO: test phase output onehot/int
        phase = []  # sub_agents
        for i in range(len(self.phase_generator)):
            phase.append((self.phase_generator[i][1].generate()))
        phase = (np.concatenate(phase)).astype(np.int8)
        # phase = np.concatenate(phase, dtype=np.int8)
        return phase

    def get_queue(self):
        """
        get delay of intersection
        return: value(one intersection) or [intersections,](multiple intersections)
        """
        queue = []
        for item in self.queue:
            item = item[1].generate()
            item = np.pad(item, (0, self.ob_length - item.shape[-1]))
            queue.append(item)
            
        tmp_queue = np.squeeze(np.array(queue, dtype=np.float32))
        queue = np.sum(tmp_queue, axis=1 if len(tmp_queue.shape)==2 else 0)
        return queue

    def get_delay(self):
        delay = []
        for i in range(len(self.delay)):
            delay.append((self.delay[i][1].generate()))
        delay = np.squeeze(np.array(delay, dtype=np.float32))
        return delay # [intersections,]

    def get_action(self, ob, phase, test=False):
        """
        input are np.array here
        # TODO: support irregular input in the future
        :param ob: [agents, ob_length] -> [batch, agents, ob_length]
        :param phase: [agents] -> [batch, agents]
        :param test: boolean, exploit while training and determined while testing
        :return: [batch, agents] -> action taken by environment
        """
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        observation = torch.tensor(self._network_input(ob, phase), dtype=torch.float32)
        edge = self.edge_idx
        dp = Data(x=observation, edge_index=edge)

        if self.get_attention:
            # TODO: collect attention matrix later
            actions = self.model(x=dp.x, edge_index=dp.edge_index, train=False)
            att = None
            actions = actions.clone().detach().numpy()
            # action = np.argmax(actions, axis=1)
            action_list = []
            for action_vec, phase_length in zip(actions, self.phase_lengths):
                action_list.append(np.argmax(action_vec[0:phase_length]))
            # action = np.clip(action, 0, self.phase_lengths - 1)
            action = np.array(action_list)
            # action = np.clip(action, 0, self.phase_lengths - 1)
            return action, att  # [batch, agents], [batch, agents, nv, neighbor]
        else:
            actions = self.model(x=dp.x, edge_index=dp.edge_index, train=False)
            actions = actions.clone().detach().numpy()
            
            action_list = []
            for action_vec, phase_length in zip(actions, self.phase_lengths):
                action_list.append(np.argmax(action_vec[0:phase_length]))
            # action = np.clip(action, 0, self.phase_lengths - 1)
            action = np.array(action_list)
            
            return action  # [batch, agents] TODO: check here

    def sample(self):
        action = np.random.randint(0, self.action_space.n, self.sub_agents)
        action = np.clip(action, 0, self.phase_lengths - 1)
        return action

    def _build_model(self):
        model = ColightNet(self.ob_length, self.action_space.n, self.phase_lengths, **self.model_dict)
        return model

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs,
                 cur_phase, done, key, *, scene_t=None, scene_t1=None,
                 truncated=False):
        if self.phase:
            # Own the phase snapshots even if a caller reuses its arrays.
            last_phase = np.array(last_phase, copy=True)
            cur_phase = np.array(cur_phase, copy=True)
        if scene_t is None and scene_t1 is None and not truncated:
            # Keep ordinary CoLight checkpoints byte-compatible with the
            # historical six-field replay until scene collection is opted in.
            payload = (last_obs, last_phase, actions, rewards, obs, cur_phase)
        else:
            payload = pack_scene_transition(
                last_obs, last_phase, scene_t, actions, rewards, obs,
                cur_phase, scene_t1, done, truncated)
        self.replay_buffer.append((key, payload))

    def _batchwise(self, samples):
        # load onto tensor

        batch_list = []
        batch_list_p = []
        actions = []
        rewards = []
        for item in samples:
            dp = unpack_scene_transition(item[1])
            state = torch.tensor(self._network_input(dp.obs_t, dp.phase_t), dtype=torch.float32)
            batch_list.append(Data(x=state, edge_index=self.edge_idx))

            state_p = torch.tensor(self._network_input(dp.obs_t1, dp.phase_t1), dtype=torch.float32)
            batch_list_p.append(Data(x=state_p, edge_index=self.edge_idx))
            rewards.append(dp.reward)
            actions.append(dp.action)
        batch_t = Batch.from_data_list(batch_list)
        batch_tp = Batch.from_data_list(batch_list_p)
        # TODO reshape slow warning
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        if self.sub_agents > 1:
            rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
            actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here
        # rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
        # actions = torch.tensor(np.array(actions), dtype=torch.long)
        # actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here

        return batch_t, batch_tp, rewards, actions

    def train(self):
        samples = random.sample(self.replay_buffer, self.batch_size)
        b_t, b_tp, rewards, actions = self._batchwise(samples)

        out = self.target_model(x=b_tp.x, edge_index=b_tp.edge_index, train=False)
        target = rewards + self.gamma * torch.max(out, dim=1)[0]
        target_f = self.model(x=b_t.x, edge_index=b_t.edge_index, train=False)

        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        loss = self.criterion(self.model(x=b_t.x, edge_index=b_t.edge_index, train=True), target_f)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.clone().detach().numpy()

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                'model', f'{e}_{self.rank}.pt')
        self.model.load_state_dict(torch.load(model_name))
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.target_model.state_dict(), model_name)


class ColightNet(nn.Module):
    def __init__(self, input_dim, output_dim, phase_lengths, **kwargs):
        super(ColightNet, self).__init__()
        self.model_dict = kwargs
        self.sga_enabled = bool(self.model_dict.get('sga_enabled', False))
        self.sga_input_mode = self.model_dict.get('sga_input_mode', 'structured')
        self.batch_size = self.model_dict['batch_size']
        self.action_space = gym.spaces.Discrete(output_dim)
        self.features = input_dim
        if self.model_dict.get('phase', False):
            # Old phase=True weights have the same shape but were trained on
            # zero padding. A persistent marker makes strict checkpoint loads
            # reject that incompatible input meaning. phase=False is unchanged.
            self.register_buffer('_phase_input_version', torch.tensor(1, dtype=torch.int64))
        self.module_list = nn.ModuleList()
        self.embedding_MLP = Embedding_MLP(self.features, layers=self.model_dict.get('NODE_EMB_DIM'))
        self.sga_hidden_dim = int(self.model_dict.get('NODE_EMB_DIM')[-1])
        if self.sga_enabled:
            if self.sga_input_mode in ('text_multi_event', 'concat', 'text_multi_event_flx'):
                self.scene_encoder = SceneContextEncoder(
                    text_hidden_dim=self.sga_hidden_dim,
                    meta_dim=int(self.model_dict.get('sga_meta_dim', 64)),
                    fusion_hidden_dim=self.sga_hidden_dim,
                )
                if self.sga_input_mode == 'text_multi_event':
                    self.scene_attention = MultiEventSceneGuidedAttention(
                        hidden_dim=self.sga_hidden_dim,
                        attention_dim=int(self.model_dict.get('sga_attention_dim', self.sga_hidden_dim)),
                        temperature=float(self.model_dict.get('sga_temperature', 1.0)),
                        residual_scale=float(self.model_dict.get('sga_residual_scale', 1.0)),
                        grounding_bias=float(self.model_dict.get('sga_grounding_bias', 1.0)),
                    )
                elif self.sga_input_mode == 'text_multi_event_flx':
                    self.scene_attention = SceneCrossAttentionResidual(
                        hidden_dim=self.sga_hidden_dim,
                        attention_dim=int(self.model_dict.get('sga_attention_dim', self.sga_hidden_dim)),
                        heads=int(self.model_dict.get('sga_flx_heads', 4)),
                        grounding_bias=float(self.model_dict.get('sga_grounding_bias', 1.0)),
                    )
                    if float(self.model_dict.get('sga_align_lambda', 0.0)) > 0:
                        # Auxiliary alignment (stage-3 arms): the window
                        # encoder only participates in the auxiliary loss,
                        # never in the control forward path.  mode='traffic'
                        # is CAREL-style (event<->traffic window); mode=
                        # 'transition' is GRIF-style (event<->traffic change).
                        align_mode = str(self.model_dict.get(
                            'sga_align_mode', 'traffic'))
                        feat_dim = int(self.model_dict.get(
                            'sga_align_feat_dim', self.features))
                        if align_mode == 'transition':
                            self.align_window_encoder = \
                                TrafficTransitionEncoder(
                                    feat_dim=feat_dim,
                                    out_dim=self.sga_hidden_dim)
                        elif align_mode == 'traffic':
                            self.align_window_encoder = TrafficWindowEncoder(
                                feat_dim=feat_dim,
                                out_dim=self.sga_hidden_dim)
                        else:
                            raise ValueError(
                                f'unknown sga_align_mode {align_mode!r}')
                        # 'meta' strips location tokens from the alignment
                        # event embedding (meta branch never sees location),
                        # forcing z_e to carry event semantics rather than
                        # instance identity; 'fused' keeps text+meta.
                        if str(self.model_dict.get(
                                'sga_align_event_repr', 'fused')) == 'meta':
                            self.align_event_proj = nn.Linear(
                                int(self.model_dict.get('sga_meta_dim', 64)),
                                self.sga_hidden_dim)
                else:
                    # The concat control is deliberately a separate fusion
                    # path: it consumes the same frozen event embeddings and
                    # Gscene masks as SGA, but uses a deterministic per-node
                    # masked mean followed by one shared projection.  This
                    # keeps the traffic encoder, graph and Q head identical
                    # while removing learned event selection.
                    self.concat_scene_fusion = nn.Sequential(
                        nn.Linear(2 * self.sga_hidden_dim, self.sga_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.sga_hidden_dim, self.sga_hidden_dim),
                    )
            else:
                self.scene_encoder = StructuredSceneEncoder(
                    output_dim=self.sga_hidden_dim,
                    hidden_dim=max(32, self.sga_hidden_dim // 2),
                )
                self.scene_attention = SceneGuidedAttention(
                    hidden_dim=self.sga_hidden_dim,
                    attention_dim=int(self.model_dict.get('sga_attention_dim', self.sga_hidden_dim)),
                    temperature=float(self.model_dict.get('sga_temperature', 1.0)),
                    dropout=float(self.model_dict.get('sga_dropout', 0.0)),
                    residual_scale=float(self.model_dict.get('sga_residual_scale', 1.0)),
                )
        for i in range(self.model_dict.get('N_LAYERS')):
            block = MultiHeadAttModel(d=self.model_dict.get('INPUT_DIM')[i],
                                      dv=self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[i],
                                      d_out=self.model_dict.get('OUTPUT_DIM')[i],
                                      nv=self.model_dict.get('NUM_HEADS')[i],
                                      suffix=i)
            self.module_list.append(block)
        output_dict = OrderedDict()

        if len(self.model_dict['OUTPUT_LAYERS']) != 0:
            # TODO: dubug this branch
            for l_idx, l_size in enumerate(self.model_dict['OUTPUT_LAYERS']):
                name = f'output_{l_idx}'
                if l_idx == 0:
                    h = nn.Linear(block.d_out, l_size)
                else:
                    h = nn.Linear(self.model_dict.get('OUTPUT_LAYERS')[l_idx - 1], l_size)
                output_dict.update({name: h})
                name = f'relu_{l_idx}'
                output_dict.update({name: nn.ReLU})
            out = nn.Linear(self.model_dict['OUTPUT_LAYERS'][-1], self.action_space.n)
        else:
            out = nn.Linear(block.d_out, self.action_space.n)
        name = f'output'
        output_dict.update({name: out})
        
        # make mask
        unpadded_phase_mask = [torch.ones(length, dtype=torch.bool) for length in phase_lengths]
        phase_mask = torch.nn.utils.rnn.pad_sequence(unpadded_phase_mask, batch_first=True)
        mask_layer = MaskedOutput(mask=phase_mask, batch_size=self.batch_size, action_space=self.action_space)
        output_dict.update({'out_mask': mask_layer})

        self.output_layer = nn.Sequential(output_dict)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if self.model_dict.get('phase', False):
            version = state_dict.get('_phase_input_version')
            if not torch.is_tensor(version) or version.numel() != 1 or version.item() != 1:
                raise RuntimeError('CoLight phase input version mismatch: legacy zero-padded phase weights are incompatible')
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def encode_scene(self, metadata, train=True):
        if not self.sga_enabled:
            raise RuntimeError('Scene encoder is disabled for this network')
        if self.sga_input_mode in ('text_multi_event', 'concat', 'text_multi_event_flx'):
            from utils.scene_context import SceneContext
            context = getattr(metadata, 'context', metadata)
            grounding = getattr(metadata, 'grounding', None)
            if not isinstance(context, SceneContext) or grounding is None:
                raise ValueError('text_multi_event SGA requires SceneContext plus Gscene grounding')
            if train:
                return self.scene_encoder.encode_context_events(context, grounding)
            with torch.no_grad():
                return self.scene_encoder.encode_context_events(context, grounding)
        values = metadata if isinstance(metadata, (tuple, list)) else (metadata,)
        if train:
            return self.scene_encoder(values)
        with torch.no_grad():
            return self.scene_encoder(values)

    def forward(self, x, edge_index, train=True, scene_embedding=None):
        return self.control_from_features(
            self.encode_traffic(x, train=train), edge_index, train=train,
            scene_embedding=scene_embedding)

    def encode_traffic(self, observation, train=True):
        """Map per-node traffic observations to CoLight node features."""
        return self.embedding_MLP.forward(observation, train)

    def control_from_features(self, traffic_features, graph_context, train=True,
                              scene_embedding=None):
        """Run the unchanged CoLight GAT and Q head on node features."""
        if self.sga_enabled:
            if scene_embedding is None:
                raise ValueError('SGA-enabled CoLight requires scene_embedding')
            if self.sga_input_mode in ('text_multi_event', 'text_multi_event_flx'):
                if isinstance(scene_embedding, EventSceneRepresentation):
                    batch = 1
                    nodes = traffic_features.shape[0]
                    condition = SceneCondition(
                        scene_embedding.z_events.unsqueeze(0),
                        scene_embedding.event_mask.unsqueeze(0),
                        scene_embedding.node_event_mask.unsqueeze(0),
                        None if scene_embedding.node_event_relation is None else
                        scene_embedding.node_event_relation.unsqueeze(0))
                elif isinstance(scene_embedding, SceneCondition):
                    condition = scene_embedding
                    batch = condition.event_embeddings.shape[0]
                    if traffic_features.shape[0] % batch:
                        raise ValueError('traffic features are not divisible by scene batch')
                    nodes = traffic_features.shape[0] // batch
                else:
                    raise ValueError('text_multi_event/flx requires EventSceneRepresentation or SceneCondition')
                modulated, alpha, node_gate, feature_gate = self.scene_attention(
                    traffic_features.reshape(batch, nodes, -1), condition)
                traffic_features = modulated.reshape(batch * nodes, -1)
                self.last_scene_attention = alpha.detach()
                self.last_scene_node_gate = node_gate.detach()
                self.last_scene_feature_gate = feature_gate.detach()
            elif self.sga_input_mode == 'concat':
                if isinstance(scene_embedding, EventSceneRepresentation):
                    condition = SceneCondition(
                        scene_embedding.z_events.unsqueeze(0),
                        scene_embedding.event_mask.unsqueeze(0),
                        scene_embedding.node_event_mask.unsqueeze(0),
                        None if scene_embedding.node_event_relation is None else
                        scene_embedding.node_event_relation.unsqueeze(0))
                elif isinstance(scene_embedding, SceneCondition):
                    condition = scene_embedding
                    if traffic_features.shape[0] % condition.event_embeddings.shape[0]:
                        raise ValueError('traffic features are not divisible by scene batch')
                else:
                    raise ValueError('concat requires EventSceneRepresentation or SceneCondition')
                batch = condition.event_embeddings.shape[0]
                if traffic_features.shape[0] % batch:
                    raise ValueError('traffic features are not divisible by scene batch')
                nodes = traffic_features.shape[0] // batch
                condition.validate(batch, nodes, traffic_features.shape[-1])
                # Use direct and route-aligned Gscene relations as the
                # deterministic association mask.  An event with no relation
                # to a node contributes zero; normal scenes remain an exact
                # zero scene vector and retain the traffic path.
                relation = condition.node_event_relation
                if relation is None:
                    relation_mask = condition.node_event_mask
                else:
                    relation_mask = relation.any(dim=-1)
                weights = relation_mask.to(traffic_features.dtype)
                event_count = condition.event_embeddings.shape[1]
                if event_count == 0:
                    node_scene = traffic_features.new_zeros((batch, nodes, self.sga_hidden_dim))
                else:
                    node_scene = torch.einsum(
                        'bne,bed->bnd', weights, condition.event_embeddings)
                    node_scene = node_scene / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
                fused = torch.cat((traffic_features.reshape(batch, nodes, -1), node_scene), dim=-1)
                traffic_features = self.concat_scene_fusion(fused).reshape(batch * nodes, -1)
                self.last_concat_scene = node_scene.detach()
            else:
                if scene_embedding.ndim != 2 or traffic_features.ndim != 2:
                    raise ValueError('SGA inputs must be [batch, dim] and [batch*num_nodes, dim]')
                batch = scene_embedding.shape[0]
                if traffic_features.shape[0] % batch:
                    raise ValueError('traffic features are not divisible by scene batch')
                nodes = traffic_features.shape[0] // batch
                traffic_features, alpha = self.scene_attention(
                    traffic_features.reshape(batch, nodes, -1), scene_embedding)
                self.last_scene_attention = alpha.detach()
            traffic_features = traffic_features.reshape(batch * nodes, -1)
        if train:
            h = traffic_features
            for mdl in self.module_list:
                h = mdl.forward(h, graph_context, train=True)
            return self.output_layer(h)
        with torch.no_grad():
            h = traffic_features
            for mdl in self.module_list:
                h = mdl.forward(h, graph_context, train=False)
            return self.output_layer(h)

class MaskedOutput(nn.Module):
    def __init__(self, mask, batch_size, action_space):
        super(MaskedOutput, self).__init__()
        self.batch_size = batch_size
        self.mask = mask
        self.action_space = action_space

    def forward(self, x):
        # Apply the mask to the output
        # x = torch.exp(x)
        masked_output = x.reshape(-1 ,self.mask.shape[0], self.action_space.n) * self.mask
        masked_output = masked_output.reshape(-1, self.mask.shape[-1])
        return masked_output

class Embedding_MLP(nn.Module):
    def __init__(self, in_size, layers):
        super(Embedding_MLP, self).__init__()
        constructor_dict = OrderedDict()
        for l_idx, l_size in enumerate(layers):
            name = f"node_embedding_{l_idx}"
            if l_idx == 0:
                h = nn.Linear(in_size, l_size)
                constructor_dict.update({name: h})
            else:
                h = nn.Linear(layers[l_idx - 1], l_size)
                constructor_dict.update({name: h})
            name = f"n_relu_{l_idx}"
            constructor_dict.update({name: nn.ReLU()})
        self.embedding_node = nn.Sequential(constructor_dict)

    def _forward(self, x):
        x = self.embedding_node(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


class MultiHeadAttModel(MessagePassing):
    """
    inputs:
        In_agent [bacth,agents,128]
        In_neighbor [agents, neighbor_num]
        l: number of neighborhoods (in my code, l=num_neighbor+1,because l include itself)
        d: dimension of agents's embedding
        dv: dimension of each head
        dout: dimension of output
        nv: number of head (multi-head attention)
    output:
        -hidden state: [batch,agents,32]
        -attention: [batch,agents,neighbor]
    """
    def __init__(self, d, dv, d_out, nv, suffix):
        super(MultiHeadAttModel, self).__init__(aggr='add')
        self.d = d
        self.dv = dv
        self.d_out = d_out
        self.nv = nv
        self.suffix = suffix
        # target is center
        self.W_target = nn.Linear(d, dv * nv)
        self.W_source = nn.Linear(d, dv * nv)
        self.hidden_embedding = nn.Linear(d, dv * nv)
        self.out = nn.Linear(dv, d_out)
        self.att_list = []
        self.att = None

    def _forward(self, x, edge_index):
        # TODO: test batch is shared or not

        # x has shape [N, d], edge_index has shape [E, 2]
        edge_index, _ = add_self_loops(edge_index=edge_index)
        aggregated = self.propagate(x=x, edge_index=edge_index)  # [16, 16]
        out = self.out(aggregated)
        out = F.relu(out)  # [ 16, 128]
        #self.att = torch.tensor(self.att_list)
        return out

    def forward(self, x, edge_index, train=True):
        if train:
            return self._forward(x, edge_index)
        else:
            with torch.no_grad():
                return self._forward(x, edge_index)

    def message(self, x_i, x_j, edge_index):
        h_target = F.relu(self.W_target(x_i))
        h_target = h_target.view(h_target.shape[:-1][0], self.nv, self.dv)
        agent_repr = h_target.permute(1, 0, 2)

        h_source = F.relu(self.W_source(x_j))
        h_source = h_source.view(h_source.shape[:-1][0], self.nv, self.dv)

        neighbor_repr = h_source.permute(1, 0, 2)  # [nv, E, dv]
        index = edge_index[1]  # which is target

        e_i = torch.mul(agent_repr, neighbor_repr).sum(-1)  # [5, 64]
        max_node = torch_scatter.scatter_max(e_i, index=index)[0]  # [5, 16]
        max_i = max_node.index_select(1, index=index)  # [5, 64]
        ec_i = torch.add(e_i, -max_i)
        ecexp_i = torch.exp(ec_i)
        norm_node = torch_scatter.scatter_sum(ecexp_i, index=index)  # [5, 16]
        normst_node = torch.add(norm_node, 1e-12)  # [5, 16]
        normst_i = normst_node.index_select(1, index)  # [5, 64]

        alpha_i = ecexp_i / normst_i  # [5, 64]
        alpha_i_expand = alpha_i.repeat(self.dv, 1, 1)
        alpha_i_expand = alpha_i_expand.permute((1, 2, 0))  # [5, 64, 16]

        hidden_neighbor = F.relu(self.hidden_embedding(x_j))
        hidden_neighbor = hidden_neighbor.view(hidden_neighbor.shape[:-1][0], self.nv, self.dv)
        hidden_neighbor_repr = hidden_neighbor.permute(1, 0, 2)  # [5, 64, 16]
        out = torch.mul(hidden_neighbor_repr, alpha_i_expand).mean(0)

        # TODO: attention ouput in the future
        # self.att_list.append(alpha_i)  # [64, 16]
        return out

    def get_att(self):
        if self.att is None:
            print('invalid att')
        return self.att


@Registry.register_model('sga_colight')
class SGAColightAgent(CoLightAgent):
    """CoLight with replay-safe structured or event-level Scene Attention."""

    uses_scene_attention = True

    def __init__(self, world, rank):
        super().__init__(world, rank)
        self.scene_replay_enabled = True
        self._live_scene = None
        self._sga_attention_log = None
        self._scene_log_context = {
            'episode': None, 'decision_step': None, 'global_decision_step': None,
        }
        try:
            path = Registry.mapping['logger_mapping']['path'].path
            self._sga_attention_log = os.path.join(path, 'sga_attention.jsonl')
            self._sga_branch_log = os.path.join(path, 'sga_branch.jsonl')
        except (KeyError, AttributeError):
            self._sga_branch_log = None
        self._branch_update_count = 0

    def set_scene_log_context(self, episode=None, decision_step=None,
                              global_decision_step=None):
        """Attach stable run coordinates to the next attention record."""
        self._scene_log_context = {
            'episode': None if episode is None else int(episode),
            'decision_step': None if decision_step is None else int(decision_step),
            'global_decision_step': (
                None if global_decision_step is None else int(global_decision_step)),
        }

    @staticmethod
    def _metadata(scene_state):
        from agent.scene_attention import scene_metadata_from_context
        from utils.scene_context import SceneContext
        context = SceneContext(0.0, ()) if scene_state is None else getattr(
            scene_state, 'context', scene_state)
        return scene_metadata_from_context(context)

    def _scene_embedding(self, network, scene_state, train):
        if network.sga_input_mode in ('text_multi_event', 'concat', 'text_multi_event_flx'):
            return network.encode_scene(scene_state, train=train)
        return network.encode_scene(self._metadata(scene_state), train=train)

    def get_action(self, ob, phase, test=False):
        if not test and np.random.rand() <= self.epsilon:
            return self.sample()
        observation = torch.tensor(self._network_input(ob, phase), dtype=torch.float32)
        dp = Data(x=observation, edge_index=self.edge_idx)
        scene = self._scene_embedding(self.model, self._live_scene, train=False)
        actions = self.model(x=dp.x, edge_index=dp.edge_index, train=False,
                             scene_embedding=scene).detach().numpy()
        if self._sga_attention_log and hasattr(self.model, 'last_scene_attention'):
            alpha = self.model.last_scene_attention.detach().cpu().numpy().tolist()
            context = getattr(self._live_scene, 'context', self._live_scene)
            row = {
                **self._scene_log_context,
                'simulation_time': float(getattr(context, 'observed_at', -1.0)),
                'scene_state': getattr(context, 'state', 'unknown'),
                'event_ids': list(getattr(context, 'event_ids', ())),
                'attention': alpha,
            }
            row['event_count'] = len(row['event_ids'])
            row['node_ids'] = list(getattr(
                self, 'intersection_ids', getattr(self.world, 'intersection_ids', ())))
            grounding = getattr(self._live_scene, 'grounding', None)
            if grounding is not None:
                row['direct_grounding_mask'] = grounding.node_report_mask.astype(int).tolist()
                row['node_event_relation'] = grounding.node_event_relation.astype(int).tolist()
            if hasattr(self.model, 'last_scene_node_gate'):
                row['node_gate'] = self.model.last_scene_node_gate.detach().cpu().numpy().tolist()
            if hasattr(self.model, 'last_scene_feature_gate'):
                row['feature_gate_mean'] = self.model.last_scene_feature_gate.detach().mean(-1).cpu().numpy().tolist()
            for name in ('semantic_score', 'direct_bias', 'relation_bias', 'relevance'):
                value = getattr(self.model.scene_attention, 'last_' + name, None)
                if value is None:
                    continue
                flat = value.detach().float().reshape(-1)
                if flat.numel():
                    row[name + '_stats'] = {
                        'mean': float(flat.mean()), 'std': float(flat.std(unbiased=False)),
                        'min': float(flat.min()), 'max': float(flat.max()),
                    }
                else:
                    row[name + '_stats'] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            branch_norm = getattr(self.model.scene_attention, 'branch_norm', None)
            if callable(branch_norm):
                row['branch_head_norm'] = branch_norm()
            with open(self._sga_attention_log, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        elif self._sga_attention_log and hasattr(self.model, 'last_concat_scene'):
            # Concat has no learned attention weights.  Preserve the same
            # per-decision coordinates and expose the deterministic scene
            # contribution for the explanation audit.
            context = getattr(self._live_scene, 'context', self._live_scene)
            scene = self.model.last_concat_scene.detach()
            grounding = getattr(self._live_scene, 'grounding', None)
            row = {
                **self._scene_log_context,
                'fusion': 'masked_mean_concat',
                'simulation_time': float(getattr(context, 'observed_at', -1.0)),
                'scene_state': getattr(context, 'state', 'unknown'),
                'event_ids': list(getattr(context, 'event_ids', ())),
                'event_count': len(getattr(context, 'event_ids', ())),
                'scene_norm_mean': float(scene.norm(dim=-1).mean()),
                'scene_norm_max': float(scene.norm(dim=-1).max()) if scene.numel() else 0.0,
            }
            if grounding is not None:
                row['direct_grounding_mask'] = grounding.node_report_mask.astype(int).tolist()
                row['node_event_relation'] = grounding.node_event_relation.astype(int).tolist()
            with open(self._sga_attention_log, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        return np.asarray([np.argmax(row[:length])
                           for row, length in zip(actions, self.phase_lengths)])

    def _batchwise(self, samples):
        batch_t, batch_tp, rewards, actions = super()._batchwise(samples)
        if self.model.sga_input_mode in ('text_multi_event', 'concat', 'text_multi_event_flx'):
            scene_t = [self._scene_embedding(self.model,
                                             unpack_scene_transition(item[1]).scene_t,
                                             train=True) for item in samples]
            scene_tp = [self._scene_embedding(self.target_model,
                                              unpack_scene_transition(item[1]).scene_t1,
                                              train=False) for item in samples]
            return batch_t, batch_tp, rewards, actions, \
                batch_event_scene_representations(scene_t), \
                batch_event_scene_representations(scene_tp)
        scene_t = [self._metadata(unpack_scene_transition(item[1]).scene_t)
                   for item in samples]
        scene_tp = [self._metadata(unpack_scene_transition(item[1]).scene_t1)
                    for item in samples]
        return batch_t, batch_tp, rewards, actions, scene_t, scene_tp

    def train(self):
        samples = random.sample(self.replay_buffer, self.batch_size)
        b_t, b_tp, rewards, actions, scene_t, scene_tp = self._batchwise(samples)
        if self.model.sga_input_mode in ('text_multi_event', 'concat', 'text_multi_event_flx'):
            target_scene, online_scene = scene_tp, scene_t
        else:
            target_scene = self.target_model.encode_scene(scene_tp, train=False)
            online_scene = self.model.encode_scene(scene_t, train=True)
        out = self.target_model(x=b_tp.x, edge_index=b_tp.edge_index, train=False,
                                scene_embedding=target_scene)
        target = rewards + self.gamma * torch.max(out, dim=1)[0]
        target_f = self.model(x=b_t.x, edge_index=b_t.edge_index, train=False,
                              scene_embedding=online_scene)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        q_values = self.model(x=b_t.x, edge_index=b_t.edge_index, train=True,
                              scene_embedding=online_scene)
        loss = self.criterion(q_values, target_f)
        align = None
        if getattr(self, '_align_enabled', False):
            align = self._compute_align_loss()
            if align is not None:
                loss = loss + self._align_lambda * align
        self.optimizer.zero_grad()
        loss.backward()
        self._log_branch_diagnostics(
            scene_t,
            align_loss=None if align is None else float(align.detach()))
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.detach().numpy()

    def _log_branch_diagnostics(self, scene_t, align_loss=None):
        """Per-update branch evidence: head value norm, head/inner grad norms.

        Required by the ATT-ENTITY-003 gate-opening contract: distinguishes
        'branch never opened' from 'text carried no usable signal'.  Written
        pre-clip so the raw TD gradient magnitude is what is recorded.
        """
        attention = getattr(self.model, 'scene_attention', None)
        if attention is None or not getattr(self, '_sga_branch_log', None):
            return
        head = getattr(attention, 'out', None)
        inner_sq = 0.0
        for name, param in attention.named_parameters():
            if param.grad is None or (head is not None and name.startswith('out.')):
                continue
            inner_sq += float(param.grad.norm()) ** 2
        branch_norm = getattr(attention, 'branch_norm', None)
        # scene_t arrives as a batched SceneCondition (see _batchwise); count
        # batch rows that carry at least one valid event.
        mask = getattr(scene_t, 'event_mask', None)
        event_rows = int(mask.any(dim=1).sum()) if mask is not None else 0
        batch = int(mask.shape[0]) if mask is not None else 0
        self._branch_update_count += 1
        row = {
            'update': self._branch_update_count,
            'head_norm': None if not callable(branch_norm) else branch_norm(),
            'head_grad_norm': (None if head is None or head.weight.grad is None
                               else float(head.weight.grad.norm())),
            'inner_grad_norm': inner_sq ** 0.5,
            'event_rows': event_rows,
            'batch': batch,
            'align_loss': align_loss,
            'align_pairs': (len(self._align_buffer)
                            if getattr(self, '_align_enabled', False) else None),
        }
        with open(self._sga_branch_log, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')


@Registry.register_model('concat_colight')
class ConcatColightAgent(SGAColightAgent):
    """CoLight with deterministic per-node scene concatenation.

    This is the formal concat comparator for ``sga_colight``.  It shares the
    frozen text encoder, event replay contract, traffic encoder and graph Q
    head; only the scene fusion module differs.  The class is kept separate
    so a resolved experiment manifest records an unambiguous model identity.
    """

    uses_scene_attention = True

    def __init__(self, world, rank):
        super().__init__(world, rank)
        if getattr(self, 'model', None) is not None and self.model.sga_input_mode != 'concat':
            raise ValueError('concat_colight requires model.sga_input_mode=concat')


@Registry.register_model('sga_flx_colight')
class SGAFlxColightAgent(SGAColightAgent):
    """CoLight with zero-init cross-attention event injection (SGA-FLX).

    Shares the frozen text encoder, event replay contract, traffic encoder
    and graph Q head; only the fusion module differs from ``sga_colight``.
    The separate registry name keeps the experiment manifest identity
    unambiguous, mirroring the ``concat_colight`` precedent.
    """

    uses_scene_attention = True

    def __init__(self, world, rank):
        super().__init__(world, rank)
        if getattr(self, 'model', None) is not None \
                and self.model.sga_input_mode != 'text_multi_event_flx':
            raise ValueError('sga_flx_colight requires sga_input_mode=text_multi_event_flx')
        # CAREL-style auxiliary event-traffic alignment (ATT-ENTITY-003 stage
        # 3).  Enabled only via model.sga_align_lambda > 0; the accumulator
        # runs inside remember() and the loss is folded into the TD update.
        self._align_lambda = float(self.model_dict.get('sga_align_lambda', 0.0))
        self._align_enabled = self._align_lambda > 0 and hasattr(
            self.model, 'align_window_encoder')
        if self.model_dict.get('sga_align_lambda', 0.0) > 0 \
                and not self._align_enabled:
            raise RuntimeError(
                'sga_align_lambda>0 but model has no align_window_encoder')
        if self._align_enabled:
            feat_dim = self.model.align_window_encoder.feat_dim
            if feat_dim != self.lane_ob_length:
                raise ValueError(
                    f'align window encoder feat_dim={feat_dim} disagrees with '
                    f'lane_ob_length={self.lane_ob_length}; set '
                    'model.sga_align_feat_dim to the raw lane feature width')
            self._align_tau = float(self.model_dict.get('sga_align_tau', 0.07))
            self._align_max_events = int(self.model_dict.get(
                'sga_align_max_events', 16))
            self._align_min_events = int(self.model_dict.get(
                'sga_align_min_events', 2))
            self._align_rng = np.random.default_rng(
                int(self.model_dict.get('sga_align_seed', 0)) + int(rank))
            self._align_acc = EpisodeAlignAccumulator(
                window_cap_steps=int(self.model_dict.get(
                    'sga_align_window_cap_steps', 60)),
                min_active_steps=int(self.model_dict.get(
                    'sga_align_min_active_steps', 3)))
            self._align_buffer = AlignmentBuffer(
                max_episodes=int(self.model_dict.get(
                    'sga_align_buffer_episodes', 32)))
            self._last_align_loss = None

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards,
                 obs, cur_phase, done, key, *, scene_t=None, scene_t1=None,
                 truncated=False):
        super().remember(last_obs, last_phase, actions, actions_prob, rewards,
                         obs, cur_phase, done, key, scene_t=scene_t,
                         scene_t1=scene_t1, truncated=truncated)
        if not self._align_enabled:
            return
        obs_rows = np.stack([np.asarray(row, dtype=np.float32)
                             for row in last_obs])
        self._align_acc.step(obs_rows, scene_t)
        if done or truncated:
            self._align_buffer.append(self._align_acc.finish(self._align_rng))
            self._align_acc.reset()

    def _encode_events_for_align(self, reports):
        """Re-encode grounded reports with current (grad-enabled) encoder."""
        enc = self.model.scene_encoder
        metas = tuple(StructuredSceneMeta.from_report(r) for r in reports)
        if hasattr(self.model, 'align_event_proj'):
            # location-free semantic embedding for the auxiliary objective
            z = self.model.align_event_proj(enc.meta_branch(metas))
            return F.normalize(z, dim=-1)
        texts = tuple(r.text for r in reports)
        z = enc.fusion(torch.cat([enc.text_branch(texts),
                                  enc.meta_branch(metas)], dim=-1))
        return F.normalize(z, dim=-1)

    def _compute_align_loss(self):
        pairs = self._align_buffer.all_pairs()
        if len(pairs) < self._align_min_events:
            return None
        pairs = pairs[-self._align_max_events:]
        z_e = self._encode_events_for_align([p.report for p in pairs])
        if isinstance(self.model.align_window_encoder,
                      TrafficTransitionEncoder):
            # GRIF-style: windows become (baseline, observed) pairs measured
            # at the same nodes; skip negatives lacking a baseline.
            befores, afters, pos_idx = [], [], []
            keep = []          # event indices with a valid baseline window
            for i, pair in enumerate(pairs):
                if pair.pos_before is None:
                    continue
                keep.append(i)
                pos_idx.append(len(befores))
                befores.append(pair.pos_before)
                afters.append(pair.pos)
                for name, w in pair.negs.items():
                    nb = pair.negs_before.get(name)
                    if nb is not None:
                        befores.append(nb)
                        afters.append(w)
            if len(keep) < self._align_min_events:
                return None
            if len(keep) != len(pairs):
                z_e = z_e[keep]
            z_w = self.model.align_window_encoder.encode_batch(
                befores, afters)
            return symmetric_infonce(z_e, z_w, pos_idx,
                                     tau=self._align_tau)
        windows, pos_idx = [], []
        for pair in pairs:
            pos_idx.append(len(windows))
            windows.append(pair.pos)
            windows.extend(pair.negs.values())
        z_w = self.model.align_window_encoder.encode_batch(windows)
        return symmetric_infonce(z_e, z_w, pos_idx, tau=self._align_tau)
