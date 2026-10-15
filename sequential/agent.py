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


class DuelingDQNNet(nn.Module):
    """Small dueling head with the same input/action contract as ``DQNNet``."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
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


def normalize_algorithm_id(value):
    value = str(value or 'independent_dqn').strip().lower()
    aliases = {
        'dqn': 'independent_dqn',
        'independent': 'independent_dqn',
        'ddqn': 'double_dqn',
        'dueling_ddqn': 'dueling_double_dqn',
    }
    value = aliases.get(value, value)
    allowed = {'independent_dqn', 'double_dqn', 'dueling_double_dqn'}
    if value not in allowed:
        raise ValueError(f'Unsupported sequential algorithm: {value}')
    return value


def build_q_network(algorithm_id, input_dim, output_dim):
    algorithm_id = normalize_algorithm_id(algorithm_id)
    if algorithm_id == 'dueling_double_dqn':
        return DuelingDQNNet(input_dim, output_dim)
    return DQNNet(input_dim, output_dim)


def _readonly_array(value):
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


class SequentialDQNAgent:
    """DQN implementation isolated from the repository's default Online path."""

    def __init__(self, world, rank, model_config, trainer_config,
                 binding_factory=None, algorithm_id='independent_dqn'):
        self.rank = int(rank)
        self.algorithm_id = normalize_algorithm_id(algorithm_id)
        self.target_operator = (
            'double_dqn' if self.algorithm_id != 'independent_dqn'
            else 'plain_dqn'
        )
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
        self.model = build_q_network(
            self.algorithm_id, self.ob_length, self.action_space.n,
        )
        self.target_model = build_q_network(
            self.algorithm_id, self.ob_length, self.action_space.n,
        )
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
        # Built lazily on the first transition: lane lengths come from the
        # live SUMO connection, which is not loaded until the first reset.
        self._reward_shaper = None
        self._reward_shaper_pending = (
            str(self.trainer_config.get('reward_mode', 'none')).lower()
            not in ('none', 'b0'))
        self._shaper_episode = None
        # Adjudication layer (B'): records store raw r0 + scoring features;
        # the effective reward is computed per sampled batch, identically
        # across online/offline/expert legs.  Mutually exclusive with the
        # legacy remember()-time shaping modes b1/b2.
        from .adjudication import build_adjudicator
        self.adjudicator = build_adjudicator(
            self.trainer_config, self.action_space.n)
        if (self.adjudicator is not None
                and self._reward_shaper_pending):
            raise ValueError(
                'adjudication and legacy reward_mode (b1/b2) are mutually '
                'exclusive shaping schemes')
        self._adjudication_spec = self.trainer_config.get('adjudication')
        self._adj_ctx = None
        self._adj_tracker = None
        self._adj_episode = None

    def _build_reward_shaper(self):
        """Build the optional HITL reward shaper (spec §6.2 B channel).

        trainer_config keys (all optional; default keeps r0-only behavior):
          reward_mode: 'none'|'b0'|'b1'|'b2'
          reward_lambda: shaping amplitude (spec default 0.1)
          reward_calibration_path: JSON with frozen mu/sigma for 'h'/'f_hat'
          reward_model_path: frozen PreferenceRewardNet checkpoint (B2 only)
          reward_ctx: optional explicit {'lane_capacities': [...],
                      'phase_served_lanes': [[...]]} overriding the live
                      world derivation (used by tests and worlds lacking
                      lane-length APIs).
        """
        mode = str(self.trainer_config.get('reward_mode', 'none')).lower()
        if mode in ('none', 'b0'):
            return None
        from .reward_shaping import (
            RewardShaper, lane_capacities_from_lengths, load_calibration)
        ctx = self.trainer_config.get('reward_ctx')
        if ctx is not None:
            capacities = list(ctx['lane_capacities'])
            served = [list(s) for s in ctx['phase_served_lanes']]
        else:
            lane_ids = list(self.environment_signature['incoming_lane_mapping'])
            lane_lengths = [
                float(self.world.eng.lane.getLength(l)) for l in lane_ids
            ]
            capacities = lane_capacities_from_lengths(lane_lengths)
            lane_index = {l: i for i, l in enumerate(lane_ids)}
            served = [
                [lane_index[l] for l in self.inter.phase_available_startlanes[a]
                 if l in lane_index]
                for a in range(self.action_space.n)
            ]
        calib_path = self.trainer_config.get('reward_calibration_path')
        calibration = load_calibration(calib_path) if calib_path else {}
        return RewardShaper(
            mode, capacities, served,
            float(self.trainer_config.get('action_interval', 10.0)),
            self.action_space.n,
            lam=float(self.trainer_config.get('reward_lambda', 0.1)),
            calibration=calibration,
            model_path=self.trainer_config.get('reward_model_path'),
        )

    def _shaped_reward(self, state, phase, action, reward, next_state,
                       next_phase, local_episode):
        if local_episode != self._shaper_episode:
            self._reward_shaper.tracker.reset()
            self._shaper_episode = local_episode
        lane_counts = np.asarray(next_state, dtype=np.float64).reshape(-1)
        n_lanes = self._reward_shaper.tracker.n_lanes
        occupancy = lane_counts[:n_lanes]
        try:
            queue_l = np.asarray(
                self.queue.generate(), dtype=np.float64).reshape(-1)
            if queue_l.size != n_lanes:
                queue_l = occupancy
        except Exception:
            queue_l = occupancy
        obs = np.asarray(state, dtype=np.float64).reshape(-1)
        if self.phase and self.one_hot:
            obs = np.concatenate([
                obs, agent_utils.idx2onehot(
                    np.asarray(phase).reshape(-1), self.action_space.n)[0],
            ])
            next_obs = np.concatenate([
                lane_counts, agent_utils.idx2onehot(
                    np.asarray(next_phase).reshape(-1),
                    self.action_space.n)[0],
            ])
        else:
            next_obs = lane_counts
        r_train, aux = self._reward_shaper.shape(
            float(np.asarray(reward).reshape(-1)[0]), obs,
            int(np.asarray(action).reshape(-1)[0]), next_obs,
            queue_l, occupancy, int(np.asarray(action).reshape(-1)[0]),
        )
        return r_train, aux

    def _build_adj_ctx(self):
        """Lane capacities + phase-served map for adjudication.

        Explicit override via trainer_config['adjudication_ctx'] =
        {'lane_capacities': [...], 'phase_served_lanes': [[...]]}
        (tests / worlds lacking lane-length APIs); otherwise derived
        from the live world like the reward shaper."""
        from .adjudication import build_ctx_from_world
        ctx = self.trainer_config.get('adjudication_ctx')
        if ctx is not None:
            return {
                'lane_capacities': list(ctx['lane_capacities']),
                'phase_served_lanes': [
                    list(v) for v in ctx['phase_served_lanes']],
                'action_interval': float(
                    self.trainer_config.get('action_interval', 10.0)),
                'T_u': float(ctx.get('T_u', 60.0)),
            }
        topo_path = self.trainer_config.get('adjudication_ctx_path')
        if topo_path:
            from .adjudication import load_topology_ctx
            ctx = load_topology_ctx(
                topo_path, float(
                    self.trainer_config.get('action_interval', 10.0)))
            ctx['action_interval'] = float(
                self.trainer_config.get('action_interval', 10.0))
            return ctx
        ctx = build_ctx_from_world(
            self.world, self.inter, self.environment_signature,
            self.action_space.n)
        ctx['action_interval'] = float(
            self.trainer_config.get('action_interval', 10.0))
        ctx['T_u'] = 60.0
        return ctx

    def _adj_features(self, action, next_state, local_episode):
        """Advance the service-gap tracker and return this transition's
        adjudication feature dict {q_ratio, gap_norm}."""
        from .reward_shaping import ServiceGapTracker
        if self._adj_ctx is None:
            self._adj_ctx = self._build_adj_ctx()
            self._adj_tracker = ServiceGapTracker(
                self._adj_ctx['phase_served_lanes'],
                len(self._adj_ctx['lane_capacities']),
                self._adj_ctx['action_interval'])
        episode_key = (self.current_stage_index, local_episode)
        if episode_key != self._adj_episode:
            self._adj_tracker.reset()
            self._adj_episode = episode_key
        lane_counts = np.asarray(next_state, dtype=np.float64).reshape(-1)
        n_lanes = self._adj_tracker.n_lanes
        occupancy = lane_counts[:n_lanes]
        gaps = self._adj_tracker.advance(
            int(np.asarray(action).reshape(-1)[0]), occupancy)
        return {
            'q_ratio': (
                occupancy / np.asarray(
                    self._adj_ctx['lane_capacities'], dtype=np.float64)
            ).tolist(),
            'gap_norm': np.minimum(
                gaps / self._adj_ctx['T_u'], 2.0).tolist(),
        }

    def _apply_adjudication(self, records, rewards):
        """r_eff = r_base + lam*z for a list of online ReplayRecords.

        Reads each record's stored adjudication features (written by
        remember() under adjudication mode); records missing features
        contribute z=0 and are counted for the audit."""
        adjudicator = getattr(self, 'adjudicator', None)
        if adjudicator is None:
            return rewards, None
        adjusted = np.asarray(rewards, dtype=np.float64).copy()
        zs = np.zeros(len(records), dtype=np.float64)
        missing = 0
        judge_sums = {}
        judge_counts = {}
        fallback_count = 0
        for i, record in enumerate(records):
            aux = getattr(record.metadata, 'reward_aux', None) or {}
            feat = aux.get('adjudication')
            if not feat:
                missing += 1
                continue
            obs = np.asarray(
                record.payload.state, dtype=np.float64).reshape(-1)
            next_state = np.asarray(
                record.payload.next_state, dtype=np.float64).reshape(-1)
            if self.phase:
                if self.one_hot:
                    obs = np.concatenate([
                        obs, agent_utils.idx2onehot(
                            np.asarray(record.payload.phase).reshape(-1),
                            self.action_space.n)[0]])
                    next_obs = np.concatenate([
                        next_state, agent_utils.idx2onehot(
                            np.asarray(record.payload.next_phase).reshape(-1),
                            self.action_space.n)[0]])
                else:
                    obs = np.concatenate([
                        obs, np.asarray(record.payload.phase).reshape(-1)])
                    next_obs = np.concatenate([
                        next_state,
                        np.asarray(record.payload.next_phase).reshape(-1)])
            else:
                next_obs = next_state
            z, detail = adjudicator.z(
                obs, int(np.asarray(record.payload.action).reshape(-1)[0]),
                next_obs, feat['q_ratio'], feat['gap_norm'])
            zs[i] = z
            adjusted[i] = adjudicator.effective_reward(
                float(rewards[i]), z)
            fallback_count += len(detail['fallback'])
            for name, entry in detail['judges'].items():
                judge_sums[name] = judge_sums.get(name, 0.0) + entry['z']
                judge_counts[name] = judge_counts.get(name, 0) + 1
        stats = {
            'z_mean': float(np.mean(zs)), 'z_std': float(np.std(zs)),
            'missing_features': int(missing),
            'judge_fallbacks': int(fallback_count),
            'judge_z_mean': {
                k: judge_sums[k] / judge_counts[k] for k in judge_sums},
        }
        return torch.as_tensor(adjusted, dtype=torch.float32), stats

    @classmethod
    def from_parent(cls, world, rank, model_config, trainer_config,
                    parent_manifest, binding_factory=None,
                    skip_replay_digest=False, algorithm_id='independent_dqn'):
        agent = cls(
            world, rank, model_config, trainer_config,
            binding_factory=binding_factory,
            algorithm_id=algorithm_id,
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
        aux = None
        if self._reward_shaper is None and self._reward_shaper_pending:
            self._reward_shaper = self._build_reward_shaper()
            self._reward_shaper_pending = False
        if self._reward_shaper is not None:
            reward, aux = self._shaped_reward(
                state, phase, action, reward, next_state, next_phase,
                local_episode,
            )
        if getattr(self, 'adjudicator', None) is not None:
            aux = dict(aux or {})
            aux['adjudication'] = self._adj_features(
                action, next_state, local_episode)
            aux['r_base'] = float(np.asarray(reward).reshape(-1)[0])
        payload = TrainingPayload(
            _readonly_array(state), _readonly_array(phase),
            _readonly_array(action), _readonly_array(reward),
            _readonly_array(next_state), _readonly_array(next_phase),
            bool(terminated), bool(truncated), aux,
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
            reward_aux=aux,
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
        rewards, _adj_stats = self._apply_adjudication(records, rewards)
        with torch.no_grad():
            target = rewards + self.gamma * self._bootstrap_values(next_state)
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
            'adjudication': (
                {'manifest_digest': self.adjudicator.digest, 'online': _adj_stats}
                if _adj_stats else None),
            'target_synced': target_synced,
        }

    def _bootstrap_values(self, next_state):
        """Return the target-network value under the frozen algorithm operator."""
        target_operator = getattr(self, 'target_operator', 'plain_dqn')
        if target_operator == 'plain_dqn':
            return torch.max(self.target_model(next_state), dim=1)[0]
        next_actions = self.model(next_state).argmax(dim=1, keepdim=True)
        return self.target_model(next_state).gather(1, next_actions).squeeze(1)

    def full_state_dict(self):
        algorithm_id = normalize_algorithm_id(
            getattr(self, 'algorithm_id', 'independent_dqn')
        )
        target_operator = getattr(
            self, 'target_operator',
            'double_dqn' if algorithm_id != 'independent_dqn' else 'plain_dqn',
        )
        return {
            'schema_version': 1,
            'algorithm_id': algorithm_id,
            'target_operator': target_operator,
            'architecture_name': (
                'dueling_mlp' if algorithm_id == 'dueling_double_dqn'
                else 'dqn_mlp'
            ),
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
            'adjudication': (
                {
                    'manifest_digest': self.adjudicator.digest,
                    'spec': getattr(self, '_adjudication_spec', None),
                }
                if getattr(self, 'adjudicator', None) is not None else None
            ),
            'rng_state': capture_rng_state(),
        }

    def load_full_state_dict(self, state):
        if state.get('schema_version') != 1:
            raise ValueError('Unsupported Sequential agent state schema')
        state_algorithm = normalize_algorithm_id(
            state.get('algorithm_id', 'independent_dqn')
        )
        expected_algorithm = normalize_algorithm_id(
            getattr(self, 'algorithm_id', 'independent_dqn')
        )
        if state_algorithm != expected_algorithm:
            raise ValueError(
                'Sequential checkpoint algorithm mismatch: '
                f'{state_algorithm} != {expected_algorithm}'
            )
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
        saved_adj = state.get('adjudication')
        adjudicator = getattr(self, 'adjudicator', None)
        if saved_adj is not None:
            if adjudicator is None:
                raise ValueError(
                    'Checkpoint was trained under an adjudication manifest '
                    'but this run configures none — the reward '
                    'interpretation would silently change')
            if adjudicator.digest != saved_adj.get('manifest_digest'):
                raise ValueError(
                    'Adjudication manifest changed since checkpoint: '
                    f'{saved_adj.get("manifest_digest")} != '
                    f'{adjudicator.digest}; judge changes require a new '
                    'stage, not a silent resume')
        elif adjudicator is not None:
            raise ValueError(
                'This run configures adjudication but the checkpoint was '
                'trained without it — the reward interpretation would '
                'silently change')
        rng = state['rng_state']
        random.setstate(rng['python_random_state'])
        np.random.set_state(rng['numpy_random_state'])
        torch.set_rng_state(rng['torch_cpu_rng_state'])
        if torch.cuda.is_available() and rng['torch_cuda_rng_states']:
            torch.cuda.set_rng_state_all(rng['torch_cuda_rng_states'])
        self.last_observation = None
        return self
