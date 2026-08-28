"""Formal Plan5 DDQN-family adapters on the shared Sequential runtime."""
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from agent import utils as agent_utils
from agent.dqn import DQNNet
from sequential.agent import SequentialDQNAgent
from sequential.core import TargetUpdateScheduler, canonical_digest, capture_rng_state
from .algorithms import dqn_full_vector_target


class Plan5DDQNAgent(SequentialDQNAgent):
    algorithm_id = 'DDQN'

    def __init__(self, world, rank, model_config, trainer_config,
                 binding_factory=None):
        super().__init__(
            world, rank, model_config, trainer_config,
            binding_factory=binding_factory,
        )
        # Fresh Plan5 agents synchronize after successful gradients 10, 20, ...
        self.target_scheduler = TargetUpdateScheduler(
            int(trainer_config['target_update_interval']),
            int(trainer_config['target_update_interval']),
        )

    def begin_stage(self, stage_index, network, policy='clear'):
        if policy != 'clear':
            raise ValueError('Plan5 DDQN-family Stage2 only authorizes Clear')
        result = super().begin_stage(stage_index, network, policy)
        if result['replay_size'] != 0:
            raise RuntimeError('Plan5 Clear did not empty replay')
        return result

    def should_use_random_warmup_action(self):
        # Match the formal DQN pre-decision boundary exactly: fresh anchors use
        # agent.sample() through global decision 1000.  Stage2 retains the
        # Stage1 global counter, so Clear does not re-enter this branch.
        return self.counters.global_decision_step <= self.learning_start

    def _plan5_batchwise(self, records):
        state, next_state, rewards, actions = self._batchwise(records)
        terminated = torch.as_tensor(
            [item.payload.terminated for item in records], dtype=torch.bool,
        )
        truncated = torch.as_tensor(
            [item.payload.truncated for item in records], dtype=torch.bool,
        )
        return state, next_state, rewards, actions, terminated, truncated

    def successful_gradient_update(self):
        if not self.is_update_ready():
            return None
        records = self.replay.sample(self.batch_size, random)
        state, next_state, rewards, actions, terminated, truncated = (
            self._plan5_batchwise(records)
        )
        with torch.no_grad():
            online_next = self.model(next_state)
            selected_actions = torch.argmax(online_next, dim=1)
            selected_values = self.target_model(next_state).gather(
                1, selected_actions[:, None]
            ).squeeze(1)
            # SUMO's 3600-second horizon is a truncation and still bootstraps.
            target = rewards.reshape(-1) + self.gamma * (
                ~terminated
            ).float() * selected_values
        predicted = self.model(state)
        target_full = dqn_full_vector_target(predicted, actions, target)
        # Keep the original formal DQN's full 8-vector mean-MSE reduction.
        # Only the selected-action target operator changes to Double-DQN.
        loss = self.criterion(predicted, target_full)
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
            'truncated_count': int(truncated.sum().item()),
            'terminated_count': int(terminated.sum().item()),
            'target_operator': 'double_dqn',
        }

    def full_state_dict(self):
        state = super().full_state_dict()
        state['algorithm_id'] = self.algorithm_id
        state['target_operator'] = 'double_dqn'
        return state

    def load_full_state_dict(self, state):
        if state.get('algorithm_id') != self.algorithm_id:
            raise ValueError('Plan5 algorithm checkpoint identity mismatch')
        if state.get('target_operator') != 'double_dqn':
            raise ValueError('Plan5 target operator checkpoint mismatch')
        return super().load_full_state_dict(state)


class Plan5CTXDDQNAgent(Plan5DDQNAgent):
    algorithm_id = 'CTXDDQN'
    context_dim = 4

    def __init__(self, world, rank, model_config, trainer_config,
                 binding_factory=None):
        if not getattr(world, 'plan5_context_enabled', False):
            raise ValueError('CTXDDQN requires SUMO Plan5 causal entry tracking')
        super().__init__(
            world, rank, model_config, trainer_config,
            binding_factory=binding_factory,
        )
        self.raw_ob_length = self.ob_length
        self.raw_lane_dim = self.raw_ob_length - self.action_space.n
        if not self.phase or not self.one_hot or self.raw_lane_dim <= 0:
            raise ValueError('Plan5 CTXDDQN requires raw lane counts + phase one-hot')
        self.ob_length = self.raw_ob_length + self.context_dim
        self.model = DQNNet(self.ob_length, self.action_space.n)
        self.target_model = DQNNet(self.ob_length, self.action_space.n)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.RMSprop(
            self.model.parameters(), lr=float(model_config['learning_rate']),
            alpha=0.9, centered=False, eps=1e-7,
        )
        self.environment_signature['raw_model_input_dim'] = self.raw_ob_length
        self.environment_signature['context_dim'] = self.context_dim
        self.environment_signature['model_input_dim'] = self.ob_length
        self.last_context = None

    def get_ob(self):
        raw = np.asarray([self.ob_generator.generate()], dtype=np.float32)
        context = self.world.get_plan5_arrival_context()
        rates = np.asarray([context['rates']], dtype=np.float32)
        self.last_context = copy.deepcopy(context)
        observation = np.concatenate([raw, rates], axis=1)
        self.last_observation = observation
        return observation

    def _ctx_feature(self, observation, phase):
        observation = np.asarray(observation, dtype=np.float32)
        lane = observation[:, :self.raw_lane_dim]
        context = observation[:, self.raw_lane_dim:]
        if context.shape[1] != self.context_dim:
            raise ValueError('CTXDDQN observation context dimension mismatch')
        phase_feature = agent_utils.idx2onehot(
            phase, self.action_space.n
        ).astype(np.float32)
        return np.concatenate([lane, phase_feature, context], axis=1)

    def get_action(self, observation, phase, test=False):
        if not test and np.random.rand() <= self.epsilon:
            return self.sample()
        feature = self._ctx_feature(observation, phase)
        with torch.no_grad():
            values = self.model(torch.as_tensor(feature, dtype=torch.float32))
        return np.argmax(values.cpu().numpy(), axis=1)

    def _batchwise(self, records):
        state = np.concatenate([
            self._ctx_feature(item.payload.state, item.payload.phase)
            for item in records
        ])
        next_state = np.concatenate([
            self._ctx_feature(item.payload.next_state, item.payload.next_phase)
            for item in records
        ])
        rewards = torch.as_tensor(
            np.asarray([item.payload.reward for item in records]),
            dtype=torch.float32,
        )
        actions = torch.as_tensor(
            np.asarray([item.payload.action for item in records]),
            dtype=torch.long,
        )
        return (
            torch.as_tensor(state, dtype=torch.float32),
            torch.as_tensor(next_state, dtype=torch.float32), rewards, actions,
        )

    def begin_stage(self, stage_index, network, policy='clear'):
        result = super().begin_stage(stage_index, network, policy)
        self.world._reset_plan5_entry_tracking()
        self.last_context = None
        result['context_history_size'] = 0
        return result

    def training_state_digests(self):
        result = super().training_state_digests()
        result['context_state_digest'] = canonical_digest(
            self.world.plan5_context_state_dict()
        )
        result['last_context_digest'] = canonical_digest(self.last_context)
        return result

    def rebind_environment(self, world, expected_signature=None):
        before = self.training_state_digests()
        binding = self._make_binding(world)
        raw_signature = binding['signature']
        raw_ob_length = (
            int(raw_signature['state_dim'])
            + int(raw_signature['action_dim'])
            if self.phase and self.one_hot
            else int(raw_signature['state_dim']) + (1 if self.phase else 0)
        )
        if raw_ob_length != self.raw_ob_length:
            raise ValueError('CTXDDQN raw16 environment changed')
        if tuple(raw_signature['phase_action_mapping']) != tuple(
            self.environment_signature['phase_action_mapping']
        ):
            raise ValueError('CTXDDQN phase/action mapping changed')
        if tuple(raw_signature['incoming_lane_mapping']) != tuple(
            self.environment_signature['incoming_lane_mapping']
        ):
            raise ValueError('CTXDDQN lane mapping changed')
        self._install_binding(binding)
        # _install_binding restores the raw observation width.  Preserve that
        # width explicitly before adding ctx4; using the previous augmented
        # self.ob_length here would grow 20 -> 24 on the first Stage2 rebind.
        self.raw_ob_length = raw_ob_length
        self.raw_lane_dim = self.raw_ob_length - self.action_space.n
        self.ob_length = self.raw_ob_length + self.context_dim
        self.environment_signature['raw_model_input_dim'] = self.raw_ob_length
        self.environment_signature['context_dim'] = self.context_dim
        self.environment_signature['model_input_dim'] = self.ob_length
        if expected_signature is not None \
                and self.environment_signature != expected_signature:
            raise ValueError('CTXDDQN environment signature mismatch')
        self.last_observation = None
        self.last_context = None
        after = self.training_state_digests()
        allowed = {
            'environment_signature_digest', 'context_state_digest',
            'last_context_digest',
        }
        changed = {
            key for key in before if before[key] != after[key]
        } - allowed
        if changed:
            raise RuntimeError(
                f'CTXDDQN environment rebind changed training state: {changed}'
            )
        return {
            'state_dim': self.ob_length, 'action_dim': self.action_space.n,
            'training_state_preserved': True,
            'context_history_reset': True,
        }

    def full_state_dict(self):
        state = super().full_state_dict()
        state['context_state'] = self.world.plan5_context_state_dict()
        state['last_context'] = copy.deepcopy(self.last_context)
        return state

    def load_full_state_dict(self, state):
        result = super().load_full_state_dict(state)
        self.world.load_plan5_context_state_dict(state['context_state'])
        self.last_context = copy.deepcopy(state.get('last_context'))
        return result


def nested_initialize(ddqn, ctxddqn, raw_batch):
    """Deterministically embed one DDQN initialization into CTXDDQN."""
    if not isinstance(ddqn, Plan5DDQNAgent) or isinstance(
        ddqn, Plan5CTXDDQNAgent
    ):
        raise TypeError('Nested source must be Plan5 DDQN')
    if not isinstance(ctxddqn, Plan5CTXDDQNAgent):
        raise TypeError('Nested target must be Plan5 CTXDDQN')
    with torch.no_grad():
        for target_net, source_net in (
            (ctxddqn.model, ddqn.model),
            (ctxddqn.target_model, ddqn.target_model),
        ):
            target_net.dense_1.weight.zero_()
            target_net.dense_1.weight[:, :ddqn.ob_length].copy_(
                source_net.dense_1.weight
            )
            target_net.dense_1.bias.copy_(source_net.dense_1.bias)
            target_net.dense_2.load_state_dict(source_net.dense_2.state_dict())
            target_net.dense_3.load_state_dict(source_net.dense_3.state_dict())
    if ddqn.optimizer.state or ctxddqn.optimizer.state:
        raise ValueError('Nested initialization requires empty optimizer state')
    raw = torch.as_tensor(raw_batch, dtype=torch.float32)
    augmented = torch.cat([
        raw,
        torch.zeros((raw.shape[0], ctxddqn.context_dim), dtype=raw.dtype),
    ], dim=1)
    with torch.no_grad():
        source_q = ddqn.model(raw)
        target_q = ctxddqn.model(augmented)
    if not torch.equal(source_q, target_q):
        raise AssertionError('Nested initialization Q outputs are not exact equal')
    def layer_digests(model):
        return {
            name: canonical_digest(value)
            for name, value in sorted(model.state_dict().items())
        }
    return {
        'schema_version': 1,
        'ddqn_online_digest': canonical_digest(ddqn.model.state_dict()),
        'ctxddqn_online_digest': canonical_digest(ctxddqn.model.state_dict()),
        'ddqn_target_digest': canonical_digest(ddqn.target_model.state_dict()),
        'ctxddqn_target_digest': canonical_digest(
            ctxddqn.target_model.state_dict()
        ),
        'ddqn_online_layer_digests': layer_digests(ddqn.model),
        'ctxddqn_online_layer_digests': layer_digests(ctxddqn.model),
        'ddqn_target_layer_digests': layer_digests(ddqn.target_model),
        'ctxddqn_target_layer_digests': layer_digests(ctxddqn.target_model),
        'ctx_columns_digest': canonical_digest(
            ctxddqn.model.dense_1.weight[:, ddqn.ob_length:]
        ),
        'raw_q_digest': canonical_digest(source_q),
        'ctx_q_digest': canonical_digest(target_q),
        'raw_q_exact_equal': True,
        'optimizer_states_empty': True,
    }
