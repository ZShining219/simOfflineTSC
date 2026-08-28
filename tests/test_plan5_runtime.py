import copy
import random

import gym
import numpy as np
import pytest
import torch

from plan5.agent import Plan5CTXDDQNAgent, Plan5DDQNAgent, nested_initialize
from plan5.checkpoint import (
    load_checkpoint, restore_checkpoint, save_checkpoint,
)
from plan5.checkpoint_index import CheckpointIndex
from plan5.ppo_adapter import Plan5PPOAdapter
from plan5.runtime import Plan5FormalRunner
from plan5.validation import validate_checkpoint_binding
from sequential.core import canonical_digest
from sequential.io import sha256_file


class FakeIntersection:
    def __init__(self):
        self.id = 'intersection_1_1'


class FakeGenerator:
    def __init__(self, value):
        self.value = np.asarray(value)
    def generate(self):
        return np.array(self.value, copy=True)


class FakeWorld:
    plan5_context_enabled = True
    def __init__(self):
        self.intersection = FakeIntersection()
        self.steps = 0
        self.context_events = []
        self.plan5_boundary_entry_mapping = {'edge': {'direction': 'North'}}
    def reset(self):
        self.steps = 0; self._reset_plan5_entry_tracking()
    def step(self, action): self.steps += 1
    def _reset_plan5_entry_tracking(self): self.context_events = []
    def get_plan5_arrival_context(self):
        return {'time_s': self.steps, 'window_s': 60.0,
                'counts': [len(self.context_events), 0, 0, 0],
                'rates': [len(self.context_events) / 60.0, 0, 0, 0]}
    def plan5_context_state_dict(self):
        return {'entry_events': list(self.context_events),
                'all_entry_events': list(self.context_events),
                'seen_departed_vehicles': [],
                'boundary_entry_mapping': copy.deepcopy(
                    self.plan5_boundary_entry_mapping)}
    def load_plan5_context_state_dict(self, state):
        assert state['boundary_entry_mapping'] == self.plan5_boundary_entry_mapping
        self.context_events = list(state['entry_events'])


def binding(world, rank):
    return {
        'world': world, 'inter': world.intersection,
        'ob_generator': FakeGenerator(np.zeros(8, dtype=np.float32)),
        'phase_generator': FakeGenerator([0]),
        'reward_generator': FakeGenerator([-1.0]),
        'queue_generator': FakeGenerator(np.zeros(8)),
        'delay_generator': FakeGenerator([0.0]),
        'signature': {
            'intersection_id': world.intersection.id,
            'incoming_lane_mapping': tuple(f'lane_{i}' for i in range(8)),
            'phase_action_mapping': tuple(f'phase_{i}' for i in range(8)),
            'state_dim': 8, 'action_dim': 8,
        },
    }


MODEL = {
    'phase': True, 'one_hot': True, 'gamma': .95, 'grad_clip': 5.0,
    'epsilon_decay': .995, 'epsilon_min': .01, 'epsilon': 1.0,
    'learning_rate': .001,
}
TRAINER = {
    'batch_size': 2, 'learning_start': 0, 'buffer_size': 20,
    'target_update_interval': 10, 'steps': 20, 'action_interval': 10,
    'update_model_rate': 1,
}


def make_ddqn(seed=1):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return Plan5DDQNAgent(
        FakeWorld(), 0, MODEL, TRAINER, binding_factory=binding,
    )


def make_ctx(seed=1):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return Plan5CTXDDQNAgent(
        FakeWorld(), 0, MODEL, TRAINER, binding_factory=binding,
    )


def append(agent, index, terminated=False, truncated=False):
    state = np.full((1, 8), index, dtype=np.float32)
    phase = np.asarray([index % 8], dtype=np.int8)
    if isinstance(agent, Plan5CTXDDQNAgent):
        state = np.concatenate([state, np.zeros((1, 4), dtype=np.float32)], 1)
    agent.remember(
        state, phase, np.asarray([index % 8]), np.asarray(-1.0),
        state + 1, phase, local_episode=1, decision_index=index,
        terminated=terminated, truncated=truncated,
    )


def test_target_sync_is_tenth_successful_gradient():
    agent = make_ddqn(); agent.begin_stage(1, 'S2', 'clear')
    append(agent, 1); append(agent, 2, truncated=True)
    for update in range(1, 11):
        result = agent.successful_gradient_update()
        assert result['target_synced'] is (update == 10)
    assert agent.counters.gradient_updates == 10
    assert agent.counters.target_updates == 1


def test_ddqn_random_warmup_uses_retained_global_counter_boundary():
    agent = make_ddqn()
    agent.learning_start = 1000
    agent.begin_stage(1, 'S2', 'clear')
    assert agent.should_use_random_warmup_action()
    agent.counters.global_decision_step = 1000
    assert agent.should_use_random_warmup_action()
    agent.counters.global_decision_step = 1001
    assert not agent.should_use_random_warmup_action()
    agent.begin_stage(2, 'S3', 'clear')
    assert not agent.should_use_random_warmup_action()


def test_nested_initialization_is_exact_and_context_columns_zero():
    source = make_ddqn(7); target = make_ctx(8)
    raw = np.arange(64, dtype=np.float32).reshape(4, 16)
    manifest = nested_initialize(source, target, raw)
    assert manifest['raw_q_exact_equal']
    assert torch.equal(
        target.model.dense_1.weight[:, 16:],
        torch.zeros_like(target.model.dense_1.weight[:, 16:]),
    )
    expected_layers = {
        'dense_1.bias', 'dense_1.weight', 'dense_2.bias',
        'dense_2.weight', 'dense_3.bias', 'dense_3.weight',
    }
    for field in (
        'ddqn_online_layer_digests', 'ctxddqn_online_layer_digests',
        'ddqn_target_layer_digests', 'ctxddqn_target_layer_digests',
    ):
        assert set(manifest[field]) == expected_layers


def test_ctx_clear_resets_context_only_and_checkpoint_roundtrip(tmp_path):
    agent = make_ctx(); agent.world.context_events = [{'time_s': 1}]
    append(agent, 1)
    model_before = canonical_digest(agent.model.state_dict())
    agent.begin_stage(2, 'S2', 'clear')
    assert len(agent.replay.records) == 0 and agent.world.context_events == []
    assert canonical_digest(agent.model.state_dict()) == model_before
    path = tmp_path / 'ctx.pt'
    payload = save_checkpoint(
        agent, path, {'algorithm_id': 'CTXDDQN', 'training_seed': 0,
                      'scene': 'S2'}, episode=2, stage=1,
    )
    clone = make_ctx(99)
    restore_checkpoint(
        clone, path, {'algorithm_id': 'CTXDDQN', 'training_seed': 0},
    )
    assert canonical_digest(clone.full_state_dict()) == canonical_digest(
        agent.full_state_dict())
    assert load_checkpoint(path)['canonical_state_digest'] == payload[
        'canonical_state_digest'
    ]


def test_ctx_rebind_preserves_raw16_plus_ctx4_width_and_training_state():
    agent = make_ctx(7)
    model_before = canonical_digest(agent.model.state_dict())
    optimizer_before = canonical_digest(agent.optimizer.state_dict())
    for _ in range(2):
        target_world = FakeWorld()
        result = agent.rebind_environment(target_world)
        assert agent.world is target_world
        assert agent.raw_ob_length == 16
        assert agent.raw_lane_dim == 8
        assert agent.ob_length == 20
        assert result == {
            'state_dim': 20,
            'action_dim': 8,
            'training_state_preserved': True,
            'context_history_reset': True,
        }
        observation = agent.get_ob()
        assert observation.shape == (1, 12)
        assert agent._ctx_feature(observation, agent.get_phase()).shape == (1, 20)
        assert canonical_digest(agent.model.state_dict()) == model_before
        assert canonical_digest(agent.optimizer.state_dict()) == optimizer_before


def test_checkpoint_binding_rejects_wrong_sha_and_scene(tmp_path):
    agent = make_ddqn()
    path = tmp_path / 'ddqn.pt'
    save_checkpoint(
        agent, path,
        {'algorithm_id': 'DDQN', 'training_seed': 0, 'scene': 'S3'},
        episode=100, stage=1,
    )
    row = {
        'run_type': 'TRANSITION', 'algorithm_id': 'DDQN',
        'training_seed': 0, 'source_scene': 'S3',
        'source_checkpoint_path': str(path),
        'source_checkpoint_sha256': 'bad-sha',
    }
    with pytest.raises(ValueError, match='SHA mismatch'):
        validate_checkpoint_binding(row)
    row['source_checkpoint_sha256'] = sha256_file(path)
    row['source_scene'] = 'S4'
    with pytest.raises(ValueError, match='scene mismatch'):
        validate_checkpoint_binding(row)


def test_formal_full_checkpoint_commits_complete_resumable_episode(tmp_path):
    agent = make_ddqn(11)
    runner = Plan5FormalRunner.__new__(Plan5FormalRunner)
    runner.agent = agent
    runner.row = {
        'logical_run_id': 'P5-ANCHOR-DDQN-S2-SD0',
        'algorithm_id': 'DDQN', 'training_seed': 0,
        'run_type': 'ANCHOR', 'scene': 'S2',
        'config_sha256': 'config', 'source_commit': 'source',
    }
    runner.attempt_dir = tmp_path
    runner.index = CheckpointIndex(tmp_path / 'checkpoint_index.json')
    runner.episode_summaries = [{'episode': 100}]
    runner.evaluations = [{'episode': 100, 'scene': 'S2'}]
    runner.probe_records = [{'episode': 100, 'split': 'main'}]

    path = runner._save_full_checkpoint(100)
    payload = load_checkpoint(path)
    assert payload['episode'] == 100
    assert payload['extra_state'] == {
        'completed_episode': 100,
        'episode_summaries': runner.episode_summaries,
        'evaluations': runner.evaluations,
        'probe_records': runner.probe_records,
    }
    assert runner.index.validate() == {'valid': True, 'count': 1}
    clone = make_ddqn(99)
    restored = restore_checkpoint(
        clone, path,
        {'logical_run_id': runner.row['logical_run_id']},
    )
    assert restored['extra_state']['completed_episode'] == restored['episode']
    assert canonical_digest(clone.full_state_dict()) == canonical_digest(
        agent.full_state_dict()
    )


class FakePPO(Plan5PPOAdapter):
    def _bind(self, world):
        self.world = world; self.inter = world.intersection
        self.intersection = world.intersection
        self.ob_generator = FakeGenerator(np.zeros(8, dtype=np.float32))
        self.phase_generator = FakeGenerator([0])
        self.reward_generator = FakeGenerator([-1.0])
        self.queue = FakeGenerator(np.zeros(8)); self.delay = FakeGenerator([0.0])
        self.action_space = gym.spaces.Discrete(8)
        self.raw_lane_dim = 8; self.ob_length = 16


def test_pfrl_ppo_360_truncation_bootstraps_and_empties_rollout():
    agent = FakePPO(FakeWorld(), 0, 2.5e-4, .001, training_seed=3)
    captured = {}
    original_update = agent.core._update
    def capture_update(dataset):
        captured['last_transition'] = copy.deepcopy(dataset[-1])
        return original_update(dataset)
    agent.core._update = capture_update
    observation = np.zeros((1, 8), dtype=np.float32)
    phase = np.asarray([0], dtype=np.int8)
    for decision in range(360):
        agent.get_action(observation, phase, test=False)
        result = agent.observe(
            observation, phase, -1.0,
            terminated=False, truncated=decision == 359,
        )
    assert result['minibatch_updates'] == 16
    assert agent.core.n_updates == 16
    assert agent.rollout_updates == 1
    assert agent.assert_rollout_empty()
    final = captured['last_transition']
    assert final['nonterminal'] == 1.0
    assert final['v_teacher'] == pytest.approx(
        final['reward'] + 0.95 * final['next_v_pred']
    )
    state = agent.full_state_dict()
    assert state['memory'] == [] and state['batch_last_state'] == [None]
    assert agent.parameter_ownership() == {
        'shared_parameters': 5248,
        'actor_only_parameters': 520,
        'critic_only_parameters': 65,
        'total_parameters': 5833,
    }


def test_ppo_custom_checkpoint_covers_empty_rollout_and_statistics(tmp_path):
    agent = FakePPO(FakeWorld(), 0, 2.5e-4, .001, training_seed=4)
    path = tmp_path / 'ppo.pt'
    save_checkpoint(
        agent, path,
        {'algorithm_id': 'PPO', 'training_seed': 4, 'run_type': 'smoke'},
        episode=0, stage=1,
    )
    payload = load_checkpoint(path)
    state = payload['agent_state']
    assert state['memory'] == []
    assert state['batch_last_episode'] is None
    assert state['batch_last_state'] is None
    assert set(state['statistics']) == {
        'value_record', 'entropy_record', 'value_loss_record',
        'policy_loss_record',
    }
