import random

import numpy as np
import json
import pytest

from arterial.control import (actions_by_intersection, build_action_mask,
                              stable_intersection_order)
from arterial.experiment import SCENE_ORDERS, stage_overlay, validate_experiment_config
from arterial.replay import (HistoricalPool, LocalTransition,
                             HistoryArchiveWriter, TraceableReplayBuffer, TransitionMetadata,
                             mixed_batch, split_local_transitions)


IDS = tuple(f'intersection_{index}_1' for index in range(1, 7))
SCENES = ('300_0.6', '300_0.3', '700_0.3', '700_0.6')


def transition(scene, intersection, number, source='offline_history'):
    metadata = TransitionMetadata(
        transition_id=f'{scene}:{intersection}:{number}', scene_id=scene,
        intersection_id=intersection, episode_id=1, decision_step=number,
        source=source, training_stage=0, policy_version=0)
    return LocalTransition(
        state=np.asarray([number], dtype=np.float32), phase=np.asarray([0]),
        action=0, reward=0.0,
        next_state=np.asarray([number + 1], dtype=np.float32),
        next_phase=np.asarray([0]), terminated=False, truncated=False,
        metadata=metadata)


def base_config(**updates):
    config = {
        'roadnet': 'arterial_1x6', 'scene_order': 'order_1',
        'num_intersections': 6, 'shared_parameters': True,
        'state_variant': 'base', 'reward_variant': 'original',
        'use_position_encoding': False, 'use_neighbor_summary': False,
        'replay_clear_on_scene_switch': True, 'offline_enabled': True,
        'offline_ratio': 0.5,
        'offline_sampling_strategy': 'scene_intersection_balanced',
        'history_access_mode': 'causal', 'epsilon_mode': 'reset_schedule',
        'training_seed': 0, 'episode_budget': 2,
        'evaluation_interval': 1, 'checkpoint_interval': 1,
    }
    config.update(updates)
    return config


def test_stable_six_intersection_order_and_action_mapping():
    shuffled = ['intersection_6_1', 'intersection_2_1', 'intersection_1_1',
                'intersection_5_1', 'intersection_4_1', 'intersection_3_1']
    assert stable_intersection_order(shuffled) == IDS
    assert actions_by_intersection(IDS, range(6)) == dict(zip(IDS, range(6)))


def test_one_decision_splits_into_six_traceable_transitions():
    records = split_local_transitions(
        IDS, np.zeros((6, 4)), np.zeros(6), np.arange(6), np.arange(6),
        np.ones((6, 4)), np.ones(6), False, False, '300_0.6', 2, 9,
        collector_training_seed=3)
    assert len(records) == 6
    assert [x.metadata.intersection_id for x in records] == list(IDS)
    assert [x.action for x in records] == list(range(6))
    assert all(x.metadata.scene_id == '300_0.6' for x in records)
    assert all(x.metadata.collector_training_seed == 3 for x in records)
    assert len({x.metadata.transition_id for x in records}) == 6
    assert all(':seed3:' in x.metadata.transition_id for x in records)


def test_collector_seed_prevents_cross_run_transition_id_collisions():
    arguments = (
        IDS, np.zeros((6, 4)), np.zeros(6), np.arange(6), np.arange(6),
        np.ones((6, 4)), np.ones(6), False, False, '300_0.6', 2, 9)
    seed0 = split_local_transitions(*arguments, collector_training_seed=0)
    seed1 = split_local_transitions(*arguments, collector_training_seed=1)
    assert {item.metadata.transition_id for item in seed0}.isdisjoint(
        item.metadata.transition_id for item in seed1)


def test_action_mask_supports_heterogeneous_local_spaces():
    mask = build_action_mask([8, 8, 4, 8, 6, 8])
    assert mask.shape == (6, 8)
    assert mask[2].sum() == 4 and not mask[2, 4]
    assert mask[4].sum() == 6 and not mask[4, 6]


@pytest.mark.parametrize('ratio,expected', [
    (0.0, (8, 0)), (0.25, (6, 2)), (0.5, (4, 4)), (0.75, (2, 6))])
def test_mixed_batch_counts(ratio, expected):
    online = TraceableReplayBuffer(20)
    for number in range(10):
        online.append(transition('current', IDS[0], number, 'online_current'))
    offline = HistoricalPool([
        transition('300_0.6', IDS[number % 6], number) for number in range(20)])
    batch = mixed_batch(online, offline, 8, ratio, ['300_0.6'], rng=random.Random(3))
    sources = [source for _, source in batch]
    assert (sources.count('online_current'), sources.count('offline_history')) == expected


def test_zero_offline_ratio_never_accesses_offline_pool():
    online = TraceableReplayBuffer(10)
    for number in range(8):
        online.append(transition('current', IDS[0], number, 'online_current'))
    class Forbidden:
        def sample(self, *args, **kwargs):
            raise AssertionError('offline pool was accessed')
    assert len(mixed_batch(online, Forbidden(), 8, 0.0, rng=random.Random(1))) == 8


def test_causal_history_excludes_current_and_future_and_full_sees_all():
    assert HistoricalPool.visible_scenes(SCENES, 2, 'causal') == SCENES[:2]
    assert HistoricalPool.visible_scenes(SCENES, 2, 'full') == SCENES


def test_scene_intersection_balancing_is_hierarchical_not_volume_weighted():
    records = [transition('300_0.6', IDS[0], i) for i in range(1000)]
    records += [transition('300_0.3', IDS[1], 2000)]
    pool = HistoricalPool(records)
    sampled = pool.sample(400, ['300_0.6', '300_0.3'],
                          rng=random.Random(7))
    low_volume = sum(x.metadata.scene_id == '300_0.3' for x in sampled)
    assert 150 <= low_volume <= 250


def test_stage_overlay_marks_full_history_as_noncausal_upper_bound():
    overlay = stage_overlay(base_config(history_access_mode='full'), 1,
                            history_paths=['archive/a'])
    assert overlay['run_metadata']['full_history_noncausal_upper_bound'] is True
    assert overlay['model']['scene_id'] == SCENE_ORDERS['order_1'][1]
    assert overlay['model']['history_paths'] == ['archive/a']


def test_single_intersection_compatibility_contract_is_valid():
    config = validate_experiment_config(base_config(num_intersections=1))
    assert config['num_intersections'] == 1


def test_chunked_history_archive_round_trip(tmp_path):
    writer = HistoryArchiveWriter(tmp_path)
    writer.append_episode(1, [transition('300_0.6', IDS[0], 1)])
    writer.append_episode(2, [transition('300_0.6', IDS[1], 2)])
    manifest = {
        'roadnet_id': 'arterial_1x6', 'scene_id': '300_0.6',
        'intersection_ids': list(IDS), 'state_schema': {},
        'action_schema': {}, 'reward_schema': {}, 'num_episodes': 2,
        'num_decision_steps': 2, 'training_seed': 0,
        'source_policy': 'shared_dqn', 'collection_stage': 0,
        'created_at': '2026-07-29T00:00:00Z',
    }
    writer.finalize(manifest)
    loaded = HistoricalPool.load_many([tmp_path])
    assert len(loaded.records) == 2
    assert loaded.scenes == ('300_0.6',)


def test_mapped_history_preserves_transition_and_visibility(tmp_path):
    record = LocalTransition(
        state=np.arange(12, dtype=np.float32), phase=np.asarray([2]),
        action=3, reward=-4.5,
        next_state=np.arange(12, dtype=np.float32) + 1,
        next_phase=np.asarray([3]), terminated=False, truncated=False,
        metadata=TransitionMetadata(
            transition_id=f'300_0.6:seed0:1:7:{IDS[0]}',
            scene_id='300_0.6', intersection_id=IDS[0], episode_id=1,
            decision_step=7, source='offline_history', training_stage=0,
            policy_version=0, collector_training_seed=0),
    )
    dtype = np.dtype([
        ('state', '<f4', (12,)), ('next_state', '<f4', (12,)),
        ('phase', 'u1'), ('next_phase', 'u1'), ('action', 'u1'),
        ('reward', '<f4'), ('terminated', '?'), ('truncated', '?'),
        ('episode_id', '<u2'), ('decision_step', '<u2'),
        ('collector_seed', 'u1'), ('training_stage', 'u1'),
        ('policy_version', '<u4'),
    ])
    array = np.zeros(1, dtype=dtype)
    array['state'][0] = record.state
    array['next_state'][0] = record.next_state
    array['phase'][0] = record.phase[0]
    array['next_phase'][0] = record.next_phase[0]
    array['action'][0] = record.action
    array['reward'][0] = record.reward
    array['episode_id'][0] = record.metadata.episode_id
    array['decision_step'][0] = record.metadata.decision_step
    array['collector_seed'][0] = record.metadata.collector_training_seed or 0
    np.save(tmp_path / 'pool.npy', array, allow_pickle=False)
    archive = tmp_path / 'archive'
    archive.mkdir()
    (archive / 'manifest.json').write_text(json.dumps({
        'scene_id': '300_0.6', 'num_transitions': 1,
        'intersection_ids': list(IDS), 'state_schema': {},
        'action_schema': {}, 'reward_schema': {},
    }))
    (tmp_path / 'index.json').write_text(json.dumps({
        'status': 'completed', 'source_hoa_archive_hash': 'a' * 64,
        'pools': [{'scene_id': '300_0.6', 'intersection_id': IDS[0],
                   'file': 'pool.npy', 'count': 1}],
    }))
    pool = HistoricalPool.load_sampling_index(tmp_path / 'index.json', [archive])
    loaded = pool.sample(1, ['300_0.6'])[0]
    assert loaded.metadata.transition_id == record.metadata.transition_id
    assert np.array_equal(loaded.state, record.state)
    assert np.array_equal(loaded.next_state, record.next_state)
    with pytest.raises(ValueError, match='No visible'):
        pool.sample(1, ['700_0.6'])
