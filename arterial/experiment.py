"""Configuration and result contracts for arterial shared-DQN experiments."""

from __future__ import annotations

import json
from pathlib import Path


ARTERIAL_SCENES = {
    '300_0.3': 'sumoarterial1x6_300_03',
    '300_0.6': 'sumoarterial1x6_300_06',
    '700_0.3': 'sumoarterial1x6_700_03',
    '700_0.6': 'sumoarterial1x6_700_06',
}

SCENE_ORDERS = {
    'order_1': ['300_0.6', '300_0.3', '700_0.3', '700_0.6'],
    'order_2': ['700_0.6', '700_0.3', '300_0.3', '300_0.6'],
    'order_3': ['300_0.6', '700_0.6', '300_0.3', '700_0.3'],
}


def validate_experiment_config(config):
    required = {
        'roadnet', 'scene_order', 'num_intersections', 'shared_parameters',
        'state_variant', 'reward_variant', 'use_position_encoding',
        'use_neighbor_summary', 'replay_clear_on_scene_switch',
        'offline_enabled', 'offline_ratio', 'offline_sampling_strategy',
        'history_access_mode', 'epsilon_mode', 'training_seed',
        'episode_budget', 'evaluation_interval', 'checkpoint_interval',
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f'Experiment config missing fields: {missing}')
    order = config['scene_order']
    if isinstance(order, str):
        if order not in SCENE_ORDERS:
            raise ValueError(f'Unknown scene order: {order}')
        order = SCENE_ORDERS[order]
    if not order or any(scene not in ARTERIAL_SCENES for scene in order):
        raise ValueError('scene_order contains an unsupported arterial scene')
    if int(config['num_intersections']) not in {1, 6}:
        raise ValueError('Only compatibility sizes 1 and 6 are supported')
    if not config['shared_parameters']:
        raise ValueError('arterial shared DQN requires shared_parameters=true')
    if config['history_access_mode'] not in {
            'causal', 'full', 'non_causal_full_history_upper_bound'}:
        raise ValueError(
            'history_access_mode must be causal, full, or '
            'non_causal_full_history_upper_bound')
    if config['epsilon_mode'] not in {
            'reset_schedule', 'continue_schedule', 'fixed_low'}:
        raise ValueError('Unsupported epsilon_mode')
    ratio = float(config['offline_ratio'])
    if ratio not in {0.0, 0.25, 0.5, 0.75, 1.0}:
        raise ValueError('offline_ratio is outside the supported experiment grid')
    if config['use_neighbor_summary']:
        raise ValueError('Neighbor summary is reserved but not enabled in this Goal')
    return {**config, 'scene_order': list(order)}


def stage_overlay(config, stage_index, history_paths=(), stage_checkpoint=None):
    config = validate_experiment_config(config)
    order = config['scene_order']
    scene = order[stage_index]
    replay_policy = 'clear' if config['replay_clear_on_scene_switch'] else 'fifo'
    budget = int(config['episode_budget'])
    def schedule(interval):
        interval = int(interval)
        if interval <= 0:
            raise ValueError('Evaluation/checkpoint intervals must be positive')
        return sorted(set([0, budget, *range(interval, budget + 1, interval)]))
    checkpoint_schedule = schedule(config['checkpoint_interval'])
    evaluation_schedule = sorted(set(
        schedule(config['evaluation_interval']) + checkpoint_schedule))
    configured_history_mode = config['history_access_mode']
    full_history = configured_history_mode in {
        'full', 'non_causal_full_history_upper_bound'}
    formal_history_mode = (
        'non_causal_full_history_upper_bound' if full_history else 'causal')
    return {
        'model': {
            'scene_id': scene, 'scene_order': order,
            'training_stage': int(stage_index),
            'state_variant': config['state_variant'],
            'reward_variant': config['reward_variant'],
            'use_position_encoding': bool(config['use_position_encoding']),
            'use_neighbor_summary': False,
            'offline_ratio': float(config['offline_ratio'])
                if config['offline_enabled'] else 0.0,
            'offline_sampling_strategy': config['offline_sampling_strategy'],
            # HistoricalPool uses the compact internal value while formal
            # experiment identity uses the frozen research label.
            'history_access_mode': 'full' if full_history else 'causal',
            'history_paths': list(history_paths),
            'epsilon_mode': config['epsilon_mode'],
            'replay_policy': replay_policy,
            'stage_checkpoint': stage_checkpoint,
            'expected_num_intersections': int(config['num_intersections']),
            'shared_parameters': True,
            'roadnet_id': config['roadnet'],
            'hoa_manifest_path': config.get('hoa_manifest_path'),
            'hoa_archive_version': config.get('hoa_archive_version'),
            'hoa_archive_hash': config.get('hoa_archive_hash'),
            'hoa_sampling_index_path': config.get('hoa_sampling_index_path'),
        },
        'trainer': {
            'episodes': budget,
            'evaluation_episodes': evaluation_schedule,
            'resumable_checkpoint_episodes': checkpoint_schedule,
        },
        'run_metadata': {
            'roadnet': config['roadnet'], 'scene': scene,
            'scene_order': order, 'stage_index': int(stage_index),
            'training_seed': int(config['training_seed']),
            'history_mode': formal_history_mode,
            'non_causal_full_history_upper_bound': full_history,
            # Backward compatibility for existing smoke artifact readers.
            'full_history_noncausal_upper_bound': full_history,
            'simulator_network': ARTERIAL_SCENES[scene],
            'system_profile_path': config.get('system_profile_path'),
            'system_profile_hash': config.get('system_profile_hash'),
            'hoa_manifest_path': config.get('hoa_manifest_path'),
            'hoa_archive_version': config.get('hoa_archive_version'),
            'hoa_archive_hash': config.get('hoa_archive_hash'),
            'hoa_sampling_index_path': config.get('hoa_sampling_index_path'),
        },
    }


class EvaluationMatrixWriter:
    FIELDS = {
        'training_stage', 'trained_until_scene', 'evaluation_scene',
        'checkpoint_path', 'metric', 'value',
    }

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record):
        missing = sorted(self.FIELDS - set(record))
        if missing:
            raise ValueError(f'Evaluation matrix row missing fields: {missing}')
        if record['evaluation_scene'] not in ARTERIAL_SCENES:
            raise ValueError('Unknown evaluation scene')
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')


def build_cross_scene_evaluation_manifest(
        output_path, source_run_dir, target_run_dirs, checkpoint,
        checkpoint_episode, trained_until_scene, training_stage,
        training_seed=0, evaluation_seeds=(0,), evaluation_steps=3600):
    """Build a v2 manifest consumable by run.py's frozen evaluator."""
    source_run_dir = str(Path(source_run_dir).resolve())
    source_manifest = json.loads(
        (Path(source_run_dir) / 'run_manifest.json').read_text(encoding='utf-8'))
    if source_manifest['agent'] != 'shared_dqn':
        raise ValueError('Cross-scene arterial evaluation requires shared_dqn source')
    if trained_until_scene not in ARTERIAL_SCENES:
        raise ValueError('Unknown trained_until_scene')
    controllers = []
    for evaluation_scene, target_run_dir in target_run_dirs.items():
        if evaluation_scene not in ARTERIAL_SCENES:
            raise ValueError(f'Unknown evaluation scene: {evaluation_scene}')
        controllers.append({
            'controller_id': (
                f'stage{int(training_stage)}_{trained_until_scene.replace(chr(46), "")}_'
                f'to_{evaluation_scene.replace(chr(46), "")}'),
            'agent': 'shared_dqn',
            'network': ARTERIAL_SCENES[evaluation_scene],
            'source_network': source_manifest['network'],
            'target_network': ARTERIAL_SCENES[evaluation_scene],
            'training_seed': int(training_seed),
            'run_dir': source_run_dir,
            'target_run_dir': str(Path(target_run_dir).resolve()),
            'checkpoint': str(checkpoint),
            'checkpoint_role': 'resumable',
            'checkpoint_episode': int(checkpoint_episode),
            'source_policy': 'shared_dqn',
            'training_stage': int(training_stage),
            'trained_until_scene': trained_until_scene,
            'evaluation_scene': evaluation_scene,
            'evaluation_steps': int(evaluation_steps),
        })
    payload = {
        'schema_version': 2,
        'package_id': f'arterial_stage{int(training_stage)}',
        'world': 'sumo', 'evaluation_seeds': list(evaluation_seeds),
        'sampling_interval_seconds': 10, 'smoothing_window_seconds': 60,
        'metrics': ['reward', 'queue', 'delay', 'throughput', 'travel_time'],
        'expected_controller_count': len(controllers),
        'expected_episode_count': len(controllers) * len(evaluation_seeds),
        'record_state_diagnostics': True, 'controllers': controllers,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    return payload
