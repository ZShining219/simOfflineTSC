"""Build schema-frozen Plan5 run summaries and inferential result tables."""
import math
from pathlib import Path

import numpy as np

from sequential.io import read_json, sha256_file

from .analysis import (
    HISTORICAL_SCHEDULE, context_distinction, cross_algorithm_cluster_bootstrap,
    current_adaptation, historical_metrics, paired_cluster_bootstrap,
    probe_internal_metrics,
)
from .config import SCENES, SEEDS
from .probe import load_probe_rows
from .schema import TABLE_SCHEMAS, write_table


PRIMARY_METRICS = {
    'worst_historical_degradation': 'worst_historical_degradation',
    'episode100_degradation': 'episode100_degradation',
    'max_policy_disagreement_main': 'max_policy_disagreement_main',
    'current_adaptation_aulc': 'current_adaptation_aulc',
}


def _hashed_json(path, digest, label):
    path = Path(path)
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f'Plan5 {label} hash/path mismatch')
    return read_json(path)


def load_run_analysis_assets(run_summary_path):
    path = Path(run_summary_path)
    summary = read_json(path)
    if summary.get('status') != 'completed' or not summary.get('valid'):
        raise ValueError('Plan5 analysis requires a valid completed run')
    validation = _hashed_json(
        summary['run_validation'], summary['run_validation_sha256'],
        'run validation',
    )
    if not validation.get('valid'):
        raise ValueError('Plan5 run validation is not valid')
    indexes = {}
    for name in ('episode_summaries', 'evaluation_index', 'probe_index'):
        descriptor = summary.get('analysis_indexes', {}).get(name, {})
        payload = _hashed_json(
            descriptor.get('path'), descriptor.get('sha256'), name,
        )
        records = payload.get('records', [])
        if len(records) != int(descriptor.get('record_count', -1)):
            raise ValueError(f'Plan5 {name} record count mismatch')
        if payload.get('identity') != summary.get('identity'):
            raise ValueError(f'Plan5 {name} identity mismatch')
        indexes[name] = records
    checkpoint_index = _hashed_json(
        summary['checkpoint_index'], summary['checkpoint_index_sha256'],
        'checkpoint index',
    )
    return {
        'path': str(path.resolve()), 'sha256': sha256_file(path),
        'summary': summary, 'validation': validation,
        'episode_summaries': indexes['episode_summaries'],
        'evaluations': indexes['evaluation_index'],
        'probe_records': indexes['probe_index'],
        'checkpoints': checkpoint_index['checkpoints'],
    }


def _evaluation_metrics(record):
    summary = record['summary']
    result = {}
    for key in (
            'travel_time', 'mean_queue', 'mean_delay', 'throughput',
            'mean_reward'):
        value = float(summary[key])
        if not math.isfinite(value):
            raise ValueError('Plan5 evaluation metric is non-finite')
        result[key] = value
    return result


def specialist_travel_time(evaluations):
    late = [
        float(record['summary']['travel_time'])
        for record in evaluations
        if record['role'] == 'current' and 91 <= int(record['episode']) <= 100
    ]
    if len(late) != 10 or not all(math.isfinite(value) for value in late):
        raise ValueError('Plan5 specialist late evaluation matrix is incomplete')
    return float(np.median(late))


def summarize_anchor_run(run_summary_path, fixedtime):
    assets = load_run_analysis_assets(run_summary_path)
    identity = assets['summary']['identity']
    if identity.get('run_type') != 'ANCHOR':
        raise ValueError('Plan5 anchor summary received a non-anchor run')
    current = {
        int(record['episode']): float(record['summary']['travel_time'])
        for record in assets['evaluations'] if record['role'] == 'current'
    }
    if set(current) != set(range(101)):
        raise ValueError('Plan5 anchor current evaluation matrix is incomplete')
    late = specialist_travel_time(assets['evaluations'])
    episode100 = next(
        item for item in assets['checkpoints']
        if item['checkpoint_type'] == 'full_resumable'
        and int(item['episode']) == 100
    )
    validation = assets['validation']
    scene = identity['scene']
    row = {
        'logical_run_id': identity['logical_run_id'],
        'algorithm_id': identity['algorithm_id'], 'scene': scene,
        'network': SCENES[scene],
        'training_seed': int(identity['training_seed']),
        'status': 'completed', 'completed': True,
        'finite_metrics': validation['finite_metrics'],
        'checkpoint_valid': validation['checkpoint_valid'],
        'resume_valid': validation['episode100_resume_valid'],
        'evaluation_isolation_valid': validation[
            'evaluation_isolation_valid'
        ],
        'fixed_probe_valid': validation['fixed_probe_valid'],
        'late_metrics_present': validation['late_metrics_present'],
        'tt_episode0': current[0], 'tt_late': late,
        'tt_fixedtime': float(fixedtime['scenes'][scene]['travel_time']),
        'improvement_fraction': (current[0] - late) / current[0],
        'valid': True,
        'episode100_checkpoint_path': episode100['path'],
        'episode100_checkpoint_sha256': episode100['sha256'],
        'run_summary_path': assets['path'],
        'run_summary_sha256': assets['sha256'],
        'attempt_id': identity.get(
            'attempt_id', Path(run_summary_path).parent.name
        ),
        'failure_class': '',
    }
    return row, assets


def summarize_calibration_run(run_summary_path, fixedtime):
    """Derive one immutable PPO calibration selection row from run evidence."""
    assets = load_run_analysis_assets(run_summary_path)
    identity = assets['summary']['identity']
    if identity.get('run_type') != 'CALIBRATION' \
            or identity.get('algorithm_id') != 'PPO' \
            or identity.get('scene') != 'S2':
        raise ValueError('Plan5 calibration summary identity is invalid')
    current = {
        int(record['episode']): float(record['summary']['travel_time'])
        for record in assets['evaluations'] if record['role'] == 'current'
    }
    if set(current) != set(range(101)):
        raise ValueError('Plan5 calibration evaluation matrix is incomplete')
    late = float(np.median([current[episode] for episode in range(91, 101)]))
    adaptation = current_adaptation(current, late)
    return {
        'learning_rate': float(identity['learning_rate']),
        'entropy_coefficient': float(identity['entropy_coefficient']),
        'training_seed': int(identity['training_seed']),
        'logical_run_id': identity['logical_run_id'], 'status': 'completed',
        'tt_episode0': current[0], 'tt_late_median': late,
        'tt_fixedtime': float(fixedtime['scenes']['S2']['travel_time']),
        'current_adaptation_aulc': adaptation['current_adaptation_aulc'],
        'run_summary_path': assets['path'],
        'run_summary_sha256': assets['sha256'],
    }


def _load_probe_outputs(probe_records):
    outputs = {}
    for record in probe_records:
        payload = _hashed_json(
            record['path'], record['sha256'], 'probe inference',
        )
        if payload.get('actions_digest') != record['actions_digest'] \
                or payload.get('policy_vectors_digest') != record[
                    'policy_vectors_digest'
                ]:
            raise ValueError('Plan5 probe inference digest mismatch')
        key = (record['split'], int(record['episode']))
        if key in outputs:
            raise ValueError('Plan5 duplicate probe inference cell')
        outputs[key] = payload
    return outputs


def summarize_transition_run(run_summary_path, specialist_tt):
    assets = load_run_analysis_assets(run_summary_path)
    identity = assets['summary']['identity']
    if identity.get('run_type') != 'TRANSITION':
        raise ValueError('Plan5 transition summary received a non-transition run')
    current_records = [
        record for record in assets['evaluations']
        if record['role'] == 'current'
    ]
    historical_records = [
        record for record in assets['evaluations']
        if record['role'] == 'historical'
    ]
    current_tt = {
        int(record['episode']): float(record['summary']['travel_time'])
        for record in current_records
    }
    historical_tt = {
        int(record['episode']): float(record['summary']['travel_time'])
        for record in historical_records
    }
    adaptation = current_adaptation(current_tt, specialist_tt)
    historical = historical_metrics(historical_tt)
    probe_outputs = _load_probe_outputs(assets['probe_records'])
    policy_rows = []
    internal_rows = []
    disagreement = {}
    for split in ('main', 'heldout'):
        baseline = probe_outputs[(split, 0)]
        for episode in HISTORICAL_SCHEDULE:
            current_probe = probe_outputs[(split, episode)]
            metrics = probe_internal_metrics(baseline, current_probe)
            disagreement[(split, episode)] = metrics['greedy_disagreement']
            policy_rows.append({
                'algorithm_id': identity['algorithm_id'],
                'transition_id': identity['transition_id'],
                'training_seed': int(identity['training_seed']),
                'source_scene': identity['source_scene'], 'episode': episode,
                'split': split,
                'probe_row_count': int(current_probe['row_count']),
                'policy_disagreement': metrics['greedy_disagreement'],
                'episode0_actions_digest': baseline['actions_digest'],
                'episode_actions_digest': current_probe['actions_digest'],
                'valid': True,
            })
            for metric, value in metrics.items():
                internal_rows.append({
                    'algorithm_id': identity['algorithm_id'],
                    'transition_id': identity['transition_id'],
                    'training_seed': int(identity['training_seed']),
                    'scene': identity['source_scene'], 'episode': episode,
                    'split': split, 'metric': metric,
                    'value': value, 'valid': True,
                })
    for episode in assets['episode_summaries']:
        episode_id = int(episode['episode'])
        training_metrics = {
            'training_mean_reward': episode['mean_reward'],
            'training_reward_std': episode['reward_std'],
            **{
                f'training_action_frequency_{index}': value
                for index, value in enumerate(episode['action_frequencies'])
            },
        }
        if episode.get('loss_mean') is not None:
            training_metrics['training_loss_mean'] = episode['loss_mean']
        if episode.get('epsilon_end') is not None:
            training_metrics['training_epsilon_end'] = episode['epsilon_end']
        for statistic in ('mean', 'std', 'min', 'max'):
            for index, value in enumerate(
                    episode['model_input_summary'][statistic]):
                training_metrics[
                    f'model_input_{statistic}_{index}'
                ] = value
        for metric, value in training_metrics.items():
            internal_rows.append({
                'algorithm_id': identity['algorithm_id'],
                'transition_id': identity['transition_id'],
                'training_seed': int(identity['training_seed']),
                'scene': identity['target_scene'], 'episode': episode_id,
                'split': 'training', 'metric': metric,
                'value': value, 'valid': True,
            })
    historical_rows = []
    for record in historical_records:
        episode = int(record['episode'])
        metrics = _evaluation_metrics(record)
        historical_rows.append({
            'algorithm_id': identity['algorithm_id'],
            'transition_id': identity['transition_id'],
            'training_seed': int(identity['training_seed']),
            'source_scene': identity['source_scene'], 'episode': episode,
            'travel_time': metrics['travel_time'],
            'degradation': historical['degradation'][episode],
            'mean_queue': metrics['mean_queue'],
            'mean_delay': metrics['mean_delay'],
            'throughput': metrics['throughput'],
            'mean_reward': metrics['mean_reward'],
            'evaluation_manifest': record['manifest_path'],
            'evaluation_manifest_sha256': record['manifest_sha256'],
            'valid': True,
        })
    current_rows = []
    for record in current_records:
        episode = int(record['episode'])
        metrics = _evaluation_metrics(record)
        current_rows.append({
            'algorithm_id': identity['algorithm_id'],
            'transition_id': identity['transition_id'],
            'training_seed': int(identity['training_seed']),
            'target_scene': identity['target_scene'], 'episode': episode,
            'travel_time': metrics['travel_time'],
            'specialist_tt': specialist_tt,
            'normalized_travel_time': adaptation['normalized'][episode],
            'mean_queue': metrics['mean_queue'],
            'mean_delay': metrics['mean_delay'],
            'throughput': metrics['throughput'],
            'mean_reward': metrics['mean_reward'],
            'evaluation_manifest': record['manifest_path'],
            'evaluation_manifest_sha256': record['manifest_sha256'],
            'valid': True,
        })
    validation = assets['validation']
    summary_row = {
        'logical_run_id': identity['logical_run_id'],
        'algorithm_id': identity['algorithm_id'],
        'transition_id': identity['transition_id'],
        'source_scene': identity['source_scene'],
        'target_scene': identity['target_scene'],
        'training_seed': int(identity['training_seed']),
        'status': 'completed', 'completed': True,
        'finite_metrics': validation['finite_metrics'],
        'checkpoint_valid': validation['checkpoint_valid'],
        'resume_valid': validation['episode100_resume_valid'],
        'evaluation_isolation_valid': validation[
            'evaluation_isolation_valid'
        ],
        'fixed_probe_valid': validation['fixed_probe_valid'],
        'same_start_valid': True,
        'worst_historical_degradation': historical[
            'worst_historical_degradation'
        ],
        'episode100_degradation': historical['episode100_degradation'],
        'recovery_gap': historical['recovery_gap'],
        'specialist_tt': specialist_tt,
        'current_adaptation_aulc': adaptation['current_adaptation_aulc'],
        'episode100_normalized_performance': adaptation[
            'episode100_normalized_performance'
        ],
        'time_to_reference': adaptation['time_to_reference'],
        'time_to_reference_censored': adaptation[
            'time_to_reference_censored'
        ],
        'max_policy_disagreement_main': max(
            disagreement[('main', episode)] for episode in HISTORICAL_SCHEDULE
        ),
        'policy_disagreement100_main': disagreement[('main', 100)],
        'max_policy_disagreement_heldout': max(
            disagreement[('heldout', episode)]
            for episode in HISTORICAL_SCHEDULE
        ),
        'policy_disagreement100_heldout': disagreement[('heldout', 100)],
        'valid': True,
        'attempt_id': identity.get(
            'attempt_id', Path(run_summary_path).parent.name
        ),
        'failure_class': '',
    }
    return {
        'summary': summary_row, 'historical': historical_rows,
        'current': current_rows, 'policy': policy_rows,
        'internal': internal_rows, 'assets': assets,
        'source_checkpoint_sha256': identity['source_checkpoint_sha256'],
    }


def build_inference_tables(transition_runs):
    """Build primary, secondary, and DDQN-vs-CTXDDQN seed-cluster rows."""
    index = {
        (run['summary']['algorithm_id'], run['summary']['transition_id'],
         int(run['summary']['training_seed'])): run
        for run in transition_runs
    }
    algorithms = sorted({key[0] for key in index})
    same_start_rows = []
    secondary_rows = []
    primary_effects = {}
    for algorithm in algorithms:
        for metric, field in PRIMARY_METRICS.items():
            h34 = [index[(algorithm, 'H34', seed)] for seed in SEEDS]
            l32 = [index[(algorithm, 'L32', seed)] for seed in SEEDS]
            left = [float(run['summary'][field]) for run in h34]
            right = [float(run['summary'][field]) for run in l32]
            inference = paired_cluster_bootstrap(left, right)
            primary_effects[(algorithm, metric)] = {
                seed: inference['effects'][seed] for seed in SEEDS
            }
            for seed in SEEDS:
                source_left = h34[seed]['source_checkpoint_sha256']
                source_right = l32[seed]['source_checkpoint_sha256']
                if source_left != source_right:
                    raise ValueError('Plan5 H34/L32 source SHA is not exact')
                same_start_rows.append({
                    'algorithm_id': algorithm, 'training_seed': seed,
                    'metric': metric, 'h34': left[seed], 'l32': right[seed],
                    'effect': inference['effects'][seed],
                    'source_checkpoint_sha256': source_left,
                    'same_start_valid': True, 'row_type': 'seed',
                    'effect_mean': '', 'effect_median': '',
                    'direction_consistency': '', 'ci95_lower': '',
                    'ci95_upper': '', 'bootstrap_resamples': '',
                    'analysis_seed': '',
                })
            same_start_rows.append({
                'algorithm_id': algorithm, 'training_seed': '',
                'metric': metric, 'h34': '', 'l32': '', 'effect': '',
                'source_checkpoint_sha256': '',
                'same_start_valid': True, 'row_type': 'aggregate',
                'effect_mean': inference['mean'],
                'effect_median': inference['median'],
                'direction_consistency': inference['direction_consistency'],
                'ci95_lower': inference['ci95'][0],
                'ci95_upper': inference['ci95'][1],
                'bootstrap_resamples': inference['bootstrap_resamples'],
                'analysis_seed': inference['analysis_seed'],
            })
        for metric, field in PRIMARY_METRICS.items():
            for seed in SEEDS:
                h43 = index[(algorithm, 'H43', seed)]['summary']
                l23 = index[(algorithm, 'L23', seed)]['summary']
                secondary_rows.append({
                    'algorithm_id': algorithm, 'training_seed': seed,
                    'metric': metric, 'h43': h43[field], 'l23': l23[field],
                    'effect': float(h43[field]) - float(l23[field]),
                    'same_target_scene': (
                        h43['target_scene'] == l23['target_scene'] == 'S3'
                    ),
                    'valid': True,
                })
    cross_rows = []
    if {'DDQN', 'CTXDDQN'} <= set(algorithms):
        for metric in PRIMARY_METRICS:
            effects = {
                algorithm: primary_effects[(algorithm, metric)]
                for algorithm in ('DDQN', 'CTXDDQN')
            }
            inference = cross_algorithm_cluster_bootstrap(effects)
            for seed in SEEDS:
                left = effects['DDQN'][seed]
                right = effects['CTXDDQN'][seed]
                cross_rows.append({
                    'algorithm_left': 'DDQN', 'algorithm_right': 'CTXDDQN',
                    'metric': metric, 'training_seed': seed,
                    'effect_left': left, 'effect_right': right,
                    'difference': right - left, 'row_type': 'seed',
                    'difference_mean': '', 'difference_median': '',
                    'seed_consistency_positive': '', 'ci95_lower': '',
                    'ci95_upper': '', 'bootstrap_resamples': '',
                    'analysis_seed': '', 'valid': True,
                })
            cross_rows.append({
                'algorithm_left': 'DDQN', 'algorithm_right': 'CTXDDQN',
                'metric': metric, 'training_seed': '', 'effect_left': '',
                'effect_right': '', 'difference': '',
                'row_type': 'aggregate',
                'difference_mean': inference['mean'],
                'difference_median': inference['median'],
                'seed_consistency_positive': inference[
                    'seed_consistency_positive'
                ],
                'ci95_lower': inference['ci95'][0],
                'ci95_upper': inference['ci95'][1],
                'bootstrap_resamples': inference['bootstrap_resamples'],
                'analysis_seed': inference['analysis_seed'], 'valid': True,
            })
    return {
        'same_start': same_start_rows, 'secondary': secondary_rows,
        'cross_algorithm': cross_rows,
    }


def build_analysis_package(anchor_summary_paths, transition_summary_paths,
                           fixedtime_path, probe_manifest_path, output_root):
    """Validate complete formal matrices and atomically write final tables."""
    fixedtime = read_json(fixedtime_path)
    anchors = [
        summarize_anchor_run(path, fixedtime)
        for path in anchor_summary_paths
    ]
    anchor_rows = [item[0] for item in anchors]
    expected_anchors = {
        (algorithm, scene, seed)
        for algorithm in ('DDQN', 'CTXDDQN', 'PPO')
        for scene in SCENES for seed in SEEDS
    }
    actual_anchors = {
        (row['algorithm_id'], row['scene'], int(row['training_seed']))
        for row in anchor_rows
    }
    if actual_anchors != expected_anchors or len(anchor_rows) != 60:
        raise ValueError('Plan5 anchor analysis matrix is incomplete')
    specialist = {
        (row['algorithm_id'], row['scene'], int(row['training_seed'])):
        float(row['tt_late']) for row in anchor_rows
    }
    transitions = []
    for path in transition_summary_paths:
        identity = read_json(path)['identity']
        key = (
            identity['algorithm_id'], identity['target_scene'],
            int(identity['training_seed']),
        )
        transitions.append(summarize_transition_run(path, specialist[key]))
    expected_transitions = {
        (algorithm, transition, seed)
        for algorithm in ('DDQN', 'CTXDDQN', 'PPO')
        for transition in ('H34', 'H43', 'L23', 'L32') for seed in SEEDS
    }
    actual_transitions = {
        (item['summary']['algorithm_id'], item['summary']['transition_id'],
         int(item['summary']['training_seed']))
        for item in transitions
    }
    if actual_transitions != expected_transitions or len(transitions) != 60:
        raise ValueError('Plan5 transition analysis matrix is incomplete')
    inference = build_inference_tables(transitions)
    probe_rows = load_probe_rows(probe_manifest_path, 'main') \
        + load_probe_rows(probe_manifest_path, 'heldout')
    distinction = context_distinction(probe_rows)
    context_rows = [{
        'probe_id': 'plan5_fixed_probe_v1', **distinction, 'valid': True,
    }]
    tables = {
        'plan5_anchor_summary.csv': anchor_rows,
        'plan5_transition_summary.csv': [
            item['summary'] for item in transitions
        ],
        'plan5_historical_timeline.csv': [
            row for item in transitions for row in item['historical']
        ],
        'plan5_current_adaptation.csv': [
            row for item in transitions for row in item['current']
        ],
        'plan5_policy_displacement.csv': [
            row for item in transitions for row in item['policy']
        ],
        'plan5_internal_metrics.csv': [
            row for item in transitions for row in item['internal']
        ],
        'plan5_same_start_H34_L32.csv': inference['same_start'],
        'plan5_secondary_H43_L23.csv': inference['secondary'],
        'plan5_cross_algorithm_summary.csv': inference['cross_algorithm'],
        'plan5_context_distinction.csv': context_rows,
    }
    output_root = Path(output_root)
    result = {
        name: write_table(output_root / name, name, rows)
        for name, rows in tables.items()
    }
    return {'valid': True, 'tables': result}
