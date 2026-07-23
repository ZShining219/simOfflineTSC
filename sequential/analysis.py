import json
import os

import numpy as np

from .io import atomic_json, read_json
from .validation import compare_recovery_pair, validate_attempt


def normalized_auc(travel_times, baseline, horizon=None):
    values = np.asarray(travel_times, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError('AUC requires a one-dimensional local 0..N series')
    horizon = len(values) - 1 if horizon is None else int(horizon)
    if len(values) != horizon + 1 or horizon <= 0 or baseline <= 0:
        raise ValueError('AUC series, horizon, or baseline is invalid')
    return float(np.trapz(values / float(baseline), dx=1.0) / horizon)


def exact_sign_flip_test(paired_differences):
    differences = np.asarray(paired_differences, dtype=float)
    if differences.ndim != 1 or not len(differences):
        raise ValueError('Sign-flip test requires paired differences')
    if len(differences) > 20:
        raise ValueError('Exact sign-flip enumeration is limited to 20 pairs')
    observed = abs(float(np.mean(differences)))
    extreme = 0
    total = 1 << len(differences)
    tolerance = np.finfo(float).eps * max(1.0, observed) * 8
    bit_positions = np.arange(len(differences), dtype=np.uint64)
    for start in range(0, total, 65536):
        masks = np.arange(start, min(total, start + 65536), dtype=np.uint64)
        signs = 2.0 * ((masks[:, None] >> bit_positions[None, :]) & 1) - 1.0
        statistics = np.abs(np.mean(signs * differences[None, :], axis=1))
        extreme += int(np.count_nonzero(statistics + tolerance >= observed))
    return {'statistic': float(np.mean(differences)), 'p_value': extreme / total,
            'enumerations': total}


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind='stable')
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_bootstrap(paired_differences, seed, resamples=10000):
    differences = np.asarray(paired_differences, dtype=float)
    if differences.ndim != 1 or not len(differences):
        raise ValueError('Paired bootstrap requires paired differences')
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(differences), size=(int(resamples), len(differences)))
    estimates = np.mean(differences[indices], axis=1)
    return {
        'effect_mean': float(np.mean(differences)),
        'confidence_interval_95': [
            float(np.percentile(estimates, 2.5)),
            float(np.percentile(estimates, 97.5)),
        ],
        'resamples': int(resamples), 'seed': int(seed),
    }


def replay_uniformity_envelope(windows, seed, resamples=10000, level=0.99):
    if not windows:
        return {'available': False, 'reason': 'no_sampling_windows'}
    scenes = sorted({
        scene for window in windows
        for scene in window['population_count_by_scene']
    })
    observed_max = 0.0
    prepared = []
    for scene in scenes:
        population = np.asarray([
            int(window['population_count_by_scene'].get(scene, 0))
            for window in windows
        ], dtype=np.int64)
        total = np.asarray([int(window['replay_size']) for window in windows], dtype=np.int64)
        sample_size = np.asarray([int(window['sample_size']) for window in windows], dtype=np.int64)
        observed = np.asarray([
            int(window['sample_count_by_scene'].get(scene, 0))
            for window in windows
        ], dtype=float)
        probability = population / total
        expected = sample_size * probability
        correction = np.where(total > 1, (total - sample_size) / (total - 1), 0.0)
        variance = sample_size * probability * (1.0 - probability) * correction
        active = variance > 0
        if np.any(active):
            observed_max = max(observed_max, float(np.max(
                np.abs(observed[active] - expected[active]) / np.sqrt(variance[active])
            )))
            prepared.append((population[active], total[active], sample_size[active],
                             expected[active], variance[active]))
    if not prepared:
        return {'available': False, 'reason': 'all_sampling_probabilities_degenerate'}
    rng = np.random.default_rng(int(seed))
    maxima = np.zeros(int(resamples), dtype=float)
    chunk_size = 128
    for start in range(0, int(resamples), chunk_size):
        stop = min(int(resamples), start + chunk_size)
        chunk_max = np.zeros(stop - start, dtype=float)
        for population, total, sample_size, expected, variance in prepared:
            draws = rng.hypergeometric(
                population[None, :], (total - population)[None, :],
                sample_size[None, :], size=(stop - start, len(population)),
            )
            scores = np.abs(draws - expected[None, :]) / np.sqrt(variance[None, :])
            chunk_max = np.maximum(chunk_max, np.max(scores, axis=1))
        maxima[start:stop] = chunk_max
    try:
        threshold = float(np.quantile(maxima, level, method='higher'))
    except TypeError:
        threshold = float(np.quantile(maxima, level, interpolation='higher'))
    return {
        'available': True, 'valid': observed_max <= threshold,
        'observed_max_standardized_deviation': observed_max,
        'simultaneous_threshold': threshold, 'level': float(level),
        'resamples': int(resamples), 'seed': int(seed),
        'window_count': len(windows), 'scene_count': len(scenes),
    }


def plan1_final_means(parent_catalog_path):
    catalog = read_json(parent_catalog_path)
    values = {}
    seen = set()
    for parent in catalog['parents']:
        key = (parent['network'], int(parent['training_seed']))
        if key in seen:
            continue
        seen.add(key)
        records_path = os.path.join(
            parent['source_run_path'], 'metrics', 'records.jsonl'
        )
        final = None
        with open(records_path, encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                if record.get('record_type') == 'FINAL_EVALUATION':
                    final = record
        if final is None or int(final['episode']) != 400:
            raise ValueError(f'Missing Plan 1 final evaluation: {records_path}')
        values.setdefault(parent['network'], []).append(float(final['travel_time']))
    if any(len(network_values) != 5 for network_values in values.values()):
        raise ValueError('Plan 1 normalization requires five seeds per network')
    return {network: float(np.mean(network_values))
            for network, network_values in values.items()}


def _first_sustained(values, threshold, length=5):
    for start in range(0, len(values) - length + 1):
        if all(value <= threshold for value in values[start:start + length]):
            return start
    return None


def summarize_run(validated, baselines):
    child = validated['child_manifest']
    cells = validated['evaluation_cells']
    stage_summaries = []
    own_stage_end = {}
    for stage in range(1, len(child['networks']) + 1):
        budget = int(child['stage_episodes'][stage - 1])
        network = child['networks'][stage - 1]
        if stage == 1:
            own_stage_end[network] = float(
                cells[(1, budget, network)]['summary']['travel_time']
            )
            continue
        series = [float(cells[(stage, local, network)]['summary']['travel_time'])
                  for local in range(0, budget + 1)]
        baseline = baselines[network]
        own_stage_end[network] = series[-1]
        stage_summaries.append({
            'stage_index': stage, 'network': network, 'horizon': budget,
            'normalized_auc': normalized_auc(series, baseline, budget),
            'zero_shot_normalized': series[0] / baseline,
            'first_five_within_110_percent': _first_sustained(
                series, 1.1 * baseline, length=5
            ),
            'final_normalized': series[-1] / baseline,
        })
    final_stage = len(child['networks'])
    final_budget = int(child['stage_episodes'][-1])
    retention = {}
    forgetting = {}
    for network in child['networks']:
        final_value = float(cells[(final_stage, final_budget, network)][
            'summary']['travel_time'])
        retention[network] = final_value / baselines[network]
        forgetting[network] = final_value - own_stage_end[network]
    diagnostic_windows = []
    replay_diagnostics = []
    for key, operation in validated['state']['completed_operations'].items():
        if key.endswith(':diagnostic'):
            diagnostic = read_json(operation['artifact'])
            diagnostic_windows.extend(diagnostic.get('sampling_windows', []))
            replay_diagnostics.append({
                key: diagnostic.get(key) for key in (
                    'stage_index', 'local_episode', 'global_episode',
                    'replay_count_by_scene', 'replay_ratio_by_scene',
                    'transitions_written_by_scene', 'samples_drawn_by_scene',
                    'current_sample_fraction', 'historical_sample_fraction',
                    'replacement_thresholds',
                )
            })
    previous_networks = child['networks'][:-1]
    secondary = {
        'zero_shot_mean_normalized': float(np.mean([
            stage['zero_shot_normalized'] for stage in stage_summaries
        ])),
        'first_five_within_110_percent_mean': float(np.mean([
            stage['first_five_within_110_percent']
            if stage['first_five_within_110_percent'] is not None
            else stage['horizon'] + 1
            for stage in stage_summaries
        ])),
        'stage_end_forgetting_mean_travel_time': float(np.mean([
            forgetting[network] for network in previous_networks
        ])),
        'final_retention_mean_normalized': float(np.mean(list(retention.values()))),
    }
    return {
        'logical_run_id': validated['logical_run_id'],
        'order_id': child['order_id'], 'training_seed': child['training_seed'],
        'policy': child['policy'],
        'primary_normalized_auc': float(np.mean([
            stage['normalized_auc'] for stage in stage_summaries
        ])),
        'stages': stage_summaries,
        'stage_end_forgetting_travel_time': forgetting,
        'final_retention_normalized': retention,
        'secondary_metrics': secondary,
        'replay_diagnostics': sorted(
            replay_diagnostics,
            key=lambda row: (row['stage_index'], row['local_episode']),
        ),
        'replay_sampling_windows': diagnostic_windows,
    }


def _comparison(runs_by_key, comparator, seed, resamples, inferential):
    differences = []
    units = []
    for unit in sorted({(run['order_id'], int(run['training_seed']))
                        for run in runs_by_key.values()}):
        clear = runs_by_key.get((*unit, 'clear'))
        other = runs_by_key.get((*unit, comparator))
        if clear is None or other is None:
            continue
        differences.append(clear['primary_normalized_auc'] - other[
            'primary_normalized_auc'
        ])
        units.append({'order_id': unit[0], 'training_seed': unit[1]})
    result = {
        'contrast': f'clear_minus_{comparator}', 'paired_units': units,
        'paired_differences': differences,
        'bootstrap': paired_bootstrap(differences, seed, resamples),
        'sign_flip': exact_sign_flip_test(differences) if inferential else None,
    }
    secondary_names = (
        'zero_shot_mean_normalized',
        'first_five_within_110_percent_mean',
        'stage_end_forgetting_mean_travel_time',
        'final_retention_mean_normalized',
    )
    result['secondary_paired_effects'] = {}
    for offset, metric in enumerate(secondary_names, start=1):
        metric_differences = []
        for unit in [(row['order_id'], int(row['training_seed'])) for row in units]:
            clear = runs_by_key[(*unit, 'clear')]
            other = runs_by_key[(*unit, comparator)]
            metric_differences.append(
                clear['secondary_metrics'][metric]
                - other['secondary_metrics'][metric]
            )
        result['secondary_paired_effects'][metric] = paired_bootstrap(
            metric_differences, seed + offset, resamples
        )
    return result


def analyze_experiment(manifest_path, output_root, parent_catalog_path, output_path):
    plan = read_json(manifest_path)
    baselines = plan1_final_means(parent_catalog_path)
    validated_runs = []
    pair_checks = []
    for child in plan['children']:
        if plan['mode'] == 'pilot' and child.get('variant') == 'fault':
            continue
        logical_root = os.path.join(output_root, child['logical_run_id'])
        validated = validate_attempt(logical_root)
        validated_runs.append(validated)
        if plan['mode'] == 'pilot':
            fault_id = child['logical_run_id'].replace('_control', '_fault')
            pair_checks.append({
                'policy': child['policy'],
                **compare_recovery_pair(
                    logical_root, os.path.join(output_root, fault_id)
                ),
            })
    summaries = [summarize_run(run, baselines) for run in validated_runs]
    keyed = {(run['order_id'], int(run['training_seed']), run['policy']): run
             for run in summaries}
    analysis_config = plan['analysis']
    inferential = plan['mode'] == 'formal'
    if inferential and len({(run['order_id'], run['training_seed'])
                            for run in summaries}) != 20:
        raise ValueError('Formal analysis requires exactly 20 paired units')
    comparisons = [
        _comparison(keyed, 'fifo', analysis_config['seed'],
                    analysis_config['resamples'], inferential),
        _comparison(keyed, 'fifo_matched_wait', analysis_config['seed'] + 1,
                    analysis_config['resamples'], inferential),
    ]
    if inferential:
        adjusted = holm_adjust([item['sign_flip']['p_value'] for item in comparisons])
        for comparison, value in zip(comparisons, adjusted):
            comparison['sign_flip']['holm_adjusted_p_value'] = value
    uniformity = {}
    for run in summaries:
        uniformity[run['logical_run_id']] = replay_uniformity_envelope(
            run.pop('replay_sampling_windows'), analysis_config['seed'],
            analysis_config['resamples'], level=0.99,
        )
    report = {
        'schema_version': 1, 'mode': plan['mode'],
        'formal_inference_performed': inferential,
        'pilot_notice': (
            None if inferential else
            'Pilot results are engineering diagnostics only; no formal significance conclusion.'
        ),
        'plan1_episode400_five_seed_means': baselines,
        'runs': summaries, 'comparisons': comparisons,
        'recovery_pairs': pair_checks, 'replay_uniformity': uniformity,
    }
    atomic_json(output_path, report)
    return report
