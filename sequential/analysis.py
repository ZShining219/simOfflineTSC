import json
import os
from concurrent.futures import ProcessPoolExecutor

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


def raw_auc(values, horizon=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError('AUC requires a one-dimensional local 0..N series')
    horizon = len(values) - 1 if horizon is None else int(horizon)
    if len(values) != horizon + 1 or horizon <= 0:
        raise ValueError('AUC series or horizon is invalid')
    return float(np.trapz(values, dx=1.0) / horizon)


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
    adaptation_curves = []
    own_stage_end = {}
    for stage in range(1, len(child['networks']) + 1):
        budget = int(child['stage_episodes'][stage - 1])
        network = child['networks'][stage - 1]
        if stage == 1:
            own_stage_end[network] = float(
                cells[(1, budget, network)]['summary']['travel_time']
            )
            continue
        metric_names = (
            'travel_time', 'delay', 'real_delay', 'queue', 'throughput',
            'reward_mean',
        )
        metric_series = {
            metric: [
                float(cells[(stage, local, network)]['summary'][metric])
                for local in range(0, budget + 1)
            ]
            for metric in metric_names
        }
        series = metric_series['travel_time']
        baseline = baselines[network]
        for local_episode in range(0, budget + 1):
            adaptation_curves.append({
                'stage_index': stage, 'network': network,
                'local_episode': local_episode,
                'normalized_travel_time': (
                    metric_series['travel_time'][local_episode] / baseline
                ),
                **{
                    metric: values[local_episode]
                    for metric, values in metric_series.items()
                },
            })
        own_stage_end[network] = series[-1]
        stage_summaries.append({
            'stage_index': stage, 'network': network, 'horizon': budget,
            'normalized_auc': normalized_auc(series, baseline, budget),
            'zero_shot_normalized': series[0] / baseline,
            'first_five_within_110_percent': _first_sustained(
                series, 1.1 * baseline, length=5
            ),
            'final_normalized': series[-1] / baseline,
            'metrics': {
                metric: {
                    'auc_0_horizon': raw_auc(values, budget),
                    'zero_shot': values[0], 'final': values[-1],
                }
                for metric, values in metric_series.items()
            },
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
    stage_scene_evaluations = []
    for stage in range(1, final_stage + 1):
        stage_budget = int(child['stage_episodes'][stage - 1])
        eval_local = stage_budget
        evaluation_networks = (
            child['networks'][:stage]
            if child.get('condition') in {'M0', 'M1', 'M2', 'M3'}
            else child['networks']
        )
        for evaluation_network in evaluation_networks:
            cell = cells[(stage, eval_local, evaluation_network)]
            value = float(cell['summary']['travel_time'])
            introduced_stage = next(
                index for index, network in enumerate(child['networks'], start=1)
                if network == evaluation_network
            )
            own_end = own_stage_end.get(evaluation_network)
            stage_scene_evaluations.append({
                'stage_index': stage,
                'stage_local_episode': eval_local,
                'training_network': child['networks'][stage - 1],
                'evaluation_network': evaluation_network,
                'evaluation_scene_introduced_stage': introduced_stage,
                'travel_time': value,
                'plan1_reference_travel_time': baselines[evaluation_network],
                'final_retention_normalized': value / baselines[evaluation_network],
                'forgetting_travel_time': (
                    value - own_end if own_end is not None and stage > introduced_stage
                    else 0.0
                ),
            })
    diagnostic_windows = []
    replay_diagnostics = []
    for key, operation in validated['state']['completed_operations'].items():
        if key.endswith(':diagnostic'):
            diagnostic = read_json(operation['artifact'])
            diagnostic_windows.extend(diagnostic.get('sampling_windows', []))
            replay_diagnostics.append({
                key: diagnostic.get(key) for key in (
                    'stage_index', 'local_episode', 'global_episode',
                    'current_network',
                    'replay_count_by_scene', 'replay_ratio_by_scene',
                    'transitions_written_by_scene', 'samples_drawn_by_scene',
                    'current_sample_fraction', 'historical_sample_fraction',
                    'online_sample_fraction',
                    'historical_sample_fraction_by_kind',
                    'samples_drawn_by_kind', 'sample_age_mean',
                    'sample_age_p50', 'sample_age_p95',
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
        'policy': child['policy'], 'condition': child.get('condition'),
        'budget_id': child.get('budget_id'),
        'parent_checkpoint_episode': child.get('parent_checkpoint_episode'),
        'primary_normalized_auc': float(np.mean([
            stage['normalized_auc'] for stage in stage_summaries
        ])),
        'stages': stage_summaries,
        'adaptation_curves': adaptation_curves,
        'stage_end_forgetting_travel_time': forgetting,
        'final_retention_normalized': retention,
        'stage_scene_evaluations': stage_scene_evaluations,
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


def _select_analysis_children(plan, selected_orders=None):
    children = list(plan['children'])
    if selected_orders is None:
        return children, sorted({child['order_id'] for child in children})
    requested = list(dict.fromkeys(selected_orders))
    known = {child['order_id'] for child in children}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f'Unknown selected order(s): {unknown}')
    selected = [child for child in children if child['order_id'] in requested]
    expected = {
        (order, int(seed), policy)
        for order in requested
        for seed in plan['training_seeds']
        for policy in plan['policies']
    }
    actual = {
        (child['order_id'], int(child['training_seed']), child['policy'])
        for child in selected
    }
    if actual != expected:
        raise ValueError(
            'Selected-order matrix is incomplete; '
            f'missing={sorted(expected - actual)}, extra={sorted(actual - expected)}'
        )
    return selected, requested


def _validate_and_summarize(arguments):
    logical_root, baselines = arguments
    return summarize_run(validate_attempt(logical_root), baselines)


def analyze_experiment(
    manifest_path, output_root, parent_catalog_path, output_path,
    selected_orders=None, validation_workers=1,
):
    plan = read_json(manifest_path)
    children, included_orders = _select_analysis_children(plan, selected_orders)
    baselines = plan1_final_means(parent_catalog_path)
    logical_roots = [
        os.path.join(output_root, child['logical_run_id']) for child in children
        if not (plan['mode'] == 'pilot' and child.get('variant') == 'fault')
    ]
    pair_checks = []
    if plan['mode'] == 'pilot':
        validated_runs = [validate_attempt(path) for path in logical_roots]
        summaries = [summarize_run(run, baselines) for run in validated_runs]
        for child in children:
            if child.get('variant') == 'fault':
                continue
            logical_root = os.path.join(output_root, child['logical_run_id'])
            fault_id = child['logical_run_id'].replace('_control', '_fault')
            pair_checks.append({
                'policy': child['policy'],
                **compare_recovery_pair(
                    logical_root, os.path.join(output_root, fault_id)
                ),
            })
    else:
        arguments = [(path, baselines) for path in logical_roots]
        if int(validation_workers) > 1:
            with ProcessPoolExecutor(max_workers=int(validation_workers)) as executor:
                summaries = list(executor.map(_validate_and_summarize, arguments))
        else:
            summaries = [_validate_and_summarize(item) for item in arguments]
    keyed = {(run['order_id'], int(run['training_seed']), run['policy']): run
             for run in summaries}
    analysis_config = plan['analysis']
    paired_unit_count = len({
        (run['order_id'], int(run['training_seed'])) for run in summaries
    })
    expected_formal = {
        (order, int(seed), policy)
        for order in plan['orders']
        for seed in plan['training_seeds']
        for policy in plan['policies']
    }
    full_formal_matrix = (
        plan['mode'] == 'formal' and set(keyed) == expected_formal
    )
    inferential = full_formal_matrix
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
        windows = run.pop('replay_sampling_windows')
        uniformity[run['logical_run_id']] = (
            replay_uniformity_envelope(
                windows, analysis_config['seed'],
                analysis_config['resamples'], level=0.99,
            ) if inferential else {
                'available': False,
                'reason': 'deferred_until_complete_four_order_formal_matrix',
                'window_count': len(windows),
            }
        )
    report = {
        'schema_version': 2,
        'mode': (
            plan['mode'] if full_formal_matrix or plan['mode'] != 'formal'
            else 'formal_partial'
        ),
        'source_mode': plan['mode'],
        'analysis_scope': {
            'included_orders': included_orders,
            'planned_orders': sorted(plan['orders']),
            'included_logical_runs': len(summaries),
            'planned_logical_runs': len(plan['children']),
            'paired_unit_count': paired_unit_count,
            'complete_selected_order_matrix': True,
            'complete_formal_matrix': full_formal_matrix,
        },
        'budget_id': plan.get('budget_id'),
        'parent_checkpoint_episode': plan.get('parent_checkpoint_episode'),
        'primary_metric': 'normalized_travel_time_auc_0_100',
        'normalization_protocol': {
            'reference': 'plan1_formal_episode_400_final_evaluation_five_seed_mean',
            'shared_across_budgets': True,
            'scene_denominators': baselines,
        },
        'analysis_seed': analysis_config['seed'],
        'formal_inference_performed': inferential,
        'pilot_notice': (
            None if inferential else (
                'Partial formal-batch result: descriptive estimates only; '
                'no full four-order formal significance conclusion.'
                if plan['mode'] == 'formal' else
                'Pilot results are engineering diagnostics only; no formal significance conclusion.'
            )
        ),
        'plan1_episode400_five_seed_means': baselines,
        'runs': summaries, 'comparisons': comparisons,
        'recovery_pairs': pair_checks, 'replay_uniformity': uniformity,
    }
    atomic_json(output_path, report)
    return report


def analyze_cross_budget(b100_report_path, b400_report_path, output_path):
    reports = {'b100': read_json(b100_report_path), 'b400': read_json(b400_report_path)}
    protocols = [report.get('normalization_protocol') for report in reports.values()]
    if protocols[0] != protocols[1] or not protocols[0].get('shared_across_budgets'):
        raise ValueError('Cross-budget normalization protocols are not identical')
    indexes = {budget: {
        (run['order_id'], int(run['training_seed']), run['policy']): run
        for run in report['runs']
    } for budget, report in reports.items()}
    expected = {(f'O{o}', s, p) for o in range(1, 5) for s in range(5)
                for p in ('clear', 'fifo', 'fifo_matched_wait')}
    if any(report.get('budget_id') != budget or set(indexes[budget]) != expected
           for budget, report in reports.items()):
        raise ValueError('Cross-budget formal report matrix is invalid')
    seed, resamples = int(reports['b100']['analysis_seed']), 10000
    policy_effects = {}
    for offset, policy in enumerate(('clear', 'fifo', 'fifo_matched_wait')):
        differences = [indexes['b100'][key]['primary_normalized_auc']
                       - indexes['b400'][key]['primary_normalized_auc']
                       for key in sorted(expected) if key[2] == policy]
        policy_effects[policy] = {
            'contrast': 'b100_minus_b400', 'paired_differences': differences,
            'bootstrap': paired_bootstrap(differences, seed + offset, resamples),
            'sign_flip': exact_sign_flip_test(differences),
        }
    interactions = {}
    for offset, comparator in enumerate(('fifo', 'fifo_matched_wait'), start=10):
        differences, units = [], []
        for order in range(1, 5):
            for training_seed in range(5):
                unit = (f'O{order}', training_seed)
                within = {budget: indexes[budget][(*unit, 'clear')][
                    'primary_normalized_auc'] - indexes[budget][(*unit, comparator)][
                    'primary_normalized_auc'] for budget in reports}
                differences.append(within['b100'] - within['b400'])
                units.append({'order_id': unit[0], 'training_seed': unit[1]})
        interactions[f'budget_x_{comparator}'] = {
            'contrast': f'(clear_minus_{comparator})_b100_minus_b400',
            'paired_units': units, 'paired_differences': differences,
            'bootstrap': paired_bootstrap(differences, seed + offset, resamples),
            'sign_flip': exact_sign_flip_test(differences),
        }
    payload = {
        'schema_version': 1, 'budgets': ['b100', 'b400'],
        'primary_metric': 'normalized_travel_time_auc_0_100',
        'normalization_protocol': protocols[0],
        'training_seed_interpretation': 'agent-side randomness; SUMO fixed_default',
        'within_budget_effects': {b: r['comparisons'] for b, r in reports.items()},
        'cross_budget_policy_effects': policy_effects,
        'budget_replay_policy_interactions': interactions,
    }
    atomic_json(output_path, payload)
    return payload
