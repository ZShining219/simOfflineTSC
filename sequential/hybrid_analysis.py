"""Validated run-list and paired analysis for the CS-HR M0--M3 matrix."""

import os

import numpy as np

from .analysis import (
    exact_sign_flip_test, holm_adjust, paired_bootstrap, plan1_final_means,
    summarize_run,
)
from .io import atomic_json, read_json
from .validation import validate_attempt


CONDITIONS = ('M0', 'M1', 'M2', 'M3')
PRESPECIFIED_CONTRASTS = (
    ('M0', 'M1'), ('M0', 'M2'), ('M0', 'M3'), ('M2', 'M3'),
)


def _attempt_number(path):
    name = os.path.basename(os.path.normpath(path))
    try:
        return int(name.rsplit('_', 1)[1])
    except (IndexError, ValueError):
        return -1


def _completed_attempts(logical_root):
    attempts_root = os.path.join(logical_root, 'attempts')
    if not os.path.isdir(attempts_root):
        return []
    paths = []
    for name in os.listdir(attempts_root):
        path = os.path.join(attempts_root, name)
        if (os.path.isdir(path) and os.path.isfile(os.path.join(path, 'completed.json'))
                and os.path.isfile(os.path.join(path, 'current_state.json'))):
            paths.append(path)
    return sorted(paths, key=_attempt_number, reverse=True)


def _compatible_hybrid_plans(manifest_paths):
    if set(manifest_paths) != set(CONDITIONS):
        raise ValueError('CS-HR analysis requires exactly M0, M1, M2, and M3 manifests')
    plans = {condition: read_json(manifest_paths[condition]) for condition in CONDITIONS}
    for condition, plan in plans.items():
        if plan.get('mode') != 'hybrid_formal' or plan.get('condition') != condition:
            raise ValueError(f'Invalid CS-HR manifest identity: {condition}')
    reference = plans['M0']
    frozen_fields = ('orders', 'training_seeds', 'budget_id',
                     'parent_checkpoint_episode', 'stage_episodes')
    for condition, plan in plans.items():
        for field in frozen_fields:
            if plan.get(field) != reference.get(field):
                raise ValueError(
                    f'CS-HR manifests differ on frozen field {field}: {condition}'
                )
    return plans


def build_hybrid_run_list(manifest_paths, output_roots, output_path,
                          selected_orders=None):
    """Select the newest validator-passing completed attempt for every unit."""
    plans = _compatible_hybrid_plans(manifest_paths)
    if set(output_roots) != set(CONDITIONS):
        raise ValueError('CS-HR run-list requires one output root per condition')
    known_orders = set(plans['M0']['orders'])
    included_orders = sorted(known_orders if selected_orders is None else selected_orders)
    unknown = set(included_orders) - known_orders
    if unknown:
        raise ValueError(f'Unknown CS-HR order(s): {sorted(unknown)}')
    accepted, missing, rejected = [], [], []
    for condition in CONDITIONS:
        children = {
            (child['order_id'], int(child['training_seed'])): child
            for child in plans[condition]['children']
            if child['order_id'] in included_orders
        }
        for unit, child in sorted(children.items()):
            logical_root = os.path.join(output_roots[condition], child['logical_run_id'])
            selected = None
            failures = []
            for attempt_dir in _completed_attempts(logical_root):
                try:
                    validated = validate_attempt(attempt_dir)
                    if validated['logical_run_id'] != child['logical_run_id']:
                        raise ValueError('logical run identity mismatch')
                    selected = (attempt_dir, validated)
                    break
                except Exception as error:  # preserve every excluded attempt and reason
                    failures.append({
                        'attempt_dir': os.path.abspath(attempt_dir),
                        'error_type': type(error).__name__, 'error': str(error),
                    })
            rejected.extend({
                'condition': condition, 'order_id': unit[0],
                'training_seed': unit[1], 'logical_run_id': child['logical_run_id'],
                **failure,
            } for failure in failures)
            if selected is None:
                missing.append({
                    'condition': condition, 'order_id': unit[0],
                    'training_seed': unit[1], 'logical_run_id': child['logical_run_id'],
                    'logical_root': os.path.abspath(logical_root),
                })
                continue
            attempt_dir, validated = selected
            accepted.append({
                'condition': condition, 'order_id': unit[0],
                'training_seed': unit[1], 'logical_run_id': child['logical_run_id'],
                'attempt_dir': os.path.abspath(attempt_dir),
                'manifest_path': os.path.abspath(manifest_paths[condition]),
                'plan_digest': plans[condition]['plan_digest'],
                'trajectory_digest': validated['trajectory_digest'],
                'final_training_state_digest': validated['final_training_state_digest'],
                'evaluation_cell_count': len(validated['evaluation_cells']),
                'validator': 'sequential.validation.validate_attempt',
            })
    expected = len(CONDITIONS) * len(included_orders) * len(
        plans['M0']['training_seeds']
    )
    payload = {
        'schema_version': 1, 'experiment': 'cs_hr', 'budget_id': 'b100',
        'conditions': list(CONDITIONS), 'included_orders': included_orders,
        'planned_run_count': expected, 'valid_run_count': len(accepted),
        'complete': len(accepted) == expected and not missing,
        'runs': accepted, 'missing_runs': missing, 'rejected_attempts': rejected,
    }
    atomic_json(output_path, payload)
    return payload


def _standardized_paired_effect(differences):
    values = np.asarray(differences, dtype=float)
    if len(values) < 2:
        return None
    spread = float(np.std(values, ddof=1))
    return None if spread == 0 else float(np.mean(values) / spread)


def _paired_effect(index, baseline, candidate, metric, seed, resamples, inferential):
    units = sorted(set(index[baseline]) & set(index[candidate]))
    differences = [
        float(metric(index[baseline][unit])) - float(metric(index[candidate][unit]))
        for unit in units
    ]
    result = {
        'contrast': f'{baseline}_minus_{candidate}',
        'direction': 'positive_means_candidate_has_lower_cost',
        'paired_units': [
            {'order_id': order, 'training_seed': seed_value}
            for order, seed_value in units
        ],
        'paired_differences': differences,
        'effect_mean': float(np.mean(differences)) if differences else None,
        'paired_standardized_effect_dz': _standardized_paired_effect(differences),
        'bootstrap': (
            paired_bootstrap(differences, seed, resamples) if inferential else None
        ),
        'sign_flip': exact_sign_flip_test(differences) if inferential else None,
    }
    return result


def _stratified_effect(index, baseline, candidate, extractor):
    """Return paired descriptive effects for every shared stratum."""
    by_stratum = {}
    units = sorted(set(index[baseline]) & set(index[candidate]))
    for unit in units:
        left = extractor(index[baseline][unit])
        right = extractor(index[candidate][unit])
        for stratum in sorted(set(left) & set(right)):
            by_stratum.setdefault(stratum, []).append(float(left[stratum]) - float(
                right[stratum]
            ))
    return {
        str(stratum): {
            'paired_count': len(values),
            'effect_mean': float(np.mean(values)),
            'paired_differences': values,
        }
        for stratum, values in sorted(by_stratum.items())
    }


def _stage_adaptation(run):
    return {int(row['stage_index']): row['normalized_auc'] for row in run['stages']}


def _final_retention_by_scene_position(run):
    final_stage = max(row['stage_index'] for row in run['stage_scene_evaluations'])
    return {
        int(row['evaluation_scene_introduced_stage']): row[
            'final_retention_normalized'
        ]
        for row in run['stage_scene_evaluations']
        if int(row['stage_index']) == final_stage
    }


def _final_forgetting_by_scene_position(run):
    final_stage = max(row['stage_index'] for row in run['stage_scene_evaluations'])
    return {
        int(row['evaluation_scene_introduced_stage']): row['forgetting_travel_time']
        for row in run['stage_scene_evaluations']
        if (int(row['stage_index']) == final_stage
            and int(row['evaluation_scene_introduced_stage']) < final_stage)
    }


def _replay_performance_association(summaries):
    rows = []
    for run in summaries:
        if run['condition'] not in {'M2', 'M3'}:
            continue
        fractions = [
            row.get('historical_sample_fraction_by_kind')
            for row in run['replay_diagnostics']
            if int(row.get('stage_index') or 0) > 1
            and row.get('historical_sample_fraction_by_kind') is not None
        ]
        if not fractions:
            continue
        mean_fraction = float(np.mean(fractions))
        rows.append({
            'logical_run_id': run['logical_run_id'],
            'condition': run['condition'], 'order_id': run['order_id'],
            'training_seed': int(run['training_seed']),
            'historical_sample_fraction_mean': mean_fraction,
            'absolute_target_deviation': abs(mean_fraction - 0.5),
            'adaptation_auc': run['primary_normalized_auc'],
            'final_retention': run['secondary_metrics'][
                'final_retention_mean_normalized'
            ],
            'forgetting_travel_time': run['secondary_metrics'][
                'stage_end_forgetting_mean_travel_time'
            ],
        })
    correlations = {}
    for condition in ('M2', 'M3'):
        selected = [row for row in rows if row['condition'] == condition]
        correlations[condition] = {}
        for metric in ('adaptation_auc', 'final_retention', 'forgetting_travel_time'):
            if len(selected) < 3:
                value = None
            else:
                x = np.asarray([row['absolute_target_deviation'] for row in selected])
                y = np.asarray([row[metric] for row in selected])
                value = (None if np.std(x) == 0 or np.std(y) == 0 else
                         float(np.corrcoef(x, y)[0, 1]))
            correlations[condition][metric] = {
                'pearson_r': value, 'run_count': len(selected),
                'interpretation': 'descriptive_association_not_causal',
            }
    return {'runs': rows, 'target_deviation_correlations': correlations}


def analyze_hybrid_experiment(run_list_path, parent_catalog_path, output_path):
    run_list = read_json(run_list_path)
    if run_list.get('experiment') != 'cs_hr':
        raise ValueError('Not a CS-HR run-list')
    baselines = plan1_final_means(parent_catalog_path)
    summaries = []
    for record in run_list['runs']:
        summary = summarize_run(validate_attempt(record['attempt_dir']), baselines)
        summary['condition'] = record['condition']
        summaries.append(summary)
    index = {condition: {} for condition in CONDITIONS}
    for run in summaries:
        unit = (run['order_id'], int(run['training_seed']))
        if unit in index[run['condition']]:
            raise ValueError(f'Duplicate CS-HR paired unit: {run["condition"]} {unit}')
        index[run['condition']][unit] = run
    planned_pairs = len(run_list['included_orders']) * 5
    complete = (run_list.get('complete') is True and all(
        len(index[condition]) == planned_pairs for condition in CONDITIONS
    ))
    analysis_seed, resamples = 20260723, 10000
    metric_specs = {
        'adaptation_auc': lambda run: run['primary_normalized_auc'],
        'final_retention': lambda run: run['secondary_metrics'][
            'final_retention_mean_normalized'
        ],
        'forgetting_travel_time': lambda run: run['secondary_metrics'][
            'stage_end_forgetting_mean_travel_time'
        ],
    }
    effects = {}
    primary = []
    for offset, (baseline, candidate) in enumerate(PRESPECIFIED_CONTRASTS):
        contrast = f'{baseline}_minus_{candidate}'
        effects[contrast] = {}
        for metric_offset, (name, accessor) in enumerate(metric_specs.items()):
            effect = _paired_effect(
                index, baseline, candidate, accessor,
                analysis_seed + 10 * offset + metric_offset,
                resamples, complete,
            )
            effects[contrast][name] = effect
            if name == 'adaptation_auc':
                primary.append(effect)
    if complete:
        adjusted = holm_adjust([row['sign_flip']['p_value'] for row in primary])
        for row, value in zip(primary, adjusted):
            row['sign_flip']['holm_adjusted_p_value'] = value
    order_strata = {}
    for order in run_list['included_orders']:
        order_strata[order] = {}
        for baseline, candidate in PRESPECIFIED_CONTRASTS:
            contrast = f'{baseline}_minus_{candidate}'
            units = sorted(set(index[baseline]) & set(index[candidate]))
            units = [unit for unit in units if unit[0] == order]
            differences = [
                index[baseline][unit]['primary_normalized_auc']
                - index[candidate][unit]['primary_normalized_auc']
                for unit in units
            ]
            order_strata[order][contrast] = {
                'paired_count': len(differences),
                'effect_mean': float(np.mean(differences)) if differences else None,
                'paired_differences': differences,
            }
    stage_strata, position_retention, position_forgetting = {}, {}, {}
    for baseline, candidate in PRESPECIFIED_CONTRASTS:
        contrast = f'{baseline}_minus_{candidate}'
        stage_strata[contrast] = _stratified_effect(
            index, baseline, candidate, _stage_adaptation,
        )
        position_retention[contrast] = _stratified_effect(
            index, baseline, candidate, _final_retention_by_scene_position,
        )
        position_forgetting[contrast] = _stratified_effect(
            index, baseline, candidate, _final_forgetting_by_scene_position,
        )
    report = {
        'schema_version': 1, 'experiment': 'cs_hr', 'budget_id': 'b100',
        'mode': 'formal' if complete else 'formal_partial',
        'formal_inference_performed': complete,
        'notice': (None if complete else
                   'Incomplete 20-pair matrix: descriptive paired effects only; '
                   'bootstrap confidence intervals and significance tests are withheld.'),
        'analysis_scope': {
            'included_orders': run_list['included_orders'],
            'valid_run_count': len(summaries), 'paired_unit_target': planned_pairs,
            'complete_formal_matrix': complete,
        },
        'normalization_protocol': {
            'reference': 'plan1_formal_episode_400_final_evaluation_five_seed_mean',
            'scene_denominators': baselines,
        },
        'prespecified_contrasts': [
            f'{left}_minus_{right}' for left, right in PRESPECIFIED_CONTRASTS
        ],
        'runs': summaries, 'paired_effects': effects,
        'order_stratified_adaptation_effects': order_strata,
        'stage_stratified_adaptation_effects': stage_strata,
        'scene_position_stratified_retention_effects': position_retention,
        'scene_position_stratified_forgetting_effects': position_forgetting,
        'replay_composition_performance_association': (
            _replay_performance_association(summaries)
        ),
        'run_list_path': os.path.abspath(run_list_path),
    }
    atomic_json(output_path, report)
    return report
