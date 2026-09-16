"""Analysis for the preregistered HA cross-algorithm matrices."""

import itertools
import os

import numpy as np

from .analysis import exact_sign_flip_test, paired_bootstrap
from .ha_analysis import (
    _plan1_reference_means,
    _resolve_analysis_whitelist,
    _write_csv,
    summarize_ha_run,
)
from .io import atomic_json, read_json
from .validation import validate_ha_attempt


def _analysis_whitelist(plan, whitelist_path):
    """Reuse the HA audit admission rules with the shared reference catalog."""
    adapted = dict(plan)
    adapted['initial_state_catalog'] = plan['reference_initial_state_catalog']
    return _resolve_analysis_whitelist(adapted, whitelist_path)


def _algorithm_effects(runs, analysis_config):
    baseline = {
        (run['algorithm_id'], run['order_id'], int(run['training_seed'])): run
        for run in runs if run['condition_id'] == 'CONT_FIFO'
    }
    groups = {}
    for run in runs:
        if run['condition_id'] == 'CONT_FIFO':
            continue
        key = (run['algorithm_id'], run['condition_id'])
        control = baseline.get((run['algorithm_id'], run['order_id'],
                               int(run['training_seed'])))
        if control is None:
            continue
        groups.setdefault(key, []).append((control, run))
    rows = []
    for offset, ((algorithm, condition), pairs) in enumerate(sorted(groups.items())):
        adaptation = [c['current_adaptation_aulc_mean'] - r['current_adaptation_aulc_mean']
                      for c, r in pairs]
        forgetting = [c['average_forgetting'] - r['average_forgetting']
                      for c, r in pairs]
        retention = [r['average_retention'] - c['average_retention']
                     for c, r in pairs]
        effect = paired_bootstrap(
            adaptation, int(analysis_config['seed']) + offset,
            int(analysis_config['resamples']),
        )
        rows.append({
            'algorithm_id': algorithm, 'condition_id': condition,
            'paired_unit_count': len(pairs),
            'adaptation_effect_mean': effect['effect_mean'],
            'adaptation_ci95_low': effect['confidence_interval_95'][0],
            'adaptation_ci95_high': effect['confidence_interval_95'][1],
            'adaptation_sign_flip_p': exact_sign_flip_test(adaptation)['p_value'],
            'forgetting_effect_mean': float(np.mean(forgetting)),
            'retention_effect_mean': float(np.mean(retention)),
        })
    return rows, groups


def _algorithm_pair_effects(groups, analysis_config):
    conditions = sorted({condition for _, condition in groups})
    algorithms = sorted({algorithm for algorithm, _ in groups})
    rows = []
    offset = 0
    for condition in conditions:
        per_algorithm = {
            algorithm: {
                (c['order_id'], int(c['training_seed'])):
                c['current_adaptation_aulc_mean'] - r['current_adaptation_aulc_mean']
                for c, r in groups.get((algorithm, condition), [])
            }
            for algorithm in algorithms
        }
        for left, right in itertools.combinations(algorithms, 2):
            units = sorted(set(per_algorithm[left]) & set(per_algorithm[right]))
            differences = [per_algorithm[right][unit] - per_algorithm[left][unit]
                           for unit in units]
            if not differences:
                continue
            effect = paired_bootstrap(
                differences, int(analysis_config['seed']) + offset,
                int(analysis_config['resamples']),
            )
            rows.append({
                'condition_id': condition,
                'left_algorithm_id': left,
                'right_algorithm_id': right,
                'paired_unit_count': len(differences),
                'right_minus_left_adaptation_improvement_mean': effect['effect_mean'],
                'ci95_low': effect['confidence_interval_95'][0],
                'ci95_high': effect['confidence_interval_95'][1],
                'sign_flip_p': exact_sign_flip_test(differences)['p_value'],
            })
            offset += 1
    return rows


def analyze_cross_algorithm_experiment(manifest_path, output_root,
                                       whitelist_path, output_dir):
    plan = read_json(manifest_path)
    if plan.get('protocol_id') != 'ha_cross_algorithm_v1':
        raise ValueError('Cross-analysis requires a cross-algorithm manifest')
    whitelist = _analysis_whitelist(plan, whitelist_path)
    references = _plan1_reference_means(whitelist['reference_whitelist_path'])
    runs = []
    for child in plan['children']:
        validated = validate_ha_attempt(
            os.path.join(output_root, child['logical_run_id'])
        )
        if (whitelist['attempts'] is not None and
                os.path.abspath(validated['attempt_dir']) !=
                whitelist['attempts'][child['logical_run_id']]):
            raise ValueError('Cross-analysis effective attempt differs from whitelist')
        run = summarize_ha_run(validated, references)
        run['algorithm_id'] = child['algorithm_id']
        run['condition_id'] = child['condition_id']
        runs.append(run)
    analysis_config = plan.get('analysis', {'seed': 20260909, 'resamples': 10000})
    algorithm_effects, groups = _algorithm_effects(runs, analysis_config)
    pair_effects = _algorithm_pair_effects(groups, analysis_config)
    os.makedirs(output_dir, exist_ok=True)
    run_rows = [{key: run[key] for key in (
        'logical_run_id', 'algorithm_id', 'condition_id', 'archive_mode',
        'method', 'offline_ratio', 'order_id', 'training_seed',
        'current_adaptation_aulc_mean', 'average_forgetting',
        'worst_forgetting', 'average_retention',
    )} for run in runs]
    _write_csv(os.path.join(output_dir, 'runs.csv'), run_rows)
    _write_csv(os.path.join(output_dir, 'adaptation_curves.csv'), [
        {**row, 'algorithm_id': run['algorithm_id'],
         'condition_id': run['condition_id']}
        for run in runs for row in run['adaptation']
    ])
    _write_csv(os.path.join(output_dir, 'algorithm_effects.csv'), algorithm_effects)
    _write_csv(os.path.join(output_dir, 'algorithm_pair_effects.csv'), pair_effects)
    report = {
        'schema_version': 1, 'valid': True,
        'manifest_path': os.path.abspath(manifest_path),
        'output_root': os.path.abspath(output_root),
        'analysis_whitelist': {
            key: value for key, value in whitelist.items() if key != 'attempts'
        },
        'run_count': len(runs),
        'algorithm_effects': algorithm_effects,
        'algorithm_pair_effects': pair_effects,
    }
    atomic_json(os.path.join(output_dir, 'cross_algorithm_analysis.json'), report)
    return report
