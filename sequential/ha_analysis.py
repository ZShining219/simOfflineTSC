"""Strict-validator-backed HA-SODQN tables, selection, and plots."""

import csv
import hashlib
import os

import numpy as np

from .analysis import exact_sign_flip_test, paired_bootstrap
from .io import atomic_json, read_json
from .validation import validate_ha_attempt


METRICS = ('travel_time', 'delay', 'real_delay', 'queue', 'throughput', 'reward_mean')


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plan1_reference_means(whitelist_path):
    references = {}
    with open(whitelist_path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            final = None
            records = os.path.join(row['run_path'], 'metrics', 'records.jsonl')
            with open(records, encoding='utf-8') as source:
                for line in source:
                    record = read_json_line(line)
                    if record.get('record_type') == 'FINAL_EVALUATION':
                        final = record
            if final is None:
                raise ValueError(f'Missing Plan 1 final evaluation: {records}')
            references.setdefault(row['network'], []).append(float(final['travel_time']))
    if any(len(values) != 5 for values in references.values()):
        raise ValueError('HA analysis requires five Plan 1 references per network')
    return {network: float(np.mean(values)) for network, values in references.items()}


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_analysis_whitelist(plan, whitelist_path):
    """Resolve strict HA-audit admission and the Plan 1 normalization source.

    Historical callers passed the Plan 1 trajectory CSV directly.  Formal HA
    analysis instead passes the successful HA audit as its admission whitelist;
    the Plan 1 reference CSV is then recovered from the frozen initial-state
    catalog referenced by the experiment manifest.
    """
    absolute = os.path.abspath(whitelist_path)
    try:
        audit = read_json(absolute)
    except (UnicodeDecodeError, ValueError):
        audit = None
    if not isinstance(audit, dict) or not {'valid', 'runs'} <= set(audit):
        return {
            'kind': 'plan1_trajectory_csv',
            'path': absolute,
            'sha256': _file_sha256(absolute),
            'run_count': None,
            'attempts': None,
            'reference_whitelist_path': absolute,
        }
    if audit.get('valid') is not True:
        raise ValueError('HA analysis whitelist audit is not valid')
    rows = audit.get('runs')
    if not isinstance(rows, list) or audit.get('run_count') != len(rows):
        raise ValueError('HA analysis whitelist audit run count mismatch')
    attempts = {}
    for row in rows:
        logical_id = row.get('logical_run_id')
        attempt_dir = row.get('attempt_dir')
        if not logical_id or not attempt_dir or logical_id in attempts:
            raise ValueError('HA analysis whitelist audit identity is invalid')
        if (row.get('ha_audit') or {}).get('valid') is not True:
            raise ValueError('HA analysis whitelist contains an invalid run')
        attempts[logical_id] = os.path.abspath(attempt_dir)
    expected = {child['logical_run_id'] for child in plan.get('children', [])}
    if set(attempts) != expected:
        raise ValueError('HA analysis whitelist identity set mismatch')
    catalog_path = os.path.abspath(plan['initial_state_catalog'])
    catalog = read_json(catalog_path)
    reference_path = os.path.abspath(catalog['whitelist_path'])
    return {
        'kind': 'ha_audit_json',
        'path': absolute,
        'sha256': _file_sha256(absolute),
        'run_count': len(rows),
        'attempts': attempts,
        'reference_whitelist_path': reference_path,
        'reference_whitelist_sha256': _file_sha256(reference_path),
        'initial_state_catalog_path': catalog_path,
        'initial_state_catalog_sha256': _file_sha256(catalog_path),
    }


def read_json_line(line):
    import json
    return json.loads(line)


def _series_auc(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    if len(values) == 1:
        return float(values[0])
    return float(np.trapz(values, dx=1.0) / (len(values) - 1))


def summarize_ha_run(validated, references):
    child = validated['child_manifest']
    cells = validated['evaluation_cells']
    adaptation = []
    stage_rows = []
    lower_triangle = []
    own_end = {}
    for stage, network in enumerate(child['networks'], start=1):
        budget = int(child['stage_episodes'][stage - 1])
        locals_ = range(1, budget + 1) if stage == 1 else range(0, budget + 1)
        series = []
        for local in locals_:
            summary = cells[(stage, local, network)]['summary']
            normalized = float(summary['travel_time']) / references[network]
            series.append(normalized)
            adaptation.append({
                'logical_run_id': child['logical_run_id'],
                'archive_mode': child['archive_mode'], 'method': child['method'],
                'offline_ratio': child['offline_ratio'], 'order_id': child['order_id'],
                'training_seed': child['training_seed'], 'stage_index': stage,
                'network': network, 'local_episode': local,
                'normalized_travel_time': normalized,
                **{metric: float(summary[metric]) for metric in METRICS},
            })
        own_end[network] = float(cells[(stage, budget, network)]['summary']['travel_time'])
        horizons = {}
        for horizon in (20, 50, 100):
            count = min(horizon if stage == 1 else horizon + 1, len(series))
            horizons[f'normalized_travel_time_aulc_0_{horizon}'] = _series_auc(series[:count])
        stage_rows.append({
            'stage_index': stage, 'network': network,
            'stage_final_travel_time': own_end[network],
            **horizons,
        })
        for evaluation_network in child['networks'][:stage]:
            summary = cells[(stage, budget, evaluation_network)]['summary']
            lower_triangle.append({
                'logical_run_id': child['logical_run_id'],
                'stage_index': stage, 'training_network': network,
                'evaluation_network': evaluation_network,
                **{metric: float(summary[metric]) for metric in METRICS},
            })
    final_stage = len(child['networks'])
    final_budget = int(child['stage_episodes'][-1])
    forgetting = []
    for position, network in enumerate(child['networks'], start=1):
        final = float(cells[(final_stage, final_budget, network)]['summary']['travel_time'])
        absolute = final - own_end[network]
        forgetting.append({
            'network': network, 'scene_position': position,
            'own_stage_end_travel_time': own_end[network],
            'final_travel_time': final,
            'absolute_forgetting': absolute,
            'relative_forgetting': absolute / own_end[network],
            'retention_ratio': own_end[network] / final if final else None,
        })
    historical_forgetting = [row for row in forgetting if row['scene_position'] < final_stage]
    mechanism = []
    for key, operation in validated['state']['completed_operations'].items():
        if not key.endswith(':diagnostic'):
            continue
        diagnostic = read_json(operation['artifact'])
        windows = diagnostic.get('sampling_windows', [])
        mechanism.append({
            'logical_run_id': child['logical_run_id'],
            'stage_index': diagnostic['stage_index'],
            'local_episode': diagnostic['local_episode'],
            'orb_size': diagnostic['replay_size'],
            'visible_hoa_count': (
                0 if diagnostic.get('visible_archive') is None
                else diagnostic['visible_archive']['transition_count']
            ),
            'owp_digest': (diagnostic.get('owp_manifest') or {}).get('owp_digest'),
            'owp_selected_count': (diagnostic.get('owp_manifest') or {}).get('selected_count'),
            'mean_actual_offline_ratio': (
                None if not windows else float(np.mean([
                    row['actual_offline_ratio'] for row in windows
                ]))
            ),
            'online_loss_mean': (
                None if not windows else float(np.mean([row['loss_online'] for row in windows]))
            ),
            'offline_loss_mean': (
                None if not any(row['loss_offline'] is not None for row in windows)
                else float(np.mean([
                    row['loss_offline'] for row in windows
                    if row['loss_offline'] is not None
                ]))
            ),
            'offline_unique_samples_used': diagnostic.get('historical_unique_samples_used', 0),
        })
    full_aulc = [row['normalized_travel_time_aulc_0_100'] for row in stage_rows]
    return {
        'logical_run_id': child['logical_run_id'],
        'archive_mode': child['archive_mode'], 'method': child['method'],
        'offline_ratio': float(child['offline_ratio']),
        'order_id': child['order_id'], 'training_seed': int(child['training_seed']),
        'stage_rows': stage_rows, 'adaptation': adaptation,
        'lower_triangle': lower_triangle, 'forgetting': forgetting,
        'mechanism': mechanism,
        'current_adaptation_aulc_mean': float(np.mean(full_aulc)),
        'average_forgetting': float(np.mean([
            row['absolute_forgetting'] for row in historical_forgetting
        ])) if historical_forgetting else 0.0,
        'worst_forgetting': float(np.max([
            row['absolute_forgetting'] for row in historical_forgetting
        ])) if historical_forgetting else 0.0,
        'average_retention': float(np.mean([
            row['retention_ratio'] for row in historical_forgetting
        ])) if historical_forgetting else 1.0,
    }


def _selection_table(runs):
    baseline = next((run for run in runs if run['archive_mode'] == 'NONE'), None)
    if baseline is None:
        return [], []
    rows = []
    for run in runs:
        adaptation_degradation = (
            run['current_adaptation_aulc_mean'] / baseline['current_adaptation_aulc_mean'] - 1
        )
        improves = (
            run['average_forgetting'] < baseline['average_forgetting']
            or run['worst_forgetting'] < baseline['worst_forgetting']
            or run['average_retention'] > baseline['average_retention']
        )
        rows.append({
            'logical_run_id': run['logical_run_id'],
            'archive_mode': run['archive_mode'], 'method': run['method'],
            'offline_ratio': run['offline_ratio'],
            'adaptation_degradation_vs_cont': adaptation_degradation,
            'average_forgetting': run['average_forgetting'],
            'worst_forgetting': run['worst_forgetting'],
            'average_retention': run['average_retention'],
            'passes_five_percent_adaptation_gate': adaptation_degradation <= 0.05,
            'improves_retention_or_forgetting': improves,
            'eligible': run['archive_mode'] == 'NONE' or (
                adaptation_degradation <= 0.05 and improves
            ),
        })
    selected = [next(row for row in rows if row['archive_mode'] == 'NONE')]
    for method in ('DHOA', 'CQ', 'CQA'):
        candidates = [row for row in rows if row['method'] == method and row['eligible']]
        if candidates:
            selected.append(min(candidates, key=lambda row: (
                row['average_forgetting'], row['adaptation_degradation_vs_cont'],
            )))
    return rows, selected


def _paired_effects(runs, analysis_config):
    baseline = {
        (run['order_id'], run['training_seed']): run
        for run in runs if run['archive_mode'] == 'NONE'
    }
    effects = []
    candidates = sorted({
        (run['archive_mode'], run['method'], run['offline_ratio'])
        for run in runs if run['archive_mode'] != 'NONE'
    })
    for offset, identity in enumerate(candidates):
        paired = []
        for run in runs:
            if (run['archive_mode'], run['method'], run['offline_ratio']) != identity:
                continue
            control = baseline.get((run['order_id'], run['training_seed']))
            if control is not None:
                paired.append(
                    control['current_adaptation_aulc_mean']
                    - run['current_adaptation_aulc_mean']
                )
        if paired:
            effects.append({
                'archive_mode': identity[0], 'method': identity[1],
                'offline_ratio': identity[2], 'paired_unit_count': len(paired),
                **paired_bootstrap(
                    paired, int(analysis_config['seed']) + offset,
                    int(analysis_config['resamples']),
                ),
                'sign_flip_p_value': exact_sign_flip_test(paired)['p_value'],
            })
    return effects


def _plots(output_dir, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plots = []
    figure, axis = plt.subplots(figsize=(10, 6))
    for run in runs:
        values = sorted(run['adaptation'], key=lambda row: (
            row['stage_index'], row['local_episode'],
        ))
        axis.plot(
            range(len(values)), [row['normalized_travel_time'] for row in values],
            label=run['logical_run_id'], alpha=.8,
        )
    axis.set_xlabel('Sequential adaptation evaluation index')
    axis.set_ylabel('Normalized travel time')
    axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    path = os.path.join(output_dir, 'adaptation_curves.png')
    figure.savefig(path, dpi=180)
    plt.close(figure)
    plots.append(path)

    figure, axis = plt.subplots(figsize=(10, 6))
    labels = [run['logical_run_id'] for run in runs]
    axis.bar(np.arange(len(runs)) - .2, [run['average_forgetting'] for run in runs], .4,
             label='average forgetting')
    axis.bar(np.arange(len(runs)) + .2, [run['worst_forgetting'] for run in runs], .4,
             label='worst forgetting')
    axis.set_xticks(np.arange(len(runs)), labels, rotation=90, fontsize=6)
    axis.legend()
    figure.tight_layout()
    path = os.path.join(output_dir, 'forgetting_summary.png')
    figure.savefig(path, dpi=180)
    plt.close(figure)
    plots.append(path)
    return plots


def analyze_ha_experiment(manifest_path, output_root, whitelist_path, output_dir):
    plan = read_json(manifest_path)
    whitelist = _resolve_analysis_whitelist(plan, whitelist_path)
    references = _plan1_reference_means(whitelist['reference_whitelist_path'])
    runs = []
    for child in plan['children']:
        validated = validate_ha_attempt(
            os.path.join(output_root, child['logical_run_id'])
        )
        if (whitelist['attempts'] is not None and
                os.path.abspath(validated['attempt_dir']) !=
                whitelist['attempts'][child['logical_run_id']]):
            raise ValueError('HA analysis effective attempt differs from whitelist')
        runs.append(summarize_ha_run(validated, references))
    os.makedirs(output_dir, exist_ok=True)
    run_rows = [{key: run[key] for key in (
        'logical_run_id', 'archive_mode', 'method', 'offline_ratio',
        'order_id', 'training_seed', 'current_adaptation_aulc_mean',
        'average_forgetting', 'worst_forgetting', 'average_retention',
    )} for run in runs]
    selection_rows, selected = _selection_table(runs)
    effects = _paired_effects(runs, plan['analysis'])
    _write_csv(os.path.join(output_dir, 'runs.csv'), run_rows)
    _write_csv(os.path.join(output_dir, 'adaptation_curves.csv'), [
        row for run in runs for row in run['adaptation']
    ])
    _write_csv(os.path.join(output_dir, 'lower_triangle.csv'), [
        row for run in runs for row in run['lower_triangle']
    ])
    _write_csv(os.path.join(output_dir, 'mechanism.csv'), [
        row for run in runs for row in run['mechanism']
    ])
    _write_csv(os.path.join(output_dir, 'selection.csv'), selection_rows)
    _write_csv(os.path.join(output_dir, 'paired_effects.csv'), effects)
    plots = _plots(output_dir, runs)
    report = {
        'schema_version': 1, 'valid': True,
        'manifest_path': os.path.abspath(manifest_path),
        'output_root': os.path.abspath(output_root),
        'analysis_whitelist': {
            key: value for key, value in whitelist.items() if key != 'attempts'
        },
        'run_count': len(runs), 'runs': runs,
        'selection_table': selection_rows,
        'preregistered_selection': selected,
        'paired_effects': effects,
        'plots': plots,
    }
    atomic_json(os.path.join(output_dir, 'ha_analysis.json'), report)
    return report
