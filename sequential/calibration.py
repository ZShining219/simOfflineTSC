import json
import os

from .io import atomic_json, read_json


METRIC_FIELDS = (
    'travel_time', 'reward_mean', 'queue', 'delay', 'real_delay',
    'throughput', 'waiting_time', 'unfinished_vehicles', 'phase_switches',
    'phase_switch_frequency', 'action_distribution',
)


def compare_evaluator_calibration(legacy_path, new_alias_path, output_path=None):
    legacy = read_json(legacy_path)
    alias = read_json(new_alias_path)
    committed = read_json(alias['physical_committed_path'])
    summary = read_json(committed['summary_path'])
    decisions = []
    with open(committed['decisions_path'], encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                decisions.append(json.loads(line))
    new_actions = [row['actions'] for row in decisions]
    legacy_metrics = legacy['metrics']
    metric_checks = {
        field: legacy_metrics[field] == summary[field] for field in METRIC_FIELDS
    }
    report = {
        'schema_version': 1, 'network': legacy['network'],
        'checkpoint_digest': alias['identity']['checkpoint_digest'],
        'decision_count_legacy': legacy['decision_count'],
        'decision_count_new': len(new_actions),
        'action_sequence_equal': legacy['action_sequence'] == new_actions,
        'metric_checks': metric_checks,
    }
    report['valid'] = (
        report['decision_count_legacy'] == report['decision_count_new'] == 360
        and report['action_sequence_equal'] and all(metric_checks.values())
    )
    if output_path is not None:
        atomic_json(output_path, report)
    return report
