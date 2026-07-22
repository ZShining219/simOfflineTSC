import copy
import json
import os

import yaml


ALLOWED_COMMAND_FIELDS = {'network', 'prefix', 'seed'}
ALLOWED_WORLD_FIELDS = {
    'combined_file', 'roadnetFile', 'flowFile',
    'convertroadnetFile', 'convertflowFile',
}
RUNTIME_WORLD_METADATA_FIELDS = {'roadnetLogFile', 'replayLogFile'}


def _load_yaml(path):
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _normalize_resolved_config(config):
    normalized = copy.deepcopy(config)
    for field in ALLOWED_COMMAND_FIELDS:
        if field in normalized.get('command', {}):
            normalized['command'][field] = '<ALLOWED>'
    for field in ALLOWED_WORLD_FIELDS:
        if field in normalized.get('world', {}):
            normalized['world'][field] = '<ALLOWED>'
    for field in RUNTIME_WORLD_METADATA_FIELDS:
        if field in normalized.get('world', {}):
            normalized['world'][field] = '<RUN_METADATA>'
    record = normalized.get('config_record')
    if isinstance(record, dict):
        record.pop('created_at_utc', None)
        sources = record.get('sources')
        if isinstance(sources, list):
            record['sources'] = [
                'configs/sim/<ALLOWED_NETWORK>.cfg'
                if source.startswith('configs/sim/') else source
                for source in sources
            ]
    return normalized


def _normalize_runtime_model(model):
    normalized = copy.deepcopy(model)
    normalized.pop('reproducibility_probe', None)
    for agent in normalized.get('agents', []):
        agent.pop('online_model_state_hash', None)
        agent.pop('target_model_state_hash', None)
    return normalized


def load_normalized_run(run_path):
    config_path = os.path.join(run_path, 'config')
    return {
        'resolved_config': _normalize_resolved_config(
            _load_yaml(os.path.join(config_path, 'resolved_config.yaml'))
        ),
        'runtime_model': _normalize_runtime_model(
            _load_json(os.path.join(config_path, 'model_resolved.json'))
        ),
    }


def _differences(left, right, path=''):
    if type(left) is not type(right):
        return [f'{path}: type {type(left).__name__} != {type(right).__name__}']
    if isinstance(left, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            child = f'{path}.{key}' if path else key
            if key not in left:
                differences.append(f'{child}: missing from reference')
            elif key not in right:
                differences.append(f'{child}: missing from candidate')
            else:
                differences.extend(_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [f'{path}: length {len(left)} != {len(right)}']
        differences = []
        for index, (first, second) in enumerate(zip(left, right)):
            differences.extend(_differences(first, second, f'{path}[{index}]'))
        return differences
    return [] if left == right else [f'{path}: {left!r} != {right!r}']


def _dimensions(run):
    dimensions = []
    for agent in run['runtime_model'].get('agents', []):
        model = agent.get('model') or {}
        dimensions.append((model.get('input_dim'), agent.get('action_dim')))
    return dimensions


def compare_runs(run_paths):
    if len(run_paths) < 2:
        raise ValueError('At least two run paths are required')
    loaded = [load_normalized_run(path) for path in run_paths]
    reference_dimensions = _dimensions(loaded[0])
    results = []
    for path, candidate in zip(run_paths[1:], loaded[1:]):
        candidate_dimensions = _dimensions(candidate)
        if candidate_dimensions != reference_dimensions:
            results.append({
                'run': path,
                'compatible': False,
                'differences': [
                    'runtime input_dim/action_dim mismatch: '
                    f'{reference_dimensions!r} != {candidate_dimensions!r}'
                ],
            })
            continue
        differences = _differences(loaded[0], candidate)
        results.append({
            'run': path,
            'compatible': not differences,
            'differences': differences,
        })
    return {
        'reference': run_paths[0],
        'compatible': all(result['compatible'] for result in results),
        'comparisons': results,
    }
