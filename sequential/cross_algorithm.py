"""Cross-algorithm HA-SODQN pilot manifest construction and validation."""

import copy
import os

import yaml

from .core import canonical_digest
from .ha_manifest import _child, _load_inputs, _repository_identity
from .ha_manifest import load_ha_config
from .initial_state import validate_initial_state_catalog
from .io import atomic_json, read_json, sha256_file


CROSS_PROTOCOL = 'ha_cross_algorithm_v1'
ALGORITHMS = ('independent_dqn', 'double_dqn', 'dueling_double_dqn')
CONDITION_IDS = (
    'CONT_FIFO', 'P1C_DHOA_R25', 'P1C_CQ_R75', 'P1C_CQA_R75',
)


def load_cross_algorithm_config(path):
    path = os.path.abspath(path)
    with open(path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if config.get('schema_version') != 1 or config.get('protocol_id') != CROSS_PROTOCOL:
        raise ValueError('Unsupported cross-algorithm config identity')
    algorithms = tuple(config.get('algorithms', ()))
    if algorithms != ALGORITHMS:
        raise ValueError('Cross-algorithm pilot must contain the frozen algorithms')
    pilot = config.get('pilot', {})
    if pilot.get('order_id') != 'O2' or int(pilot.get('training_seed', -1)) != 0:
        raise ValueError('Cross-algorithm pilot must use O2/seed0')
    if pilot.get('stage_episodes') != [12, 12, 12, 12]:
        raise ValueError('Cross-algorithm pilot stage budget changed')
    conditions = tuple(pilot.get('conditions', ()))
    if tuple(item.get('condition_id') for item in conditions) != CONDITION_IDS:
        raise ValueError('Cross-algorithm pilot conditions changed')
    formal = config.get('formal', {})
    if formal.get('orders') != ['O1', 'O2', 'O3', 'O4']:
        raise ValueError('Cross-algorithm formal orders changed')
    if formal.get('training_seeds') != [0, 1, 2, 3, 4]:
        raise ValueError('Cross-algorithm formal seeds changed')
    if formal.get('stage_episodes') != [100, 100, 100, 100]:
        raise ValueError('Cross-algorithm formal stage budget changed')
    catalogs = config.get('initial_state_catalogs', {})
    if set(catalogs) != set(ALGORITHMS):
        raise ValueError('Cross-algorithm initial-state catalogs are incomplete')
    base_path = os.path.abspath(config['base_config'])
    base = load_ha_config(base_path)
    return {
        'path': path, 'config': copy.deepcopy(config), 'base_path': base_path,
        'base': base,
    }


def _load_catalog(path):
    path = os.path.abspath(path)
    validate_initial_state_catalog(path, revalidate_sources=False)
    catalog = read_json(path)
    return path, catalog, {
        (entry['order_id'], int(entry['training_seed'])): entry
        for entry in catalog['entries']
    }


def _condition_config(base, algorithm_id):
    config = copy.deepcopy(base)
    config['algorithm_id'] = algorithm_id
    return config


def _cross_child(base, initial_catalog_path, initial_index, algorithm_id,
                 condition, stage_episodes, order_id='O2', training_seed=0,
                 experiment_stage='pilot'):
    item = _child(
        _condition_config(base, algorithm_id), initial_index,
        order_id, int(training_seed), condition['archive_mode'],
        condition['method'], float(condition['offline_ratio']), stage_episodes,
        experiment_stage,
    )
    item['condition_id'] = condition['condition_id']
    item['initial_state_catalog'] = initial_catalog_path
    item['logical_run_id'] = (
        f'{algorithm_id}-{condition["condition_id"]}-{order_id}-SD{training_seed}'
    )
    return item


def build_cross_algorithm_pilot_plan(output_path, config_path):
    loaded = load_cross_algorithm_config(config_path)
    config = loaded['config']
    base = loaded['base']
    archive_path, archive, base_initial_path, _ = _load_inputs(base)
    catalogs = {}
    for algorithm_id, relative_path in config['initial_state_catalogs'].items():
        catalogs[algorithm_id] = _load_catalog(relative_path)
    children = []
    for algorithm_id in ALGORITHMS:
        catalog_path, _, initial_index = catalogs[algorithm_id]
        for condition in config['pilot']['conditions']:
            children.append(_cross_child(
                base, catalog_path, initial_index, algorithm_id, condition,
                config['pilot']['stage_episodes'],
            ))
    payload = {
        'schema_version': 1,
        'protocol_id': CROSS_PROTOCOL,
        'mode': 'pilot',
        'launch_authorized': False,
        'child_count': len(children),
        'algorithms': list(ALGORITHMS),
        'conditions': list(CONDITION_IDS),
        'pilot': config['pilot'],
        'base_config_path': loaded['base_path'],
        'base_config_sha256': sha256_file(loaded['base_path']),
        'archive_root_manifest': archive_path,
        'archive_root_manifest_sha256': sha256_file(archive_path),
        'archive_digest': archive['archive_digest'],
        'analysis': base['analysis'],
        'reference_initial_state_catalog': base_initial_path,
        'reference_initial_state_catalog_sha256': sha256_file(base_initial_path),
        'initial_state_catalogs': {
            algorithm_id: {
                'path': path, 'sha256': sha256_file(path),
            }
            for algorithm_id, (path, _, _) in catalogs.items()
        },
        'config_path': loaded['path'],
        'config_sha256': sha256_file(loaded['path']),
        'experiment_stage': 'cross_algorithm_pilot',
        'rng_config': base['rng'],
        **_repository_identity(),
        'children': children,
    }
    payload['plan_digest'] = canonical_digest(payload)
    atomic_json(output_path, payload)
    return payload


def build_cross_algorithm_formal_plan(output_path, config_path):
    loaded = load_cross_algorithm_config(config_path)
    config = loaded['config']
    base = loaded['base']
    archive_path, archive, base_initial_path, _ = _load_inputs(base)
    catalogs = {}
    for algorithm_id, relative_path in config['initial_state_catalogs'].items():
        catalogs[algorithm_id] = _load_catalog(relative_path)
    children = []
    for algorithm_id in ALGORITHMS:
        catalog_path, _, initial_index = catalogs[algorithm_id]
        for order_id in config['formal']['orders']:
            for training_seed in config['formal']['training_seeds']:
                for condition in config['pilot']['conditions']:
                    children.append(_cross_child(
                        base, catalog_path, initial_index, algorithm_id,
                        condition, config['formal']['stage_episodes'],
                        order_id=order_id, training_seed=training_seed,
                        experiment_stage='formal',
                    ))
    payload = {
        'schema_version': 1,
        'protocol_id': CROSS_PROTOCOL,
        'mode': 'formal',
        'launch_authorized': False,
        'child_count': len(children),
        'algorithms': list(ALGORITHMS),
        'conditions': list(CONDITION_IDS),
        'orders': list(config['formal']['orders']),
        'training_seeds': list(config['formal']['training_seeds']),
        'formal': config['formal'],
        'analysis': base['analysis'],
        'base_config_path': loaded['base_path'],
        'base_config_sha256': sha256_file(loaded['base_path']),
        'archive_root_manifest': archive_path,
        'archive_root_manifest_sha256': sha256_file(archive_path),
        'archive_digest': archive['archive_digest'],
        'reference_initial_state_catalog': base_initial_path,
        'reference_initial_state_catalog_sha256': sha256_file(base_initial_path),
        'initial_state_catalogs': {
            algorithm_id: {'path': path, 'sha256': sha256_file(path)}
            for algorithm_id, (path, _, _) in catalogs.items()
        },
        'config_path': loaded['path'],
        'config_sha256': sha256_file(loaded['path']),
        'experiment_stage': 'cross_algorithm_formal',
        'rng_config': base['rng'],
        **_repository_identity(),
        'children': children,
    }
    payload['plan_digest'] = canonical_digest(payload)
    atomic_json(output_path, payload)
    return payload


def validate_cross_algorithm_pilot(path):
    plan = read_json(path)
    if plan.get('schema_version') != 1 or plan.get('protocol_id') != CROSS_PROTOCOL:
        raise ValueError('Unsupported cross-algorithm pilot manifest')
    unsigned = dict(plan)
    recorded = unsigned.pop('plan_digest', None)
    if canonical_digest(unsigned) != recorded:
        raise ValueError('Cross-algorithm pilot plan digest mismatch')
    if plan.get('child_count') != 12:
        raise ValueError('Cross-algorithm pilot must contain 12 children')
    if sha256_file(plan['config_path']) != plan['config_sha256']:
        raise ValueError('Cross-algorithm config changed after plan creation')
    if sha256_file(plan['base_config_path']) != plan['base_config_sha256']:
        raise ValueError('Base HA config changed after plan creation')
    if sha256_file(plan['archive_root_manifest']) != plan['archive_root_manifest_sha256']:
        raise ValueError('HA archive changed after plan creation')
    if sha256_file(plan['reference_initial_state_catalog']) != plan[
            'reference_initial_state_catalog_sha256']:
        raise ValueError('Cross-algorithm reference catalog changed')
    if _repository_identity()['git_commit'] != plan.get('git_commit'):
        raise ValueError('Current checkout differs from cross-algorithm pilot commit')
    for item in plan['initial_state_catalogs'].values():
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError('Cross-algorithm initial-state catalog changed')
    expected = {
        (algorithm, condition)
        for algorithm in ALGORITHMS for condition in CONDITION_IDS
    }
    actual = {
        (child.get('algorithm_id'), child.get('condition_id'))
        for child in plan['children']
    }
    if actual != expected:
        raise ValueError('Cross-algorithm pilot identity matrix mismatch')
    for child in plan['children']:
        if child['stage_episodes'] != [12, 12, 12, 12]:
            raise ValueError('Cross-algorithm pilot stage budget mismatch')
        if child['order_id'] != 'O2' or int(child['training_seed']) != 0:
            raise ValueError('Cross-algorithm pilot order/seed mismatch')
        if child['archive_mode'] == 'NONE':
            if child['condition_id'] != 'CONT_FIFO' or child['offline_ratio'] != 0.0:
                raise ValueError('Invalid cross-algorithm CONT identity')
        elif child['archive_mode'] != 'P1C':
            raise ValueError('Cross-algorithm pilot must use P1C or CONT')
    return {
        'valid': True, 'child_count': plan['child_count'],
        'plan_digest': recorded,
    }


def validate_cross_algorithm_formal_plan(path):
    plan = read_json(path)
    if plan.get('schema_version') != 1 or plan.get('protocol_id') != CROSS_PROTOCOL:
        raise ValueError('Unsupported cross-algorithm formal manifest')
    if plan.get('mode') != 'formal' or plan.get('child_count') != 240:
        raise ValueError('Cross-algorithm formal manifest must contain 240 children')
    unsigned = dict(plan)
    recorded = unsigned.pop('plan_digest', None)
    if canonical_digest(unsigned) != recorded:
        raise ValueError('Cross-algorithm formal plan digest mismatch')
    if sha256_file(plan['config_path']) != plan['config_sha256']:
        raise ValueError('Cross-algorithm config changed after plan creation')
    if sha256_file(plan['base_config_path']) != plan['base_config_sha256']:
        raise ValueError('Base HA config changed after plan creation')
    if sha256_file(plan['archive_root_manifest']) != plan['archive_root_manifest_sha256']:
        raise ValueError('HA archive changed after plan creation')
    if sha256_file(plan['reference_initial_state_catalog']) != plan[
            'reference_initial_state_catalog_sha256']:
        raise ValueError('Cross-algorithm reference catalog changed')
    if _repository_identity()['git_commit'] != plan.get('git_commit'):
        raise ValueError('Current checkout differs from cross-algorithm formal commit')
    for item in plan['initial_state_catalogs'].values():
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError('Cross-algorithm initial-state catalog changed')
    expected = {
        (algorithm, condition, order, seed)
        for algorithm in ALGORITHMS for condition in CONDITION_IDS
        for order in ('O1', 'O2', 'O3', 'O4') for seed in range(5)
    }
    actual = {
        (child.get('algorithm_id'), child.get('condition_id'),
         child.get('order_id'), int(child.get('training_seed')))
        for child in plan['children']
    }
    if actual != expected:
        raise ValueError('Cross-algorithm formal identity matrix mismatch')
    for child in plan['children']:
        if child['stage_episodes'] != [100, 100, 100, 100]:
            raise ValueError('Cross-algorithm formal stage budget mismatch')
        if child['archive_mode'] == 'NONE':
            if child['condition_id'] != 'CONT_FIFO' or child['offline_ratio'] != 0.0:
                raise ValueError('Invalid cross-algorithm formal CONT identity')
        elif child['archive_mode'] != 'P1C':
            raise ValueError('Cross-algorithm formal must use P1C or CONT')
    return {
        'valid': True, 'child_count': plan['child_count'],
        'plan_digest': recorded,
    }
