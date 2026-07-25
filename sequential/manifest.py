import os

from .config import FORMAL_POLICIES, load_sequential_config
from .core import canonical_digest
from .io import atomic_json, read_json


PLAN_SCHEMA_VERSION = 1


def build_formal_plan(parent_catalog_path, output_path,
                      config_path='configs/sequential/plan34.yml'):
    config = load_sequential_config(config_path)
    catalog = read_json(parent_catalog_path)
    budget_id = config['budget_id']
    parent_episode = int(config['parent_checkpoint_episode'])
    if (
        catalog.get('budget_id') != budget_id
        or catalog.get('parent_checkpoint_episode') != parent_episode
    ):
        raise ValueError('Parent catalog does not match sequential budget config')
    parent_index = {
        (item['order_id'], int(item['training_seed'])): item
        for item in catalog.get('parents', [])
    }
    children = []
    for order_id, networks in config['orders'].items():
        for seed in config['training_seeds']:
            parent = parent_index.get((order_id, seed))
            if parent is None:
                raise ValueError(f'Missing parent for {order_id} seed {seed}')
            for policy in FORMAL_POLICIES:
                logical_run_id = f'plan34_{budget_id}_{order_id}_seed{seed}_{policy}'
                children.append({
                    'logical_run_id': logical_run_id,
                    'budget_id': budget_id,
                    'parent_checkpoint_episode': parent_episode,
                    'order_id': order_id,
                    'training_seed': seed,
                    'policy': policy,
                    'networks': list(networks),
                    'stage_episodes': list(config['trainer']['stage_episodes']),
                    'parent_import_manifest': parent['import_manifest_path'],
                    'parent_checkpoint': parent['checkpoint_path'],
                    'parent_checkpoint_file_sha256': parent['checkpoint_file_sha256'],
                    'parent_digests': parent['digests'],
                    'trace_replay_samples': False,
                    'estimated_output_bytes': 2 * 1024 ** 3,
                    'status': 'planned',
                })
    payload = {
        'schema_version': PLAN_SCHEMA_VERSION,
        'budget_id': budget_id,
        'parent_checkpoint_episode': parent_episode,
        'mode': 'formal',
        'launch_authorized': False,
        'child_count': len(children),
        'orders': config['orders'],
        'training_seeds': config['training_seeds'],
        'policies': list(FORMAL_POLICIES),
        'trainer': config['trainer'],
        'model': config['model'],
        'analysis': config['analysis'],
        'parent_catalog_path': os.path.abspath(parent_catalog_path),
        'children': children,
    }
    payload['plan_digest'] = canonical_digest(payload)
    atomic_json(output_path, payload)
    return payload


def build_hybrid_plan(parent_catalog_path, output_path,
                      config_path='configs/sequential/plan34_b100.yml',
                      condition='M3'):
    """Build the b100 CS-HR child matrix (20 order/seed pairs).

    Existing Plan 3/4 builders remain unchanged.  The hybrid runtime is
    selected explicitly through ``policy=hybrid`` and carries a condition
    label so M0/M1/M2/M3 results cannot be mixed accidentally.
    """
    config = load_sequential_config(config_path)
    if config['budget_id'] != 'b100':
        raise ValueError('CS-HR formal plan requires b100')
    condition_settings = {
        'M0': {'policy': 'clear', 'online_ratio': None,
               'historical_sampling': None},
        'M1': {'policy': 'fifo', 'online_ratio': None,
               'historical_sampling': None},
        'M2': {'policy': 'hybrid', 'online_ratio': 0.5,
               'historical_sampling': 'uniform_cumulative'},
        'M3': {'policy': 'hybrid', 'online_ratio': 0.5,
               'historical_sampling': 'stage_balanced_episode_stratified'},
    }
    settings = condition_settings.get(condition)
    if settings is None:
        raise ValueError(f'Unsupported hybrid condition: {condition}')
    catalog = read_json(parent_catalog_path)
    if (catalog.get('budget_id') != 'b100' or
            catalog.get('parent_checkpoint_episode') != 100):
        raise ValueError('Parent catalog does not match CS-HR b100')
    parent_index = {(item['order_id'], int(item['training_seed'])): item
                    for item in catalog.get('parents', [])}
    children = []
    for order_id, networks in config['orders'].items():
        for seed in config['training_seeds']:
            parent = parent_index.get((order_id, seed))
            if parent is None:
                raise ValueError(f'Missing parent for {order_id} seed {seed}')
            children.append({
                'logical_run_id': f'cs_hr_b100_{condition}_{order_id}_seed{seed}',
                'budget_id': 'b100', 'parent_checkpoint_episode': 100,
                'order_id': order_id, 'training_seed': seed,
                'policy': settings['policy'], 'condition': condition,
                'hybrid_online_ratio': settings['online_ratio'],
                'historical_sampling': settings['historical_sampling'],
                'networks': list(networks), 'stage_episodes': [100] * 4,
                'parent_import_manifest': parent['import_manifest_path'],
                'parent_checkpoint': parent['checkpoint_path'],
                'parent_checkpoint_file_sha256': parent['checkpoint_file_sha256'],
                'parent_digests': parent['digests'],
                'trace_replay_samples': False,
                'estimated_output_bytes': 2 * 1024 ** 3, 'status': 'planned',
            })
    payload = {
        'schema_version': PLAN_SCHEMA_VERSION, 'budget_id': 'b100',
        'parent_checkpoint_episode': 100, 'mode': 'hybrid_formal',
        'launch_authorized': False, 'child_count': len(children),
        'orders': config['orders'], 'training_seeds': config['training_seeds'],
        'policies': [settings['policy']], 'condition': condition,
        'trainer': config['trainer'], 'model': config['model'],
        'analysis': config['analysis'],
        'parent_catalog_path': os.path.abspath(parent_catalog_path),
        'children': children,
    }
    payload['plan_digest'] = canonical_digest(payload)
    atomic_json(output_path, payload)
    return payload


def validate_hybrid_plan(path):
    plan = read_json(path)
    if plan.get('schema_version') != PLAN_SCHEMA_VERSION or plan.get('mode') != 'hybrid_formal':
        raise ValueError('Unsupported CS-HR formal plan')
    if plan.get('budget_id') != 'b100' or plan.get('child_count') != 20:
        raise ValueError('CS-HR plan must contain exactly 20 b100 children')
    if plan.get('condition') not in {'M0', 'M1', 'M2', 'M3'}:
        raise ValueError('CS-HR plan condition identity mismatch')
    condition = plan['condition']
    expected_policy = {'M0': 'clear', 'M1': 'fifo', 'M2': 'hybrid', 'M3': 'hybrid'}[condition]
    if plan.get('policies') != [expected_policy]:
        raise ValueError('CS-HR plan policy identity mismatch')
    identities = {(c['order_id'], int(c['training_seed'])) for c in plan['children']}
    expected = {(f'O{o}', s) for o in range(1, 5) for s in range(5)}
    if identities != expected or any(c.get('policy') != expected_policy for c in plan['children']):
        raise ValueError('CS-HR child matrix is incomplete or duplicated')
    digest_payload = dict(plan); recorded = digest_payload.pop('plan_digest', None)
    if canonical_digest(digest_payload) != recorded:
        raise ValueError('CS-HR plan digest mismatch')
    return {'valid': True, 'child_count': 20, 'plan_digest': recorded}


def validate_formal_plan(path):
    plan = read_json(path)
    if plan.get('schema_version') != PLAN_SCHEMA_VERSION:
        raise ValueError('Unsupported sequential plan schema')
    if plan.get('mode') != 'formal' or plan.get('launch_authorized') is not False:
        raise ValueError('Formal plan must be frozen and not implicitly launch-authorized')
    children = plan.get('children', [])
    if len(children) != 60 or plan.get('child_count') != 60:
        raise ValueError('Formal plan must contain exactly 60 children')
    budget_id = plan.get('budget_id')
    parent_episode = plan.get('parent_checkpoint_episode')
    expected_stages = {
        'b400': [400, 100, 100, 100],
        'b100': [100, 100, 100, 100],
    }.get(budget_id)
    if expected_stages is None or parent_episode != expected_stages[0]:
        raise ValueError('Formal plan budget/checkpoint identity mismatch')
    for child in children:
        expected_id = (
            f'plan34_{budget_id}_{child["order_id"]}_seed'
            f'{child["training_seed"]}_{child["policy"]}'
        )
        if (
            child.get('logical_run_id') != expected_id
            or child.get('budget_id') != budget_id
            or child.get('parent_checkpoint_episode') != parent_episode
            or child.get('stage_episodes') != expected_stages
        ):
            raise ValueError('Formal child budget identity is inconsistent')
    identities = {
        (child['order_id'], int(child['training_seed']), child['policy'])
        for child in children
    }
    expected = {
        (f'O{order}', seed, policy)
        for order in range(1, 5) for seed in range(5)
        for policy in FORMAL_POLICIES
    }
    if identities != expected:
        raise ValueError('Formal child matrix is incomplete or duplicated')
    if any(child.get('trace_replay_samples') for child in children):
        raise ValueError('Formal plan must disable full replay sample tracing')
    digest_payload = dict(plan)
    recorded = digest_payload.pop('plan_digest', None)
    if canonical_digest(digest_payload) != recorded:
        raise ValueError('Formal plan digest mismatch')
    return {'valid': True, 'child_count': 60, 'plan_digest': recorded}


def build_pilot_plan(parent_catalog_path, output_path, later_stage_episodes=15,
                     config_path='configs/sequential/plan34.yml'):
    config = load_sequential_config(config_path)
    budget_id = config['budget_id']
    parent_episode = int(config['parent_checkpoint_episode'])
    if int(later_stage_episodes) <= 0:
        raise ValueError('Pilot stage episodes must be positive')
    catalog = read_json(parent_catalog_path)
    if (
        catalog.get('budget_id') != budget_id
        or catalog.get('parent_checkpoint_episode') != parent_episode
    ):
        raise ValueError('Parent catalog does not match pilot budget config')
    parent = next(
        item for item in catalog['parents']
        if item['order_id'] == 'O1' and int(item['training_seed']) == 0
    )
    fault_points = {
        'clear': 'after_REPLAY_POLICY_APPLIED_before_local0',
        'fifo': 'stage2_matrix_half_complete',
        'fifo_matched_wait': 'stage3_episode4_simulation_step180',
    }
    children = []
    for policy in FORMAL_POLICIES:
        for variant in ('control', 'fault'):
            children.append({
                'logical_run_id': f'pilot_{budget_id}_O1_seed0_{policy}_{variant}',
                'budget_id': budget_id,
                'parent_checkpoint_episode': parent_episode,
                'order_id': 'O1', 'training_seed': 0,
                'policy': policy, 'variant': variant,
                'fault_point': fault_points[policy] if variant == 'fault' else None,
                'networks': list(config['orders']['O1']),
                'stage_episodes': [parent_episode] + [int(later_stage_episodes)] * 3,
                'parent_import_manifest': parent['import_manifest_path'],
                'parent_checkpoint': parent['checkpoint_path'],
                'parent_checkpoint_file_sha256': parent['checkpoint_file_sha256'],
                'parent_digests': parent['digests'],
                'trace_replay_samples': False,
                'estimated_output_bytes': 2 * 1024 ** 3,
                'interface': 'libsumo', 'status': 'planned',
            })
    payload = {
        'schema_version': PLAN_SCHEMA_VERSION,
        'budget_id': budget_id,
        'parent_checkpoint_episode': parent_episode,
        'mode': 'pilot', 'launch_authorized': True,
        'child_count': 6,
        'orders': {'O1': config['orders']['O1']},
        'training_seeds': [0], 'policies': list(FORMAL_POLICIES),
        'trainer': config['trainer'], 'model': config['model'],
        'analysis': config['analysis'],
        'parent_catalog_path': os.path.abspath(parent_catalog_path),
        'children': children,
    }
    payload['plan_digest'] = canonical_digest(payload)
    atomic_json(output_path, payload)
    return payload
