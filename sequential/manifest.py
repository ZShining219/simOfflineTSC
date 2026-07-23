import os

from .config import FORMAL_POLICIES, load_sequential_config
from .core import canonical_digest
from .io import atomic_json, read_json


PLAN_SCHEMA_VERSION = 1


def build_formal_plan(parent_catalog_path, output_path,
                      config_path='configs/sequential/plan34.yml'):
    config = load_sequential_config(config_path)
    catalog = read_json(parent_catalog_path)
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
                logical_run_id = f'plan34_{order_id}_seed{seed}_{policy}'
                children.append({
                    'logical_run_id': logical_run_id,
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


def validate_formal_plan(path):
    plan = read_json(path)
    if plan.get('schema_version') != PLAN_SCHEMA_VERSION:
        raise ValueError('Unsupported sequential plan schema')
    if plan.get('mode') != 'formal' or plan.get('launch_authorized') is not False:
        raise ValueError('Formal plan must be frozen and not implicitly launch-authorized')
    children = plan.get('children', [])
    if len(children) != 60 or plan.get('child_count') != 60:
        raise ValueError('Formal plan must contain exactly 60 children')
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
    if int(later_stage_episodes) <= 0:
        raise ValueError('Pilot stage episodes must be positive')
    catalog = read_json(parent_catalog_path)
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
                'logical_run_id': f'pilot_O1_seed0_{policy}_{variant}',
                'order_id': 'O1', 'training_seed': 0,
                'policy': policy, 'variant': variant,
                'fault_point': fault_points[policy] if variant == 'fault' else None,
                'networks': list(config['orders']['O1']),
                'stage_episodes': [400] + [int(later_stage_episodes)] * 3,
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
