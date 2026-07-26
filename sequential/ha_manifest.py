"""Configuration and manifest support for HA-SODQN experiment stages."""

import copy
import os
import subprocess

import yaml

from .config import FORMAL_NETWORKS
from .core import canonical_digest
from .historical_archive import ARCHIVE_SCHEMA_VERSION
from .initial_state import validate_initial_state_catalog
from .io import atomic_json, read_json, sha256_file


HA_PLAN_SCHEMA_VERSION = 2


def load_ha_config(path='configs/sequential/ha_sodqn_b100.yml'):
    with open(path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    validate_ha_config(config)
    return copy.deepcopy(config)


def validate_ha_config(config):
    if config.get('schema_version') != 1 or config.get('protocol_id') != 'ha_sodqn_b100_v1':
        raise ValueError('Unsupported HA-SODQN config identity')
    orders = config.get('orders', {})
    if set(orders) != {'O1', 'O2', 'O3', 'O4'}:
        raise ValueError('HA-SODQN config must define O1..O4')
    for order_id, networks in orders.items():
        if len(networks) != 4 or set(networks) != set(FORMAL_NETWORKS):
            raise ValueError(f'{order_id} must contain every formal network once')
    if config.get('training_seeds') != [0, 1, 2, 3, 4]:
        raise ValueError('HA-SODQN training seeds must be 0..4')
    if config.get('archive_modes') != ['P1C', 'P1F']:
        raise ValueError('HA-SODQN archive modes changed')
    if config.get('methods') != ['DHOA', 'RAND', 'COV', 'CQ', 'CQA']:
        raise ValueError('HA-SODQN method definitions changed')
    if config.get('offline_ratios') != [0.25, 0.5, 0.75]:
        raise ValueError('HA-SODQN offline ratios changed')
    trainer = config.get('trainer', {})
    required = {
        'stage_episodes': [100, 100, 100, 100],
        'steps': 3600, 'test_steps': 3600, 'action_interval': 10,
        'learning_start': 1000, 'buffer_size': 5000, 'batch_size': 64,
        'update_model_rate': 1, 'target_update_interval': 10,
        'max_child_concurrency': 8,
    }
    for key, expected in required.items():
        if trainer.get(key) != expected:
            raise ValueError(f'Invalid frozen HA-SODQN trainer setting: {key}')
    archive = config.get('archive', {})
    if archive.get('source_episode_range') != [1, 100]:
        raise ValueError('HA-SODQN source episode range changed')
    if archive.get('owp_capacity') != 5000:
        raise ValueError('HA-SODQN OWP capacity changed')
    if archive.get('alignment_warmup_episodes') != 10:
        raise ValueError('HA-SODQN alignment warm-up changed')
    if archive.get('cqa_global_fraction') != 0.75:
        raise ValueError('HA-SODQN CQA global fraction changed')
    return config


def _git(command):
    return subprocess.check_output(['git', *command], text=True).strip()


def _repository_identity():
    branch = _git(['branch', '--show-current'])
    remote = _git(['remote', 'get-url', 'origin'])
    return {'git_commit': _git(['rev-parse', 'HEAD']), 'git_branch': branch, 'git_remote': remote}


def _load_inputs(config):
    archive_path = os.path.abspath(config['archive']['root_manifest'])
    initial_path = os.path.abspath(config['archive']['initial_state_catalog'])
    archive = read_json(archive_path)
    if archive.get('schema_version') != ARCHIVE_SCHEMA_VERSION:
        raise ValueError('Invalid HA archive root manifest')
    validate_initial_state_catalog(initial_path, revalidate_sources=False)
    initial = read_json(initial_path)
    initial_index = {
        (entry['order_id'], int(entry['training_seed'])): entry
        for entry in initial['entries']
    }
    return archive_path, archive, initial_path, initial_index


def _child(config, initial_index, order_id, seed, archive_mode, method, ratio,
           stage_episodes, experiment_stage):
    networks = config['orders'][order_id]
    initial = initial_index[(order_id, int(seed))]
    if archive_mode == 'NONE':
        logical_id = f'CONT-FIFO-{order_id}-SD{seed}'
        condition = 'CONT_FIFO'
    else:
        ratio_name = f'R{int(round(float(ratio) * 100)):02d}'
        logical_id = f'{archive_mode}-{method}-{ratio_name}-{order_id}-SD{seed}'
        condition = 'HA_SODQN'
    return {
        'logical_run_id': logical_id,
        'experiment_stage': experiment_stage,
        'condition': condition,
        'policy': 'ha_sodqn',
        'order_id': order_id,
        'training_seed': int(seed),
        'networks': list(networks),
        'stage_episodes': list(stage_episodes),
        'initial_state': initial,
        'initial_checkpoint': initial['checkpoint_path'],
        'initial_checkpoint_file_sha256': initial['checkpoint_file_sha256'],
        'archive_mode': archive_mode,
        'method': method,
        'offline_ratio': float(ratio),
        'owp_capacity': int(config['archive']['owp_capacity']),
        'alignment_warmup_episodes': int(config['archive']['alignment_warmup_episodes']),
        'trace_replay_samples': False,
        'estimated_output_bytes': 2 * 1024 ** 3,
        'interface': 'libsumo',
        'status': 'planned',
    }


def stage_specifications(config, experiment_stage, selection=None):
    if experiment_stage == 'smoke':
        return [
            ('O2', 0, 'NONE', 'CONT', 0.0),
            ('O2', 0, 'P1C', 'DHOA', 0.5),
            ('O2', 0, 'P1C', 'CQA', 0.5),
            ('O2', 0, 'P1F', 'DHOA', 0.5),
        ], config['trainer']['smoke_stage_episodes']
    if experiment_stage == 'E0':
        return [('O2', 0, 'P1C', 'DHOA', 0.5)], config['trainer']['stage_episodes']
    if experiment_stage == 'E1':
        items = [('O2', 0, 'NONE', 'CONT', 0.0)]
        items.extend(
            ('O2', 0, 'P1C', method, ratio)
            for ratio in config['offline_ratios'] for method in config['methods']
        )
        return items, config['trainer']['stage_episodes']
    if selection is None:
        raise ValueError(f'{experiment_stage} requires an explicit preregistered selection')
    return [(
        item['order_id'], int(item['training_seed']), item['archive_mode'],
        item['method'], float(item['offline_ratio']),
    ) for item in selection], config['trainer']['stage_episodes']


def build_ha_plan(output_path, experiment_stage, config_path,
                  selection=None, launch_authorized=None):
    config = load_ha_config(config_path)
    archive_path, archive, initial_path, initial_index = _load_inputs(config)
    specifications, stage_episodes = stage_specifications(
        config, experiment_stage, selection=selection,
    )
    children = [
        _child(config, initial_index, *specification, stage_episodes, experiment_stage)
        for specification in specifications
    ]
    identities = [child['logical_run_id'] for child in children]
    if len(identities) != len(set(identities)):
        raise ValueError('HA-SODQN plan contains duplicate logical identities')
    formal = experiment_stage != 'smoke'
    if launch_authorized is None:
        launch_authorized = not formal
    payload = {
        'schema_version': HA_PLAN_SCHEMA_VERSION,
        'protocol_id': config['protocol_id'],
        'mode': 'formal' if formal else 'smoke',
        'experiment_stage': experiment_stage,
        'launch_authorized': bool(launch_authorized),
        'child_count': len(children),
        'max_child_concurrency': 8,
        'orders': config['orders'],
        'trainer': config['trainer'],
        'model': config['model'],
        'archive_config': config['archive'],
        'rng_config': config['rng'],
        'analysis': config['analysis'],
        'config_path': os.path.abspath(config_path),
        'config_sha256': sha256_file(config_path),
        'archive_root_manifest': archive_path,
        'archive_root_manifest_sha256': sha256_file(archive_path),
        'archive_digest': archive['archive_digest'],
        'initial_state_catalog': initial_path,
        'initial_state_catalog_sha256': sha256_file(initial_path),
        **_repository_identity(),
        'children': children,
    }
    payload['plan_digest'] = canonical_digest(payload)
    atomic_json(output_path, payload)
    return payload


def build_reproduction_audit_plan(output_path, config_path,
                                  order_id='O2', training_seed=0):
    config = load_ha_config(config_path)
    initial_path = os.path.abspath(config['archive']['initial_state_catalog'])
    validate_initial_state_catalog(initial_path, revalidate_sources=False)
    initial_catalog = read_json(initial_path)
    initial_index = {
        (entry['order_id'], int(entry['training_seed'])): entry
        for entry in initial_catalog['entries']
    }
    child = _child(
        config, initial_index, order_id, int(training_seed),
        'NONE', 'CONT', 0.0, [100], 'behavior_seed_audit',
    )
    child['networks'] = [config['orders'][order_id][0]]
    payload = {
        'schema_version': HA_PLAN_SCHEMA_VERSION,
        'protocol_id': config['protocol_id'],
        'mode': 'audit',
        'experiment_stage': 'behavior_seed_audit',
        'launch_authorized': True,
        'child_count': 1,
        'max_child_concurrency': 8,
        'orders': {order_id: child['networks']},
        'trainer': config['trainer'],
        'model': config['model'],
        'archive_config': config['archive'],
        'rng_config': config['rng'],
        'analysis': config['analysis'],
        'config_path': os.path.abspath(config_path),
        'config_sha256': sha256_file(config_path),
        'initial_state_catalog': initial_path,
        'initial_state_catalog_sha256': sha256_file(initial_path),
        **_repository_identity(),
        'children': [child],
    }
    payload['plan_digest'] = canonical_digest(payload)
    atomic_json(output_path, payload)
    return payload


def validate_ha_plan(path):
    plan = read_json(path)
    if plan.get('schema_version') != HA_PLAN_SCHEMA_VERSION:
        raise ValueError('Unsupported HA-SODQN plan schema')
    recorded = plan.get('plan_digest')
    unsigned = dict(plan)
    unsigned.pop('plan_digest', None)
    if canonical_digest(unsigned) != recorded:
        raise ValueError('HA-SODQN plan digest mismatch')
    if plan.get('max_child_concurrency') != 8:
        raise ValueError('HA-SODQN plans must default to 8 child slots')
    if sha256_file(plan['config_path']) != plan['config_sha256']:
        raise ValueError('HA-SODQN config changed after plan creation')
    if sha256_file(plan['archive_root_manifest']) != plan['archive_root_manifest_sha256']:
        raise ValueError('HA archive root changed after plan creation')
    if sha256_file(plan['initial_state_catalog']) != plan['initial_state_catalog_sha256']:
        raise ValueError('HA initial-state catalog changed after plan creation')
    if _git(['rev-parse', 'HEAD']) != plan['git_commit']:
        raise ValueError('Current checkout differs from HA-SODQN frozen commit')
    children = plan.get('children', ())
    if len(children) != plan.get('child_count'):
        raise ValueError('HA-SODQN child count mismatch')
    for child in children:
        expected_budget = (
            [100] if plan['mode'] == 'audit' else
            plan['trainer']['smoke_stage_episodes']
            if plan['mode'] == 'smoke' else plan['trainer']['stage_episodes']
        )
        if child['stage_episodes'] != expected_budget:
            raise ValueError('HA-SODQN child training budget mismatch')
        if sha256_file(child['initial_checkpoint']) != child['initial_checkpoint_file_sha256']:
            raise ValueError('HA-SODQN initial checkpoint changed')
        if child['archive_mode'] == 'NONE':
            if child['method'] != 'CONT' or child['offline_ratio'] != 0.0:
                raise ValueError('Invalid CONT-FIFO identity')
        elif child['archive_mode'] not in ('P1C', 'P1F'):
            raise ValueError('Invalid HA archive mode')
    return {'valid': True, 'child_count': len(children), 'plan_digest': recorded}
