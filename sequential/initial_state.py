"""Audited Plan 1 episode-0 initial states for HA-SODQN."""

import os

import torch

from dataset.offline_trajectory_dataset import _validate_source_run

from .config import load_sequential_config
from .core import (
    online_parameter_digest, optimizer_state_digest, rng_state_digest,
    target_parameter_digest,
)
from .io import atomic_json, read_json, sha256_file
from .parents import PLAN1_FORMAL_EPISODES, ParentSource, read_parent_whitelist


INITIAL_STATE_SCHEMA_VERSION = 1


def validate_initial_source(source, require_plan1_formal=True):
    source_validation = _validate_source_run(
        source.run_path, source.network, source.training_seed,
        PLAN1_FORMAL_EPISODES, require_plan1_formal=require_plan1_formal,
    )
    path = os.path.join(
        source.run_path, 'checkpoints', 'resumable', 'episode_0000.pt',
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing Plan 1 episode-0 checkpoint: {path}')
    checkpoint = torch.load(path, map_location='cpu')
    if checkpoint.get('schema_version') != 1:
        raise ValueError(f'{path}: unsupported checkpoint schema')
    if checkpoint.get('checkpoint_type') != 'resumable' or checkpoint.get('episode') != 0:
        raise ValueError(f'{path}: not an episode-0 resumable checkpoint')
    counters = checkpoint.get('training_counters', {})
    for key in ('global_decision_step', 'gradient_updates', 'target_updates'):
        if int(counters.get(key, -1)) != 0 or int(checkpoint.get(key, 0)) != 0:
            raise ValueError(f'{path}: episode-0 {key} must be zero')
    agents = checkpoint.get('agents', ())
    if len(agents) != 1:
        raise ValueError(f'{path}: expected exactly one DQN agent')
    agent = agents[0]
    replay = agent.get('replay_state', {})
    if replay.get('items') not in ([], ()):
        raise ValueError(f'{path}: episode-0 replay must be empty')
    if replay.get('capacity') != 5000:
        raise ValueError(f'{path}: episode-0 replay capacity must be 5000')
    if not 0.0 < float(agent.get('epsilon', -1)) <= 1.0:
        raise ValueError(f'{path}: episode-0 epsilon is invalid')
    semantics = source_validation['scene_semantics']
    return {
        'schema_version': INITIAL_STATE_SCHEMA_VERSION,
        'source_run_path': os.path.abspath(source.run_path),
        'source_run_id': read_json(os.path.join(source.run_path, 'run_manifest.json'))['run_id'],
        'network': source.network,
        'training_seed': int(source.training_seed),
        'checkpoint_path': os.path.abspath(path),
        'checkpoint_file_sha256': sha256_file(path),
        'checkpoint_episode': 0,
        'epsilon': float(agent['epsilon']),
        'counters': {key: 0 for key in (
            'global_decision_step', 'gradient_updates', 'target_updates',
        )},
        'dimensions': {
            'observation_dim': 16, 'state_dim': 8,
            'phase_dim': 8, 'action_dim': 8,
        },
        'scene_semantics': semantics,
        'digests': {
            'online_parameter_digest': online_parameter_digest(
                agent['online_model_state_dict']
            ),
            'target_parameter_digest': target_parameter_digest(
                agent['target_model_state_dict']
            ),
            'optimizer_state_digest': optimizer_state_digest(
                agent['optimizer_state_dict']
            ),
            'rng_state_digest': rng_state_digest(
                checkpoint['python_random_state'], checkpoint['numpy_random_state'],
                checkpoint['torch_cpu_rng_state'], checkpoint['torch_cuda_rng_states'],
            ),
        },
    }


def build_initial_state_catalog(whitelist_path, output_path, config_path):
    config = load_sequential_config(config_path)
    sources = read_parent_whitelist(whitelist_path)
    by_identity = {
        (source.network, source.training_seed): validate_initial_source(source)
        for source in sources
    }
    entries = []
    for order_id, networks in config['orders'].items():
        for seed in config['training_seeds']:
            source = by_identity[(networks[0], seed)]
            entries.append({
                'order_id': order_id,
                'ordered_networks': list(networks),
                'first_network': networks[0],
                'training_seed': int(seed),
                **source,
            })
    equivalence = {}
    for seed in config['training_seeds']:
        records = [record for (network, item_seed), record in by_identity.items()
                   if item_seed == seed]
        keys = (
            'online_parameter_digest', 'target_parameter_digest',
            'optimizer_state_digest', 'rng_state_digest',
        )
        equivalence[str(seed)] = {
            key: len({record['digests'][key] for record in records}) == 1
            for key in keys
        }
        equivalence[str(seed)]['all_training_state_equal'] = all(
            equivalence[str(seed)][key] for key in keys
        )
    payload = {
        'schema_version': INITIAL_STATE_SCHEMA_VERSION,
        'catalog_kind': 'plan1_episode0_resumable',
        'whitelist_path': os.path.abspath(whitelist_path),
        'config_path': os.path.abspath(config_path),
        'entry_count': len(entries),
        'entries': entries,
        'same_seed_cross_network_equivalence': equivalence,
        'runtime_rule': 'always load the actual first-network episode-0 source',
        'orb_initial_state': 'empty',
    }
    atomic_json(output_path, payload)
    return payload


def validate_initial_state_catalog(path, revalidate_sources=True):
    catalog = read_json(path)
    entries = catalog.get('entries', ())
    if catalog.get('schema_version') != INITIAL_STATE_SCHEMA_VERSION or len(entries) != 20:
        raise ValueError('Initial-state catalog must contain 20 schema-v1 entries')
    identities = {(entry['order_id'], int(entry['training_seed'])) for entry in entries}
    expected = {(f'O{order}', seed) for order in range(1, 5) for seed in range(5)}
    if identities != expected:
        raise ValueError('Initial-state catalog order/seed matrix mismatch')
    for entry in entries:
        if entry['network'] != entry['first_network']:
            raise ValueError('Initial-state source is not the actual first network')
        if sha256_file(entry['checkpoint_path']) != entry['checkpoint_file_sha256']:
            raise ValueError('Initial-state checkpoint hash mismatch')
        if revalidate_sources:
            fresh = validate_initial_source(ParentSource(
                entry['source_run_path'], entry['network'], int(entry['training_seed']),
            ))
            if fresh['digests'] != entry['digests']:
                raise ValueError('Initial-state source changed after catalog creation')
    return {'valid': True, 'entry_count': 20}
