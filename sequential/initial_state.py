"""Audited Plan 1 episode-0 initial states for HA-SODQN."""

import copy
import os

import torch
import yaml

from dataset.offline_trajectory_dataset import _validate_source_run

from .config import load_sequential_config
from .core import (
    online_parameter_digest, optimizer_state_digest, rng_state_digest,
    target_parameter_digest,
)
from .io import atomic_json, read_json, sha256_file
from .parents import PLAN1_FORMAL_EPISODES, ParentSource, read_parent_whitelist


INITIAL_STATE_SCHEMA_VERSION = 1


def _dueling_state_dict(source):
    """Convert a frozen DQN MLP into an equivalent dueling representation."""
    required = {
        'dense_1.weight', 'dense_1.bias', 'dense_2.weight', 'dense_2.bias',
        'dense_3.weight', 'dense_3.bias',
    }
    if set(source) != required:
        raise ValueError(
            'Dueling conversion requires the frozen six-tensor DQN MLP; '
            f'found {sorted(source)}'
        )
    advantage_weight = source['dense_3.weight'].detach().clone()
    advantage_bias = source['dense_3.bias'].detach().clone()
    value_weight = advantage_weight.mean(dim=0, keepdim=True)
    value_bias = advantage_bias.mean().reshape(1)
    advantage_weight = advantage_weight - value_weight
    advantage_bias = advantage_bias - value_bias
    return {
        'feature.0.weight': source['dense_1.weight'].detach().clone(),
        'feature.0.bias': source['dense_1.bias'].detach().clone(),
        'feature.2.weight': source['dense_2.weight'].detach().clone(),
        'feature.2.bias': source['dense_2.bias'].detach().clone(),
        'value.weight': value_weight,
        'value.bias': value_bias,
        'advantage.weight': advantage_weight,
        'advantage.bias': advantage_bias,
    }


def _dueling_optimizer_state(source):
    groups = copy.deepcopy(source.get('param_groups', []))
    if not groups:
        groups = [{'lr': 0.001, 'alpha': 0.9, 'centered': False,
                   'eps': 1e-7, 'weight_decay': 0, 'momentum': 0,
                   'dampening': 0, 'nesterov': False}]
    groups[0]['params'] = list(range(8))
    return {'state': {}, 'param_groups': groups}


def build_dueling_initial_state_catalog(source_catalog_path, output_catalog_path,
                                        output_root):
    """Create algorithm-specific episode-0 assets for Dueling Double DQN.

    The conversion preserves the initial Q function exactly: the value head
    receives the mean of the frozen DQN action weights and the advantage head
    receives the centered residual. Episode-0 optimizer state is empty, so a
    fresh eight-parameter RMSprop group is equivalent to the source state.
    """
    validate_initial_state_catalog(source_catalog_path, revalidate_sources=False)
    source_catalog = read_json(source_catalog_path)
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)
    entries = []
    for source_entry in source_catalog['entries']:
        checkpoint = torch.load(source_entry['checkpoint_path'], map_location='cpu')
        if checkpoint.get('checkpoint_type') != 'resumable' or checkpoint.get('episode') != 0:
            raise ValueError('Dueling conversion requires episode-0 resumable checkpoints')
        converted = copy.deepcopy(checkpoint)
        for agent in converted.get('agents', ()):
            agent['online_model_state_dict'] = _dueling_state_dict(
                agent['online_model_state_dict']
            )
            agent['target_model_state_dict'] = _dueling_state_dict(
                agent['target_model_state_dict']
            )
            agent['optimizer_state_dict'] = _dueling_optimizer_state(
                agent['optimizer_state_dict']
            )
            agent['algorithm_id'] = 'dueling_double_dqn'
        converted['algorithm_id'] = 'dueling_double_dqn'
        relative = os.path.join(
            source_entry['order_id'], f'seed{int(source_entry["training_seed"])}',
            'checkpoints', 'resumable', 'episode_0000.pt',
        )
        checkpoint_path = os.path.join(output_root, relative)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(converted, checkpoint_path)
        agent = converted['agents'][0]
        entry = copy.deepcopy(source_entry)
        entry.update({
            'algorithm_id': 'dueling_double_dqn',
            'source_algorithm_id': 'independent_dqn',
            'checkpoint_path': os.path.abspath(checkpoint_path),
            'checkpoint_file_sha256': sha256_file(checkpoint_path),
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
                    converted['python_random_state'],
                    converted['numpy_random_state'],
                    converted['torch_cpu_rng_state'],
                    converted['torch_cuda_rng_states'],
                ),
            },
        })
        entries.append(entry)
    payload = {
        'schema_version': INITIAL_STATE_SCHEMA_VERSION,
        'catalog_kind': 'dueling_double_dqn_episode0_resumable',
        'source_catalog_path': os.path.abspath(source_catalog_path),
        'entry_count': len(entries), 'entries': entries,
        'algorithm_id': 'dueling_double_dqn',
        'runtime_rule': 'load the converted first-network episode-0 source',
        'orb_initial_state': 'empty',
        'source_episodes': int(source_catalog.get(
            'source_episodes', PLAN1_FORMAL_EPISODES
        )),
    }
    atomic_json(output_catalog_path, payload)
    return payload


def validate_initial_source(source, require_plan1_formal=True,
                            expected_episodes=PLAN1_FORMAL_EPISODES):
    source_validation = _validate_source_run(
        source.run_path, source.network, source.training_seed,
        int(expected_episodes), require_plan1_formal=require_plan1_formal,
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


def build_initial_state_catalog(whitelist_path, output_path, config_path,
                                source_episodes=PLAN1_FORMAL_EPISODES):
    try:
        config = load_sequential_config(config_path)
    except ValueError:
        with open(config_path, encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        if config.get('protocol_id') != 'ha_sodqn_b100_v1':
            raise
        if set(config.get('orders', {})) != {'O1', 'O2', 'O3', 'O4'}:
            raise ValueError('HA initial-state config must define O1..O4')
        if config.get('training_seeds') != [0, 1, 2, 3, 4]:
            raise ValueError('HA initial-state config seeds must be 0..4')
    sources = read_parent_whitelist(whitelist_path)
    if int(source_episodes) == PLAN1_FORMAL_EPISODES:
        by_identity = {
            (source.network, source.training_seed): validate_initial_source(source)
            for source in sources
        }
    else:
        by_identity = {
            (source.network, source.training_seed): validate_initial_source(
                source, expected_episodes=source_episodes,
            )
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
        'source_episodes': int(source_episodes),
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
    source_episodes = int(catalog.get('source_episodes', PLAN1_FORMAL_EPISODES))
    for entry in entries:
        if entry['network'] != entry['first_network']:
            raise ValueError('Initial-state source is not the actual first network')
        if sha256_file(entry['checkpoint_path']) != entry['checkpoint_file_sha256']:
            raise ValueError('Initial-state checkpoint hash mismatch')
        if revalidate_sources:
            fresh = validate_initial_source(
                ParentSource(
                    entry['source_run_path'], entry['network'],
                    int(entry['training_seed']),
                ),
                expected_episodes=source_episodes,
            )
            if fresh['digests'] != entry['digests']:
                raise ValueError('Initial-state source changed after catalog creation')
    return {'valid': True, 'entry_count': 20}
