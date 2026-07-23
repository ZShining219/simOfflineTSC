import csv
import json
import os
import re
from dataclasses import dataclass

import torch
import yaml

from dataset.offline_trajectory_dataset import _validate_source_run

from .config import FORMAL_NETWORKS, load_sequential_config
from .core import (
    ReplayMetadata, ReplayRecord, SequentialReplay, TargetUpdateScheduler,
    TrainingPayload, environment_signature_digest, online_parameter_digest,
    optimizer_state_digest, replay_content_digest, replay_metadata_digest,
    rng_state_digest, target_parameter_digest,
)
from .io import atomic_json, read_json, sha256_file


PARENT_SCHEMA_VERSION = 1
EXPECTED_PARENT_EPISODE = 400
EXPECTED_GLOBAL_DECISIONS = 144000
EXPECTED_GRADIENT_UPDATES = 143000
EXPECTED_TARGET_UPDATES = 14300
EXPECTED_REPLAY_CAPACITY = 5000
EXPECTED_REPLAY_SIZE = 5000
LEGACY_KEY = re.compile(r'^(?P<episode>\d+)_(?P<decision>\d+)_(?P<agent>.+)$')
SEMANTIC_SIGNATURE_KEYS = (
    'roadnet_control_topology_sha256', 'tl_program_sha256',
    'action_mapping_sha256', 'feature_schema_sha256',
    'reward_definition_sha256', 'controlled_intersection_count',
    'green_action_count',
)


@dataclass(frozen=True)
class ParentSource:
    run_path: str
    network: str
    training_seed: int


def read_parent_whitelist(path):
    sources = []
    seen = set()
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {'run_path', 'network', 'behavior_training_seed'}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f'Parent whitelist fields must be {sorted(required)}')
        for row in reader:
            source = ParentSource(
                run_path=os.path.abspath(row['run_path']),
                network=row['network'],
                training_seed=int(row['behavior_training_seed']),
            )
            identity = (source.network, source.training_seed)
            if source.network not in FORMAL_NETWORKS:
                raise ValueError(f'Unsupported parent network: {source.network}')
            if identity in seen:
                raise ValueError(f'Duplicate parent identity: {identity}')
            seen.add(identity)
            sources.append(source)
    expected = {(network, seed) for network in FORMAL_NETWORKS for seed in range(5)}
    if seen != expected:
        raise ValueError(
            f'Parent whitelist matrix mismatch: missing={sorted(expected-seen)}, '
            f'extra={sorted(seen-expected)}'
        )
    return sources


def _checkpoint_path(run_path):
    return os.path.join(
        run_path, 'checkpoints', 'resumable',
        f'episode_{EXPECTED_PARENT_EPISODE:04d}.pt',
    )


def _parse_legacy_key(key):
    match = LEGACY_KEY.match(str(key))
    if match is None:
        raise ValueError(f'Invalid Plan 1 replay key: {key!r}')
    return int(match.group('episode')), int(match.group('decision'))


def convert_parent_replay(items, network, global_decision_step, capacity):
    if len(items) > capacity:
        raise ValueError('Parent replay exceeds capacity')
    first_global = global_decision_step - len(items) + 1
    records = []
    previous_identity = None
    for offset, item in enumerate(items):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError('Invalid Plan 1 replay record')
        key, legacy_payload = item
        zero_based_episode, decision_index = _parse_legacy_key(key)
        written_global_step = first_global + offset
        identity = (zero_based_episode, decision_index)
        if previous_identity is not None and identity <= previous_identity:
            raise ValueError('Parent replay order is not strictly increasing')
        previous_identity = identity
        metadata = ReplayMetadata(
            transition_id=(
                f'{network}:stage1:global{written_global_step:09d}'
            ),
            source_network=network,
            stage_index=1,
            local_episode=zero_based_episode + 1,
            decision_index=decision_index,
            written_global_step=written_global_step,
        )
        records.append(ReplayRecord(
            payload=TrainingPayload.from_legacy(legacy_payload),
            metadata=metadata,
        ))
    return SequentialReplay(capacity, records)


def _validate_frozen_config(run_path, source, checkpoint, source_validation):
    with open(os.path.join(run_path, 'config', 'resolved_config.yaml'), encoding='utf-8') as handle:
        resolved = yaml.safe_load(handle)
    with open(os.path.join(run_path, 'config', 'model_resolved.json'), encoding='utf-8') as handle:
        model = json.load(handle)
    trainer = resolved.get('trainer', {})
    model_config = resolved.get('model', {})
    expected_trainer = {
        'episodes': 400, 'steps': 3600, 'test_steps': 3600,
        'action_interval': 10, 'learning_start': 1000,
        'buffer_size': 5000, 'update_model_rate': 1,
        'update_target_rate': 10,
    }
    expected_model = {
        'batch_size': 64, 'gamma': 0.95, 'learning_rate': 0.001,
        'epsilon_decay': 0.995, 'epsilon_min': 0.01,
        'phase': True, 'one_hot': True,
    }
    errors = []
    for key, expected in expected_trainer.items():
        if trainer.get(key) != expected:
            errors.append(f'trainer.{key}={trainer.get(key)!r}, expected {expected!r}')
    for key, expected in expected_model.items():
        if model_config.get(key) != expected:
            errors.append(f'model.{key}={model_config.get(key)!r}, expected {expected!r}')
    agent_model = model.get('agents', [{}])[0]
    if agent_model.get('action_dim') != 8:
        errors.append('runtime action_dim is not 8')
    network_model = agent_model.get('model', {})
    if network_model.get('input_dim') != 16:
        errors.append('runtime model input_dim is not 16')
    if network_model.get('output_dim') != 8:
        errors.append('runtime model output_dim is not 8')
    if checkpoint.get('config_hash') != read_json(
        os.path.join(run_path, 'run_manifest.json')
    ).get('config_hash'):
        errors.append('checkpoint config hash does not match source run')
    if source_validation.get('action_dim') != 8:
        errors.append('trajectory action_dim is not 8')
    if errors:
        raise ValueError(f'{source.run_path}: frozen config mismatch: ' + '; '.join(errors))
    return {
        'state_dim': 8,
        'model_input_dim': 16,
        'action_dim': 8,
        'phase_dim': 8,
        'lane_feature_count': 8,
    }

def validate_parent_source(source):
    source_validation = _validate_source_run(
        source.run_path, source.network, source.training_seed,
        EXPECTED_PARENT_EPISODE,
    )
    checkpoint_path = _checkpoint_path(source.run_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Missing parent checkpoint: {checkpoint_path}')
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as error:
        raise IOError(f'Cannot load parent checkpoint: {checkpoint_path}') from error
    required = {
        'schema_version', 'checkpoint_type', 'episode', 'global_decision_step',
        'gradient_updates', 'config_hash', 'agents', 'training_counters',
        'python_random_state', 'numpy_random_state', 'torch_cpu_rng_state',
        'torch_cuda_rng_states',
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f'{checkpoint_path}: missing fields {missing}')
    if checkpoint['schema_version'] != 1 or checkpoint['checkpoint_type'] != 'resumable':
        raise ValueError(f'{checkpoint_path}: not a Plan 1 resumable checkpoint')
    if checkpoint['episode'] != EXPECTED_PARENT_EPISODE:
        raise ValueError(f'{checkpoint_path}: parent episode is not 400')
    counters = checkpoint['training_counters']
    expected_counters = {
        'global_decision_step': EXPECTED_GLOBAL_DECISIONS,
        'gradient_updates': EXPECTED_GRADIENT_UPDATES,
        'target_updates': EXPECTED_TARGET_UPDATES,
    }
    for key, expected in expected_counters.items():
        if checkpoint.get(key, counters.get(key)) != expected or counters.get(key) != expected:
            raise ValueError(f'{checkpoint_path}: invalid {key}')
    if len(checkpoint['agents']) != 1:
        raise ValueError(f'{checkpoint_path}: expected exactly one DQN agent')
    agent = checkpoint['agents'][0]
    agent_required = {
        'online_model_state_dict', 'target_model_state_dict',
        'optimizer_state_dict', 'epsilon', 'replay_state',
    }
    missing_agent = sorted(agent_required - set(agent))
    if missing_agent:
        raise ValueError(f'{checkpoint_path}: missing agent fields {missing_agent}')
    replay_state = agent['replay_state']
    if replay_state.get('capacity') != EXPECTED_REPLAY_CAPACITY:
        raise ValueError(f'{checkpoint_path}: replay capacity is not 5000')
    if len(replay_state.get('items', ())) != EXPECTED_REPLAY_SIZE:
        raise ValueError(f'{checkpoint_path}: replay size is not 5000')
    if not isinstance(agent['epsilon'], (int, float)) or not 0 < agent['epsilon'] <= 0.01:
        raise ValueError(f'{checkpoint_path}: invalid terminal epsilon')
    if not agent['optimizer_state_dict'].get('param_groups'):
        raise ValueError(f'{checkpoint_path}: optimizer state is incomplete')
    replay = convert_parent_replay(
        replay_state['items'], source.network,
        EXPECTED_GLOBAL_DECISIONS, EXPECTED_REPLAY_CAPACITY,
    )
    scheduler = TargetUpdateScheduler.from_plan1_parent(
        EXPECTED_GRADIENT_UPDATES, EXPECTED_TARGET_UPDATES, 10,
    )
    dimensions = _validate_frozen_config(
        source.run_path, source, checkpoint, source_validation,
    )
    semantics = source_validation['scene_semantics']
    semantic_signature = {key: semantics[key] for key in SEMANTIC_SIGNATURE_KEYS}
    digests = {
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
        'replay_content_digest': replay_content_digest(replay.records),
        'replay_metadata_digest': replay_metadata_digest(replay.records),
        'environment_signature_digest': environment_signature_digest(
            semantic_signature
        ),
    }
    return {
        'schema_version': PARENT_SCHEMA_VERSION,
        'source_run_path': source.run_path,
        'source_run_id': read_json(os.path.join(source.run_path, 'run_manifest.json'))['run_id'],
        'network': source.network,
        'training_seed': source.training_seed,
        'checkpoint_path': checkpoint_path,
        'checkpoint_file_sha256': sha256_file(checkpoint_path),
        'checkpoint_episode': EXPECTED_PARENT_EPISODE,
        'epsilon': float(agent['epsilon']),
        'global_decision_step': EXPECTED_GLOBAL_DECISIONS,
        'gradient_updates': EXPECTED_GRADIENT_UPDATES,
        'target_updates': EXPECTED_TARGET_UPDATES,
        'next_target_sync_update': scheduler.next_update,
        'replay_size': len(replay.records),
        'replay_capacity': replay.capacity,
        'replay_metadata_rule': {
            'legacy_key': 'zero_based_episode_decision_agent_id',
            'local_episode': 'zero_based_episode + 1',
            'written_global_step': 'checkpoint_global-len(replay)+1+offset',
        },
        'dimensions': dimensions,
        'environment_signature': semantic_signature,
        'source_roadnet_sha256': semantics['roadnet_sha256'],
        'digests': digests,
    }


def validate_environment_compatibility(records):
    if not records:
        raise ValueError('No parent records supplied')
    reference = records[0]['environment_signature']
    incompatible = [
        record['network'] for record in records
        if record['environment_signature'] != reference
    ]
    if incompatible:
        raise ValueError(f'Incompatible sequential environments: {incompatible}')
    dimensions = records[0]['dimensions']
    bad_dimensions = [
        record['network'] for record in records
        if record['dimensions'] != dimensions
    ]
    if bad_dimensions:
        raise ValueError(f'Incompatible sequential dimensions: {bad_dimensions}')
    return environment_signature_digest(reference)


def import_parents(whitelist_path, output_dir, config_path='configs/sequential/plan34.yml'):
    config = load_sequential_config(config_path)
    sources = read_parent_whitelist(whitelist_path)
    records = [validate_parent_source(source) for source in sources]
    shared_environment_digest = validate_environment_compatibility(records)
    first_network_to_order = {
        networks[0]: order_id for order_id, networks in config['orders'].items()
    }
    imported = []
    for record in records:
        order_id = first_network_to_order[record['network']]
        enriched = dict(record)
        enriched.update({
            'order_id': order_id,
            'order': config['orders'][order_id],
            'shared_environment_signature_digest': shared_environment_digest,
        })
        path = os.path.join(
            output_dir, 'parents', order_id,
            f'seed_{record["training_seed"]}', 'parent_import.json',
        )
        atomic_json(path, enriched)
        enriched['import_manifest_path'] = os.path.abspath(path)
        imported.append(enriched)
    catalog_path = os.path.join(output_dir, 'parent_catalog.json')
    atomic_json(catalog_path, {
        'schema_version': PARENT_SCHEMA_VERSION,
        'whitelist_path': os.path.abspath(whitelist_path),
        'parent_count': len(imported),
        'shared_environment_signature_digest': shared_environment_digest,
        'parents': [{
            key: record[key] for key in (
                'order_id', 'network', 'training_seed', 'source_run_path',
                'checkpoint_path', 'checkpoint_file_sha256',
                'import_manifest_path', 'digests',
            )
        } for record in imported],
    })
    return os.path.abspath(catalog_path), imported


def validate_parent_catalog(path, revalidate_sources=True):
    catalog = read_json(path)
    parents = catalog.get('parents', [])
    if catalog.get('schema_version') != PARENT_SCHEMA_VERSION or len(parents) != 20:
        raise ValueError('Parent catalog must contain exactly 20 schema-v1 parents')
    identities = {(p['order_id'], int(p['training_seed'])) for p in parents}
    expected = {(f'O{order}', seed) for order in range(1, 5) for seed in range(5)}
    if identities != expected:
        raise ValueError('Parent catalog order/seed matrix mismatch')
    manifests = [read_json(parent['import_manifest_path']) for parent in parents]
    shared = validate_environment_compatibility(manifests)
    if shared != catalog.get('shared_environment_signature_digest'):
        raise ValueError('Parent catalog environment digest mismatch')
    if revalidate_sources:
        for parent, manifest in zip(parents, manifests):
            fresh = validate_parent_source(ParentSource(
                parent['source_run_path'], parent['network'],
                int(parent['training_seed']),
            ))
            for key in ('checkpoint_file_sha256', 'digests'):
                if fresh[key] != manifest[key] or fresh[key] != parent[key]:
                    raise ValueError(
                        f'Parent source changed after import: {parent["source_run_path"]}'
                    )
    return {
        'valid': True, 'parent_count': len(parents),
        'shared_environment_signature_digest': shared,
    }
