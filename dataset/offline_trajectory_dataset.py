import csv
import hashlib
import json
import os
import random
import re
import resource
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from common.registry import Registry
from utils.run_config_compare import compare_runs


OFFLINE_DATASET_SCHEMA_VERSION = 2
OFFLINE_DATASET_BUILDER_VERSION = '2.0.0'
FEATURE_SCHEMA_VERSION = 1
TRAJECTORY_SCHEMA_VERSION = 1
DEFAULT_SOURCE_ROOT_ID = 'plan1_formal_root'
PLAN2_STAGES = {
    'Q1': (1, 100),
    'Q2': (101, 200),
    'Q3': (201, 300),
    'Q4': (301, 400),
    'full': (1, 400),
}
PLAN2_NETWORKS = (
    'sumohz1x1_config2',
    'sumohz1x1',
    'sumohz1x1_config4',
    'sumohz1x1_config3',
)
PLAN2_BEHAVIOR_SEEDS = (0, 1, 2, 3, 4)
TRAJECTORY_REQUIRED_FIELDS = {
    'schema_version', 'network', 'scene_id', 'behavior_training_seed',
    'episode_id', 'decision_step', 'global_step', 'state', 'current_phase',
    'action', 'reward', 'next_state', 'next_phase', 'terminated', 'truncated',
}
SOURCE_INDEX_REQUIRED_FIELDS = {
    'schema_version', 'run_path', 'run_id', 'network',
    'behavior_training_seed', 'episode_id', 'transition_count',
    'source_root_id', 'relative_path', 'original_absolute_path',
    'sha256', 'semantic_validation',
}


@Registry.register_dataset('offline_readonly')
class OfflineReadOnlyPlaceholder:
    """No-op slot used while TSCTrainer constructs the offline evaluator."""

    def __init__(self, path):
        self.path = path

    def initiate(self, ep, step, interval):
        return None

    def __len__(self):
        return 0


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.tmp-', dir=os.path.dirname(path))
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _read_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _stable_hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()


def _scene_semantics(run_path):
    resolved = os.path.join(run_path, 'config', 'simulator_resolved.cfg')
    config = _read_json(resolved)
    roadnet = os.path.realpath(os.path.join(config['dir'], config['roadnetFile']))
    root = ET.parse(roadnet).getroot()
    tl_programs = []
    action_mapping = []
    for logic in root.findall('tlLogic'):
        phases = [
            (phase.get('duration'), phase.get('state'))
            for phase in logic.findall('phase')
        ]
        tl_programs.append((logic.get('id'), logic.get('programID'), logic.get('type'), phases))
        action_mapping.append((
            logic.get('id'),
            [state for _, state in phases if 'y' not in state and set(state) - set('rs')],
        ))
    feature_descriptor = {
        'version': FEATURE_SCHEMA_VERSION,
        'state': 'LaneVehicleGenerator(lane_count,in_only=True,average=None)',
        'state_order': 'SUMO intersection in_roads direction order then lane numeric suffix',
        'phase': 'current green-action index one-hot appended after state',
        'dtype': 'float32',
    }
    reward_descriptor = {
        'source': 'LaneVehicleGenerator(lane_waiting_count,in_only=True,average=all,negative=True)',
        'scale': 12,
        'dtype': 'float32',
    }
    topology = {
        'edges': sorted((
            edge.get('id'), edge.get('from'), edge.get('to'),
            tuple((lane.get('id'), lane.get('index')) for lane in edge.findall('lane')),
        ) for edge in root.findall('edge') if not str(edge.get('id', '')).startswith(':')),
        'connections': sorted((
            connection.get('from'), connection.get('to'),
            connection.get('fromLane'), connection.get('toLane'),
            connection.get('tl'), connection.get('linkIndex'),
        ) for connection in root.findall('connection')),
    }
    return {
        'roadnet_sha256': _sha256(roadnet),
        'roadnet_control_topology_sha256': _stable_hash(topology),
        'tl_program_sha256': _stable_hash(tl_programs),
        'action_mapping_sha256': _stable_hash(action_mapping),
        'feature_schema_sha256': _stable_hash(feature_descriptor),
        'reward_definition_sha256': _stable_hash(reward_descriptor),
        'controlled_intersection_count': len(action_mapping),
        'green_action_count': sum(len(states) for _, states in action_mapping),
        'feature_descriptor': feature_descriptor,
        'reward_descriptor': reward_descriptor,
    }


def _validate_shard_semantics(path, entry, source, action_dim):
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(TRAJECTORY_REQUIRED_FIELDS - set(data.files))
        if missing:
            raise ValueError(f'{path}: missing trajectory fields {missing}')
        if int(np.asarray(data['schema_version']).item()) != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(f'{path}: unsupported trajectory shard schema')
        if str(np.asarray(data['network']).item()) != source['network']:
            raise ValueError(f'{path}: shard network mismatch')
        if str(np.asarray(data['scene_id']).item()) != source['network']:
            raise ValueError(f'{path}: shard scene mismatch')
        if int(np.asarray(data['behavior_training_seed']).item()) != source[
            'behavior_training_seed'
        ]:
            raise ValueError(f'{path}: shard behavior seed mismatch')
        count = int(entry['transition_count'])
        episode_ids = np.asarray(data['episode_id']).reshape(-1)
        decisions = np.asarray(data['decision_step']).reshape(-1)
        globals_ = np.asarray(data['global_step']).reshape(-1)
        actions = np.asarray(data['action']).reshape(-1)
        rewards = np.asarray(data['reward']).reshape(-1)
        phases = np.asarray(data['current_phase']).reshape(-1)
        next_phases = np.asarray(data['next_phase']).reshape(-1)
        terminated = np.asarray(data['terminated']).reshape(-1)
        truncated = np.asarray(data['truncated']).reshape(-1)
        state = np.asarray(data['state'])
        next_state = np.asarray(data['next_state'])
        fields = (
            episode_ids, decisions, globals_, actions, rewards, phases,
            next_phases, terminated, truncated, state, next_state,
        )
        if any(len(field) != count for field in fields):
            raise ValueError(f'{path}: trajectory array length mismatch')
        if not np.all(episode_ids == entry['episode_id']):
            raise ValueError(f'{path}: episode identity mismatch')
        if not np.array_equal(decisions, np.arange(1, count + 1)):
            raise ValueError(f'{path}: non-contiguous decision steps')
        if not np.array_equal(
            globals_, np.arange(entry['first_global_step'], entry['last_global_step'] + 1)
        ):
            raise ValueError(f'{path}: non-contiguous global steps')
        if actions.dtype.kind not in 'iu' or phases.dtype.kind not in 'iu' or next_phases.dtype.kind not in 'iu':
            raise TypeError(f'{path}: actions and phases must use integer dtypes')
        if terminated.dtype != np.bool_ or truncated.dtype != np.bool_:
            raise TypeError(f'{path}: terminated/truncated must use bool dtype')
        if np.any(actions < 0) or np.any(actions >= action_dim):
            raise ValueError(f'{path}: action outside [0, {action_dim})')
        if np.any(phases < 0) or np.any(phases >= action_dim) or np.any(
            next_phases < 0
        ) or np.any(next_phases >= action_dim):
            raise ValueError(f'{path}: phase outside [0, {action_dim})')
        if np.any(terminated):
            raise ValueError(f'{path}: Plan 2 source has unexpected terminated=true')
        if count == 0 or np.any(truncated[:-1]) or not bool(truncated[-1]):
            raise ValueError(f'{path}: truncated flags do not mark exactly the episode end')
        for name, array in (('state', state), ('next_state', next_state), ('reward', rewards)):
            if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
                raise ValueError(f'{path}: {name} contains invalid numeric values')
        if count > 1 and not np.array_equal(next_state[:-1], state[1:]):
            raise ValueError(f'{path}: state chain is broken inside episode')
        if count > 1 and not np.array_equal(next_phases[:-1], phases[1:]):
            raise ValueError(f'{path}: phase chain is broken inside episode')
    return {
        'episode_id': int(entry['episode_id']),
        'transition_count': count,
        'first_global_step': int(entry['first_global_step']),
        'last_global_step': int(entry['last_global_step']),
    }


def _verify_config_archive(run_path):
    config_path = os.path.join(run_path, 'config')
    recorded = _read_json(os.path.join(config_path, 'config_hashes.json'))
    if recorded.get('algorithm') != 'sha256' or not isinstance(recorded.get('files'), dict):
        raise ValueError(f'{run_path}: invalid config hash manifest schema')
    errors = []
    for name, expected in recorded.get('files', {}).items():
        path = os.path.join(config_path, name)
        if not os.path.isfile(path):
            errors.append(f'missing config archive file: {name}')
        elif _sha256(path) != expected:
            errors.append(f'config archive hash mismatch: {name}')
    if errors:
        raise ValueError(f'{run_path}: ' + '; '.join(errors))


def _validate_source_run(
    run_path, network, behavior_seed, expected_episodes, require_plan1_formal=True,
):
    run_path = os.path.abspath(run_path)
    manifest = _read_json(os.path.join(run_path, 'run_manifest.json'))
    status = _read_json(os.path.join(run_path, 'run_status.json'))
    trajectory_root = os.path.join(run_path, 'trajectory')
    trajectory_manifest = _read_json(os.path.join(trajectory_root, 'manifest.json'))
    trajectory_validation = _read_json(os.path.join(trajectory_root, 'validation.json'))

    if status.get('status') != '已完成' or status.get('exit_code') != 0:
        raise ValueError(f'{run_path}: source run is not completed successfully')
    if (
        manifest.get('task') != 'tsc'
        or manifest.get('world') != 'sumo'
        or manifest.get('agent') != 'dqn'
    ):
        raise ValueError(f'{run_path}: source is not a TSC DQN run')
    if (
        require_plan1_formal
        and not str(manifest.get('prefix', '')).startswith('p1_formal_')
    ):
        raise ValueError(f'{run_path}: source is not marked as a Plan 1 formal run')
    if manifest.get('network') != network:
        raise ValueError(f'{run_path}: network does not match run-list')
    if manifest.get('training_seed') != behavior_seed:
        raise ValueError(f'{run_path}: behavior seed does not match run-list')
    if trajectory_manifest.get('network') != network:
        raise ValueError(f'{run_path}: trajectory network mismatch')
    if trajectory_manifest.get('behavior_training_seed') != behavior_seed:
        raise ValueError(f'{run_path}: trajectory behavior seed mismatch')
    if trajectory_manifest.get('config_hash') != manifest.get('config_hash'):
        raise ValueError(f'{run_path}: trajectory/run config hash mismatch')
    if trajectory_manifest.get('sumo_seed_mode') != 'fixed_default':
        raise ValueError(f'{run_path}: unsupported SUMO seed mode')
    if not trajectory_validation.get('valid'):
        raise ValueError(f'{run_path}: trajectory validation failed')
    if trajectory_validation.get('evaluation_transition_count') != 0:
        raise ValueError(f'{run_path}: evaluation transitions contaminated trajectory')
    if trajectory_validation.get('episode_count') != expected_episodes:
        raise ValueError(f'{run_path}: unexpected trajectory episode count')

    _verify_config_archive(run_path)
    resolved_config = os.path.join(run_path, 'config', 'resolved_config.yaml')
    if _sha256(resolved_config) != manifest.get('config_hash'):
        raise ValueError(f'{run_path}: run manifest config hash mismatch')
    entries = []
    with open(os.path.join(trajectory_root, 'index.jsonl'), encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    if len(entries) != expected_episodes:
        raise ValueError(f'{run_path}: trajectory index episode count mismatch')

    expected_per_episode = trajectory_manifest['expected_decisions_per_episode']
    if trajectory_manifest.get('schema_version') != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError(f'{run_path}: unsupported trajectory schema version')
    for expected_episode, entry in enumerate(entries, start=1):
        if entry.get('episode_id') != expected_episode:
            raise ValueError(f'{run_path}: non-contiguous episode index')
        if entry.get('transition_count') != expected_per_episode:
            raise ValueError(f'{run_path}: unexpected per-episode transition count')
        shard = os.path.abspath(os.path.join(trajectory_root, entry['file']))
        if not os.path.isfile(shard) or _sha256(shard) != entry.get('sha256'):
            raise ValueError(f'{run_path}: missing or corrupt trajectory shard {entry["file"]}')

    expected_total = expected_episodes * expected_per_episode
    if trajectory_validation.get('transition_count') != expected_total:
        raise ValueError(f'{run_path}: unexpected total transition count')
    scene_semantics = _scene_semantics(run_path)
    if scene_semantics['controlled_intersection_count'] != 1:
        raise ValueError(f'{run_path}: Plan 2 requires exactly one controlled intersection')
    if scene_semantics['green_action_count'] != trajectory_manifest['action_dim']:
        raise ValueError(
            f'{run_path}: tlLogic green-action count does not match trajectory action_dim'
        )
    return {
        'run_path': run_path,
        'run_id': manifest['run_id'],
        'network': network,
        'behavior_training_seed': behavior_seed,
        'config_hash': manifest['config_hash'],
        'action_dim': trajectory_manifest['action_dim'],
        'expected_decisions_per_episode': expected_per_episode,
        'entries': entries,
        'trajectory_root': trajectory_root,
        'scene_semantics': scene_semantics,
    }


def read_plan2_run_list(path):
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    required = {'run_path', 'network', 'behavior_training_seed'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'run-list must contain columns {sorted(required)}')
    normalized = []
    for row in rows:
        if not os.path.isabs(row['run_path']):
            raise ValueError('Plan 2 run-list run_path values must be absolute paths')
        normalized.append({
            'run_path': os.path.abspath(row['run_path']),
            'network': row['network'],
            'behavior_training_seed': int(row['behavior_training_seed']),
        })
    return normalized


def prepare_plan2_datasets(
    run_list_path,
    output_root,
    dataset_id,
    expected_networks=PLAN2_NETWORKS,
    expected_behavior_seeds=PLAN2_BEHAVIOR_SEEDS,
    expected_episodes=400,
    stages=PLAN2_STAGES,
    source_root=None,
    source_root_id=DEFAULT_SOURCE_ROOT_ID,
):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', dataset_id):
        raise ValueError('dataset_id contains unsupported characters')
    dataset_root = os.path.abspath(os.path.join(output_root, dataset_id))
    if os.path.exists(dataset_root):
        raise FileExistsError(f'Offline dataset id already exists: {dataset_root}')
    rows = read_plan2_run_list(run_list_path)
    canonical_runs = [os.path.realpath(row['run_path']) for row in rows]
    if len(canonical_runs) != len(set(canonical_runs)):
        raise ValueError('run-list contains duplicate or symlink-equivalent run paths')
    source_root = os.path.realpath(
        source_root or os.path.commonpath(canonical_runs)
    )
    if not os.path.isdir(source_root):
        raise ValueError(f'Offline source root does not exist: {source_root}')
    expected_pairs = {
        (network, seed)
        for network in expected_networks
        for seed in expected_behavior_seeds
    }
    actual_pairs = [(row['network'], row['behavior_training_seed']) for row in rows]
    if len(actual_pairs) != len(set(actual_pairs)):
        raise ValueError('run-list contains duplicate network/behavior seed pairs')
    if set(actual_pairs) != expected_pairs:
        missing = sorted(expected_pairs - set(actual_pairs))
        extra = sorted(set(actual_pairs) - expected_pairs)
        raise ValueError(f'run-list matrix mismatch; missing={missing}, extra={extra}')

    sources = [
        _validate_source_run(
            row['run_path'], row['network'], row['behavior_training_seed'],
            expected_episodes,
        )
        for row in rows
    ]
    comparison = compare_runs([source['run_path'] for source in sources])
    if not comparison['compatible']:
        raise ValueError(f'Plan 1 source runs are not configuration-compatible: {comparison}')
    action_dims = {source['action_dim'] for source in sources}
    decisions = {source['expected_decisions_per_episode'] for source in sources}
    if len(action_dims) != 1 or len(decisions) != 1:
        raise ValueError('Source runs have incompatible action dimensions or episode lengths')
    semantic_fields = (
        'roadnet_control_topology_sha256', 'tl_program_sha256',
        'action_mapping_sha256', 'feature_schema_sha256',
        'reward_definition_sha256',
    )
    for field in semantic_fields:
        values = {source['scene_semantics'][field] for source in sources}
        if len(values) != 1:
            raise ValueError(f'Plan 1 source semantic mismatch: {field}={sorted(values)}')

    os.makedirs(dataset_root)
    source_index_path = os.path.join(dataset_root, 'source_shards.jsonl')
    seen_run_episodes = set()
    seen_canonical_shards = set()
    seen_shard_hashes = set()
    raw_transition_count = 0
    with open(source_index_path, 'x', encoding='utf-8') as handle:
        for source in sorted(
            sources, key=lambda item: (item['network'], item['behavior_training_seed'])
        ):
            for entry in source['entries']:
                shard_path = os.path.abspath(
                    os.path.join(source['trajectory_root'], entry['file'])
                )
                canonical_shard = os.path.realpath(shard_path)
                logical_key = (source['run_id'], entry['episode_id'])
                if logical_key in seen_run_episodes:
                    raise ValueError(f'duplicate run/episode source: {logical_key}')
                if canonical_shard in seen_canonical_shards:
                    raise ValueError(f'duplicate or symlink-equivalent shard: {shard_path}')
                if entry['sha256'] in seen_shard_hashes:
                    raise ValueError(f'duplicate shard content hash: {entry["sha256"]}')
                try:
                    relative_path = os.path.relpath(canonical_shard, source_root)
                except ValueError as error:
                    raise ValueError(f'shard is outside source root: {shard_path}') from error
                if relative_path == os.pardir or relative_path.startswith(os.pardir + os.sep):
                    raise ValueError(f'shard is outside source root: {shard_path}')
                semantic_validation = _validate_shard_semantics(
                    shard_path, entry, source, next(iter(action_dims))
                )
                seen_run_episodes.add(logical_key)
                seen_canonical_shards.add(canonical_shard)
                seen_shard_hashes.add(entry['sha256'])
                raw_transition_count += entry['transition_count']
                indexed = {
                    'schema_version': OFFLINE_DATASET_SCHEMA_VERSION,
                    'run_path': source['run_path'],
                    'run_id': source['run_id'],
                    'network': source['network'],
                    'behavior_training_seed': source['behavior_training_seed'],
                    'episode_id': entry['episode_id'],
                    'transition_count': entry['transition_count'],
                    'source_root_id': source_root_id,
                    'relative_path': relative_path,
                    'original_absolute_path': canonical_shard,
                    'sha256': entry['sha256'],
                    'semantic_validation': semantic_validation,
                }
                handle.write(json.dumps(indexed, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())

    copied_run_list = os.path.join(dataset_root, 'source_runs.csv')
    with open(copied_run_list, 'x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['run_path', 'network', 'behavior_training_seed']
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda item: (item['network'], item['behavior_training_seed'])))

    dataset_manifests = []
    per_episode = next(iter(decisions))
    scene_semantics = {
        source['network']: source['scene_semantics'] for source in sources
        if source['behavior_training_seed'] == expected_behavior_seeds[0]
    }
    for network in expected_networks:
        for stage, episode_range in stages.items():
            path = os.path.join(dataset_root, 'datasets', network, stage, 'manifest.json')
            payload = {
                'schema_version': OFFLINE_DATASET_SCHEMA_VERSION,
                'dataset_builder_version': OFFLINE_DATASET_BUILDER_VERSION,
                'feature_schema_version': FEATURE_SCHEMA_VERSION,
                'trajectory_schema_version': TRAJECTORY_SCHEMA_VERSION,
                'created_by_commit': _git_commit(),
                'dataset_id': dataset_id,
                'dataset_kind': 'single_scene',
                'dataset_stage': stage,
                'evaluation_network': network,
                'source_networks': [network],
                'behavior_training_seeds': list(expected_behavior_seeds),
                'episode_range': list(episode_range),
                'sampling_strategy': 'uniform_transition',
                'source_index': os.path.relpath(source_index_path, os.path.dirname(path)),
                'source_index_sha256': _sha256(source_index_path),
                'source_root_id': source_root_id,
                'original_source_root': source_root,
                'action_dim': next(iter(action_dims)),
                'expected_decisions_per_episode': per_episode,
                'expected_transition_count': (
                    len(expected_behavior_seeds)
                    * (episode_range[1] - episode_range[0] + 1)
                    * per_episode
                ),
                'transition_unique_key': [
                    'run_id', 'shard_sha256', 'episode_id', 'decision_step'
                ],
                'scene_semantics': {network: scene_semantics[network]},
                'semantic_compatibility_fields': list(semantic_fields),
            }
            _atomic_json(path, payload)
            dataset_manifests.append(path)

    for target in expected_networks:
        source_networks = [network for network in expected_networks if network != target]
        path = os.path.join(
            dataset_root, 'datasets', 'leave_one_out', target, 'manifest.json'
        )
        payload = {
            'schema_version': OFFLINE_DATASET_SCHEMA_VERSION,
            'dataset_builder_version': OFFLINE_DATASET_BUILDER_VERSION,
            'feature_schema_version': FEATURE_SCHEMA_VERSION,
            'trajectory_schema_version': TRAJECTORY_SCHEMA_VERSION,
            'created_by_commit': _git_commit(),
            'dataset_id': dataset_id,
            'dataset_kind': 'leave_one_out',
            'dataset_stage': 'full',
            'evaluation_network': target,
            'source_networks': source_networks,
            'behavior_training_seeds': list(expected_behavior_seeds),
            'episode_range': [1, expected_episodes],
            'sampling_strategy': 'scene_balanced_batch',
            'source_index': os.path.relpath(source_index_path, os.path.dirname(path)),
            'source_index_sha256': _sha256(source_index_path),
            'source_root_id': source_root_id,
            'original_source_root': source_root,
            'action_dim': next(iter(action_dims)),
            'expected_decisions_per_episode': per_episode,
            'expected_transition_count': (
                len(source_networks)
                * len(expected_behavior_seeds)
                * expected_episodes
                * per_episode
            ),
            'transition_unique_key': [
                'run_id', 'shard_sha256', 'episode_id', 'decision_step'
            ],
            'scene_semantics': {
                network: scene_semantics[network] for network in source_networks
            },
            'semantic_compatibility_fields': list(semantic_fields),
        }
        _atomic_json(path, payload)
        dataset_manifests.append(path)

    root_manifest = {
        'schema_version': OFFLINE_DATASET_SCHEMA_VERSION,
        'dataset_builder_version': OFFLINE_DATASET_BUILDER_VERSION,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'trajectory_schema_version': TRAJECTORY_SCHEMA_VERSION,
        'created_by_commit': _git_commit(),
        'dataset_id': dataset_id,
        'created_from_run_list': os.path.abspath(run_list_path),
        'source_runs': os.path.relpath(copied_run_list, dataset_root),
        'source_index': os.path.relpath(source_index_path, dataset_root),
        'source_index_sha256': _sha256(source_index_path),
        'source_root_id': source_root_id,
        'original_source_root': source_root,
        'networks': list(expected_networks),
        'behavior_training_seeds': list(expected_behavior_seeds),
        'episode_count_per_run': expected_episodes,
        'expected_decisions_per_episode': per_episode,
        'action_dim': next(iter(action_dims)),
        'dataset_manifests': [os.path.relpath(path, dataset_root) for path in dataset_manifests],
        'source_run_compatibility': comparison,
        'scene_semantics': scene_semantics,
        'semantic_compatibility_fields': list(semantic_fields),
        'raw_source_shard_count': len(seen_canonical_shards),
        'deduplicated_source_shard_count': len(seen_canonical_shards),
        'raw_transition_count': raw_transition_count,
        'deduplicated_transition_count': raw_transition_count,
        'transition_unique_key': [
            'run_id', 'shard_sha256', 'episode_id', 'decision_step'
        ],
    }
    _atomic_json(os.path.join(dataset_root, 'manifest.json'), root_manifest)
    return dataset_root


def _resolve_shard_path(entry, manifest, source_root=None):
    resolved_root = os.path.realpath(
        source_root or manifest.get('original_source_root', '')
    )
    if not resolved_root:
        raise ValueError('Offline source root is not configured')
    if entry.get('source_root_id') != manifest.get('source_root_id'):
        raise ValueError('Offline source root identity mismatch')
    relative = entry.get('relative_path')
    if not isinstance(relative, str) or os.path.isabs(relative):
        raise ValueError('Offline shard relative_path is invalid')
    path = os.path.realpath(os.path.join(resolved_root, relative))
    if os.path.commonpath((resolved_root, path)) != resolved_root:
        raise ValueError('Offline shard path escapes source root')
    return path


def _load_source_entries(manifest_path, verify_hashes=True, source_root=None):
    manifest = _read_json(manifest_path)
    if manifest.get('schema_version') != OFFLINE_DATASET_SCHEMA_VERSION:
        raise ValueError(
            'Unsupported offline dataset schema version; rebuild the dataset '
            f'with builder {OFFLINE_DATASET_BUILDER_VERSION}'
        )
    expected_versions = {
        'dataset_builder_version': OFFLINE_DATASET_BUILDER_VERSION,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'trajectory_schema_version': TRAJECTORY_SCHEMA_VERSION,
    }
    mismatched = {
        field: (manifest.get(field), expected)
        for field, expected in expected_versions.items()
        if manifest.get(field) != expected
    }
    if mismatched:
        raise ValueError(f'Incompatible offline dataset semantics: {mismatched}')
    source_index = os.path.abspath(
        os.path.join(os.path.dirname(manifest_path), manifest['source_index'])
    )
    if _sha256(source_index) != manifest['source_index_sha256']:
        raise ValueError('Offline source index hash mismatch')
    entries = []
    seen_logical = set()
    seen_paths = set()
    seen_hashes = set()
    with open(source_index, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            missing_entry = sorted(SOURCE_INDEX_REQUIRED_FIELDS - set(entry))
            if missing_entry:
                raise ValueError(
                    f'Offline source index entry missing fields: {missing_entry}'
                )
            if entry['schema_version'] != OFFLINE_DATASET_SCHEMA_VERSION:
                raise ValueError('Offline source index schema version mismatch')
            if entry['network'] not in manifest['source_networks']:
                continue
            if entry['behavior_training_seed'] not in manifest['behavior_training_seeds']:
                continue
            if not manifest['episode_range'][0] <= entry['episode_id'] <= manifest['episode_range'][1]:
                continue
            shard_path = _resolve_shard_path(entry, manifest, source_root=source_root)
            logical_key = (entry['run_id'], entry['episode_id'])
            if logical_key in seen_logical:
                raise ValueError(f'Duplicate offline run/episode entry: {logical_key}')
            if shard_path in seen_paths:
                raise ValueError(f'Duplicate or symlink-equivalent offline shard: {shard_path}')
            if entry['sha256'] in seen_hashes:
                raise ValueError(f'Duplicate offline shard content: {entry["sha256"]}')
            if verify_hashes and _sha256(shard_path) != entry['sha256']:
                raise ValueError(f'Offline source shard hash mismatch: {shard_path}')
            entry = dict(entry)
            entry['resolved_shard_path'] = shard_path
            seen_logical.add(logical_key)
            seen_paths.add(shard_path)
            seen_hashes.add(entry['sha256'])
            entries.append(entry)
    count = sum(entry['transition_count'] for entry in entries)
    if count != manifest['expected_transition_count']:
        raise ValueError(
            f'Offline dataset transition count {count} != '
            f'{manifest["expected_transition_count"]}'
        )
    if manifest['dataset_kind'] == 'leave_one_out' and manifest['evaluation_network'] in {
        entry['network'] for entry in entries
    }:
        raise ValueError('Leave-one-out dataset contains target-network transitions')
    return manifest, entries


def validate_offline_dataset(manifest_path, verify_hashes=True, source_root=None):
    manifest, entries = _load_source_entries(
        manifest_path, verify_hashes=verify_hashes, source_root=source_root
    )
    result = {
        'schema_version': OFFLINE_DATASET_SCHEMA_VERSION,
        'valid': True,
        'dataset_id': manifest['dataset_id'],
        'dataset_kind': manifest['dataset_kind'],
        'dataset_stage': manifest['dataset_stage'],
        'evaluation_network': manifest['evaluation_network'],
        'source_networks': manifest['source_networks'],
        'source_shard_count': len(entries),
        'transition_count': sum(entry['transition_count'] for entry in entries),
        'source_index_sha256': manifest['source_index_sha256'],
        'source_root_id': manifest['source_root_id'],
        'resolved_source_root': os.path.realpath(
            source_root or manifest['original_source_root']
        ),
        'dataset_builder_version': manifest['dataset_builder_version'],
        'feature_schema_version': manifest['feature_schema_version'],
        'trajectory_schema_version': manifest['trajectory_schema_version'],
    }
    validation_path = os.path.join(os.path.dirname(manifest_path), 'validation.json')
    _atomic_json(validation_path, result)
    return result


def _one_hot(phases, action_dim):
    phases = np.asarray(phases, dtype=np.int64).reshape(-1)
    if np.any(phases < 0) or np.any(phases >= action_dim):
        raise ValueError('Offline trajectory contains an invalid phase index')
    result = np.zeros((len(phases), action_dim), dtype=np.float32)
    result[np.arange(len(phases)), phases] = 1.0
    return result


@dataclass
class OfflineBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    source_network: str


class OfflineTrajectoryDataset:
    """Read-only Plan 2 transition dataset backed by Plan 1 NPZ shards."""

    def __init__(self, manifest_path, seed, verify_hashes=True, source_root=None):
        self.manifest_path = os.path.abspath(manifest_path)
        self.load_started_at = time.perf_counter()
        usage_before = resource.getrusage(resource.RUSAGE_SELF)
        rss_before = usage_before.ru_maxrss
        self.manifest, entries = _load_source_entries(
            self.manifest_path, verify_hashes=verify_hashes, source_root=source_root
        )
        self.source_shard_count = len(entries)
        self.source_npz_bytes = sum(
            os.path.getsize(entry['resolved_shard_path']) for entry in entries
        )
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.action_dim = int(self.manifest['action_dim'])
        self.by_network = {}
        for entry in entries:
            self.by_network.setdefault(entry['network'], []).append(entry)
        self._arrays = {}
        self._load_arrays()
        self.load_seconds = time.perf_counter() - self.load_started_at
        usage_after = resource.getrusage(resource.RUSAGE_SELF)
        rss_after = usage_after.ru_maxrss
        self.peak_rss_delta_bytes = max(0, rss_after - rss_before) * 1024
        self.load_cpu_user_seconds = usage_after.ru_utime - usage_before.ru_utime
        self.load_cpu_system_seconds = usage_after.ru_stime - usage_before.ru_stime
        self.load_block_input_operations = usage_after.ru_inblock - usage_before.ru_inblock
        self.resident_array_bytes = sum(
            array.nbytes for arrays in self._arrays.values()
            for array in arrays.values() if isinstance(array, np.ndarray)
        )
        self.storage_strategy = 'eager_in_memory_npz_load_once'

    def _load_arrays(self):
        for network, entries in self.by_network.items():
            observations = []
            next_observations = []
            actions = []
            rewards = []
            terminated = []
            truncated = []
            episode_ids = []
            decision_steps = []
            global_steps = []
            behavior_training_seeds = []
            for entry in entries:
                with np.load(entry['resolved_shard_path'], allow_pickle=False) as data:
                    state = np.asarray(data['state'], dtype=np.float32).reshape(
                        len(data['state']), -1
                    )
                    next_state = np.asarray(data['next_state'], dtype=np.float32).reshape(
                        len(data['next_state']), -1
                    )
                    phase = np.asarray(data['current_phase']).reshape(len(state), -1)
                    next_phase = np.asarray(data['next_phase']).reshape(len(state), -1)
                    if phase.shape[1] != 1 or next_phase.shape[1] != 1:
                        raise ValueError('Plan 2 currently requires one controlled intersection')
                    observations.append(
                        np.concatenate([state, _one_hot(phase, self.action_dim)], axis=1)
                    )
                    next_observations.append(
                        np.concatenate(
                            [next_state, _one_hot(next_phase, self.action_dim)], axis=1
                        )
                    )
                    shard_actions = np.asarray(data['action'])
                    if shard_actions.dtype.kind not in 'iu':
                        raise TypeError('Offline actions must use an integer dtype')
                    shard_actions = shard_actions.astype(np.int64, copy=False).reshape(-1)
                    if np.any(shard_actions < 0) or np.any(shard_actions >= self.action_dim):
                        raise ValueError('Offline trajectory contains an invalid action')
                    shard_rewards = np.asarray(data['reward'], dtype=np.float32).reshape(-1)
                    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(next_state)):
                        raise ValueError('Offline state contains NaN/Inf')
                    if not np.all(np.isfinite(shard_rewards)):
                        raise ValueError('Offline reward contains NaN/Inf')
                    shard_terminated = np.asarray(data['terminated'])
                    shard_truncated = np.asarray(data['truncated'])
                    if shard_terminated.dtype != np.bool_ or shard_truncated.dtype != np.bool_:
                        raise TypeError('Offline terminal flags must use bool dtype')
                    actions.append(shard_actions)
                    rewards.append(shard_rewards)
                    terminated.append(shard_terminated.reshape(-1))
                    truncated.append(shard_truncated.reshape(-1))
                    episode_ids.append(
                        np.asarray(data['episode_id'], dtype=np.int64).reshape(-1)
                    )
                    decision_steps.append(
                        np.asarray(data['decision_step'], dtype=np.int64).reshape(-1)
                    )
                    global_steps.append(
                        np.asarray(data['global_step'], dtype=np.int64).reshape(-1)
                    )
                    behavior_training_seeds.append(np.full(
                        len(shard_actions), entry['behavior_training_seed'], dtype=np.int64
                    ))
            self._arrays[network] = {
                'observations': np.ascontiguousarray(np.concatenate(observations), dtype=np.float32),
                'next_observations': np.ascontiguousarray(
                    np.concatenate(next_observations), dtype=np.float32
                ),
                'actions': np.concatenate(actions),
                'rewards': np.concatenate(rewards),
                'terminated': np.concatenate(terminated),
                'truncated': np.concatenate(truncated),
                'episode_id': np.concatenate(episode_ids),
                'decision_step': np.concatenate(decision_steps),
                'global_step': np.concatenate(global_steps),
                'behavior_training_seed': np.concatenate(behavior_training_seeds),
            }

    @property
    def transition_count(self):
        return sum(len(arrays['actions']) for arrays in self._arrays.values())

    def __len__(self):
        return self.transition_count

    @property
    def observation_dim(self):
        first = next(iter(self._arrays.values()))
        return first['observations'].shape[1]

    def sample_batch(self, batch_size):
        networks = sorted(self._arrays)
        if self.manifest['sampling_strategy'] == 'scene_balanced_batch':
            network = networks[self.rng.randrange(len(networks))]
        else:
            network = networks[0]
        arrays = self._arrays[network]
        if batch_size > len(arrays['actions']):
            raise ValueError(
                f'batch_size {batch_size} exceeds source transition count '
                f'{len(arrays["actions"])}'
            )
        indices = self.rng.sample(range(len(arrays['actions'])), batch_size)
        return OfflineBatch(
            observations=arrays['observations'][indices],
            actions=arrays['actions'][indices],
            rewards=arrays['rewards'][indices],
            next_observations=arrays['next_observations'][indices],
            terminated=arrays['terminated'][indices],
            truncated=arrays['truncated'][indices],
            source_network=network,
        )

    def rng_state(self):
        return self.rng.getstate()

    def set_rng_state(self, state):
        self.rng.setstate(state)

    def statistics(self):
        result = {}
        for network, arrays in sorted(self._arrays.items()):
            actions, counts = np.unique(arrays['actions'], return_counts=True)
            result[network] = {
                'transition_count': int(len(arrays['actions'])),
                'reward_mean': float(np.mean(arrays['rewards'])),
                'reward_std': float(np.std(arrays['rewards'])),
                'action_counts': {
                    str(int(action)): int(count) for action, count in zip(actions, counts)
                },
            }
        return result

    def performance_profile(self, batch_size=64, sample_batches=1000):
        if not isinstance(sample_batches, int) or sample_batches <= 0:
            raise ValueError('sample_batches must be a positive integer')
        usage_before = resource.getrusage(resource.RUSAGE_SELF)
        started = time.perf_counter()
        for _ in range(sample_batches):
            self.sample_batch(batch_size)
        elapsed = time.perf_counter() - started
        usage_after = resource.getrusage(resource.RUSAGE_SELF)
        return {
            'storage_strategy': self.storage_strategy,
            'source_shards_open_after_load': 0,
            'dataset_load_seconds': self.load_seconds,
            'dataset_load_cpu_user_seconds': self.load_cpu_user_seconds,
            'dataset_load_cpu_system_seconds': self.load_cpu_system_seconds,
            'dataset_load_block_input_operations': self.load_block_input_operations,
            'source_shard_count': self.source_shard_count,
            'source_npz_bytes': self.source_npz_bytes,
            'effective_source_npz_bytes_per_second': (
                self.source_npz_bytes / self.load_seconds if self.load_seconds else None
            ),
            'resident_array_bytes': self.resident_array_bytes,
            'peak_rss_delta_bytes': self.peak_rss_delta_bytes,
            'sample_batches': sample_batches,
            'sample_batch_size': batch_size,
            'sampling_seconds': elapsed,
            'sampling_cpu_user_seconds': usage_after.ru_utime - usage_before.ru_utime,
            'sampling_cpu_system_seconds': usage_after.ru_stime - usage_before.ru_stime,
            'sampling_block_input_operations': (
                usage_after.ru_inblock - usage_before.ru_inblock
            ),
            'mean_sample_batch_seconds': elapsed / sample_batches,
            'sample_batches_per_second': sample_batches / elapsed if elapsed else None,
        }
