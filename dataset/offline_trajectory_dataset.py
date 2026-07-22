import csv
import hashlib
import json
import os
import random
import re
import tempfile
from dataclasses import dataclass

import numpy as np

from common.registry import Registry
from utils.run_config_compare import compare_runs


OFFLINE_DATASET_SCHEMA_VERSION = 1
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


def _validate_source_run(run_path, network, behavior_seed, expected_episodes):
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
    if not str(manifest.get('prefix', '')).startswith('p1_formal_'):
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
):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', dataset_id):
        raise ValueError('dataset_id contains unsupported characters')
    dataset_root = os.path.abspath(os.path.join(output_root, dataset_id))
    if os.path.exists(dataset_root):
        raise FileExistsError(f'Offline dataset id already exists: {dataset_root}')
    rows = read_plan2_run_list(run_list_path)
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

    os.makedirs(dataset_root)
    source_index_path = os.path.join(dataset_root, 'source_shards.jsonl')
    with open(source_index_path, 'x', encoding='utf-8') as handle:
        for source in sorted(
            sources, key=lambda item: (item['network'], item['behavior_training_seed'])
        ):
            for entry in source['entries']:
                shard_path = os.path.abspath(
                    os.path.join(source['trajectory_root'], entry['file'])
                )
                indexed = {
                    'schema_version': OFFLINE_DATASET_SCHEMA_VERSION,
                    'run_path': source['run_path'],
                    'run_id': source['run_id'],
                    'network': source['network'],
                    'behavior_training_seed': source['behavior_training_seed'],
                    'episode_id': entry['episode_id'],
                    'transition_count': entry['transition_count'],
                    'shard_path': shard_path,
                    'sha256': entry['sha256'],
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
    for network in expected_networks:
        for stage, episode_range in stages.items():
            path = os.path.join(dataset_root, 'datasets', network, stage, 'manifest.json')
            payload = {
                'schema_version': OFFLINE_DATASET_SCHEMA_VERSION,
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
                'action_dim': next(iter(action_dims)),
                'expected_decisions_per_episode': per_episode,
                'expected_transition_count': (
                    len(expected_behavior_seeds)
                    * (episode_range[1] - episode_range[0] + 1)
                    * per_episode
                ),
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
            'action_dim': next(iter(action_dims)),
            'expected_decisions_per_episode': per_episode,
            'expected_transition_count': (
                len(source_networks)
                * len(expected_behavior_seeds)
                * expected_episodes
                * per_episode
            ),
        }
        _atomic_json(path, payload)
        dataset_manifests.append(path)

    root_manifest = {
        'schema_version': OFFLINE_DATASET_SCHEMA_VERSION,
        'dataset_id': dataset_id,
        'created_from_run_list': os.path.abspath(run_list_path),
        'source_runs': os.path.relpath(copied_run_list, dataset_root),
        'source_index': os.path.relpath(source_index_path, dataset_root),
        'source_index_sha256': _sha256(source_index_path),
        'networks': list(expected_networks),
        'behavior_training_seeds': list(expected_behavior_seeds),
        'episode_count_per_run': expected_episodes,
        'expected_decisions_per_episode': per_episode,
        'action_dim': next(iter(action_dims)),
        'dataset_manifests': [os.path.relpath(path, dataset_root) for path in dataset_manifests],
        'source_run_compatibility': comparison,
    }
    _atomic_json(os.path.join(dataset_root, 'manifest.json'), root_manifest)
    return dataset_root


def _load_source_entries(manifest_path, verify_hashes=True):
    manifest = _read_json(manifest_path)
    if manifest.get('schema_version') != OFFLINE_DATASET_SCHEMA_VERSION:
        raise ValueError('Unsupported offline dataset schema version')
    source_index = os.path.abspath(
        os.path.join(os.path.dirname(manifest_path), manifest['source_index'])
    )
    if _sha256(source_index) != manifest['source_index_sha256']:
        raise ValueError('Offline source index hash mismatch')
    entries = []
    with open(source_index, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry['network'] not in manifest['source_networks']:
                continue
            if entry['behavior_training_seed'] not in manifest['behavior_training_seeds']:
                continue
            if not manifest['episode_range'][0] <= entry['episode_id'] <= manifest['episode_range'][1]:
                continue
            if verify_hashes and _sha256(entry['shard_path']) != entry['sha256']:
                raise ValueError(f'Offline source shard hash mismatch: {entry["shard_path"]}')
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


def validate_offline_dataset(manifest_path, verify_hashes=True):
    manifest, entries = _load_source_entries(manifest_path, verify_hashes=verify_hashes)
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
    source_network: str


class OfflineTrajectoryDataset:
    """Read-only Plan 2 transition dataset backed by Plan 1 NPZ shards."""

    def __init__(self, manifest_path, seed, verify_hashes=True):
        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest, entries = _load_source_entries(
            self.manifest_path, verify_hashes=verify_hashes
        )
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.action_dim = int(self.manifest['action_dim'])
        self.by_network = {}
        for entry in entries:
            self.by_network.setdefault(entry['network'], []).append(entry)
        self._arrays = {}
        self._load_arrays()

    def _load_arrays(self):
        for network, entries in self.by_network.items():
            observations = []
            next_observations = []
            actions = []
            rewards = []
            for entry in entries:
                with np.load(entry['shard_path'], allow_pickle=False) as data:
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
                    actions.append(np.asarray(data['action'], dtype=np.int64).reshape(-1))
                    rewards.append(np.asarray(data['reward'], dtype=np.float32).reshape(-1))
            self._arrays[network] = {
                'observations': np.concatenate(observations),
                'next_observations': np.concatenate(next_observations),
                'actions': np.concatenate(actions),
                'rewards': np.concatenate(rewards),
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
