import hashlib
import json
import math
import os
import tempfile

import numpy as np


TRAJECTORY_SCHEMA_VERSION = 1
TRAJECTORY_ARRAY_FIELDS = (
    'episode_id', 'decision_step', 'global_step',
    'state', 'current_phase', 'action', 'reward',
    'next_state', 'next_phase', 'terminated', 'truncated',
    'epsilon', 'behavior_mode', 'queue', 'approximate_delay',
    'real_delay', 'throughput', 'waiting_time',
)


def _atomic_json(path, payload):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix='.tmp-', dir=directory)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class EpisodeTrajectoryWriter:
    """Write immutable, episode-sharded Plan 1 training trajectories."""

    def __init__(
        self, output_path, network, behavior_training_seed, config_hash,
        simulation_steps, action_interval, action_dim,
    ):
        self.root = os.path.join(output_path, 'trajectory')
        self.episodes_path = os.path.join(self.root, 'episodes')
        self.manifest_path = os.path.join(self.root, 'manifest.json')
        self.index_path = os.path.join(self.root, 'index.jsonl')
        self.validation_path = os.path.join(self.root, 'validation.json')
        self.network = network
        self.behavior_training_seed = behavior_training_seed
        self.config_hash = config_hash
        self.expected_decisions_per_episode = math.ceil(
            simulation_steps / action_interval
        )
        self.action_dim = action_dim
        self.records = []
        self.current_episode = None
        self.total_count = 0
        self.index_entries = []
        os.makedirs(self.episodes_path, exist_ok=False)
        _atomic_json(self.manifest_path, {
            'schema_version': TRAJECTORY_SCHEMA_VERSION,
            'storage': 'episode_npz',
            'network': network,
            'scene_id': network,
            'behavior_training_seed': behavior_training_seed,
            'sumo_seed_mode': 'fixed_default',
            'config_hash': config_hash,
            'expected_decisions_per_episode': self.expected_decisions_per_episode,
            'action_dim': action_dim,
        })

    def start_episode(self, episode_id):
        if self.current_episode is not None or self.records:
            raise RuntimeError('Previous trajectory episode was not finalized')
        self.current_episode = int(episode_id)

    def append(self, **record):
        if self.current_episode is None:
            raise RuntimeError('Trajectory episode has not been started')
        missing = sorted(set(TRAJECTORY_ARRAY_FIELDS) - set(record))
        extra = sorted(set(record) - set(TRAJECTORY_ARRAY_FIELDS))
        if missing or extra:
            raise ValueError(
                f'Invalid trajectory record fields; missing={missing}, extra={extra}'
            )
        if int(record['episode_id']) != self.current_episode:
            raise ValueError('Trajectory record episode_id does not match active episode')
        copied = {}
        for field, value in record.items():
            copied[field] = value if isinstance(value, str) else np.array(value, copy=True)
        self.records.append(copied)

    def _episode_arrays(self):
        if not self.records:
            raise ValueError('Cannot finalize an empty trajectory episode')
        arrays = {
            field: np.stack([record[field] for record in self.records])
            if field not in {'behavior_mode'} else
            np.asarray([record[field] for record in self.records], dtype='U32')
            for field in TRAJECTORY_ARRAY_FIELDS
        }
        arrays.update({
            'schema_version': np.asarray(TRAJECTORY_SCHEMA_VERSION, dtype=np.int64),
            'network': np.asarray(self.network),
            'scene_id': np.asarray(self.network),
            'behavior_training_seed': np.asarray(
                self.behavior_training_seed, dtype=np.int64
            ),
            'sumo_seed_mode': np.asarray('fixed_default'),
        })
        return arrays

    def finish_episode(self):
        arrays = self._episode_arrays()
        episode_id = self.current_episode
        final_path = os.path.join(
            self.episodes_path, f'episode_{episode_id:04d}.npz'
        )
        if os.path.exists(final_path):
            raise FileExistsError(f'Trajectory shard already exists: {final_path}')
        descriptor, temporary_path = tempfile.mkstemp(
            prefix='.tmp-', suffix='.npz', dir=self.episodes_path
        )
        try:
            with os.fdopen(descriptor, 'wb') as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, final_path)
        except Exception:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise

        entry = {
            'schema_version': TRAJECTORY_SCHEMA_VERSION,
            'episode_id': episode_id,
            'file': os.path.relpath(final_path, self.root),
            'transition_count': len(self.records),
            'first_global_step': int(arrays['global_step'][0]),
            'last_global_step': int(arrays['global_step'][-1]),
            'sha256': _sha256(final_path),
        }
        with open(self.index_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        self.index_entries.append(entry)
        self.total_count += len(self.records)
        self.records = []
        self.current_episode = None
        return final_path

    def validate(self, expected_episodes=None):
        errors = []
        total = 0
        previous_global_step = 0
        entries = []
        if os.path.exists(self.index_path):
            with open(self.index_path, encoding='utf-8') as handle:
                entries = [json.loads(line) for line in handle if line.strip()]
        if expected_episodes is not None and len(entries) != expected_episodes:
            errors.append(
                f'episode shard count {len(entries)} != expected {expected_episodes}'
            )
        for expected_episode, entry in enumerate(entries, start=1):
            if entry['episode_id'] != expected_episode:
                errors.append(
                    f'episode index {entry["episode_id"]} != expected {expected_episode}'
                )
            path = os.path.join(self.root, entry['file'])
            if not os.path.isfile(path):
                errors.append(f'missing trajectory shard: {entry["file"]}')
                continue
            if _sha256(path) != entry['sha256']:
                errors.append(f'trajectory hash mismatch: {entry["file"]}')
                continue
            with np.load(path, allow_pickle=False) as data:
                missing = sorted(
                    set(TRAJECTORY_ARRAY_FIELDS) - set(data.files)
                )
                if missing:
                    errors.append(f'{entry["file"]} missing fields {missing}')
                    continue
                count = len(data['global_step'])
                total += count
                if count != entry['transition_count']:
                    errors.append(f'{entry["file"]} index count mismatch')
                if count != self.expected_decisions_per_episode:
                    errors.append(
                        f'{entry["file"]} transition count {count} != '
                        f'{self.expected_decisions_per_episode}'
                    )
                expected_decisions = np.arange(1, count + 1)
                if not np.array_equal(data['decision_step'], expected_decisions):
                    errors.append(f'{entry["file"]} decision steps are not contiguous')
                expected_globals = np.arange(
                    previous_global_step + 1, previous_global_step + count + 1
                )
                if not np.array_equal(data['global_step'], expected_globals):
                    errors.append(f'{entry["file"]} global steps are not contiguous')
                previous_global_step += count
                if count > 1 and not np.array_equal(
                    data['next_state'][:-1], data['state'][1:]
                ):
                    errors.append(f'{entry["file"]} state chain is broken')
                if count > 1 and not np.array_equal(
                    data['next_phase'][:-1], data['current_phase'][1:]
                ):
                    errors.append(f'{entry["file"]} phase chain is broken')
                numeric_fields = (
                    'state', 'reward', 'next_state', 'epsilon', 'queue',
                    'approximate_delay', 'real_delay', 'throughput', 'waiting_time',
                )
                for field in numeric_fields:
                    if not np.all(np.isfinite(data[field])):
                        errors.append(f'{entry["file"]} {field} contains NaN/Inf')
                if np.any(data['action'] < 0) or np.any(data['action'] >= self.action_dim):
                    errors.append(f'{entry["file"]} contains illegal actions')
                if np.any(data['terminated']):
                    errors.append(f'{entry["file"]} has unexpected terminated=true')
                if count and (
                    np.any(data['truncated'][:-1]) or not bool(data['truncated'][-1])
                ):
                    errors.append(f'{entry["file"]} truncated flags are invalid')
        result = {
            'schema_version': TRAJECTORY_SCHEMA_VERSION,
            'valid': not errors,
            'errors': errors,
            'episode_count': len(entries),
            'transition_count': total,
            'evaluation_transition_count': 0,
            'expected_decisions_per_episode': self.expected_decisions_per_episode,
        }
        _atomic_json(self.validation_path, result)
        return result
