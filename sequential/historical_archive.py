"""Read-only Plan 1 Q1 archive used by HA-SODQN.

The archive is a logical index over the already validated Plan 2 manifests.  It
never copies or mutates Plan 1 trajectory shards and keeps visibility separate
from sampling or working-pool policy.
"""

import bisect
import hashlib
import json
import os
import random
from dataclasses import dataclass

import numpy as np

from dataset.offline_trajectory_dataset import OfflineTrajectoryDataset

from .io import atomic_json, read_json, sha256_file


ARCHIVE_SCHEMA_VERSION = 1
ARCHIVE_MODES = ('P1C', 'P1F')


def stable_digest(payload):
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')).hexdigest()


def derive_rng_seed(training_seed, stream_name, version=1):
    """Derive a stable independent RNG seed without Python ``hash()``."""
    payload = {
        'version': int(version), 'training_seed': int(training_seed),
        'stream_name': str(stream_name),
    }
    return int(stable_digest(payload)[:16], 16)


def build_archive_root_manifest(dataset_root, output_path, seed_rule):
    dataset_root = os.path.abspath(dataset_root)
    root = read_json(os.path.join(dataset_root, 'manifest.json'))
    manifests = {}
    transition_counts = {}
    for network in root['networks']:
        path = os.path.join(dataset_root, 'datasets', network, 'Q1', 'manifest.json')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing Plan 2 Q1 manifest: {path}')
        manifest = read_json(path)
        if manifest.get('dataset_stage') != 'Q1' or manifest.get('episode_range') != [1, 100]:
            raise ValueError(f'{path}: HA-SODQN requires Plan 1 episodes 1-100')
        if manifest.get('source_networks') != [network]:
            raise ValueError(f'{path}: Q1 manifest must contain exactly its own network')
        if manifest.get('action_dim') != 8:
            raise ValueError(f'{path}: action_dim must be 8')
        manifests[network] = {
            'path': os.path.abspath(path),
            'sha256': sha256_file(path),
            'expected_transition_count': int(manifest['expected_transition_count']),
            'scene_semantics': manifest['scene_semantics'][network],
        }
        transition_counts[network] = int(manifest['expected_transition_count'])
    payload = {
        'schema_version': ARCHIVE_SCHEMA_VERSION,
        'archive_kind': 'plan1_q1_static_readonly',
        'dataset_id': root['dataset_id'],
        'dataset_root_manifest': os.path.abspath(os.path.join(dataset_root, 'manifest.json')),
        'dataset_root_manifest_sha256': sha256_file(os.path.join(dataset_root, 'manifest.json')),
        'source_root_id': root['source_root_id'],
        'original_source_root': root['original_source_root'],
        'networks': list(root['networks']),
        'behavior_training_seeds': list(root['behavior_training_seeds']),
        'episode_range': [1, 100],
        'observation_dim': 16,
        'action_dim': 8,
        'transition_counts': transition_counts,
        'total_transition_count': sum(transition_counts.values()),
        'q1_manifests': manifests,
        'behavior_seed_rule': dict(seed_rule),
        'rng_derivation': {
            'version': 1,
            'algorithm': 'sha256_first_64_bits',
            'payload_fields': ['version', 'training_seed', 'stream_name'],
        },
    }
    payload['archive_digest'] = stable_digest(payload)
    atomic_json(output_path, payload)
    return payload


@dataclass(frozen=True)
class HistoricalBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    source_networks: np.ndarray
    local_indices: np.ndarray
    transition_ids: np.ndarray
    behavior_training_seeds: np.ndarray
    episode_ids: np.ndarray
    decision_steps: np.ndarray
    run_ids: np.ndarray
    shard_sha256: np.ndarray

    def __len__(self):
        return len(self.actions)


class VisibleArchiveView:
    """Non-copying network/index view over a :class:`HistoricalArchive`."""

    def __init__(self, archive, indices_by_network, visibility):
        self.archive = archive
        self.indices_by_network = {
            network: np.asarray(indices, dtype=np.int64)
            for network, indices in sorted(indices_by_network.items())
            if len(indices)
        }
        for indices in self.indices_by_network.values():
            indices.setflags(write=False)
        self.visibility = dict(visibility)
        self.networks = tuple(self.indices_by_network)
        self._ends = []
        total = 0
        for network in self.networks:
            total += len(self.indices_by_network[network])
            self._ends.append(total)
        self.transition_count = total
        self.digest = stable_digest({
            'archive_digest': archive.manifest['archive_digest'],
            'visibility': self.visibility,
            'counts': {key: len(value) for key, value in self.indices_by_network.items()},
            'first_last_indices': {
                key: ([int(value[0]), int(value[-1])] if len(value) else [])
                for key, value in self.indices_by_network.items()
            },
        })

    def __len__(self):
        return self.transition_count

    def resolve_ordinals(self, ordinals):
        references = []
        for ordinal in np.asarray(ordinals, dtype=np.int64).reshape(-1):
            if ordinal < 0 or ordinal >= self.transition_count:
                raise IndexError('Archive ordinal outside visible view')
            network_position = bisect.bisect_right(self._ends, int(ordinal))
            previous_end = 0 if network_position == 0 else self._ends[network_position - 1]
            network = self.networks[network_position]
            local = self.indices_by_network[network][int(ordinal) - previous_end]
            references.append((network, int(local)))
        return references

    def batch_from_references(self, references):
        grouped = {network: [] for network in self.networks}
        positions = {network: [] for network in self.networks}
        for position, (network, index) in enumerate(references):
            if network not in self.indices_by_network:
                raise ValueError(f'Network {network} is not visible')
            grouped[network].append(int(index))
            positions[network].append(position)
        size = len(references)
        result = {}
        fields = (
            'observations', 'actions', 'rewards', 'next_observations',
            'terminated', 'truncated', 'transition_ids',
            'behavior_training_seeds', 'episode_ids', 'decision_steps',
            'run_ids', 'shard_sha256',
        )
        for network, indices in grouped.items():
            if not indices:
                continue
            batch = self.archive.datasets[network].select_indices(network, indices)
            for field in fields:
                values = np.asarray(getattr(batch, field))
                if field not in result:
                    result[field] = np.empty((size,) + values.shape[1:], dtype=values.dtype)
                result[field][positions[network]] = values
        source_networks = np.asarray([network for network, _ in references], dtype=object)
        local_indices = np.asarray([index for _, index in references], dtype=np.int64)
        return HistoricalBatch(
            source_networks=source_networks, local_indices=local_indices,
            **result,
        )

    def sample(self, batch_size, rng):
        batch_size = int(batch_size)
        if batch_size < 0 or batch_size > self.transition_count:
            raise ValueError('Archive sample exceeds visible population')
        ordinals = rng.sample(range(self.transition_count), batch_size)
        return self.batch_from_references(self.resolve_ordinals(ordinals))

    def source_statistics(self):
        return {
            'transition_count': self.transition_count,
            'count_by_network': {
                network: len(indices)
                for network, indices in self.indices_by_network.items()
            },
            'behavior_seed_rule': self.archive.manifest['behavior_seed_rule'],
            'visibility_digest': self.digest,
        }


class HistoricalArchive:
    def __init__(self, manifest_path, training_seed, verify_hashes=True):
        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest = read_json(self.manifest_path)
        if self.manifest.get('schema_version') != ARCHIVE_SCHEMA_VERSION:
            raise ValueError('Unsupported historical archive schema')
        expected_digest = self.manifest.get('archive_digest')
        unsigned = dict(self.manifest)
        unsigned.pop('archive_digest', None)
        if expected_digest != stable_digest(unsigned):
            raise ValueError('Historical archive manifest digest mismatch')
        self.training_seed = int(training_seed)
        self.datasets = {}
        for network, record in self.manifest['q1_manifests'].items():
            if sha256_file(record['path']) != record['sha256']:
                raise ValueError(f'Q1 manifest changed after archive creation: {network}')
            dataset = OfflineTrajectoryDataset(
                record['path'], seed=derive_rng_seed(training_seed, f'archive-load:{network}'),
                verify_hashes=verify_hashes,
            )
            if dataset.observation_dim != self.manifest['observation_dim']:
                raise ValueError(f'{network}: observation_dim mismatch')
            if dataset.action_dim != self.manifest['action_dim']:
                raise ValueError(f'{network}: action_dim mismatch')
            self.datasets[network] = dataset

    def _eligible_behavior_seeds(self):
        seeds = list(self.manifest['behavior_training_seeds'])
        rule = self.manifest['behavior_seed_rule']
        if rule.get('exclude_matching_training_seed'):
            seeds = [seed for seed in seeds if int(seed) != self.training_seed]
        if not seeds:
            raise ValueError('Behavior seed rule leaves the archive empty')
        return seeds

    def visibility(self, mode, ordered_networks, stage_index):
        mode = str(mode).upper()
        ordered_networks = tuple(ordered_networks)
        stage_index = int(stage_index)
        archive_networks = tuple(self.manifest['networks'])
        if mode not in ARCHIVE_MODES:
            raise ValueError(f'Unknown archive visibility mode: {mode}')
        if len(set(ordered_networks)) != len(ordered_networks):
            raise ValueError('Ordered networks contain duplicates')
        if set(ordered_networks) != set(archive_networks):
            raise ValueError('Ordered networks do not match the archive')
        if stage_index < 1 or stage_index > len(ordered_networks):
            raise ValueError('Stage index outside ordered networks')
        current = ordered_networks[stage_index - 1]
        future = ordered_networks[stage_index:]
        visible = ordered_networks[:stage_index - 1] if mode == 'P1C' else archive_networks
        masked = tuple(network for network in archive_networks if network not in visible)
        visibility = {
            'archive_mode': mode,
            'stage_index': stage_index,
            'ordered_networks': list(ordered_networks),
            'current_network': current,
            'visible_networks': list(visible),
            'masked_current_networks': [current] if current in masked else [],
            'masked_future_networks': [network for network in future if network in masked],
            'masked_networks': list(masked),
        }
        seeds = self._eligible_behavior_seeds()
        indices = {
            network: self.datasets[network].eligible_indices(
                network, behavior_seeds=seeds, episode_range=(1, 100),
            )
            for network in visible
        }
        visibility['behavior_training_seeds'] = seeds
        visibility['transition_count'] = sum(len(value) for value in indices.values())
        visibility['visibility_digest'] = stable_digest(visibility)
        view = VisibleArchiveView(self, indices, visibility)
        return view


def seeded_archive_rng(training_seed, stream_name='hoa_owp_sampler'):
    return random.Random(derive_rng_seed(training_seed, stream_name))
