"""Deterministic DHOA and OWP v1 construction for HA-SODQN."""

import collections
import math
import random
import time

import numpy as np

from .historical_archive import stable_digest


OWP_METHODS = ('DHOA', 'RAND', 'COV', 'CQ', 'CQA')
OWP_SCHEMA_VERSION = 1
OWP_DEFINITIONS = {
    'coverage_version': 'ha_coverage_v1',
    'coverage_descriptor': {
        'state_component': 'first 8 lane-count features',
        'fixed_lane_bins': [0, 1, 3, 7, 15, 31, 'inf'],
        'cell': [
            'source_network', 'behavior_training_seed', 'total_load_bin',
            'max_lane_bin', 'lane_imbalance_bin', 'phase_argmax', 'action',
        ],
        'allocation': 'lexicographically sorted cell round-robin',
        'tie_break': 'stable transition_id ascending',
    },
    'quality_version': 'ha_quality_v1',
    'quality_descriptor': {
        'score': '0.75 * episode_mean_reward_z + 0.25 * local_reward_z',
        'normalization': 'candidate-population mean/std, zero when std=0',
        'selection': 'coverage-cell allocation, score descending',
        'tie_break': 'stable transition_id ascending',
    },
    'alignment_version': 'ha_alignment_v1',
    'alignment_descriptor': {
        'candidate': 'mean of first 8 lane-count state features',
        'online_window': 'mean of first 8 lane-count features from warm-up',
        'distance': 'euclidean after candidate-population feature z-score',
        'global_fraction': 0.75,
        'alignment_fraction': 0.25,
        'tie_break': 'stable transition_id ascending',
    },
}


def _bin(values):
    return np.digitize(values, np.asarray([0, 1, 3, 7, 15, 31]), right=False)


def _candidate_table(view):
    columns = collections.defaultdict(list)
    for network in view.networks:
        indices = view.indices_by_network[network]
        arrays = view.archive.datasets[network]._arrays[network]
        states = arrays['observations'][indices, :8]
        phases = np.argmax(arrays['observations'][indices, 8:], axis=1)
        total_bin = _bin(np.sum(states, axis=1))
        maximum_bin = _bin(np.max(states, axis=1))
        imbalance_bin = _bin(np.max(states, axis=1) - np.min(states, axis=1))
        for offset, index in enumerate(indices):
            columns['network'].append(network)
            columns['index'].append(int(index))
            columns['seed'].append(int(arrays['behavior_training_seed'][index]))
            columns['episode'].append(int(arrays['episode_id'][index]))
            columns['action'].append(int(arrays['actions'][index]))
            columns['reward'].append(float(arrays['rewards'][index]))
            columns['transition_id'].append(str(arrays['transition_id'][index]))
            columns['coverage_key'].append((
                network, int(arrays['behavior_training_seed'][index]),
                int(total_bin[offset]), int(maximum_bin[offset]),
                int(imbalance_bin[offset]), int(phases[offset]),
                int(arrays['actions'][index]),
            ))
            columns['alignment_feature'].append(states[offset].astype(np.float64))
    result = {key: np.asarray(value) for key, value in columns.items()}
    if len(result.get('index', ())) != len(view):
        raise RuntimeError('OWP candidate table does not match visible archive')
    return result


def _quality_scores(table):
    rewards = table['reward'].astype(np.float64)
    sums = collections.defaultdict(float)
    counts = collections.defaultdict(int)
    for network, seed, episode, reward in zip(
        table['network'], table['seed'], table['episode'], rewards,
    ):
        key = (str(network), int(seed), int(episode))
        sums[key] += float(reward)
        counts[key] += 1
    episode_means = np.asarray([
        sums[(str(network), int(seed), int(episode))]
        / counts[(str(network), int(seed), int(episode))]
        for network, seed, episode in zip(
            table['network'], table['seed'], table['episode'],
        )
    ])

    def zscore(values):
        std = float(np.std(values))
        return np.zeros_like(values) if std == 0 else (values - np.mean(values)) / std

    return 0.75 * zscore(episode_means) + 0.25 * zscore(rewards)


def _round_robin(table, capacity, quality=None, excluded=()):
    excluded = set(int(value) for value in excluded)
    groups = collections.defaultdict(list)
    for row, key in enumerate(table['coverage_key']):
        if row not in excluded:
            groups[tuple(key)].append(row)
    for key, rows in groups.items():
        if quality is None:
            rows.sort(key=lambda row: str(table['transition_id'][row]))
        else:
            rows.sort(key=lambda row: (
                -float(quality[row]), str(table['transition_id'][row]),
            ))
    queues = [(key, collections.deque(groups[key])) for key in sorted(groups)]
    selected = []
    while queues and len(selected) < capacity:
        active = []
        for key, queue in queues:
            if queue and len(selected) < capacity:
                selected.append(queue.popleft())
            if queue:
                active.append((key, queue))
        queues = active
    return selected


class HistoricalSampler:
    """Sampler with a frozen candidate identity and private RNG state."""

    def __init__(self, view, method, references, manifest, rng=None):
        self.view = view
        self.method = method
        self.references = None if references is None else tuple(references)
        self.manifest = dict(manifest)
        self.rng = rng or random.Random()
        self.sample_count = 0
        self.reuse_counts = collections.Counter()

    def __len__(self):
        return len(self.view) if self.references is None else len(self.references)

    def sample(self, count):
        count = int(count)
        population = len(self)
        if count < 0 or count > population:
            raise ValueError('Offline quota exceeds historical population')
        if self.references is None:
            batch = self.view.sample(count, self.rng)
        else:
            positions = self.rng.sample(range(population), count)
            batch = self.view.batch_from_references([
                self.references[position] for position in positions
            ])
        self.sample_count += count
        self.reuse_counts.update(str(value) for value in batch.transition_ids)
        return batch

    def state_dict(self):
        return {
            'schema_version': OWP_SCHEMA_VERSION,
            'method': self.method,
            'manifest_digest': self.manifest['owp_digest'],
            'references': self.references,
            'rng_state': self.rng.getstate(),
            'sample_count': self.sample_count,
            'reuse_counts': dict(self.reuse_counts),
        }

    def load_state_dict(self, state):
        if state.get('schema_version') != OWP_SCHEMA_VERSION:
            raise ValueError('Unsupported historical sampler state')
        if state.get('method') != self.method:
            raise ValueError('Historical sampler method changed on resume')
        if state.get('manifest_digest') != self.manifest['owp_digest']:
            raise ValueError('Historical working-pool identity changed on resume')
        restored = state.get('references')
        if restored is not None and tuple(tuple(value) for value in restored) != self.references:
            raise ValueError('Historical working-pool references changed on resume')
        self.rng.setstate(state['rng_state'])
        self.sample_count = int(state['sample_count'])
        self.reuse_counts = collections.Counter(state.get('reuse_counts', {}))


def build_historical_sampler(view, method, capacity, rng, alignment_observations=None):
    started = time.perf_counter()
    method = str(method).upper()
    capacity = int(capacity)
    if method not in OWP_METHODS:
        raise ValueError(f'Unknown historical archive method: {method}')
    if capacity <= 0:
        raise ValueError('OWP capacity must be positive')
    if not len(view):
        raise ValueError('Cannot build an offline sampler from an empty archive')
    references = None
    table = None
    selected = None
    quality = None
    alignment_distances = None
    if method == 'DHOA':
        selected_count = len(view)
    else:
        table = _candidate_table(view)
        selected_count = min(capacity, len(view))
        if method == 'RAND':
            selected = rng.sample(range(len(view)), selected_count)
            references = view.resolve_ordinals(selected)
        else:
            quality = _quality_scores(table)
            if method == 'COV':
                selected = _round_robin(table, selected_count)
            elif method == 'CQ':
                selected = _round_robin(table, selected_count, quality=quality)
            else:
                if alignment_observations is None or not len(alignment_observations):
                    raise ValueError('CQA requires warm-up alignment observations')
                global_count = int(math.floor(selected_count * 0.75))
                selected = _round_robin(table, global_count, quality=quality)
                features = np.stack(table['alignment_feature']).astype(np.float64)
                online = np.asarray(alignment_observations, dtype=np.float64).reshape(-1, 16)[:, :8]
                mean = np.mean(features, axis=0)
                std = np.std(features, axis=0)
                std[std == 0] = 1.0
                target = (np.mean(online, axis=0) - mean) / std
                alignment_distances = np.linalg.norm((features - mean) / std - target, axis=1)
                selected_set = set(selected)
                remaining = [
                    row for row in range(len(view)) if row not in selected_set
                ]
                remaining.sort(key=lambda row: (
                    float(alignment_distances[row]), str(table['transition_id'][row]),
                ))
                selected.extend(remaining[:selected_count - len(selected)])
            references = [
                (str(table['network'][row]), int(table['index'][row]))
                for row in selected
            ]
    selected_ids = (
        [] if references is None else
        [str(value) for value in view.batch_from_references(references).transition_ids]
    )
    composition = collections.Counter(
        network for network, _ in references
    ) if references is not None else collections.Counter(
        view.source_statistics()['count_by_network']
    )
    manifest = {
        'schema_version': OWP_SCHEMA_VERSION,
        'method': method,
        'stage_visibility_digest': view.digest,
        'candidate_count': len(view),
        'configured_capacity': capacity,
        'selected_count': selected_count,
        'fixed_working_pool': method != 'DHOA',
        'source_composition': dict(sorted(composition.items())),
        'unique_transition_ids_digest': stable_digest(sorted(selected_ids)) if selected_ids else None,
        'definitions': OWP_DEFINITIONS,
        'quality_summary': None if quality is None else {
            'mean': float(np.mean(quality)), 'std': float(np.std(quality)),
            'min': float(np.min(quality)), 'max': float(np.max(quality)),
        },
        'alignment_summary': None if alignment_distances is None else {
            'mean': float(np.mean(alignment_distances)),
            'min': float(np.min(alignment_distances)),
            'max': float(np.max(alignment_distances)),
        },
        'build_seconds': time.perf_counter() - started,
    }
    digest_payload = dict(manifest)
    digest_payload.pop('build_seconds')
    manifest['owp_digest'] = stable_digest(digest_payload)
    return HistoricalSampler(view, method, references, manifest, rng=rng)
