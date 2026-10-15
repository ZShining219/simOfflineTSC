"""Self-produced historical archive (archive_mode='SELF') for HA-SODQN.

Instead of a precollected Plan-1/Plan-2 trajectory archive, the run's own
transitions from stages 1..k-1 form the offline leg's data source.  Records
are ReplayRecord objects already written to the online replay buffer; the
archive keeps append-only references so past-stage data survives ORB
eviction.  Visibility mirrors P1C at the *stage* level: at stage k only
records with ``metadata.stage_index < k`` are visible (revisit stages
naturally accumulate both occurrences).

Supported methods: DHOA (dynamic uniform sample from visible records),
RAND (fixed random pool), COV (coverage round-robin), CQ (reward-quality
anchored coverage), CQA (CQ global fraction + alignment fill).  Quality and
coverage statistics reuse ``owp.py`` scoring/cell definitions so the SELF
legs are directly comparable to their P1C counterparts.
"""

import collections
import random

import numpy as np

from .historical_archive import (
    HistoricalBatch, derive_rng_seed, stable_digest)
from .owp import OWP_METHODS, _bin, _quality_scores, _round_robin

SELF_SCHEMA_VERSION = 1
SELF_METHODS = OWP_METHODS
# Per-stage reservoir cap: keeps each stage's archive contribution a uniform
# subsample so committed checkpoints stay bounded (full archives would add
# tens of MB per episode checkpoint).
SELF_STAGE_CAPACITY = 2500


def _phase_index(record):
    return int(np.asarray(record.payload.phase).reshape(-1)[0])


def _record_observation(record, action_dim):
    """ReplayRecord -> dataset-style 16-dim observation (state + phase 1hot)."""
    state = np.asarray(record.payload.state, dtype=np.float32).reshape(-1)
    phase = np.zeros(action_dim, dtype=np.float32)
    phase[_phase_index(record)] = 1.0
    return np.concatenate((state, phase))


def _record_next_observation(record, action_dim):
    state = np.asarray(record.payload.next_state, dtype=np.float32).reshape(-1)
    phase = np.zeros(action_dim, dtype=np.float32)
    phase[int(np.asarray(record.payload.next_phase).reshape(-1)[0])] = 1.0
    return np.concatenate((state, phase))


def batch_from_records(records, positions=None, run_id='self_archive',
                       behavior_seed=0):
    """Build a HistoricalBatch-shaped bundle from own ReplayRecord list."""
    action_dim = 8  # hz1x1 phase space; consistent with one-hot blocks
    observations = np.stack([
        _record_observation(record, action_dim) for record in records
    ])
    next_observations = np.stack([
        _record_next_observation(record, action_dim) for record in records
    ])
    n = len(records)
    positions = range(n) if positions is None else positions
    return HistoricalBatch(
        observations=observations,
        actions=np.asarray([
            int(np.asarray(record.payload.action).reshape(-1)[0])
            for record in records
        ], dtype=np.int64),
        rewards=np.asarray([
            float(np.asarray(record.payload.reward).reshape(-1)[0])
            for record in records
        ], dtype=np.float64),
        next_observations=next_observations,
        terminated=np.asarray(
            [bool(record.payload.terminated) for record in records]),
        truncated=np.asarray(
            [bool(record.payload.truncated) for record in records]),
        source_networks=np.asarray(
            [record.metadata.source_network for record in records],
            dtype=object),
        local_indices=np.asarray(list(positions), dtype=np.int64),
        transition_ids=np.asarray(
            [record.metadata.transition_id for record in records],
            dtype=object),
        behavior_training_seeds=np.asarray([behavior_seed] * n, dtype=np.int64),
        episode_ids=np.asarray(
            [record.metadata.local_episode for record in records],
            dtype=np.int64),
        decision_steps=np.asarray(
            [record.metadata.decision_index for record in records],
            dtype=np.int64),
        run_ids=np.asarray([run_id] * n, dtype=object),
        shard_sha256=np.asarray(['self_archive'] * n, dtype=object),
    )


class SelfArchive:
    """Per-stage reservoir store of the run's own transitions.

    Records are ReplayRecord objects shared with the ORB.  Each stage keeps
    at most ``stage_capacity`` records via seeded reservoir sampling — a
    uniform subsample that bounds checkpoint size while preserving the
    stage's transition distribution.  The reservoir RNG rides inside
    state_dict so resume reproduces identical membership.
    """

    def __init__(self, training_seed, run_id='self_archive',
                 stage_capacity=SELF_STAGE_CAPACITY):
        self.training_seed = int(training_seed)
        self.run_id = str(run_id)
        self.stage_capacity = int(stage_capacity)
        self.records = []
        self.seen_per_stage = collections.Counter()
        self.reservoir_rng = random.Random(
            derive_rng_seed(training_seed, 'self_archive_reservoir'))

    def append(self, record):
        stage = int(record.metadata.stage_index)
        self.seen_per_stage[stage] += 1
        seen = self.seen_per_stage[stage]
        if seen <= self.stage_capacity:
            self.records.append(record)
            return
        slot = self.reservoir_rng.randrange(seen)
        if slot < self.stage_capacity:
            # replace the slot-th record belonging to this stage
            count = -1
            for index, existing in enumerate(self.records):
                if int(existing.metadata.stage_index) == stage:
                    count += 1
                    if count == slot:
                        self.records[index] = record
                        return

    def view(self, stage_index):
        stage_index = int(stage_index)
        visible = [
            record for record in self.records
            if int(record.metadata.stage_index) < stage_index
        ]
        by_network = collections.Counter(
            record.metadata.source_network for record in visible)
        by_stage = collections.Counter(
            int(record.metadata.stage_index) for record in visible)
        visibility = {
            'archive_mode': 'SELF',
            'stage_index': stage_index,
            'visible_records': len(visible),
            'count_by_network': dict(sorted(by_network.items())),
            'count_by_stage': {
                str(key): by_stage[key] for key in sorted(by_stage)},
            'producer': 'self',
            'reservoir_stage_capacity': self.stage_capacity,
            'seen_per_stage': {
                str(key): int(self.seen_per_stage[key])
                for key in sorted(self.seen_per_stage)
                if int(key) < stage_index},
        }
        visibility['visibility_digest'] = stable_digest({
            key: value for key, value in visibility.items()
            if key != 'visibility_digest'
        })
        return SelfArchiveView(self, visible, visibility)

    def state_dict(self):
        return {
            'schema_version': SELF_SCHEMA_VERSION,
            'records': list(self.records),
            'seen_per_stage': dict(self.seen_per_stage),
            'reservoir_rng_state': self.reservoir_rng.getstate(),
            'stage_capacity': self.stage_capacity,
        }

    @classmethod
    def from_state_dict(cls, state, training_seed, run_id='self_archive'):
        archive = cls(training_seed, run_id=run_id,
                      stage_capacity=int(state.get(
                          'stage_capacity', SELF_STAGE_CAPACITY)))
        archive.records = list(state['records'])
        archive.seen_per_stage = collections.Counter(
            {int(key): int(value)
             for key, value in state.get('seen_per_stage', {}).items()})
        if state.get('reservoir_rng_state') is not None:
            archive.reservoir_rng.setstate(state['reservoir_rng_state'])
        return archive


class SelfArchiveView:
    """Fixed snapshot of the records visible at one stage boundary."""

    def __init__(self, archive, records, visibility):
        self.archive = archive
        self.records = list(records)
        self.visibility = dict(visibility)

    def __len__(self):
        return len(self.records)

    @property
    def digest(self):
        return self.visibility['visibility_digest']

    def sample(self, count, rng):
        return batch_from_records(
            rng.sample(self.records, int(count)),
            run_id=self.archive.run_id,
            behavior_seed=self.archive.training_seed,
        )

    def batch_from_positions(self, positions):
        records = [self.records[int(position)] for position in positions]
        return batch_from_records(
            records, positions=positions,
            run_id=self.archive.run_id,
            behavior_seed=self.archive.training_seed,
        )

    def source_statistics(self):
        return {'count_by_network': dict(
            self.visibility.get('count_by_network', {}))}

    def candidate_table(self):
        """Mirror owp._candidate_table over own records (same cell keys)."""
        columns = collections.defaultdict(list)
        observations = np.stack([
            _record_observation(record, 8) for record in self.records
        ]).astype(np.float64)
        states = observations[:, :8]
        phases = np.argmax(observations[:, 8:], axis=1)
        total_bin = _bin(np.sum(states, axis=1))
        maximum_bin = _bin(np.max(states, axis=1))
        imbalance_bin = _bin(np.max(states, axis=1) - np.min(states, axis=1))
        for offset, record in enumerate(self.records):
            meta = record.metadata
            # stage-qualified pseudo episode keeps (net, seed, episode) keys
            # unique across stages while local_episode restarts each stage.
            episode_key = int(meta.stage_index) * 100000 + int(
                meta.local_episode)
            columns['network'].append(meta.source_network)
            columns['index'].append(offset)
            columns['seed'].append(self.archive.training_seed)
            columns['episode'].append(episode_key)
            columns['action'].append(
                int(np.asarray(record.payload.action).reshape(-1)[0]))
            columns['reward'].append(
                float(np.asarray(record.payload.reward).reshape(-1)[0]))
            columns['transition_id'].append(meta.transition_id)
            columns['coverage_key'].append((
                meta.source_network, self.archive.training_seed, episode_key,
                int(total_bin[offset]), int(maximum_bin[offset]),
                int(imbalance_bin[offset]), int(phases[offset]),
                columns['action'][-1],
            ))
            columns['alignment_feature'].append(states[offset])
        return {key: np.asarray(value) for key, value in columns.items()}


class SelfHistoricalSampler:
    """HistoricalSampler-compatible wrapper over a SelfArchiveView."""

    def __init__(self, view, method, positions=None, rng=None, manifest=None):
        self.view = view
        self.method = method
        self.positions = None if positions is None else tuple(positions)
        self.manifest = dict(manifest or {})
        self.rng = rng or random.Random()
        self.sample_count = 0
        self.reuse_counts = collections.Counter()

    def __len__(self):
        return len(self.view) if self.positions is None else len(
            self.positions)

    def sample(self, count):
        count = int(count)
        population = len(self)
        if count <= 0 or population == 0:
            raise ValueError('Offline quota exceeds self-archive population')
        # Unlike the external archive, an early self-archive can momentarily
        # be smaller than the offline quota; clamp instead of failing.
        count = min(count, population)
        if self.positions is None:
            batch = self.view.sample(count, self.rng)
        else:
            drawn = self.rng.sample(range(population), count)
            batch = self.view.batch_from_positions(
                [self.positions[position] for position in drawn])
        self.sample_count += count
        self.reuse_counts.update(str(v) for v in batch.transition_ids)
        return batch

    def state_dict(self):
        return {
            'schema_version': SELF_SCHEMA_VERSION,
            'method': self.method,
            'positions': None if self.positions is None else list(
                self.positions),
            'sample_count': self.sample_count,
            'rng_state': self.rng.getstate(),
            'manifest': self.manifest,
        }

    def load_state_dict(self, state):
        self.sample_count = int(state.get('sample_count', 0))
        if state.get('rng_state') is not None:
            self.rng.setstate(state['rng_state'])


def restore_self_sampler(view, method, state, rng):
    """Resume a self sampler with its recorded fixed pool (no redraw)."""
    sampler = SelfHistoricalSampler(
        view, method, positions=state.get('positions'), rng=rng,
        manifest=state.get('manifest'))
    sampler.load_state_dict(state)
    return sampler


def build_self_sampler(view, method, capacity, rng,
                       alignment_observations=None):
    """OWP-equivalent construction over a SelfArchiveView."""
    method = str(method).upper()
    if method not in OWP_METHODS:
        raise ValueError(f'Unknown self-archive method: {method}')
    if not len(view):
        return None
    if method == 'DHOA':
        return SelfHistoricalSampler(view, method, rng=rng)
    table = view.candidate_table()
    capacity = min(int(capacity), len(view))
    positions = None
    if method == 'RAND':
        positions = rng.sample(range(len(view)), capacity)
    else:
        quality = _quality_scores(table)
        if method == 'COV':
            positions = _round_robin(table, capacity)
        elif method == 'CQ':
            positions = _round_robin(table, capacity, quality=quality)
        else:  # CQA
            if alignment_observations is None or not len(
                    alignment_observations):
                return None
            global_count = int(capacity * 0.75)
            positions = list(
                _round_robin(table, global_count, quality=quality))
            features = np.stack(table['alignment_feature']).astype(np.float64)
            online = np.asarray(
                alignment_observations, dtype=np.float64).reshape(-1, 16)[:, :8]
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std[std == 0] = 1.0
            target = (np.mean(online, axis=0) - mean) / std
            distances = np.linalg.norm(
                (features - mean) / std - target, axis=1)
            selected = set(positions)
            remaining = [
                row for row in range(len(view)) if row not in selected]
            remaining.sort(key=lambda row: (
                float(distances[row]), str(table['transition_id'][row])))
            positions.extend(remaining[:capacity - len(positions)])
    manifest = {
        'schema_version': SELF_SCHEMA_VERSION,
        'method': method,
        'visibility_digest': view.digest,
        'candidate_count': len(view),
        'capacity': capacity,
        'selected_count': len(positions) if positions is not None else len(
            view),
        'producer': 'self',
    }
    return SelfHistoricalSampler(
        view, method, positions=positions, rng=rng, manifest=manifest)
