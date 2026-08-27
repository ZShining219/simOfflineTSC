"""Traceable replay and historical-pool primitives for shared DQN."""

from __future__ import annotations

import dataclasses
import json
import random
from collections import Counter, OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class TransitionMetadata:
    transition_id: str
    scene_id: str
    intersection_id: str
    episode_id: int
    decision_step: int
    source: str
    training_stage: int
    policy_version: int
    collector_training_seed: int | None = None


@dataclasses.dataclass(frozen=True)
class LocalTransition:
    state: np.ndarray
    phase: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_phase: np.ndarray
    terminated: bool
    truncated: bool
    metadata: TransitionMetadata


class TraceableReplayBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError('Replay capacity must be positive')
        self.capacity = int(capacity)
        self.records = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.records)

    def clear(self):
        self.records.clear()

    def append(self, transition: LocalTransition):
        if not isinstance(transition, LocalTransition):
            raise TypeError('Replay accepts LocalTransition records only')
        self.records.append(transition)

    def sample(self, count: int, rng=random):
        if count < 0 or count > len(self.records):
            raise ValueError('Invalid replay sample count')
        return rng.sample(list(self.records), count)

    def composition(self):
        scenes = Counter(x.metadata.scene_id for x in self.records)
        intersections = Counter(x.metadata.intersection_id for x in self.records)
        return {
            'size': len(self.records),
            'capacity': self.capacity,
            'scene_distribution': dict(sorted(scenes.items())),
            'intersection_distribution': dict(sorted(intersections.items())),
        }

    def state_dict(self):
        return {'capacity': self.capacity, 'records': list(self.records)}

    @classmethod
    def from_state_dict(cls, state):
        replay = cls(int(state['capacity']))
        replay.records.extend(state['records'])
        return replay


def split_local_transitions(intersection_ids, state, phase, actions, rewards,
                            next_state, next_phase, terminated, truncated,
                            scene_id, episode_id, decision_step,
                            training_stage=0, policy_version=0,
                            source='online_current', collector_training_seed=None):
    values = [state, phase, actions, rewards, next_state, next_phase]
    if any(len(value) != len(intersection_ids) for value in values):
        raise ValueError('Local transition fields must match intersection count')
    records = []
    for index, intersection_id in enumerate(intersection_ids):
        metadata = TransitionMetadata(
            transition_id=(f'{scene_id}:seed{collector_training_seed}:'
                           f'{episode_id}:{decision_step}:{intersection_id}'),
            scene_id=str(scene_id), intersection_id=str(intersection_id),
            episode_id=int(episode_id), decision_step=int(decision_step),
            source=str(source), training_stage=int(training_stage),
            policy_version=int(policy_version),
            collector_training_seed=(
                None if collector_training_seed is None
                else int(collector_training_seed)))
        records.append(LocalTransition(
            state=np.array(state[index], dtype=np.float32, copy=True),
            phase=np.asarray(phase[index]).reshape(1).astype(np.int64),
            action=int(np.asarray(actions[index]).item()),
            reward=float(np.asarray(rewards[index]).item()),
            next_state=np.array(next_state[index], dtype=np.float32, copy=True),
            next_phase=np.asarray(next_phase[index]).reshape(1).astype(np.int64),
            terminated=bool(terminated), truncated=bool(truncated),
            metadata=metadata))
    return records


class HistoricalPool:
    """Immutable-by-convention archive with explicit scene visibility."""

    STRATEGIES = {'global_uniform', 'scene_intersection_balanced'}

    def __init__(self, records: Iterable[LocalTransition] = ()):
        self.records = tuple(records)
        self._index = defaultdict(list)
        for record in self.records:
            if record.metadata.source != 'offline_history':
                raise ValueError('Historical records must use offline_history source')
            self._index[(record.metadata.scene_id,
                         record.metadata.intersection_id)].append(record)

    @property
    def scenes(self):
        return tuple(sorted({key[0] for key in self._index}))

    @property
    def intersection_ids(self):
        return tuple(sorted({key[1] for key in self._index}))

    @staticmethod
    def visible_scenes(scene_order: Sequence[str], stage_index: int, mode: str):
        if mode not in {'causal', 'full'}:
            raise ValueError(f'Unknown history access mode: {mode}')
        if stage_index < 0 or stage_index >= len(scene_order):
            raise ValueError('stage_index is outside scene_order')
        return tuple(scene_order[:stage_index]) if mode == 'causal' else tuple(scene_order)

    def sample(self, count: int, visible_scenes: Sequence[str],
               strategy='scene_intersection_balanced', rng=random):
        if strategy not in self.STRATEGIES:
            raise ValueError(f'Unknown offline sampling strategy: {strategy}')
        if count < 0:
            raise ValueError('Sample count cannot be negative')
        if count == 0:
            return []
        allowed = set(visible_scenes)
        if strategy == 'global_uniform':
            population = [x for x in self.records if x.metadata.scene_id in allowed]
            if not population:
                raise ValueError('No visible offline transitions')
            return [rng.choice(population) for _ in range(count)]
        keys = sorted(key for key, values in self._index.items()
                      if key[0] in allowed and values)
        if not keys:
            raise ValueError('No visible offline scene/intersection pools')
        # Each draw first chooses a scene, then an intersection, then a record.
        scene_ids = sorted({key[0] for key in keys})
        by_scene = {scene: sorted(key[1] for key in keys if key[0] == scene)
                    for scene in scene_ids}
        result = []
        for _ in range(count):
            scene = rng.choice(scene_ids)
            intersection = rng.choice(by_scene[scene])
            result.append(rng.choice(self._index[(scene, intersection)]))
        return result

    def save(self, directory, manifest):
        import torch
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        required = {
            'roadnet_id', 'scene_id', 'intersection_ids', 'state_schema',
            'action_schema', 'reward_schema', 'num_episodes',
            'num_decision_steps', 'num_transitions', 'training_seed',
            'source_policy', 'collection_stage', 'created_at',
        }
        missing = sorted(required - set(manifest))
        if missing:
            raise ValueError(f'History manifest missing fields: {missing}')
        if int(manifest['num_transitions']) != len(self.records):
            raise ValueError('History manifest transition count mismatch')
        torch.save(self.records, directory / 'transitions.pt')
        with (directory / 'manifest.json').open('w', encoding='utf-8') as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')

    @classmethod
    def load_many(cls, directories):
        return LazyHistoricalPool(directories)

    @classmethod
    def load_sampling_index(cls, manifest_path, directories):
        return MappedHistoricalPool(manifest_path, directories)


class _LazyRecordCount:
    """Compatibility view for callers that only inspect pool availability."""

    def __init__(self, count):
        self.count = int(count)

    def __len__(self):
        return self.count

    def __bool__(self):
        return self.count > 0


class LazyHistoricalPool:
    """Read-only shard sampler that avoids materializing the frozen HOA.

    Formal arterial archives contain one balanced six-intersection shard per
    episode.  Sampling loads only the selected shard and keeps a very small LRU
    cache.  The same scene/intersection selection distribution as
    :class:`HistoricalPool` is retained.
    """

    STRATEGIES = HistoricalPool.STRATEGIES

    def __init__(self, directories, cache_size=2):
        self.manifests = []
        self._shards = []
        self._cache = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        total = 0
        for directory in directories:
            directory = Path(directory).resolve()
            with (directory / 'manifest.json').open(encoding='utf-8') as handle:
                manifest = json.load(handle)
            files = manifest.get('transition_files', ['transitions.pt'])
            if not files:
                raise ValueError(f'History archive has no transition files: {directory}')
            transition_count = int(manifest['num_transitions'])
            if transition_count <= 0:
                raise ValueError(f'History archive is empty: {directory}')
            per_file, remainder = divmod(transition_count, len(files))
            for index, name in enumerate(files):
                path = directory / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                self._shards.append({
                    'path': path, 'scene_id': str(manifest['scene_id']),
                    'count': per_file + int(index < remainder),
                    'intersection_ids': tuple(manifest['intersection_ids']),
                })
            total += transition_count
            self.manifests.append(manifest)
        self.manifests = tuple(self.manifests)
        self.records = _LazyRecordCount(total)
        self._scenes = tuple(sorted({item['scene_id'] for item in self._shards}))
        self._intersection_ids = tuple(sorted({
            intersection for item in self._shards
            for intersection in item['intersection_ids']
        }))

    @property
    def scenes(self):
        return self._scenes

    @property
    def intersection_ids(self):
        return self._intersection_ids

    @staticmethod
    def visible_scenes(scene_order, stage_index, mode):
        return HistoricalPool.visible_scenes(scene_order, stage_index, mode)

    def _load(self, shard):
        import torch

        key = str(shard['path'])
        if key in self._cache:
            records = self._cache.pop(key)
            self._cache[key] = records
            return records
        records = tuple(torch.load(shard['path'], weights_only=False))
        if len(records) != shard['count']:
            raise ValueError(f"History shard count mismatch: {shard['path']}")
        if any(record.metadata.source != 'offline_history' for record in records):
            raise ValueError(f"History shard has non-HOA records: {shard['path']}")
        self._cache[key] = records
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return records

    @staticmethod
    def _weighted_choice(items, rng):
        total = sum(int(item['count']) for item in items)
        if total <= 0:
            raise ValueError('No visible offline transitions')
        target = rng.randrange(total)
        for item in items:
            target -= int(item['count'])
            if target < 0:
                return item
        raise AssertionError('Weighted shard selection failed')

    def _sample_key(self, scene, intersection, rng):
        candidates = [item for item in self._shards
                      if item['scene_id'] == scene
                      and intersection in item['intersection_ids']]
        if not candidates:
            raise ValueError('No visible offline scene/intersection pools')
        # Formal shards are balanced and the first draw succeeds.  Trying each
        # candidate also supports compact compatibility archives whose files
        # may each contain only a subset of declared intersections.
        remaining = list(candidates)
        while remaining:
            shard = rng.choice(remaining)
            records = [record for record in self._load(shard)
                       if (record.metadata.scene_id == scene and
                           record.metadata.intersection_id == intersection)]
            if records:
                return rng.choice(records)
            remaining.remove(shard)
        raise ValueError('No records for visible offline scene/intersection')

    def sample(self, count, visible_scenes,
               strategy='scene_intersection_balanced', rng=random):
        if strategy not in self.STRATEGIES:
            raise ValueError(f'Unknown offline sampling strategy: {strategy}')
        if count < 0:
            raise ValueError('Sample count cannot be negative')
        if count == 0:
            return []
        allowed = set(visible_scenes)
        visible = [item for item in self._shards if item['scene_id'] in allowed]
        if not visible:
            raise ValueError('No visible offline transitions')
        if strategy == 'global_uniform':
            result = []
            for _ in range(count):
                shard = self._weighted_choice(visible, rng)
                result.append(rng.choice(self._load(shard)))
            return result
        scenes = sorted({item['scene_id'] for item in visible})
        intersections = {scene: sorted({
            intersection for item in visible if item['scene_id'] == scene
            for intersection in item['intersection_ids']
        }) for scene in scenes}
        return [self._sample_key(
            scene := rng.choice(scenes), rng.choice(intersections[scene]), rng
        ) for _ in range(count)]


class MappedHistoricalPool:
    """Sample audited HOA transitions from read-only NumPy memmaps."""

    STRATEGIES = HistoricalPool.STRATEGIES

    def __init__(self, manifest_path, directories):
        manifest_path = Path(manifest_path).resolve()
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        if payload.get('status') != 'completed':
            raise ValueError('HOA sampling index is not completed')
        self.source_hoa_archive_hash = payload.get('source_hoa_archive_hash')
        self.manifests = []
        available_scenes = set()
        for directory in directories:
            with (Path(directory) / 'manifest.json').open(encoding='utf-8') as handle:
                source = json.load(handle)
            self.manifests.append(source)
            available_scenes.add(str(source['scene_id']))
        self.manifests = tuple(self.manifests)
        self._arrays = {}
        total = 0
        for item in payload.get('pools', []):
            scene = str(item['scene_id'])
            if scene not in available_scenes:
                continue
            intersection = str(item['intersection_id'])
            path = (manifest_path.parent / item['file']).resolve()
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if len(array) != int(item['count']):
                raise ValueError(f'HOA sampling pool count mismatch: {path}')
            self._arrays[(scene, intersection)] = array
            total += len(array)
        expected_scenes = available_scenes
        actual_scenes = {key[0] for key in self._arrays}
        if actual_scenes != expected_scenes:
            raise ValueError('HOA sampling index does not cover visible archives')
        self.records = _LazyRecordCount(total)

    @property
    def scenes(self):
        return tuple(sorted({key[0] for key in self._arrays}))

    @property
    def intersection_ids(self):
        return tuple(sorted({key[1] for key in self._arrays}))

    @staticmethod
    def visible_scenes(scene_order, stage_index, mode):
        return HistoricalPool.visible_scenes(scene_order, stage_index, mode)

    @staticmethod
    def _transition(row, scene, intersection):
        collector_seed = int(row['collector_seed'])
        episode = int(row['episode_id'])
        decision = int(row['decision_step'])
        metadata = TransitionMetadata(
            transition_id=(f'{scene}:seed{collector_seed}:'
                           f'{episode}:{decision}:{intersection}'),
            scene_id=scene, intersection_id=intersection,
            episode_id=episode, decision_step=decision,
            source='offline_history',
            training_stage=int(row['training_stage']),
            policy_version=int(row['policy_version']),
            collector_training_seed=collector_seed,
        )
        return LocalTransition(
            state=np.array(row['state'], dtype=np.float32, copy=True),
            phase=np.asarray([row['phase']], dtype=np.int64),
            action=int(row['action']), reward=float(row['reward']),
            next_state=np.array(row['next_state'], dtype=np.float32, copy=True),
            next_phase=np.asarray([row['next_phase']], dtype=np.int64),
            terminated=bool(row['terminated']), truncated=bool(row['truncated']),
            metadata=metadata,
        )

    def sample(self, count, visible_scenes,
               strategy='scene_intersection_balanced', rng=random):
        if strategy not in self.STRATEGIES:
            raise ValueError(f'Unknown offline sampling strategy: {strategy}')
        if count < 0:
            raise ValueError('Sample count cannot be negative')
        if count == 0:
            return []
        allowed = set(visible_scenes)
        keys = sorted(key for key, array in self._arrays.items()
                      if key[0] in allowed and len(array))
        if not keys:
            raise ValueError('No visible offline transitions')
        result = []
        if strategy == 'global_uniform':
            sizes = [len(self._arrays[key]) for key in keys]
            total = sum(sizes)
            for _ in range(count):
                offset = rng.randrange(total)
                for key, size in zip(keys, sizes):
                    if offset < size:
                        row = self._arrays[key][offset]
                        result.append(self._transition(row, *key))
                        break
                    offset -= size
            return result
        scenes = sorted({key[0] for key in keys})
        intersections = {scene: sorted(key[1] for key in keys if key[0] == scene)
                         for scene in scenes}
        for _ in range(count):
            scene = rng.choice(scenes)
            intersection = rng.choice(intersections[scene])
            array = self._arrays[(scene, intersection)]
            result.append(self._transition(
                array[rng.randrange(len(array))], scene, intersection))
        return result


class HistoryArchiveWriter:
    """Write bounded per-episode chunks and finalize a traceable manifest."""

    def __init__(self, directory, resume_episode=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.files = []
        self.num_transitions = 0
        if resume_episode is not None:
            import torch
            expected = [f'episode_{episode:04d}.pt'
                        for episode in range(1, int(resume_episode) + 1)]
            existing = sorted(path.name for path in self.directory.glob('episode_*.pt'))
            if existing != expected:
                raise ValueError(
                    'Resume history shards do not exactly match checkpoint episode')
            self.files = existing
            self.num_transitions = sum(
                len(torch.load(self.directory / name, weights_only=False))
                for name in existing)

    def append_episode(self, episode_id, records):
        import torch
        records = tuple(records)
        if not records:
            return None
        name = f'episode_{int(episode_id):04d}.pt'
        if (self.directory / name).exists():
            raise FileExistsError(f'History shard already exists: {name}')
        torch.save(records, self.directory / name)
        self.files.append(name)
        self.num_transitions += len(records)
        return name

    def finalize(self, manifest):
        manifest = dict(manifest)
        manifest['transition_files'] = list(self.files)
        manifest['num_transitions'] = self.num_transitions
        required = {
            'roadnet_id', 'scene_id', 'intersection_ids', 'state_schema',
            'action_schema', 'reward_schema', 'num_episodes',
            'num_decision_steps', 'num_transitions', 'training_seed',
            'source_policy', 'collection_stage', 'created_at',
        }
        missing = sorted(required - set(manifest))
        if missing:
            raise ValueError(f'History manifest missing fields: {missing}')
        with (self.directory / 'manifest.json').open('w', encoding='utf-8') as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')
        return manifest


def mixed_batch(online: TraceableReplayBuffer, offline: HistoricalPool,
                batch_size: int, offline_ratio: float, visible_scenes=(),
                strategy='scene_intersection_balanced', rng=random):
    if batch_size <= 0 or not 0 <= offline_ratio <= 1:
        raise ValueError('Invalid mixed batch configuration')
    offline_count = int(round(batch_size * offline_ratio))
    online_count = batch_size - offline_count
    if online_count > len(online):
        raise ValueError('Not enough online transitions')
    online_records = online.sample(online_count, rng=rng)
    # Important: ratio=0 must not touch the offline pool.
    offline_records = [] if offline_count == 0 else offline.sample(
        offline_count, visible_scenes, strategy=strategy, rng=rng
    )
    tagged = [(x, 'online_current') for x in online_records]
    tagged += [(x, 'offline_history') for x in offline_records]
    rng.shuffle(tagged)
    return tagged
