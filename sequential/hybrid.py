"""Auditable causal stage-aware hybrid replay for semi-offline runs.

The pool is deliberately independent from :class:`SequentialReplay`: existing
Plan 3/4 policies retain their frozen semantics while CS-HR provides explicit
read-only historical pools and deterministic sampling state for new runs.
"""

import copy
import math
import random
from collections import Counter, defaultdict

from .core import ReplayRecord


class HybridReplayPool:
    def __init__(self, capacity, online_ratio=0.5, rng=None):
        if int(capacity) <= 0:
            raise ValueError('capacity must be positive')
        if not 0.0 <= float(online_ratio) <= 1.0:
            raise ValueError('online_ratio must be in [0, 1]')
        self.capacity = int(capacity)
        self.online_ratio = float(online_ratio)
        self.rng = rng or random.Random()
        self.stage_index = 1
        self.online = []
        self.historical = {}
        self._frozen = set()
        self.sample_count = 0
        self.last_diagnostics = {}

    def begin_stage(self, stage_index):
        stage_index = int(stage_index)
        if stage_index < 1 or stage_index > 4:
            raise ValueError('stage_index must be in 1..4')
        if stage_index < self.stage_index:
            raise ValueError('stage boundaries cannot move backwards')
        if stage_index > 1 and stage_index - 1 not in self._frozen:
            raise ValueError('previous stage historical pool is not frozen')
        self.stage_index = stage_index
        self.online = []

    def append_online(self, record):
        self._check_record(record)
        if record.metadata.stage_index != self.stage_index:
            raise ValueError('online record belongs to a different stage')
        self.online.append(record)
        if len(self.online) > self.capacity:
            self.online.pop(0)

    def freeze_stage(self, stage_index=None):
        stage = self.stage_index if stage_index is None else int(stage_index)
        if stage != self.stage_index:
            raise ValueError('only the active stage can be frozen')
        if stage in self._frozen:
            return
        self.historical[stage] = tuple(copy.deepcopy(self.online))
        self._frozen.add(stage)

    def visible_historical_stages(self):
        return tuple(sorted(s for s in self._frozen if s < self.stage_index))

    def _check_record(self, record):
        if not isinstance(record, ReplayRecord):
            raise TypeError('Hybrid replay accepts ReplayRecord only')
        if record.metadata.source_kind not in {'online', 'historical'}:
            raise ValueError('invalid transition source_kind')

    def _episode_strata(self, records, strata=4):
        groups = defaultdict(list)
        episodes = [int(r.metadata.local_episode) for r in records]
        max_episode = max(episodes, default=1)
        for record in records:
            episode = int(record.metadata.local_episode)
            bucket = min(strata - 1, ((episode - 1) * strata) // max(1, max_episode))
            groups[bucket].append(record)
        return groups

    def _sample_historical(self, count, strategy):
        stages = list(self.visible_historical_stages())
        if count == 0:
            return []
        if not stages:
            raise ValueError('historical quota requested before a pool is frozen')
        if strategy not in {'uniform_cumulative', 'stage_balanced', 'stage_balanced_episode_stratified'}:
            raise ValueError(f'unknown historical sampling strategy: {strategy}')
        result = []
        if strategy == 'uniform_cumulative':
            population = [r for stage in stages for r in self.historical[stage]]
            if len(population) < count:
                raise ValueError('historical pool is smaller than requested quota')
            return self.rng.sample(population, count)
        for _ in range(count):
            stage = self.rng.choice(stages)
            records = list(self.historical[stage])
            if strategy == 'stage_balanced_episode_stratified':
                groups = self._episode_strata(records)
                records = self.rng.choice(list(groups.values()))
            result.append(self.rng.choice(records))
        return result

    def sample(self, batch_size, strategy='stage_balanced_episode_stratified', online_ratio=None):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        ratio = self.online_ratio if online_ratio is None else float(online_ratio)
        # Stage 1 is strictly online-only; no parent replay is implicitly
        # imported into the semi-offline condition.
        if self.stage_index == 1:
            ratio = 0.0
        historical_count = int(math.floor(ratio * batch_size))
        online_count = batch_size - historical_count
        if len(self.online) < online_count:
            raise ValueError('online pool is smaller than requested quota')
        if historical_count and not self.visible_historical_stages():
            raise ValueError('historical data is not causally visible at this stage')
        records = self.rng.sample(self.online, online_count) if online_count else []
        records += self._sample_historical(historical_count, strategy)
        self.rng.shuffle(records)
        self.sample_count += len(records)
        ages = [self._age(record) for record in records]
        online_ids = {id(r) for r in self.online}
        by_stage = Counter(
            ('online' if id(r) in online_ids else f'historical_stage_{r.metadata.stage_index}')
            for r in records
        )
        self.last_diagnostics = {
            'stage_index': self.stage_index, 'batch_size': batch_size,
            'target_historical_ratio': ratio,
            'actual_historical_ratio': historical_count / batch_size,
            'count_by_source_stage': dict(by_stage),
            'sample_age_mean': sum(ages) / len(ages) if ages else 0.0,
            'sample_age_p50': sorted(ages)[len(ages) // 2] if ages else 0,
            'sample_age_p95': sorted(ages)[min(len(ages)-1, int(len(ages)*.95))] if ages else 0,
        }
        return records

    def _age(self, record):
        written = int(record.metadata.written_global_step)
        return max(0, int(self.sample_count) - written)

    def state_dict(self):
        return {
            'schema_version': 1, 'capacity': self.capacity,
            'online_ratio': self.online_ratio, 'stage_index': self.stage_index,
            'online': list(self.online), 'historical': dict(self.historical),
            'frozen': sorted(self._frozen), 'sample_count': self.sample_count,
            'rng_state': self.rng.getstate(),
        }

    def composition(self):
        counts = Counter(r.metadata.source_scene or r.metadata.source_network for r in self.online)
        for stage, records in self.historical.items():
            counts.update({f'historical_stage_{stage}': len(records)})
        total = len(self.online) + sum(len(v) for v in self.historical.values())
        return {
            'size': total, 'capacity': self.capacity,
            'count_by_scene': dict(counts),
            'ratio_by_scene': {k: v / total for k, v in counts.items()} if total else {},
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get('schema_version') != 1:
            raise ValueError('unsupported HybridReplayPool schema')
        pool = cls(state['capacity'], state['online_ratio'])
        pool.stage_index = int(state['stage_index'])
        pool.online = list(state['online'])
        pool.historical = {int(k): tuple(v) for k, v in state['historical'].items()}
        pool._frozen = set(int(v) for v in state['frozen'])
        pool.sample_count = int(state['sample_count'])
        pool.rng.setstate(state['rng_state'])
        return pool
