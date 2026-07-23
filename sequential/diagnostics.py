import os

import numpy as np

from .io import atomic_json


class ReplayDiagnostics:
    def __init__(self, output_dir, trace_samples=False, sparse_interval=1000):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.trace_samples = bool(trace_samples)
        self.sparse_interval = int(sparse_interval)
        self.transitions_written_by_scene = {}
        self.samples_drawn_by_scene = {}
        self.sample_ages = []
        self.thresholds = {}
        self.previous_historical_ratio = None
        self.full_batches = []

    @staticmethod
    def _increment(mapping, key, amount=1):
        mapping[key] = mapping.get(key, 0) + amount

    def record_transition(self, metadata):
        self._increment(
            self.transitions_written_by_scene, metadata.source_network
        )

    def record_update(self, update, gradient_updates):
        for source in update['sample_sources']:
            self._increment(self.samples_drawn_by_scene, source)
        self.sample_ages.extend(int(age) for age in update['sample_ages'])
        if self.trace_samples or gradient_updates % self.sparse_interval == 0:
            self.full_batches.append({
                'gradient_updates': int(gradient_updates),
                'transition_ids': list(update['sample_transition_ids']),
            })

    def _record_replacement_thresholds(self, composition, current_network,
                                       global_step, global_episode):
        total = composition['size']
        current = composition['count_by_scene'].get(current_network, 0)
        historical_ratio = 0.0 if not total else (total - current) / total
        for threshold in (0.5, 0.1, 0.0):
            key = f'{threshold:.1f}'
            if key not in self.thresholds and historical_ratio <= threshold:
                self.thresholds[key] = {
                    'global_step': int(global_step),
                    'global_episode': int(global_episode),
                    'historical_ratio': historical_ratio,
                }
        self.previous_historical_ratio = historical_ratio

    def episode_record(self, agent, stage_index, local_episode, global_episode):
        composition = agent.replay.composition()
        records = list(agent.replay.records)
        written_steps = [record.metadata.written_global_step for record in records]
        ages = np.asarray(self.sample_ages, dtype=float)
        current_samples = self.samples_drawn_by_scene.get(agent.current_network, 0)
        total_samples = sum(self.samples_drawn_by_scene.values())
        self._record_replacement_thresholds(
            composition, agent.current_network,
            agent.counters.global_decision_step, global_episode,
        )
        record = {
            'schema_version': 1,
            'stage_index': int(stage_index),
            'local_episode': int(local_episode),
            'global_episode': int(global_episode),
            'current_network': agent.current_network,
            'replay_size': composition['size'],
            'replay_capacity': composition['capacity'],
            'replay_count_by_scene': composition['count_by_scene'],
            'replay_ratio_by_scene': composition['ratio_by_scene'],
            'transitions_written_by_scene': dict(self.transitions_written_by_scene),
            'samples_drawn_by_scene': dict(self.samples_drawn_by_scene),
            'current_sample_fraction': (
                0.0 if not total_samples else current_samples / total_samples
            ),
            'historical_sample_fraction': (
                0.0 if not total_samples else 1 - current_samples / total_samples
            ),
            'sample_age_mean': None if not len(ages) else float(np.mean(ages)),
            'sample_age_p50': None if not len(ages) else float(np.percentile(ages, 50)),
            'sample_age_p95': None if not len(ages) else float(np.percentile(ages, 95)),
            'oldest_written_global_step': min(written_steps) if written_steps else None,
            'newest_written_global_step': max(written_steps) if written_steps else None,
            'gradient_updates': agent.counters.gradient_updates,
            'target_updates': agent.counters.target_updates,
            'replacement_thresholds': dict(self.thresholds),
            'sparse_full_batches': list(self.full_batches),
        }
        path = os.path.join(
            self.output_dir,
            f'stage_{int(stage_index):02d}_episode_{int(local_episode):04d}.json',
        )
        atomic_json(path, record)
        self.sample_ages = []
        self.full_batches = []
        return path, record

    def state_dict(self):
        return {
            'transitions_written_by_scene': dict(self.transitions_written_by_scene),
            'samples_drawn_by_scene': dict(self.samples_drawn_by_scene),
            'sample_ages': list(self.sample_ages),
            'thresholds': dict(self.thresholds),
            'previous_historical_ratio': self.previous_historical_ratio,
            'full_batches': list(self.full_batches),
        }

    def load_state_dict(self, state):
        self.transitions_written_by_scene = dict(
            state['transitions_written_by_scene']
        )
        self.samples_drawn_by_scene = dict(state['samples_drawn_by_scene'])
        self.sample_ages = list(state['sample_ages'])
        self.thresholds = dict(state['thresholds'])
        self.previous_historical_ratio = state['previous_historical_ratio']
        self.full_batches = list(state['full_batches'])
