import random
import unittest

import numpy as np

from sequential.core import ReplayMetadata, ReplayRecord, TrainingPayload
from sequential.hybrid import HybridReplayPool
from sequential.evaluation_matrix import validate_lower_triangle


def record(stage, episode, step):
    payload = TrainingPayload.from_legacy((
        np.zeros((1, 2), dtype=np.float32), np.array([0]), np.array([0]),
        np.array(0.0), np.ones((1, 2), dtype=np.float32), np.array([0]),
    ))
    return ReplayRecord(payload, ReplayMetadata(
        f'{stage}-{episode}-{step}', f'S{stage}', stage, episode, step, step,
        'online', f'S{stage}', 11, 7, None, stage,
    ))


class HybridReplayTests(unittest.TestCase):
    def test_causal_freeze_and_stage_balanced_quota(self):
        pool = HybridReplayPool(100, online_ratio=0.5, rng=random.Random(3))
        pool.begin_stage(1)
        for i in range(20):
            pool.append_online(record(1, i + 1, i + 1))
        pool.freeze_stage()
        pool.begin_stage(2)
        for i in range(20):
            pool.append_online(record(2, i + 1, 30 + i))
        batch = pool.sample(10)
        self.assertEqual(len(batch), 10)
        self.assertEqual(sum(r.metadata.stage_index == 1 for r in batch), 5)
        self.assertEqual(pool.last_diagnostics['actual_historical_ratio'], 0.5)
        with self.assertRaises(ValueError):
            pool.begin_stage(3)

    def test_stage_three_cannot_read_unfrozen_stage_two(self):
        pool = HybridReplayPool(100, rng=random.Random(1))
        pool.begin_stage(1)
        pool.append_online(record(1, 1, 1))
        pool.freeze_stage()
        pool.begin_stage(2)
        pool.append_online(record(2, 1, 2))
        with self.assertRaises(ValueError):
            pool.begin_stage(3)

    def test_state_roundtrip_preserves_rng_sampling(self):
        pool = HybridReplayPool(100, rng=random.Random(9))
        pool.begin_stage(1)
        for i in range(10):
            pool.append_online(record(1, i + 1, i + 1))
        pool.freeze_stage(); pool.begin_stage(2)
        for i in range(10):
            pool.append_online(record(2, i + 1, i + 11))
        state = pool.state_dict()
        restored = HybridReplayPool.from_state_dict(state)
        self.assertEqual(
            [r.metadata.transition_id for r in pool.sample(6)],
            [r.metadata.transition_id for r in restored.sample(6)],
        )

    def test_lower_triangle_rejects_future_scene(self):
        self.assertTrue(validate_lower_triangle(['S1', 'S2'], ['S1', 'S2', 'S3'], 2)['valid'])
        with self.assertRaises(ValueError):
            validate_lower_triangle(['S1', 'S3'], ['S1', 'S2', 'S3'], 2)
