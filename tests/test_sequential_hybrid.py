import random
import unittest
import tempfile
import json
import os

import numpy as np

from sequential.core import ReplayMetadata, ReplayRecord, TrainingPayload
from sequential.hybrid import HybridReplayPool
from sequential.evaluation_matrix import validate_lower_triangle
from sequential.manifest import build_hybrid_plan, validate_hybrid_plan
from sequential.hybrid_agent import HybridDQNAgent
from sequential.config import load_sequential_config


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
    def test_hybrid_agent_uses_online_only_warmup_and_restores_pool(self):
        class Intersection:
            id = 'intersection_1_1'
        class Generator:
            def __init__(self, value): self.value = value
            def generate(self): return np.array(self.value, copy=True)
        class World:
            def __init__(self): self.intersection = Intersection()
        def binding(world, rank):
            return {
                'world': world, 'inter': world.intersection,
                'ob_generator': Generator(np.zeros(2, dtype=np.float32)),
                'phase_generator': Generator([0]),
                'reward_generator': Generator([-1.]),
                'queue_generator': Generator([1., 1.]),
                'delay_generator': Generator([.1]),
                'signature': {
                    'intersection_id': 'intersection_1_1',
                    'incoming_lane_mapping': ('a', 'b'),
                    'phase_action_mapping': ('p0', 'p1'),
                    'state_dim': 2, 'action_dim': 2,
                },
            }
        config = load_sequential_config()
        trainer = dict(config['trainer']); trainer.update({
            'learning_start': 2, 'batch_size': 2, 'buffer_size': 20,
        })
        agent = HybridDQNAgent(
            World(), 0, config['model'], trainer, binding_factory=binding,
        )
        agent.begin_stage(1, 'S1', 'hybrid')
        for i in range(2):
            agent.remember(
                np.zeros((1, 2)), np.array([0]), np.array([0]), np.array([-1.]),
                np.ones((1, 2)), np.array([0]), i + 1, i + 1,
            )
        self.assertFalse(agent.is_update_ready())
        agent.remember(
            np.zeros((1, 2)), np.array([0]), np.array([0]), np.array([-1.]),
            np.ones((1, 2)), np.array([0]), 3, 3,
        )
        self.assertTrue(agent.is_update_ready())
        update = agent.successful_gradient_update()
        self.assertIsNotNone(update)
        self.assertAlmostEqual(update['replay_diagnostics']['actual_historical_ratio'], 0.0)
        state = agent.full_state_dict()
        restored = HybridDQNAgent(
            World(), 0, config['model'], trainer, binding_factory=binding,
        )
        restored.load_full_state_dict(state)
        self.assertEqual(len(restored.hybrid_pool.online), 3)
        self.assertEqual(restored.hybrid_pool.stage_index, 1)

    def test_hybrid_plan_is_exact_b100_matrix(self):
        catalog = {'budget_id': 'b100', 'parent_checkpoint_episode': 100, 'parents': []}
        for order in range(1, 5):
            for seed in range(5):
                catalog['parents'].append({
                    'order_id': f'O{order}', 'training_seed': seed,
                    'import_manifest_path': f'/tmp/p-{order}-{seed}.json',
                    'checkpoint_path': f'/tmp/p-{order}-{seed}.pt',
                    'checkpoint_file_sha256': 'x' * 64, 'digests': {},
                })
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = os.path.join(directory, 'catalog.json')
            plan_path = os.path.join(directory, 'plan.json')
            with open(catalog_path, 'w', encoding='utf-8') as handle:
                json.dump(catalog, handle)
            plan = build_hybrid_plan(catalog_path, plan_path)
            self.assertEqual(plan['child_count'], 20)
            self.assertTrue(validate_hybrid_plan(plan_path)['valid'])

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
