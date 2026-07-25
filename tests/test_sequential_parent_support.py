import json
import os
import random
import tempfile
import unittest

import numpy as np
import torch

from sequential.config import FORMAL_POLICIES, load_sequential_config
from sequential.core import (
    ReplayMetadata, ReplayRecord, SequentialReplay, TargetUpdateScheduler,
    TrainingPayload, canonical_digest, optimizer_state_digest,
    replay_content_digest, replay_metadata_digest,
)
from sequential.io import atomic_json
from sequential.manifest import build_formal_plan, validate_formal_plan
from sequential.parents import convert_parent_replay


class SequentialParentSupportTests(unittest.TestCase):
    def _legacy_payload(self, value):
        return (
            np.full((1, 8), value, dtype=np.float32),
            np.array([value % 8], dtype=np.int8),
            np.array([value % 8], dtype=np.int64),
            np.array(float(value), dtype=np.float32),
            np.full((1, 8), value + 1, dtype=np.float32),
            np.array([(value + 1) % 8], dtype=np.int8),
        )

    def test_plan1_scheduler_preserves_original_phase(self):
        scheduler = TargetUpdateScheduler.from_plan1_parent(143000, 14300, 10)
        self.assertEqual(scheduler.next_update, 143009)
        for completed in range(143001, 143009):
            self.assertFalse(scheduler.after_successful_gradient(completed))
        self.assertTrue(scheduler.after_successful_gradient(143009))
        self.assertEqual(scheduler.next_update, 143019)

    def test_scheduler_rejects_incompatible_parent_counter(self):
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            TargetUpdateScheduler.from_plan1_parent(143000, 14299, 10)

    def test_legacy_replay_conversion_is_immutable_and_ordered(self):
        items = [
            ('398_359_intersection_1_1', self._legacy_payload(1)),
            ('398_360_intersection_1_1', self._legacy_payload(2)),
            ('399_1_intersection_1_1', self._legacy_payload(3)),
        ]
        replay = convert_parent_replay(items, 'sumohz1x1', 144000, 5000)
        records = list(replay.records)
        self.assertEqual(
            [item.metadata.written_global_step for item in records],
            [143998, 143999, 144000],
        )
        self.assertEqual(
            [(item.metadata.local_episode, item.metadata.decision_index) for item in records],
            [(399, 359), (399, 360), (400, 1)],
        )
        self.assertFalse(records[0].payload.state.flags.writeable)
        with self.assertRaises(ValueError):
            records[0].payload.state[0, 0] = 99

    def test_replay_digests_separate_payload_and_metadata(self):
        payload = TrainingPayload.from_legacy(self._legacy_payload(4))
        first = ReplayRecord(payload, ReplayMetadata('a', 'sumohz1x1', 1, 1, 1, 1))
        second = ReplayRecord(payload, ReplayMetadata('b', 'sumohz1x1', 1, 1, 1, 1))
        self.assertEqual(replay_content_digest([first]), replay_content_digest([second]))
        self.assertNotEqual(replay_metadata_digest([first]), replay_metadata_digest([second]))

    def test_replay_checkpoint_roundtrip_preserves_fifo_and_metadata(self):
        records = [
            ReplayRecord(
                TrainingPayload.from_legacy(self._legacy_payload(index)),
                ReplayMetadata(str(index), 'sumohz1x1', 1, 1, index, index),
            )
            for index in range(4)
        ]
        replay = SequentialReplay(4, records)
        replay.stage_insertions = 9
        restored = SequentialReplay.from_state_dict(replay.state_dict())
        self.assertEqual(list(restored.records), list(replay.records))
        self.assertEqual(restored.stage_insertions, 9)
        restored.append(ReplayRecord(
            TrainingPayload.from_legacy(self._legacy_payload(5)),
            ReplayMetadata('5', 'sumohz1x1', 2, 1, 1, 5),
        ))
        self.assertEqual([r.metadata.transition_id for r in restored.records], ['1', '2', '3', '5'])

    def test_canonical_optimizer_digest_is_dict_order_independent(self):
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        first = {
            'state': {0: {'square_avg': tensor, 'step': torch.tensor(3.0)}},
            'param_groups': [{'lr': 0.001, 'params': [0], 'centered': False}],
        }
        second = {
            'param_groups': [{'params': [0], 'centered': False, 'lr': 0.001}],
            'state': {0: {'step': torch.tensor(3.0), 'square_avg': tensor.clone()}},
        }
        self.assertEqual(optimizer_state_digest(first), optimizer_state_digest(second))
        second['param_groups'][0]['lr'] = 0.002
        self.assertNotEqual(optimizer_state_digest(first), optimizer_state_digest(second))

    def test_canonical_digest_does_not_advance_rng(self):
        random.seed(41)
        np.random.seed(41)
        torch.manual_seed(41)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        canonical_digest({'tensor': torch.ones(2), 'array': np.ones(2)})
        after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        self.assertEqual(before[0], after[0])
        self.assertTrue(np.array_equal(before[1][1], after[1][1]))
        self.assertTrue(torch.equal(before[2], after[2]))

    def test_build_formal_plan_has_exact_frozen_matrix(self):
        config = load_sequential_config()
        with tempfile.TemporaryDirectory() as directory:
            parents = []
            for order in range(1, 5):
                for seed in range(5):
                    parents.append({
                        'order_id': f'O{order}', 'training_seed': seed,
                        'import_manifest_path': f'/parent/O{order}/{seed}.json',
                        'checkpoint_path': f'/checkpoint/O{order}/{seed}.pt',
                        'checkpoint_file_sha256': 'a' * 64,
                        'digests': {'online_parameter_digest': f'{order}{seed}'},
                    })
            catalog = os.path.join(directory, 'catalog.json')
            atomic_json(catalog, {
                'budget_id': 'b400', 'parent_checkpoint_episode': 400,
                'parents': parents,
            })
            output = os.path.join(directory, 'plan.json')
            plan = build_formal_plan(catalog, output)
            self.assertEqual(plan['child_count'], 60)
            self.assertFalse(plan['launch_authorized'])
            self.assertEqual(
                {child['policy'] for child in plan['children']}, set(FORMAL_POLICIES)
            )
            self.assertTrue(all(not child['trace_replay_samples'] for child in plan['children']))
            self.assertTrue(all(
                child['logical_run_id'].startswith('plan34_b400_')
                for child in plan['children']
            ))
            self.assertEqual(validate_formal_plan(output)['child_count'], 60)
            self.assertEqual(plan['orders'], config['orders'])

    def test_formal_plan_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            parents = [{
                'order_id': f'O{order}', 'training_seed': seed,
                'import_manifest_path': 'parent.json', 'checkpoint_path': 'parent.pt',
                'checkpoint_file_sha256': 'a' * 64, 'digests': {},
            } for order in range(1, 5) for seed in range(5)]
            catalog = os.path.join(directory, 'catalog.json')
            output = os.path.join(directory, 'plan.json')
            atomic_json(catalog, {
                'budget_id': 'b400', 'parent_checkpoint_episode': 400,
                'parents': parents,
            })
            build_formal_plan(catalog, output)
            with open(output, encoding='utf-8') as handle:
                plan = json.load(handle)
            plan['children'][0]['trace_replay_samples'] = True
            atomic_json(output, plan)
            with self.assertRaisesRegex(ValueError, 'disable'):
                validate_formal_plan(output)

    def test_b100_formal_plan_uses_episode100_parents_and_budget_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            parents = [{
                'order_id': f'O{order}', 'training_seed': seed,
                'import_manifest_path': f'/parent/O{order}/{seed}.json',
                'checkpoint_path': f'/checkpoint/O{order}/{seed}/episode_0100.pt',
                'checkpoint_file_sha256': 'b' * 64, 'digests': {},
            } for order in range(1, 5) for seed in range(5)]
            catalog = os.path.join(directory, 'catalog.json')
            output = os.path.join(directory, 'plan.json')
            atomic_json(catalog, {
                'budget_id': 'b100', 'parent_checkpoint_episode': 100,
                'parents': parents,
            })
            plan = build_formal_plan(
                catalog, output, 'configs/sequential/plan34_b100.yml'
            )
            self.assertEqual(plan['budget_id'], 'b100')
            self.assertEqual(plan['parent_checkpoint_episode'], 100)
            self.assertTrue(all(
                child['stage_episodes'] == [100, 100, 100, 100]
                and child['logical_run_id'].startswith('plan34_b100_')
                and child['parent_checkpoint_episode'] == 100
                for child in plan['children']
            ))
            self.assertTrue(validate_formal_plan(output)['valid'])


if __name__ == '__main__':
    unittest.main()
