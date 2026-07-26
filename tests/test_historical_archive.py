import random
import unittest

import numpy as np

from dataset.offline_trajectory_dataset import OfflineBatch
from sequential.historical_archive import (
    HistoricalArchive, derive_rng_seed, stable_digest,
)
from sequential.owp import build_historical_sampler


class DummyDataset:
    def __init__(self, network, count=6):
        self.network = network
        self.count = count
        self.seeds = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        prefix = 10 if network == 'n1' else 20 if network == 'n2' else 30
        observations = np.arange(count * 16, dtype=np.float32).reshape(count, 16)
        observations[:, :8] += prefix
        observations[:, 8:] = 0
        observations[np.arange(count), 8 + np.arange(count) % 8] = 1
        self._arrays = {network: {
            'observations': observations,
            'actions': np.arange(count, dtype=np.int64) % 8,
            'rewards': -np.arange(count, dtype=np.float32),
            'behavior_training_seed': self.seeds,
            'episode_id': np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int64),
            'transition_id': np.asarray(
                [f'{network}:{index}' for index in range(count)], dtype=object,
            ),
        }}

    def eligible_indices(self, network, behavior_seeds=None, episode_range=None):
        assert network == self.network
        mask = np.isin(self.seeds, list(behavior_seeds))
        return np.flatnonzero(mask)

    def select_indices(self, network, indices):
        assert network == self.network
        indices = np.asarray(indices, dtype=np.int64)
        count = len(indices)
        prefix = 10 if network == 'n1' else 20 if network == 'n2' else 30
        return OfflineBatch(
            observations=np.full((count, 16), prefix, dtype=np.float32),
            actions=indices % 8,
            rewards=-indices.astype(np.float32),
            next_observations=np.full((count, 16), prefix + 1, dtype=np.float32),
            terminated=np.zeros(count, dtype=np.bool_),
            truncated=np.zeros(count, dtype=np.bool_),
            source_network=network,
            indices=indices,
            transition_ids=np.asarray([f'{network}:{index}' for index in indices], dtype=object),
            behavior_training_seeds=self.seeds[indices],
            episode_ids=np.ones(count, dtype=np.int64),
            decision_steps=indices + 1,
            run_ids=np.asarray([f'run-{network}'] * count, dtype=object),
            shard_sha256=np.asarray(['a' * 64] * count, dtype=object),
        )


def archive(exclude_matching=True):
    instance = HistoricalArchive.__new__(HistoricalArchive)
    instance.training_seed = 1
    instance.manifest = {
        'archive_digest': 'archive',
        'networks': ['n1', 'n2', 'n3', 'n4'],
        'behavior_training_seeds': [0, 1, 2],
        'behavior_seed_rule': {
            'exclude_matching_training_seed': exclude_matching,
        },
    }
    instance.datasets = {
        network: DummyDataset(network)
        for network in instance.manifest['networks']
    }
    return instance


class HistoricalArchiveTest(unittest.TestCase):
    def test_rng_derivation_is_stable_and_stream_isolated(self):
        self.assertEqual(derive_rng_seed(3, 'online'), derive_rng_seed(3, 'online'))
        self.assertNotEqual(derive_rng_seed(3, 'online'), derive_rng_seed(3, 'hoa'))
        self.assertNotEqual(derive_rng_seed(3, 'hoa'), derive_rng_seed(4, 'hoa'))

    def test_p1c_visibility_masks_current_and_future(self):
        view = archive().visibility('P1C', ['n2', 'n3', 'n1', 'n4'], 3)
        self.assertEqual(('n2', 'n3'), view.networks)
        self.assertEqual(8, len(view))
        self.assertEqual(['n1'], view.visibility['masked_current_networks'])
        self.assertEqual(['n4'], view.visibility['masked_future_networks'])
        self.assertEqual([0, 2], view.visibility['behavior_training_seeds'])
        batch = view.sample(4, random.Random(9))
        self.assertEqual(4, len(batch))
        self.assertTrue(set(batch.source_networks) <= {'n2', 'n3'})
        self.assertNotIn(1, set(batch.behavior_training_seeds))

    def test_p1c_stage1_is_empty_and_p1f_is_full(self):
        causal = archive().visibility('p1c', ['n1', 'n2', 'n3', 'n4'], 1)
        self.assertEqual(0, len(causal))
        full = archive(False).visibility('P1F', ['n4', 'n3', 'n2', 'n1'], 1)
        self.assertEqual(24, len(full))
        self.assertEqual([], full.visibility['masked_networks'])

    def test_invalid_order_and_stage_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            archive().visibility('P1C', ['n1', 'n1', 'n3', 'n4'], 2)
        with self.assertRaisesRegex(ValueError, 'outside'):
            archive().visibility('P1C', ['n1', 'n2', 'n3', 'n4'], 5)

    def test_visibility_digest_is_deterministic(self):
        first = archive().visibility('P1C', ['n1', 'n2', 'n3', 'n4'], 2)
        second = archive().visibility('P1C', ['n1', 'n2', 'n3', 'n4'], 2)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(
            stable_digest(first.visibility), stable_digest(second.visibility)
        )

    def test_all_working_pool_methods_are_deterministic_and_frozen(self):
        view = archive(False).visibility('P1F', ['n1', 'n2', 'n3', 'n4'], 2)
        alignment = np.arange(160, dtype=np.float32).reshape(10, 16)
        for method in ('DHOA', 'RAND', 'COV', 'CQ', 'CQA'):
            kwargs = {'alignment_observations': alignment} if method == 'CQA' else {}
            first = build_historical_sampler(
                view, method, 12, random.Random(17), **kwargs,
            )
            second = build_historical_sampler(
                view, method, 12, random.Random(17), **kwargs,
            )
            self.assertEqual(first.manifest['owp_digest'], second.manifest['owp_digest'])
            self.assertEqual(first.references, second.references)
            self.assertEqual(24 if method == 'DHOA' else 12, len(first))
            first_ids = list(first.sample(5).transition_ids)
            second_ids = list(second.sample(5).transition_ids)
            self.assertEqual(first_ids, second_ids)
            state = first.state_dict()
            resumed = build_historical_sampler(
                view, method, 12, random.Random(17), **kwargs,
            )
            resumed.load_state_dict(state)
            self.assertEqual(
                list(first.sample(4).transition_ids),
                list(resumed.sample(4).transition_ids),
            )

    def test_cqa_requires_shared_warmup_descriptor(self):
        view = archive(False).visibility('P1F', ['n1', 'n2', 'n3', 'n4'], 1)
        with self.assertRaisesRegex(ValueError, 'warm-up'):
            build_historical_sampler(view, 'CQA', 12, random.Random(1))


if __name__ == '__main__':
    unittest.main()
