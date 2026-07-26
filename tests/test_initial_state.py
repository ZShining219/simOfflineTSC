import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sequential.initial_state import build_initial_state_catalog
from sequential.parents import ParentSource


class InitialStateCatalogTest(unittest.TestCase):
    def test_catalog_uses_each_orders_actual_first_network(self):
        orders = {
            'O1': ['n1', 'n2', 'n3', 'n4'],
            'O2': ['n2', 'n3', 'n4', 'n1'],
            'O3': ['n3', 'n4', 'n1', 'n2'],
            'O4': ['n4', 'n1', 'n2', 'n3'],
        }
        config = {'orders': orders, 'training_seeds': [0, 1, 2, 3, 4]}
        sources = [
            ParentSource(f'/source/{network}/{seed}', network, seed)
            for network in orders['O1'] for seed in range(5)
        ]

        def validated(source):
            shared = f'seed-{source.training_seed}'
            return {
                'source_run_path': source.run_path,
                'source_run_id': source.run_path,
                'network': source.network,
                'training_seed': source.training_seed,
                'checkpoint_path': source.run_path + '/episode_0000.pt',
                'checkpoint_file_sha256': 'a' * 64,
                'checkpoint_episode': 0,
                'epsilon': 1.0,
                'counters': {}, 'dimensions': {}, 'scene_semantics': {},
                'digests': {
                    'online_parameter_digest': shared,
                    'target_parameter_digest': shared,
                    'optimizer_state_digest': shared,
                    'rng_state_digest': shared,
                },
            }

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            'sequential.initial_state.load_sequential_config', return_value=config,
        ), mock.patch(
            'sequential.initial_state.read_parent_whitelist', return_value=sources,
        ), mock.patch(
            'sequential.initial_state.validate_initial_source', side_effect=validated,
        ):
            output = Path(temporary) / 'catalog.json'
            catalog = build_initial_state_catalog('whitelist.csv', output, 'config.yml')
        self.assertEqual(20, catalog['entry_count'])
        for entry in catalog['entries']:
            self.assertEqual(orders[entry['order_id']][0], entry['first_network'])
            self.assertEqual(entry['first_network'], entry['network'])
        self.assertTrue(all(
            item['all_training_state_equal']
            for item in catalog['same_seed_cross_network_equivalence'].values()
        ))


if __name__ == '__main__':
    unittest.main()
