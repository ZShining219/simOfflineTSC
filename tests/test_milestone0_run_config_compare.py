import json
import os
import tempfile
import unittest

import yaml

from utils.run_config_compare import compare_runs


def write_run(root, name, network='n1', seed=0, batch_size=64, input_dim=16, action_dim=8):
    run = os.path.join(root, name)
    config = os.path.join(run, 'config')
    os.makedirs(config)
    resolved = {
        'command': {
            'network': network, 'prefix': name, 'seed': seed,
            'interface': 'libsumo', 'delay_type': 'apx', 'agent': 'dqn',
        },
        'world': {
            'interval': 1.0, 'combined_file': f'{network}.sumocfg',
            'roadnetFile': f'{network}.net.xml', 'flowFile': f'{network}.rou.xml',
            'convertroadnetFile': f'{network}.road.json',
            'convertflowFile': f'{network}.flow.json', 'gui': False,
        },
        'trainer': {'batch_size': batch_size},
        'model': {'gamma': 0.95},
        'logger': {'schema': 1},
        'config_record': {
            'created_at_utc': '2026-01-01T00:00:00Z',
            'sources': ['configs/tsc/base.yml', 'configs/tsc/dqn.yml', f'configs/sim/{network}.cfg'],
        },
    }
    with open(os.path.join(config, 'resolved_config.yaml'), 'w', encoding='utf-8') as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    runtime = {
        'schema_version': 1,
        'agent': 'dqn',
        'reproducibility_probe': {'python_random': [seed]},
        'agents': [{
            'rank': 0, 'action_dim': action_dim,
            'online_model_state_hash': str(seed), 'target_model_state_hash': str(seed),
            'model': {'input_dim': input_dim, 'hidden_layers': [20, 20]},
            'target_model': {'input_dim': input_dim, 'hidden_layers': [20, 20]},
            'optimizer': {'class': 'RMSprop'}, 'loss': {'class': 'MSELoss'},
        }],
    }
    with open(os.path.join(config, 'model_resolved.json'), 'w', encoding='utf-8') as handle:
        json.dump(runtime, handle)
    return run


class RunConfigCompareTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_closed_allowlist_accepts_only_network_prefix_seed_paths_and_metadata(self):
        first = write_run(self.temporary_directory.name, 'a', 'n1', 0)
        second = write_run(self.temporary_directory.name, 'b', 'n2', 1)
        self.assertTrue(compare_runs([first, second])['compatible'])

    def test_trainer_and_interface_changes_fail(self):
        first = write_run(self.temporary_directory.name, 'a')
        second = write_run(self.temporary_directory.name, 'b', batch_size=32)
        result = compare_runs([first, second])
        self.assertFalse(result['compatible'])
        self.assertTrue(any('trainer.batch_size' in item for item in result['comparisons'][0]['differences']))

    def test_input_or_action_dimension_mismatch_fails_explicitly(self):
        first = write_run(self.temporary_directory.name, 'a')
        second = write_run(self.temporary_directory.name, 'b', input_dim=17)
        result = compare_runs([first, second])
        self.assertFalse(result['compatible'])
        self.assertIn('input_dim/action_dim mismatch', result['comparisons'][0]['differences'][0])


if __name__ == '__main__':
    unittest.main()
