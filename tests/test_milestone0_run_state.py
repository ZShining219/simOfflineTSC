import json
import os
import tempfile
import unittest

from utils.logger import RunStateManager


class RunStateTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.config = {
            'command': {
                'task': 'tsc', 'agent': 'dqn', 'world': 'sumo',
                'network': 'sumohz1x1', 'prefix': 'state-test', 'seed': 3,
            },
            'world': {'dir': self.temporary_directory.name},
        }
        self.output = os.path.join(
            self.temporary_directory.name, 'output_data', 'tsc', 'sumo_dqn',
            'sumohz1x1', 'state-test',
        )
        self.config_path = os.path.join(self.output, 'config')
        os.makedirs(self.config_path)
        with open(os.path.join(self.config_path, 'resolved_config.yaml'), 'wb') as handle:
            handle.write(b'fixed bytes\n')

    def tearDown(self):
        self.temporary_directory.cleanup()

    def read(self, name):
        with open(os.path.join(self.output, name), encoding='utf-8') as handle:
            return json.load(handle)

    def test_completed_lifecycle_has_fixed_identity_and_times(self):
        state = RunStateManager(self.config, self.config_path)
        manifest = self.read('run_manifest.json')
        created = self.read('run_status.json')
        self.assertEqual('tsc/sumo_dqn/sumohz1x1/state-test', manifest['run_id'])
        self.assertEqual(3, manifest['training_seed'])
        self.assertEqual('fixed_default', manifest['sumo_seed_mode'])
        self.assertEqual('已创建', created['status'])
        self.assertIsNone(created['started_at_utc'])
        state.transition('运行中')
        state.transition('已完成', exit_code=0)
        completed = self.read('run_status.json')
        self.assertEqual('已完成', completed['status'])
        self.assertIsNotNone(completed['started_at_utc'])
        self.assertIsNotNone(completed['finished_at_utc'])
        self.assertEqual(0, completed['exit_code'])

    def test_failure_is_terminal_and_redacts_environment_values(self):
        state = RunStateManager(self.config, self.config_path)
        state.transition('运行中')
        secret = os.environ.get('PATH', '')
        state.transition('失败', exit_code=1, error=RuntimeError('failure ' + secret))
        failed = self.read('run_status.json')
        self.assertEqual('失败', failed['status'])
        self.assertEqual('RuntimeError', failed['error_type'])
        self.assertNotIn(secret, failed['error_message'])
        with self.assertRaisesRegex(ValueError, 'Invalid run status transition'):
            state.transition('已完成')

    def test_invalid_status_and_invalid_completion_are_rejected(self):
        state = RunStateManager(self.config, self.config_path)
        with self.assertRaisesRegex(ValueError, 'Invalid run status'):
            state.transition('未知')
        state.transition('运行中')
        with self.assertRaisesRegex(ValueError, 'requires exit_code=0'):
            state.transition('已完成', exit_code=2)


if __name__ == '__main__':
    unittest.main()
