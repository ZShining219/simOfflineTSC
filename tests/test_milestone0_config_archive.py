import copy
import hashlib
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from utils.logger import (
    archive_run_config,
    reserve_run_output,
    resolve_simulator_config,
    verify_config_archive,
)


class ConfigArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.source = json.dumps({
            'network': 'unit-network',
            'interval': 1.0,
            'dir': 'unused',
        }, sort_keys=True).encode('utf-8')
        self.source_hash = hashlib.sha256(self.source).hexdigest()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def config(self, prefix):
        return {
            'command': {
                'task': 'tsc', 'agent': 'fixedtime', 'world': 'sumo',
                'network': 'unit-network', 'prefix': prefix, 'seed': 7,
            },
            'world': {'dir': self.temporary_directory.name, 'interval': 0.5},
            'logger': {'replay_dir': 'replay'},
            'trainer': {},
            'model': {},
        }

    def test_distinct_prefixes_get_isolated_resolved_configs(self):
        def resolve(prefix):
            config = self.config(prefix)
            reserve_run_output(config)
            resolved_path, extra = resolve_simulator_config(config, self.source)
            self.assertEqual({}, extra)
            with open(resolved_path, encoding='utf-8') as file_handle:
                self.assertEqual(0.5, json.load(file_handle)['interval'])
            return resolved_path

        with ThreadPoolExecutor(max_workers=2) as executor:
            paths = list(executor.map(resolve, ('first', 'second')))
        self.assertNotEqual(paths[0], paths[1])
        self.assertEqual(self.source_hash, hashlib.sha256(self.source).hexdigest())

    def test_archive_verification_rejects_corruption_and_missing_file(self):
        config = self.config('archive')
        reserve_run_output(config)
        resolved_path, extra = resolve_simulator_config(config, self.source)
        with open(resolved_path, encoding='utf-8') as file_handle:
            resolved_world = json.load(file_handle)
        resolved_world.update(extra)
        snapshots = {
            'base.yml': b'base: true\n',
            'fixedtime.yml': b'agent: fixedtime\n',
            'simulator_source.cfg': self.source,
        }
        config_path = archive_run_config(config, snapshots, resolved_world)
        self.assertIn('resolved_config.yaml', verify_config_archive(config_path))

        resolved_config_path = os.path.join(config_path, 'resolved_config.yaml')
        with open(resolved_config_path, 'rb') as file_handle:
            original = file_handle.read()
        with open(resolved_config_path, 'ab') as file_handle:
            file_handle.write(b'corrupt')
        with self.assertRaisesRegex(IOError, 'verification failed'):
            verify_config_archive(config_path)
        with open(resolved_config_path, 'wb') as file_handle:
            file_handle.write(original)
        os.unlink(resolved_config_path)
        with self.assertRaisesRegex(IOError, 'is missing'):
            verify_config_archive(config_path)

    def test_duplicate_prefix_is_rejected(self):
        config = self.config('duplicate')
        reserve_run_output(config)
        with self.assertRaisesRegex(FileExistsError, 'Use a new --prefix'):
            reserve_run_output(copy.deepcopy(config))


if __name__ == '__main__':
    unittest.main()
