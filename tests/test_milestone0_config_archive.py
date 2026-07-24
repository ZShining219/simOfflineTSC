import copy
import hashlib
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from utils.logger import (
    archive_run_config,
    build_sumo_traffic_identity,
    compose_evaluation_world_config,
    reserve_run_output,
    resolve_simulator_config,
    SUMO_ENVIRONMENT_IDENTITY_FIELDS,
    validate_cross_scene_traffic_identity,
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

    def write_sumo_scene(self, name, vehicle_count):
        data_dir = os.path.join(self.temporary_directory.name, 'data')
        scene_dir = os.path.join(data_dir, name)
        os.makedirs(scene_dir, exist_ok=True)
        network_name = f'{name}.net.xml'
        route_name = f'{name}.rou.xml'
        combined_name = f'{name}.sumocfg'
        with open(os.path.join(scene_dir, network_name), 'w', encoding='utf-8') as handle:
            handle.write('<net/>\n')
        with open(os.path.join(scene_dir, route_name), 'w', encoding='utf-8') as handle:
            handle.write('<routes>')
            for index in range(vehicle_count):
                handle.write(f'<vehicle id="{index}" depart="{index}"/>')
            handle.write('</routes>\n')
        with open(os.path.join(scene_dir, combined_name), 'w', encoding='utf-8') as handle:
            handle.write(
                '<configuration><input>'
                f'<net-file value="{network_name}"/>'
                f'<route-files value="{route_name}"/>'
                '</input><time><begin value="0"/><end value="3600"/>'
                '</time></configuration>\n'
            )
        simulator = {
            'network': name, 'interval': 1.0, 'seed': 0,
            'dir': data_dir + os.sep,
            'combined_file': f'{name}/{combined_name}',
            'roadnetFile': f'{name}/{network_name}',
            'flowFile': f'{name}/{route_name}',
            'convertroadnetFile': f'{name}/roadnet.json',
            'convertflowFile': f'{name}/flow.json',
            'no_warning': True, 'name': 'debug', 'yellow_length': 5,
            'gui': False,
        }
        return json.dumps(simulator, sort_keys=True).encode('utf-8'), simulator

    def test_cross_scene_target_world_is_base_and_archive_is_effective_target(self):
        source_content, source_world = self.write_sumo_scene('S1', 3)
        target_content, target_world = self.write_sumo_scene('S2', 5)
        source_world.update({
            'saveReplay': True, 'report_log_mode': 'normal',
            'report_log_rate': 10, 'rlTrafficLight': True,
        })
        config = self.config('cross-scene')
        config['command']['network'] = 'S2'
        config['world'] = compose_evaluation_world_config(
            source_world, target_content
        )
        config['world']['saveReplay'] = False
        reserve_run_output(config)
        protected = tuple(
            field for field in config['world']
            if field not in {
                'saveReplay', 'report_log_mode', 'report_log_rate',
                'rlTrafficLight',
            }
        )
        resolved_path, extra = resolve_simulator_config(
            config, target_content, protected_world_fields=protected
        )
        identities = validate_cross_scene_traffic_identity(
            source_content, target_content, resolved_path, 'S1', 'S2'
        )
        self.assertEqual(5, identities['effective']['expected_vehicle_count'])
        self.assertEqual(
            identities['target']['route_sha256'],
            identities['effective']['route_sha256'],
        )
        self.assertNotEqual(
            identities['source']['route_sha256'],
            identities['effective']['route_sha256'],
        )
        resolved_world = build_sumo_traffic_identity(resolved_path)
        self.assertEqual(target_world['flowFile'], resolved_world['identity_fields']['flowFile'])
        self.assertFalse(extra['saveReplay'])
        config_path = archive_run_config(config, {
            'base.yml': b'base: true\n', 'fixedtime.yml': b'agent: fixedtime\n',
            'source_simulator_source.cfg': source_content,
            'simulator_source.cfg': target_content,
        }, {**json.loads(target_content), **extra})
        hashes = verify_config_archive(config_path)
        self.assertIn('simulator_resolved.cfg', hashes)
        with open(os.path.join(config_path, 'simulator_resolved.cfg'), 'rb') as handle:
            archived = handle.read()
        self.assertEqual(
            build_sumo_traffic_identity(target_content)['route_sha256'],
            build_sumo_traffic_identity(archived)['route_sha256'],
        )

    def test_cross_scene_identity_fails_fast_on_source_overwrite_and_same_scene_works(self):
        source_content, source_world = self.write_sumo_scene('S2', 5)
        target_content, _ = self.write_sumo_scene('S4', 2)
        config = self.config('bad-cross-scene')
        config['command']['network'] = 'S4'
        config['world'] = source_world
        reserve_run_output(config)
        resolved_path, _ = resolve_simulator_config(config, target_content)
        with self.assertRaisesRegex(ValueError, 'does not match target'):
            validate_cross_scene_traffic_identity(
                source_content, target_content, resolved_path, 'S2', 'S4'
            )

        same_config = self.config('same-scene')
        same_config['command']['network'] = 'S2'
        same_config['world'] = compose_evaluation_world_config(
            source_world, source_content
        )
        reserve_run_output(same_config)
        same_path, _ = resolve_simulator_config(
            same_config, source_content,
            protected_world_fields=SUMO_ENVIRONMENT_IDENTITY_FIELDS,
        )
        identities = validate_cross_scene_traffic_identity(
            source_content, source_content, same_path, 'S2', 'S2'
        )
        self.assertEqual(
            identities['source']['route_sha256'],
            identities['effective']['route_sha256'],
        )


if __name__ == '__main__':
    unittest.main()
