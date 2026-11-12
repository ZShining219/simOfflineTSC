import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

from semi_offline.offline.agent import BatchDQNAgent, CQLDQNAgent
from common.registry import Registry
from semi_offline.offline.dataset import (
    OfflineBatch, _reward_transform, _source_reward_descriptor,
)
from dev.hz1x1_offline.prepare import build_specs
from semi_offline.offline import build_parser
from semi_offline.offline.trainer import source_network_counts
from utils.logger import build_config, merge_dicts
from world.world_sumo import World


ROOT = Path(__file__).resolve().parents[1]


class Hz1x1OfflineAvailabilityTest(unittest.TestCase):
    def setUp(self):
        self.overlay = yaml.safe_load((ROOT / 'configs/offline_tsc/hz1x1_v1.yml').read_text())

    def config(self, algorithm):
        args = build_parser().parse_args([
            'train', '-a', algorithm, '-n', 'sumohz1x1', '--seed', '1000',
            '--prefix', 'unit_only', '--dataset-manifest', 'not_opened.json',
            '--backend', 'native', '--experiment-config', 'configs/offline_tsc/hz1x1_v1.yml',
        ])
        base, _ = build_config(args)
        config, _ = merge_dicts(base, self.overlay)
        config['model']['offline_backend'] = args.backend
        return config

    def test_matrix_and_merged_protocol(self):
        cfg = self.overlay['offline_campaign']
        specs = build_specs(cfg)
        self.assertEqual(27, len(specs))
        self.assertEqual(162, len(specs) * len(cfg['algorithms']) * len(cfg['offline_seeds']))
        counts = {'e0': 72000, 'e1': 144000, 'e2frap': 144000, 'e2sac': 144000,
                  'e2mplight': 144000, 'e3': 288000, 'e4': 216000,
                  'e4loo': 216000, 'e5': 288000}
        expected_e5_cells = [
            {'network': cfg['scenes'][scene]['network'], 'arm': arm}
            for scene, arm in [('A', 'd3qn'), ('B', 'frap'), ('C', 'sac'), ('D', 'mplight')]
        ]
        for spec in specs:
            condition = spec['name'].rsplit('_p', 1)[0]
            count = len(spec['source_cells']) * len(spec['behavior_training_seeds']) * 72000
            self.assertEqual(counts[condition], count)
            self.assertEqual('uniform_transition', spec['sampling_strategy'])
            if condition == 'e5':
                self.assertEqual(expected_e5_cells, spec['source_cells'])
            self.assertNotIn('presslight', {cell['arm'] for cell in spec['source_cells']})
            if condition == 'e4loo':
                self.assertNotIn(spec['evaluation_network'], spec['source_networks'])
        for algorithm, alpha in (('batch_dqn', 0.), ('cql_dqn', 1.)):
            config = self.config(algorithm)
            self.assertEqual(144000, config['trainer']['total_updates'])
            self.assertEqual(list(range(0, 144001, 1800)), config['trainer']['evaluation_updates'])
            self.assertEqual(64, config['model']['batch_size'])
            self.assertEqual(alpha, config['model']['cql_alpha'])
            self.assertEqual(1., config['model']['offline_reward_scale'])
            self.assertEqual('native', config['model']['offline_backend'])

    def test_mplight_reward_unit_is_identified_without_rescaling(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'config').mkdir()
            (run / 'run_manifest.json').write_text('{"agent":"mplight"}')
            (run / 'config/resolved_config.yaml').write_text('model: {}\n')
            descriptor = _source_reward_descriptor(str(run))
        self.assertEqual('negative_incoming_waiting_road_mean_v1', descriptor['semantic_id'])
        self.assertEqual(1.0, descriptor['scale'])
        transform = _reward_transform(descriptor, 'mplight', {
            'semantic_id': descriptor['semantic_id'],
            'source_scales': {'mplight': 1.0}, 'target_scale': 1.0,
        })
        self.assertEqual(1.0, transform['multiplier'])
        self.assertFalse(transform['raw_shards_modified'])

    def test_real_agent_construction_reward_greedy_action_and_optimizer_without_simulation(self):
        # Actual SUMO agent/generators, with a frozen world fixture and no engine.
        world = World.__new__(World)
        roads = ['n', 'e', 's', 'w']
        lanes = {road: [f'{road}_0', f'{road}_1'] for road in roads}
        intersection = SimpleNamespace(id='i', in_roads=roads, roads=roads,
                                       road_lane_mapping=lanes, current_phase=0, phases=list(range(8)))
        world.RIGHT = True
        world.intersection_ids = ['i']
        world.id2intersection = {'i': intersection}
        world.subscribe = lambda _: None
        world.get_info = lambda _: {lane: 2. for group in lanes.values() for lane in group}
        world.step = lambda _: self.fail('offline optimizer must not step a world')
        for cls in (BatchDQNAgent, CQLDQNAgent):
            config = self.config(cls.offline_algorithm)
            with patch.dict(Registry.mapping['command_mapping'], {'setting': SimpleNamespace(param=config['command'])}), \
                 patch.dict(Registry.mapping['trainer_mapping'], {'setting': SimpleNamespace(param=config['trainer'])}), \
                 patch.dict(Registry.mapping['model_mapping'], {'setting': SimpleNamespace(param=config['model'])}):
                torch.manual_seed(1000)
                agent = cls(world, 0)
                self.assertEqual(16, agent.ob_length)
                self.assertEqual(8, agent.action_space.n)
                self.assertEqual(-2., float(agent.get_reward()))
                action = agent.get_action(agent.get_ob(), agent.get_phase(), test=True)
                np.testing.assert_array_equal(action, agent.get_action(agent.get_ob(), agent.get_phase(), test=True))
                batch = OfflineBatch(
                    observations=np.ones((64, 16), np.float32),
                    next_observations=np.ones((64, 16), np.float32) * 2,
                    actions=np.arange(64) % 8, rewards=-np.ones(64, np.float32),
                    terminated=np.zeros(64, bool), truncated=np.arange(64) % 5 == 0,
                    source_network='mixed',
                )
                before = [parameter.detach().clone() for parameter in agent.model.parameters()]
                result = agent.train_offline_batch(batch)
                self.assertTrue(all(np.isfinite(value) for value in result.values()))
                self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, agent.model.parameters())))
                self.assertEqual(0, len(agent.replay_buffer))

    def test_training_metric_counts_mixed_transition_provenance_not_batches(self):
        batch = OfflineBatch(
            observations=np.zeros((5, 1)), actions=np.zeros(5, dtype=np.int64),
            rewards=np.zeros(5), next_observations=np.zeros((5, 1)),
            terminated=np.zeros(5, bool), truncated=np.zeros(5, bool),
            source_network='mixed', source_networks=np.asarray(['A', 'A', 'B', 'C', 'C']),
        )
        self.assertEqual({'A': 2, 'B': 1, 'C': 2}, source_network_counts(batch))
        batch.source_networks = None
        batch.source_network = 'A'
        self.assertEqual({'A': 5}, source_network_counts(batch))


if __name__ == '__main__':
    unittest.main()
