import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from agent.dqn import DQNNet
from agent.offline_dqn import BatchDQNAgent, CQLDQNAgent
from dataset.offline_trajectory_dataset import (
    OfflineBatch,
    OfflineTrajectoryDataset,
    prepare_plan2_datasets,
    validate_offline_dataset,
)
from tools.experiment_plotting.plan2 import run_plan2_analysis


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_source_run(root, network, seed, episodes=4, per_episode=2):
    run = root / f'p1_formal_{network}_seed{seed}'
    config = run / 'config'
    trajectory = run / 'trajectory'
    shards = trajectory / 'episodes'
    shards.mkdir(parents=True)

    resolved = {
        'command': {
            'task': 'tsc', 'world': 'sumo', 'agent': 'dqn',
            'network': network, 'prefix': run.name, 'seed': seed,
        },
        'world': {
            'combined_file': f'{network}.sumocfg',
            'roadnetFile': f'{network}.net.xml',
            'flowFile': f'{network}.rou.xml',
            'convertroadnetFile': f'{network}.json',
            'convertflowFile': f'{network}.flow.json',
        },
        'trainer': {'episodes': episodes},
        'model': {'gamma': 0.95},
        'config_record': {
            'created_at_utc': '2026-07-22T00:00:00Z',
            'sources': [
                'configs/tsc/base.yml', 'configs/tsc/dqn.yml',
                f'configs/sim/{network}.cfg',
            ],
        },
    }
    config.mkdir()
    with (config / 'resolved_config.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    write_json(config / 'model_resolved.json', {
        'schema_version': 1,
        'agent': 'dqn',
        'reproducibility_probe': {'python_random': [seed]},
        'agents': [{
            'rank': 0, 'action_dim': 2,
            'online_model_state_hash': str(seed),
            'target_model_state_hash': str(seed),
            'model': {'input_dim': 4},
            'target_model': {'input_dim': 4},
        }],
    })
    hashes = {
        path.name: sha256(path) for path in config.iterdir()
    }
    write_json(config / 'config_hashes.json', {'algorithm': 'sha256', 'files': hashes})

    config_hash = sha256(config / 'resolved_config.yaml')
    write_json(run / 'run_manifest.json', {
        'schema_version': 1, 'run_id': run.name, 'task': 'tsc',
        'world': 'sumo', 'agent': 'dqn', 'network': network,
        'prefix': run.name, 'training_seed': seed, 'config_hash': config_hash,
    })
    write_json(run / 'run_status.json', {
        'schema_version': 1, 'run_id': run.name, 'status': '已完成',
        'exit_code': 0,
    })
    index = []
    global_step = 1
    for episode in range(1, episodes + 1):
        shard = shards / f'episode_{episode:04d}.npz'
        values = np.arange(per_episode, dtype=np.float32) + episode * 10
        np.savez_compressed(
            shard,
            state=np.stack((values, values + .5), axis=1).reshape(per_episode, 1, 1, 2),
            current_phase=np.asarray([0, 1], dtype=np.int64).reshape(per_episode, 1, 1),
            action=np.asarray([0, 1], dtype=np.int64).reshape(per_episode, 1, 1),
            reward=(-values).reshape(per_episode, 1, 1),
            next_state=np.stack((values + 1, values + 1.5), axis=1).reshape(per_episode, 1, 1, 2),
            next_phase=np.asarray([1, 0], dtype=np.int64).reshape(per_episode, 1, 1),
            terminated=np.asarray([False, False]),
            truncated=np.asarray([False, True]),
        )
        index.append({
            'schema_version': 1,
            'episode_id': episode,
            'file': f'episodes/{shard.name}',
            'transition_count': per_episode,
            'first_global_step': global_step,
            'last_global_step': global_step + per_episode - 1,
            'sha256': sha256(shard),
        })
        global_step += per_episode
    with (trajectory / 'index.jsonl').open('w', encoding='utf-8') as handle:
        for entry in index:
            handle.write(json.dumps(entry, sort_keys=True) + '\n')
    write_json(trajectory / 'manifest.json', {
        'schema_version': 1, 'network': network,
        'behavior_training_seed': seed, 'sumo_seed_mode': 'fixed_default',
        'expected_decisions_per_episode': per_episode, 'action_dim': 2,
        'config_hash': config_hash,
    })
    write_json(trajectory / 'validation.json', {
        'schema_version': 1, 'valid': True, 'episode_count': episodes,
        'transition_count': episodes * per_episode,
        'evaluation_transition_count': 0,
    })
    return run


class Plan2OfflineDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.networks = ('n1', 'n2', 'n3')
        self.runs = [
            write_source_run(self.root, network, 0)
            for network in self.networks
        ]
        self.run_list = self.root / 'runs.csv'
        with self.run_list.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                'run_path', 'network', 'behavior_training_seed',
            ))
            writer.writeheader()
            for network, run in zip(self.networks, self.runs):
                writer.writerow({
                    'run_path': run, 'network': network,
                    'behavior_training_seed': 0,
                })
        self.dataset_root = Path(prepare_plan2_datasets(
            self.run_list, self.root / 'offline', 'fixture',
            expected_networks=self.networks,
            expected_behavior_seeds=(0,), expected_episodes=4,
            stages={'Q1': (1, 2), 'Q4': (3, 4), 'full': (1, 4)},
        ))

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_stage_split_features_and_last_transitions_are_retained(self):
        q1 = self.dataset_root / 'datasets' / 'n1' / 'Q1' / 'manifest.json'
        validation = validate_offline_dataset(q1)
        self.assertEqual(4, validation['transition_count'])
        dataset = OfflineTrajectoryDataset(q1, seed=1000)
        self.assertEqual(4, len(dataset))
        self.assertEqual(4, dataset.observation_dim)
        arrays = dataset._arrays['n1']
        self.assertEqual(4, len(arrays['actions']))
        np.testing.assert_array_equal(arrays['observations'][-1, -2:], [0, 1])
        np.testing.assert_array_equal(arrays['next_observations'][-1, -2:], [1, 0])

    def test_leave_one_out_rejects_leakage_and_sampling_is_deterministic(self):
        manifest = (
            self.dataset_root / 'datasets' / 'leave_one_out' / 'n1' / 'manifest.json'
        )
        first = OfflineTrajectoryDataset(manifest, seed=1001)
        second = OfflineTrajectoryDataset(manifest, seed=1001)
        sources_first = [first.sample_batch(3).source_network for _ in range(20)]
        sources_second = [second.sample_batch(3).source_network for _ in range(20)]
        self.assertEqual(sources_first, sources_second)
        self.assertEqual({'n2', 'n3'}, set(sources_first))

        payload = json.loads(manifest.read_text(encoding='utf-8'))
        payload['source_networks'].append('n1')
        payload['expected_transition_count'] += 8
        write_json(manifest, payload)
        with self.assertRaisesRegex(ValueError, 'target-network'):
            validate_offline_dataset(manifest)

    def test_corrupt_source_shard_is_rejected(self):
        manifest = self.dataset_root / 'datasets' / 'n2' / 'full' / 'manifest.json'
        shard = self.runs[1] / 'trajectory' / 'episodes' / 'episode_0001.npz'
        shard.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            validate_offline_dataset(manifest)


class OfflineDQNLossTest(unittest.TestCase):
    def _agent(self, cls, alpha):
        agent = cls.__new__(cls)
        agent.model = DQNNet(2, 2)
        agent.target_model = DQNNet(2, 2)
        for model in (agent.model, agent.target_model):
            for parameter in model.parameters():
                torch.nn.init.zeros_(parameter)
        agent.gamma = .95
        agent.grad_clip = 5.0
        agent.cql_alpha = alpha
        agent.criterion = torch.nn.MSELoss(reduction='mean')
        agent.optimizer = torch.optim.RMSprop(
            agent.model.parameters(), lr=0.0, alpha=.9, centered=False, eps=1e-7
        )
        return agent

    def test_batch_and_cql_differ_only_by_conservative_penalty(self):
        batch = OfflineBatch(
            observations=np.zeros((2, 2), dtype=np.float32),
            actions=np.asarray([0, 1]),
            rewards=np.ones(2, dtype=np.float32),
            next_observations=np.zeros((2, 2), dtype=np.float32),
            source_network='n1',
        )
        batch_result = self._agent(BatchDQNAgent, 0.0).train_offline_batch(batch)
        cql_result = self._agent(CQLDQNAgent, 1.0).train_offline_batch(batch)
        self.assertAlmostEqual(.5, batch_result['td_loss'], places=6)
        self.assertAlmostEqual(batch_result['td_loss'], batch_result['loss'], places=6)
        self.assertAlmostEqual(np.log(2), cql_result['conservative_loss'], places=6)
        self.assertAlmostEqual(
            batch_result['td_loss'] + np.log(2), cql_result['loss'], places=6
        )


class Plan2AnalysisTest(unittest.TestCase):
    def test_incomplete_smoke_run_can_be_validated_and_aggregated_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'run'
            config = run / 'config'
            metrics = run / 'metrics'
            evaluation = run / 'evaluation'
            config.mkdir(parents=True)
            metrics.mkdir()
            evaluation.mkdir()
            dataset_manifest = root / 'dataset_manifest.json'
            write_json(dataset_manifest, {
                'schema_version': 1, 'dataset_id': 'fixture',
                'dataset_kind': 'single_scene', 'dataset_stage': 'Q1',
                'evaluation_network': 'sumohz1x1',
                'source_networks': ['sumohz1x1'],
            })
            resolved = config / 'resolved_config.yaml'
            resolved.write_text('command:\n  task: offline_tsc\n', encoding='utf-8')
            write_json(config / 'config_hashes.json', {
                'algorithm': 'sha256',
                'files': {'resolved_config.yaml': sha256(resolved)},
            })
            config_hash = sha256(resolved)
            write_json(run / 'run_manifest.json', {
                'schema_version': 1, 'agent': 'batch_dqn',
                'training_mode': 'pure_offline', 'dataset_id': 'fixture',
                'dataset_kind': 'single_scene', 'dataset_stage': 'Q1',
                'evaluation_network': 'sumohz1x1',
                'offline_training_seed': 1000, 'config_hash': config_hash,
            })
            write_json(run / 'run_status.json', {
                'schema_version': 1, 'status': '已完成', 'exit_code': 0,
            })
            fingerprint = 'f' * 64
            dataset_hash = sha256(dataset_manifest)
            write_json(run / 'offline_run_metadata.json', {
                'training_environment_interactions': 0,
                'total_updates': 1, 'evaluation_updates': [0, 1],
                'behavior_training_seeds': [0],
                'dataset_manifest': str(dataset_manifest),
                'dataset_manifest_sha256': dataset_hash,
                'resume_fingerprint': fingerprint,
                'backend': {'name': 'native'},
                'dataset': {
                    'evaluation_network': 'sumohz1x1',
                    'source_networks': ['sumohz1x1'],
                },
                'dataset_statistics': {
                    'sumohz1x1': {
                        'transition_count': 2, 'reward_mean': -1.0,
                        'reward_std': 0.0, 'action_counts': {'0': 2},
                    },
                },
            })
            records = [
                {
                    'schema_version': 1, 'record_type': 'EVALUATION',
                    'algorithm': 'batch_dqn', 'network': 'sumohz1x1',
                    'offline_training_seed': 1000, 'dataset_id': 'fixture',
                    'dataset_kind': 'single_scene', 'dataset_stage': 'Q1',
                    'training_update': 0, 'travel_time': 10.0,
                },
                {
                    'schema_version': 1, 'record_type': 'TRAIN',
                    'algorithm': 'batch_dqn', 'network': 'sumohz1x1',
                    'offline_training_seed': 1000, 'dataset_id': 'fixture',
                    'dataset_kind': 'single_scene', 'dataset_stage': 'Q1',
                    'training_update': 1, 'loss': 1.0,
                },
                {
                    'schema_version': 1, 'record_type': 'FINAL_EVALUATION',
                    'algorithm': 'batch_dqn', 'network': 'sumohz1x1',
                    'offline_training_seed': 1000, 'dataset_id': 'fixture',
                    'dataset_kind': 'single_scene', 'dataset_stage': 'Q1',
                    'training_update': 1, 'travel_time': 9.0,
                },
            ]
            with (metrics / 'offline_records.jsonl').open('w', encoding='utf-8') as handle:
                for record in records:
                    handle.write(json.dumps(record) + '\n')
            write_json(evaluation / 'summary.json', {
                'evaluation_updates': [0, 1], 'best_update': 1,
                'best_travel_time': 9.0, 'final_update': 1,
                'final_travel_time': 9.0,
            })
            state = {'weight': torch.zeros(1)}
            for update in (0, 1):
                for checkpoint_type in ('evaluation', 'resumable'):
                    path = run / 'checkpoints' / checkpoint_type / f'update_{update:06d}.pt'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        'schema_version': 1, 'training_mode': 'pure_offline',
                        'checkpoint_type': checkpoint_type,
                        'training_update': update, 'algorithm': 'batch_dqn',
                        'dataset_manifest_sha256': dataset_hash,
                        'resume_fingerprint': fingerprint,
                        'online_model_state_dict': state,
                    }
                    if checkpoint_type == 'resumable':
                        payload.update({
                            'target_model_state_dict': state,
                            'optimizer_state_dict': {}, 'dataset_rng_state': (),
                            'python_random_state': (), 'numpy_random_state': (),
                            'torch_cpu_rng_state': torch.get_rng_state(),
                            'torch_cuda_rng_states': [],
                        })
                    torch.save(payload, path)
            run_list = root / 'runs.csv'
            with run_list.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    'algorithm', 'dataset_id', 'dataset_kind', 'dataset_stage',
                    'evaluation_network', 'offline_training_seed', 'run_dir', 'include',
                ))
                writer.writeheader()
                writer.writerow({
                    'algorithm': 'batch_dqn', 'dataset_id': 'fixture',
                    'dataset_kind': 'single_scene', 'dataset_stage': 'Q1',
                    'evaluation_network': 'sumohz1x1',
                    'offline_training_seed': 1000, 'run_dir': run,
                    'include': 'true',
                })
            output = run_plan2_analysis(SimpleNamespace(
                run_list=str(run_list), analysis_id='fixture_analysis',
                output_root=str(root / 'analysis'), allow_incomplete=True,
            ))
            self.assertTrue((output / 'tables' / 'run_summary.csv').is_file())
            self.assertTrue((output / 'tables' / 'offline_training_curves.csv').is_file())
            self.assertTrue((output / 'plan2_report.md').is_file())


if __name__ == '__main__':
    unittest.main()
