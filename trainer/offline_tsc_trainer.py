"""Pure offline training loop with SUMO interaction restricted to evaluation."""

import copy
import hashlib
import json
import os
import random
import tempfile
import time

import numpy as np
import torch

from common.registry import Registry
from dataset.offline_trajectory_dataset import OfflineTrajectoryDataset
from trainer.tsc_trainer import TSCTrainer
from utils.logger import hash_torch_state_dict


OFFLINE_METRIC_SCHEMA_VERSION = 1


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.tmp-', dir=os.path.dirname(path))
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, 'item'):
        return value.item()
    return value


class OfflineMetricLogger:
    REQUIRED_FIELDS = {
        'schema_version', 'record_type', 'algorithm', 'backend', 'network',
        'offline_training_seed', 'dataset_id', 'dataset_kind', 'dataset_stage',
        'training_update', 'gradient_updates', 'target_updates', 'loss',
        'td_loss', 'conservative_loss', 'gradient_norm', 'source_network_counts',
        'travel_time', 'reward_mean', 'reward_sum', 'queue', 'delay',
        'real_delay', 'waiting_time', 'unfinished_vehicles', 'throughput',
        'action_distribution', 'phase_switches', 'phase_switch_frequency',
        'wall_time_seconds',
    }

    def __init__(self, output_path):
        self.path = os.path.join(output_path, 'metrics', 'offline_records.jsonl')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def append(self, record):
        missing = self.REQUIRED_FIELDS - set(record)
        extra = set(record) - self.REQUIRED_FIELDS
        if missing or extra:
            raise ValueError(f'Invalid offline metric fields; missing={missing}, extra={extra}')
        if record['schema_version'] != OFFLINE_METRIC_SCHEMA_VERSION:
            raise ValueError('Invalid offline metric schema version')
        if record['record_type'] not in {'TRAIN', 'EVALUATION', 'FINAL_EVALUATION'}:
            raise ValueError('Invalid offline metric record type')
        line = json.dumps(
            _json_safe(record), ensure_ascii=False, separators=(',', ':'),
            allow_nan=False,
        ) + '\n'
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def validate(self, require_records=True):
        count = 0
        if not os.path.isfile(self.path):
            if require_records:
                raise ValueError(f'Missing offline metric file: {self.path}')
            return 0
        with open(self.path, encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f'Blank offline metric line {line_number}')
                record = json.loads(line)
                missing = self.REQUIRED_FIELDS - set(record)
                extra = set(record) - self.REQUIRED_FIELDS
                if missing or extra:
                    raise ValueError(
                        f'Invalid offline metric line {line_number}; '
                        f'missing={missing}, extra={extra}'
                    )
                count += 1
        if require_records and count == 0:
            raise ValueError('Offline metric file contains no records')
        return count


@Registry.register_trainer('offline_tsc')
class OfflineTSCTrainer(TSCTrainer):
    """Train only from frozen trajectory shards and evaluate in the simulator."""

    def __init__(self, logger, gpu=0, cpu=False, name='offline_tsc'):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        command = Registry.mapping['command_mapping']['setting'].param
        trainer = Registry.mapping['trainer_mapping']['setting'].param
        model = Registry.mapping['model_mapping']['setting'].param

        self.total_updates = int(trainer['total_updates'])
        self.episodes = self.total_updates
        self.batch_size = int(model['batch_size'])
        self.target_update_interval = int(trainer['target_update_interval'])
        self.log_interval = int(trainer.get('log_interval', 100))
        self.evaluation_updates = self._validate_evaluation_updates(
            trainer['evaluation_updates']
        )
        self.offline_training_seed = int(command['seed'])
        self.output_path = Registry.mapping['logger_mapping']['path'].path
        self.offline_dataset = OfflineTrajectoryDataset(
            command['dataset_manifest'], seed=self.offline_training_seed,
            verify_hashes=not command.get('skip_dataset_hash_check', False),
        )
        self.dataset = self.offline_dataset
        self.structured_metrics = OfflineMetricLogger(self.output_path)
        self.current_update = 0
        self.gradient_updates = 0
        self.target_updates = 0
        self.evaluation_results = {}
        self.final_evaluation_completed = False

        if len(self.agents) != 1:
            raise ValueError('Plan 2 currently supports exactly one controlled intersection')
        agent = self.agents[0]
        if self.offline_dataset.observation_dim != agent.ob_length:
            raise ValueError(
                'Offline observation dimension does not match evaluation agent: '
                f'{self.offline_dataset.observation_dim} != {agent.ob_length}'
            )
        if self.offline_dataset.action_dim != agent.action_space.n:
            raise ValueError('Offline action dimension does not match evaluation agent')
        if self.offline_dataset.manifest['evaluation_network'] != command['network']:
            raise ValueError('Dataset evaluation network does not match --network')

        self.dataset_manifest_sha256 = _sha256(self.offline_dataset.manifest_path)
        self.resume_fingerprint = self._build_resume_fingerprint()
        self._write_metadata()
        if command.get('resume'):
            self.load_resumable_checkpoint(command['resume'])

    def _validate_evaluation_updates(self, configured):
        if not isinstance(configured, list) or not all(
            isinstance(value, int) for value in configured
        ):
            raise ValueError('evaluation_updates must be a list of integers')
        if configured != sorted(set(configured)):
            raise ValueError('evaluation_updates must be sorted and unique')
        if not configured or configured[0] != 0 or configured[-1] != self.total_updates:
            raise ValueError('evaluation_updates must include 0 and total_updates')
        return tuple(configured)

    def _build_resume_fingerprint(self):
        agent = self.agents[0]
        payload = {
            'algorithm': agent.offline_algorithm,
            'dataset_manifest_sha256': self.dataset_manifest_sha256,
            'observation_dim': self.offline_dataset.observation_dim,
            'action_dim': self.offline_dataset.action_dim,
            'batch_size': self.batch_size,
            'gamma': agent.gamma,
            'learning_rate': agent.learning_rate,
            'grad_clip': agent.grad_clip,
            'cql_alpha': agent.cql_alpha,
            'target_update_interval': self.target_update_interval,
            'total_updates': self.total_updates,
            'offline_training_seed': self.offline_training_seed,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

    def _write_metadata(self):
        agent = self.agents[0]
        payload = {
            'schema_version': 1,
            'training_mode': 'pure_offline',
            'algorithm': agent.offline_algorithm,
            'backend': agent.backend,
            'offline_training_seed': self.offline_training_seed,
            'behavior_training_seeds': self.offline_dataset.manifest[
                'behavior_training_seeds'
            ],
            'sumo_seed_mode': 'fixed_default',
            'dataset_manifest': self.offline_dataset.manifest_path,
            'dataset_manifest_sha256': self.dataset_manifest_sha256,
            'dataset': self.offline_dataset.manifest,
            'dataset_statistics': self.offline_dataset.statistics(),
            'training_config': {
                'network_hidden_layers': [20, 20],
                'batch_size': self.batch_size,
                'learning_rate': agent.learning_rate,
                'gamma': agent.gamma,
                'grad_clip': agent.grad_clip,
                'optimizer': 'RMSprop',
                'optimizer_alpha': 0.9,
                'optimizer_centered': False,
                'optimizer_eps': 1e-7,
                'target_update_interval': self.target_update_interval,
                'cql_alpha': agent.cql_alpha,
                'td_target_terminal_mask': False,
                'td_target_double_dqn': False,
                'loss': 'full_q_vector_mse',
            },
            'total_updates': self.total_updates,
            'evaluation_updates': list(self.evaluation_updates),
            'resume_fingerprint': self.resume_fingerprint,
            'training_environment_interactions': 0,
            'evaluation_environment_interactions_only': True,
        }
        _atomic_json(os.path.join(self.output_path, 'offline_run_metadata.json'), payload)

    def _base_record(self, record_type, update, wall_time_seconds):
        agent = self.agents[0]
        manifest = self.offline_dataset.manifest
        return {
            'schema_version': OFFLINE_METRIC_SCHEMA_VERSION,
            'record_type': record_type,
            'algorithm': agent.offline_algorithm,
            'backend': agent.backend['name'],
            'network': Registry.mapping['command_mapping']['setting'].param['network'],
            'offline_training_seed': self.offline_training_seed,
            'dataset_id': manifest['dataset_id'],
            'dataset_kind': manifest['dataset_kind'],
            'dataset_stage': manifest['dataset_stage'],
            'training_update': update,
            'gradient_updates': self.gradient_updates,
            'target_updates': self.target_updates,
            'loss': None,
            'td_loss': None,
            'conservative_loss': None,
            'gradient_norm': None,
            'source_network_counts': {},
            'travel_time': None,
            'reward_mean': None,
            'reward_sum': None,
            'queue': None,
            'delay': None,
            'real_delay': None,
            'waiting_time': None,
            'unfinished_vehicles': None,
            'throughput': None,
            'action_distribution': {},
            'phase_switches': None,
            'phase_switch_frequency': None,
            'wall_time_seconds': wall_time_seconds,
        }

    def _write_train_record(self, update, losses, source_counts, wall_time_seconds):
        record = self._base_record('TRAIN', update, wall_time_seconds)
        record.update(losses)
        record['source_network_counts'] = dict(sorted(source_counts.items()))
        return self.structured_metrics.append(record)

    def writeStructuredLog(
        self, record_type, episode, simulation_step, decision_step, loss_mean,
        wall_time_seconds,
    ):
        record = self._base_record(record_type, episode, wall_time_seconds)
        rewards = self.metric.lane_metrics.get('rewards')
        action_total = sum(self.action_counts.values())
        previous_action_count = 0 if self.previous_actions is None else len(self.previous_actions)
        record.update({
            'travel_time': self.metric.real_average_travel_time(),
            'reward_mean': self.metric.rewards(),
            'reward_sum': None if rewards is None else float(np.sum(rewards)),
            'queue': self.metric.queue(),
            'delay': self.metric.delay(),
            'real_delay': self.metric.real_delay(),
            'waiting_time': self.metric.waiting_time(),
            'unfinished_vehicles': self.metric.unfinished_vehicles(),
            'throughput': self.metric.throughput(),
            'action_distribution': self._action_distribution(),
            'phase_switches': self.phase_switches,
            'phase_switch_frequency': (
                0.0 if action_total <= previous_action_count else
                self.phase_switches / (action_total - previous_action_count)
            ),
        })
        return self.structured_metrics.append(record)

    def _checkpoint_path(self, checkpoint_type, update):
        return os.path.join(
            self.output_path, 'checkpoints', checkpoint_type,
            f'update_{update:06d}.pt',
        )

    @staticmethod
    def _atomic_torch_save(payload, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix='.tmp-checkpoint-', suffix='.pt', dir=os.path.dirname(path)
        )
        os.close(descriptor)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        except Exception:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    def save_checkpoint(self, checkpoint_type, update):
        if checkpoint_type not in {'evaluation', 'resumable'}:
            raise ValueError(f'Invalid offline checkpoint type: {checkpoint_type}')
        agent = self.agents[0]
        payload = {
            'schema_version': 1,
            'training_mode': 'pure_offline',
            'checkpoint_type': checkpoint_type,
            'training_update': update,
            'gradient_updates': self.gradient_updates,
            'target_updates': self.target_updates,
            'algorithm': agent.offline_algorithm,
            'backend': agent.backend,
            'dataset_manifest_sha256': self.dataset_manifest_sha256,
            'resume_fingerprint': self.resume_fingerprint,
            'online_model_state_dict': copy.deepcopy(agent.model.state_dict()),
        }
        if checkpoint_type == 'resumable':
            payload.update({
                'target_model_state_dict': copy.deepcopy(agent.target_model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(agent.optimizer.state_dict()),
                'evaluation_results': copy.deepcopy(self.evaluation_results),
                'dataset_rng_state': self.offline_dataset.rng_state(),
                'python_random_state': random.getstate(),
                'numpy_random_state': np.random.get_state(),
                'torch_cpu_rng_state': torch.get_rng_state(),
                'torch_cuda_rng_states': (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            })
        path = self._checkpoint_path(checkpoint_type, update)
        self._atomic_torch_save(payload, path)
        return path

    def load_resumable_checkpoint(self, path):
        payload = torch.load(path, map_location='cpu')
        if payload.get('checkpoint_type') != 'resumable':
            raise ValueError('Offline resume requires a resumable checkpoint')
        if payload.get('resume_fingerprint') != self.resume_fingerprint:
            raise ValueError('Offline checkpoint is incompatible with this run configuration')
        agent = self.agents[0]
        agent.model.load_state_dict(payload['online_model_state_dict'])
        agent.target_model.load_state_dict(payload['target_model_state_dict'])
        agent.optimizer.load_state_dict(payload['optimizer_state_dict'])
        self.current_update = int(payload['training_update'])
        self.global_decision_step = self.current_update
        self.gradient_updates = int(payload['gradient_updates'])
        self.target_updates = int(payload['target_updates'])
        self.evaluation_results = copy.deepcopy(payload.get('evaluation_results', {}))
        self.offline_dataset.set_rng_state(payload['dataset_rng_state'])
        random.setstate(payload['python_random_state'])
        np.random.set_state(payload['numpy_random_state'])
        torch.set_rng_state(payload['torch_cpu_rng_state'])
        if torch.cuda.is_available() and payload['torch_cuda_rng_states']:
            torch.cuda.set_rng_state_all(payload['torch_cuda_rng_states'])
        return payload

    def _write_evaluation_summary(self):
        if not self.evaluation_results:
            return
        ordered = [
            self.evaluation_results[key] for key in sorted(self.evaluation_results)
        ]
        best = min(ordered, key=lambda item: (item['travel_time'], item['training_update']))
        final = self.evaluation_results.get(self.total_updates)
        payload = {
            'schema_version': 1,
            'training_mode': 'pure_offline',
            'selection_metric': 'travel_time',
            'selection_rule': 'minimum_then_earliest_update',
            'evaluation_updates': list(self.evaluation_updates),
            'best_update': best['training_update'],
            'best_travel_time': best['travel_time'],
            'best_checkpoint': os.path.relpath(
                self._checkpoint_path('evaluation', best['training_update']),
                self.output_path,
            ),
            'final_update': None if final is None else final['training_update'],
            'final_travel_time': None if final is None else final['travel_time'],
            'final_checkpoint': None if final is None else os.path.relpath(
                self._checkpoint_path('evaluation', self.total_updates), self.output_path
            ),
            'evaluations': ordered,
        }
        _atomic_json(os.path.join(self.output_path, 'evaluation', 'summary.json'), payload)

    def _evaluate(self, update):
        transition_count = len(self.offline_dataset)
        self.save_checkpoint('evaluation', update)
        self.save_checkpoint('resumable', update)
        record_type = 'FINAL_EVALUATION' if update == self.total_updates else 'EVALUATION'
        self.train_test(update, record_type=record_type)
        if len(self.offline_dataset) != transition_count:
            raise RuntimeError('Evaluation mutated the frozen offline dataset')
        record = dict(self.last_evaluation_record)
        record['online_model_state_hash'] = hash_torch_state_dict(
            self.agents[0].model.state_dict()
        )
        self.evaluation_results[update] = _json_safe(record)
        self._write_evaluation_summary()
        if update == self.total_updates:
            self.final_evaluation_completed = True

    def train(self):
        if self.current_update in self.evaluation_updates:
            self._evaluate(self.current_update)
        interval_started_at = time.perf_counter()
        accumulated = []
        source_counts = {}
        for update in range(self.current_update + 1, self.total_updates + 1):
            batch = self.offline_dataset.sample_batch(self.batch_size)
            losses = self.agents[0].train_offline_batch(batch)
            accumulated.append(losses)
            source_counts[batch.source_network] = source_counts.get(batch.source_network, 0) + 1
            self.current_update = update
            self.global_decision_step = update
            self.gradient_updates += 1
            if update % self.target_update_interval == self.target_update_interval - 1:
                self.agents[0].update_target_network()
                self.target_updates += 1

            if update % self.log_interval == 0 or update in self.evaluation_updates:
                mean_losses = {
                    key: float(np.mean([item[key] for item in accumulated]))
                    for key in ('loss', 'td_loss', 'conservative_loss', 'gradient_norm')
                }
                self._write_train_record(
                    update, mean_losses, source_counts,
                    time.perf_counter() - interval_started_at,
                )
                accumulated = []
                source_counts = {}
                interval_started_at = time.perf_counter()
            if update in self.evaluation_updates:
                self._evaluate(update)

        self.structured_metrics.validate(require_records=True)

    def test(self, drop_load=True):
        if not self.final_evaluation_completed:
            self._evaluate(self.current_update)
        return self.metric
