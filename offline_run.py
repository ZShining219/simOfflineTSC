"""Separate command-line entry for Plan 2 pure offline training.

This module intentionally does not import simulator, agent, or trainer modules
for dataset preparation/validation commands.  The existing ``run.py`` online
entry and its import graph remain unchanged.
"""

import argparse
import json
import logging
import os
import re
import tempfile
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor

from dataset.offline_trajectory_dataset import (
    prepare_plan2_datasets,
    validate_offline_dataset,
)


def _identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value):
        raise argparse.ArgumentTypeError(
            'value must contain only letters, digits, dot, underscore and hyphen'
        )
    return value


def _atomic_json(path, payload):
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


def _percentage_schedule(total_updates):
    return sorted(set(int(total_updates * fraction) for fraction in (0, .1, .25, .5, .75, 1)))


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _benchmark_dataset_worker(arguments):
    from dataset.offline_trajectory_dataset import OfflineTrajectoryDataset
    manifest, source_root, seed, batch_size, sample_batches = arguments
    dataset = OfflineTrajectoryDataset(
        manifest, seed=seed, source_root=source_root
    )
    return dataset.performance_profile(
        batch_size=batch_size, sample_batches=sample_batches
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description='Plan 2 pure Offline DQN entry (separate from run.py Online TSC)'
    )
    subparsers = parser.add_subparsers(dest='offline_command', required=True)

    prepare = subparsers.add_parser(
        'prepare-plan2', help='Build read-only Plan 2 indexes over approved Plan 1 runs'
    )
    prepare.add_argument('--run-list', required=True)
    prepare.add_argument('--dataset-id', required=True, type=_identifier)
    prepare.add_argument(
        '--output-root', default='data/output_data/offline_datasets/plan2'
    )
    prepare.add_argument('--source-root', default=None)
    prepare.add_argument('--source-root-id', default='plan1_formal_root', type=_identifier)

    validate = subparsers.add_parser(
        'validate-dataset', help='Verify a generated offline dataset manifest and shards'
    )
    validate.add_argument('--manifest', required=True)
    validate.add_argument('--skip-shard-hashes', action='store_true')
    validate.add_argument('--source-root', default=None)

    benchmark = subparsers.add_parser(
        'benchmark-dataset', help='Benchmark one-time NPZ loading and in-memory sampling'
    )
    benchmark.add_argument('--manifest', required=True)
    benchmark.add_argument('--source-root', default=None)
    benchmark.add_argument('--seed', type=int, default=0)
    benchmark.add_argument('--batch-size', type=int, default=64)
    benchmark.add_argument('--sample-batches', type=int, default=1000)
    benchmark.add_argument('--workers', type=int, default=1)

    train = subparsers.add_parser(
        'train', help='Train Batch-DQN or CQL-DQN without training-time simulation'
    )
    train.add_argument('-a', '--agent', choices=('batch_dqn', 'cql_dqn'), required=True)
    train.add_argument('-w', '--world', choices=('sumo',), default='sumo')
    train.add_argument('-n', '--network', required=True)
    train.add_argument('--dataset-manifest', required=True)
    train.add_argument('--prefix', required=True, type=_identifier)
    train.add_argument(
        '--seed', type=int, required=True,
        help='offline_training_seed (Plan 2 formal values: 1000..1004)',
    )
    train.add_argument('--model-seed', type=int, default=None)
    train.add_argument('--dataset-sampler-seed', type=int, default=None)
    train.add_argument('--evaluation-seed', type=int, default=None)
    train.add_argument('--sumo-seed', type=int, default=None)
    train.add_argument('--source-root', default=None)
    train.add_argument('--backend', choices=('d3rlpy', 'native', 'auto'), default='d3rlpy')
    train.add_argument('--resume', default=None, help='Resumable offline checkpoint path')
    train.add_argument('--thread_num', type=int, default=4)
    train.add_argument('--ngpu', default='-1')
    train.add_argument('--interface', choices=('libsumo', 'traci'), default='libsumo')
    train.add_argument('--delay_type', choices=('apx', 'real'), default='apx')
    train.add_argument('--debug', action='store_true')
    train.add_argument('--skip-dataset-hash-check', action='store_true')
    train.add_argument(
        '--total-updates', type=int, default=None,
        help='Override 144000 only for development smoke tests',
    )
    train.add_argument('--log-interval', type=int, default=None)
    train.add_argument('--evaluation-timeout-seconds', type=int, default=None)
    train.add_argument('--evaluation-retries', type=int, default=None)
    train.set_defaults(task='offline_tsc', dataset='offline_readonly')
    return parser


class OfflineRunner:
    def __init__(self, args):
        # Imports are delayed so preparation and validation never initialize SUMO.
        import agent  # noqa: F401
        import dataset  # noqa: F401
        import task  # noqa: F401
        import trainer  # noqa: F401
        import world  # noqa: F401
        import agent.offline_dqn  # noqa: F401
        import task.offline_tsc_task  # noqa: F401
        import trainer.offline_tsc_trainer  # noqa: F401
        from common import interface
        from common.registry import Registry
        from utils.logger import (
            RunStateManager, archive_run_config, archive_runtime_model,
            build_config, capture_config_sources, reserve_run_output,
            setup_logging, verify_config_archive,
        )

        self.Registry = Registry
        self.interface = interface
        self.RunStateManager = RunStateManager
        self.archive_runtime_model = archive_runtime_model
        self.setup_logging = setup_logging
        self.verify_config_archive = verify_config_archive
        self.args = args
        self.config, self.duplicate_config = build_config(args)
        command = self.config['command']
        command['model_seed'] = args.seed if args.model_seed is None else args.model_seed
        command['dataset_sampler_seed'] = (
            args.seed + 10000 if args.dataset_sampler_seed is None
            else args.dataset_sampler_seed
        )
        command['evaluation_seed'] = (
            args.seed + 20000 if args.evaluation_seed is None else args.evaluation_seed
        )
        self.config['model']['offline_backend'] = args.backend
        if args.total_updates is not None:
            if args.total_updates <= 0:
                raise ValueError('--total-updates must be positive')
            self.config['trainer']['total_updates'] = args.total_updates
            self.config['trainer']['episodes'] = args.total_updates
            self.config['trainer']['evaluation_updates'] = _percentage_schedule(
                args.total_updates
            )
        if args.log_interval is not None:
            if args.log_interval <= 0:
                raise ValueError('--log-interval must be positive')
            self.config['trainer']['log_interval'] = args.log_interval
        if args.evaluation_timeout_seconds is not None:
            if args.evaluation_timeout_seconds <= 0:
                raise ValueError('--evaluation-timeout-seconds must be positive')
            self.config['trainer']['evaluation_timeout_seconds'] = args.evaluation_timeout_seconds
        if args.evaluation_retries is not None:
            if args.evaluation_retries < 0:
                raise ValueError('--evaluation-retries must be non-negative')
            self.config['trainer']['evaluation_retries'] = args.evaluation_retries
        self.config_sources = capture_config_sources(self.config)
        self.output_path = reserve_run_output(self.config)
        self.run_state = None
        try:
            self._config_registry()
            self.config_archive_path = archive_run_config(
                self.config, self.config_sources,
                Registry.mapping['world_mapping']['setting'].param,
            )
            self.run_state = RunStateManager(self.config, self.config_archive_path)
            self._augment_manifest()
        except Exception as error:
            if self.run_state is None:
                RunStateManager.record_initialization_failure(
                    self.config, self.output_path, error, exit_code=1
                )
            elif self.run_state.status['status'] == '已创建':
                self.run_state.transition('失败', exit_code=1, error=error)
            raise

    def _config_registry(self):
        Registry = self.Registry
        interface = self.interface
        interface.Command_Setting_Interface(self.config)
        interface.Logger_param_Interface(self.config)
        interface.World_param_Interface(
            self.config, self.config_sources['simulator_source.cfg']
        )
        interface.Logger_path_Interface(self.config)
        interface.Trainer_param_Interface(self.config)
        interface.ModelAgent_param_Interface(self.config)
        os.makedirs(Registry.mapping['logger_mapping']['path'].path, exist_ok=True)

    def _augment_manifest(self):
        path = os.path.join(self.output_path, 'run_manifest.json')
        with open(path, encoding='utf-8') as handle:
            manifest = json.load(handle)
        with open(self.args.dataset_manifest, encoding='utf-8') as handle:
            dataset_manifest = json.load(handle)
        manifest.update({
            'training_mode': 'pure_offline',
            'created_by_commit': _git_commit(),
            'seed_role': 'offline_training_seed',
            'offline_training_seed': self.args.seed,
            'seed_roles': {
                'model_init_seed': self.config['command']['model_seed'],
                'dataset_sampler_seed': self.config['command']['dataset_sampler_seed'],
                'evaluation_seed': self.config['command']['evaluation_seed'],
                'sumo_seed': self.args.sumo_seed,
            },
            'behavior_training_seeds': dataset_manifest['behavior_training_seeds'],
            'dataset_id': dataset_manifest['dataset_id'],
            'dataset_kind': dataset_manifest['dataset_kind'],
            'dataset_stage': dataset_manifest['dataset_stage'],
            'evaluation_network': dataset_manifest['evaluation_network'],
        })
        _atomic_json(path, manifest)
        self.run_state.manifest = manifest

    def run(self):
        Registry = self.Registry
        self.trainer = None
        try:
            logger = self.setup_logging(logging.DEBUG if self.args.debug else logging.INFO)
            trainer_class = Registry.mapping['trainer_mapping']['offline_tsc']
            # Keep the partially constructed trainer reachable so finally can close
            # a SUMO world even if agent/metric/dataset initialization later fails.
            self.trainer = trainer_class.__new__(trainer_class)
            trainer_class.__init__(self.trainer, logger)
            self.archive_runtime_model(
                self.config_archive_path, self.trainer, self.args.agent
            )
            task_class = Registry.mapping['task_mapping']['offline_tsc']
            self.task = task_class(self.trainer)
            self.run_state.transition('运行中')
            started = time.time()
            self.task.run()
            logger.info('Offline total time taken: %s', time.time() - started)
            for handler in logger.handlers:
                handler.flush()
            self.verify_config_archive(self.config_archive_path)
            self.trainer.structured_metrics.validate(require_records=True)
            close_report = self.trainer.world.close()
            if close_report.get('close_error') or close_report.get('process_alive'):
                raise RuntimeError(f'SUMO final cleanup failed: {close_report}')
            self.run_state.transition('已完成', exit_code=0)
        except Exception as error:
            if self.run_state.status['status'] in {'已创建', '运行中'}:
                self.run_state.transition('失败', exit_code=1, error=error)
            raise
        finally:
            world = None if self.trainer is None else getattr(self.trainer, 'world', None)
            if world is not None and hasattr(world, 'close'):
                world.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.offline_command == 'prepare-plan2':
        path = prepare_plan2_datasets(
            args.run_list, args.output_root, args.dataset_id,
            source_root=args.source_root, source_root_id=args.source_root_id,
        )
        print(f'Plan 2 dataset indexes prepared: {path}')
        return path
    if args.offline_command == 'validate-dataset':
        result = validate_offline_dataset(
            args.manifest, verify_hashes=not args.skip_shard_hashes,
            source_root=args.source_root,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result
    if args.offline_command == 'benchmark-dataset':
        if args.workers <= 0:
            raise ValueError('--workers must be positive')
        started = time.perf_counter()
        arguments = [
            (
                args.manifest, args.source_root, args.seed + worker,
                args.batch_size, args.sample_batches,
            )
            for worker in range(args.workers)
        ]
        if args.workers == 1:
            profiles = [_benchmark_dataset_worker(arguments[0])]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                profiles = list(executor.map(_benchmark_dataset_worker, arguments))
        wall_seconds = time.perf_counter() - started
        total_batches = args.workers * args.sample_batches
        result = {
            'workers': args.workers,
            'wall_seconds': wall_seconds,
            'total_sample_batches': total_batches,
            'aggregate_sample_batches_per_second_including_load': (
                total_batches / wall_seconds if wall_seconds else None
            ),
            'aggregate_sampling_only_batches_per_second': sum(
                profile['sample_batches_per_second'] for profile in profiles
            ),
            'max_dataset_load_seconds': max(
                profile['dataset_load_seconds'] for profile in profiles
            ),
            'sum_resident_array_bytes': sum(
                profile['resident_array_bytes'] for profile in profiles
            ),
            'sum_peak_rss_delta_bytes': sum(
                profile['peak_rss_delta_bytes'] for profile in profiles
            ),
            'storage_strategies': sorted({
                profile['storage_strategy'] for profile in profiles
            }),
            'source_shards_open_after_load': max(
                profile['source_shards_open_after_load'] for profile in profiles
            ),
            'worker_profiles': profiles,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result
    os.environ['CUDA_VISIBLE_DEVICES'] = args.ngpu
    runner = OfflineRunner(args)
    runner.run()
    return runner.output_path


if __name__ == '__main__':
    main()
