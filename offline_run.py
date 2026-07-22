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

    validate = subparsers.add_parser(
        'validate-dataset', help='Verify a generated offline dataset manifest and shards'
    )
    validate.add_argument('--manifest', required=True)
    validate.add_argument('--skip-shard-hashes', action='store_true')

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
            'seed_role': 'offline_training_seed',
            'offline_training_seed': self.args.seed,
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
        try:
            logger = self.setup_logging(logging.DEBUG if self.args.debug else logging.INFO)
            trainer_class = Registry.mapping['trainer_mapping']['offline_tsc']
            self.trainer = trainer_class(logger)
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
            self.run_state.transition('已完成', exit_code=0)
        except Exception as error:
            if self.run_state.status['status'] in {'已创建', '运行中'}:
                self.run_state.transition('失败', exit_code=1, error=error)
            raise


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.offline_command == 'prepare-plan2':
        path = prepare_plan2_datasets(
            args.run_list, args.output_root, args.dataset_id
        )
        print(f'Plan 2 dataset indexes prepared: {path}')
        return path
    if args.offline_command == 'validate-dataset':
        result = validate_offline_dataset(
            args.manifest, verify_hashes=not args.skip_shard_hashes
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result
    os.environ['CUDA_VISIBLE_DEVICES'] = args.ngpu
    runner = OfflineRunner(args)
    runner.run()
    return runner.output_path


if __name__ == '__main__':
    main()
