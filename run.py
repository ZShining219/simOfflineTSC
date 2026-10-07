import task
import trainer
import agent
import agent.offline_dqn  # register Batch-DQN/CQL-DQN evaluators
import dataset
from common.registry import Registry
from common import interface
from common.utils import *
from utils.logger import *
import time
import argparse
import logging
import yaml
import json
import shutil
from pathlib import Path
import torch


# parseargs
parser = argparse.ArgumentParser(description='Run Experiment')
parser.add_argument('--thread_num', type=int, default=4, help='number of threads')  # used in cityflow
parser.add_argument('--ngpu', type=str, default="-1", help='gpu to be used')  # choose gpu card
parser.add_argument('--prefix', type=str, default='test', help="the number of prefix in this running process")
parser.add_argument('--seed', type=int, default=None, help="seed for pytorch backend")
parser.add_argument('--sumo_seed', type=int, default=None,
                    help="explicit SUMO traffic seed (kept separate from training seed)")
parser.add_argument('--debug', type=bool, default=True)
parser.add_argument('--interface', type=str, default="libsumo", choices=['libsumo','traci'], help="interface type") # libsumo(fast) or traci(slow)
parser.add_argument('--delay_type', type=str, default="apx", choices=['apx','real'], help="method of calculating delay") # apx(approximate) or real

parser.add_argument('-t', '--task', type=str, default="tsc", help="task type to run")
parser.add_argument('-a', '--agent', type=str, default="dqn", help="agent type of agents in RL environment")
parser.add_argument('-w', '--world', type=str, default="cityflow", choices=['cityflow','sumo'], help="simulator type")
parser.add_argument('-n', '--network', type=str, default="cityflow1x1", help="network name")
parser.add_argument('-d', '--dataset', type=str, default='onfly', help='type of dataset in training process')
parser.add_argument(
    '--evaluation-manifest', default=None,
    help='Explicit best-checkpoint evaluation collection manifest (SUMO only)',
)
parser.add_argument(
    '--evaluation-output', default=None,
    help='New output directory for the immutable evaluation data package',
)
parser.add_argument(
    '--experiment-config', default=None,
    help='Optional YAML overlay for trainer/model/world experiment settings',
)
parser.add_argument('--resume-output', default=None)
parser.add_argument('--resume-checkpoint', default=None)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

logging_level = logging.INFO
if args.debug:
    logging_level = logging.DEBUG


class Runner:
    def __init__(self, pArgs):
        """
        instantiate runner object with processed config and register config into Registry class
        """
        self.resume = bool(pArgs.resume_output or pArgs.resume_checkpoint)
        if self.resume and not (pArgs.resume_output and pArgs.resume_checkpoint):
            raise ValueError('Resume requires --resume-output and --resume-checkpoint')
        if self.resume:
            self._initialize_resume(pArgs)
            return
        self.config, self.duplicate_config = build_config(pArgs)
        if pArgs.experiment_config:
            with open(pArgs.experiment_config, encoding='utf-8') as handle:
                overlay = yaml.safe_load(handle) or {}
            self.config, overlay_duplicates = merge_dicts(self.config, overlay)
            self.duplicate_config.update(overlay_duplicates)
            # Preserve CLI identity fields after merging a research overlay.
            self.config['command'].update(vars(pArgs))
        self.config_sources = capture_config_sources(self.config)
        if pArgs.experiment_config:
            with open(pArgs.experiment_config, 'rb') as handle:
                self.config_sources['experiment_overlay.yml'] = handle.read()
        self.output_path = reserve_run_output(self.config)
        self.run_state = None
        try:
            self.config_registry()
            self.config_archive_path = archive_run_config(
                self.config,
                self.config_sources,
                Registry.mapping['world_mapping']['setting'].param,
            )
            self.run_state = RunStateManager(self.config, self.config_archive_path)
        except Exception as error:
            RunStateManager.record_initialization_failure(
                self.config, self.output_path, error, exit_code=1
            )
            raise

    @staticmethod
    def _prune_jsonl(path, checkpoint_episode, orphan_dir):
        path = Path(path)
        if not path.is_file():
            return
        kept, removed = [], []
        for line in path.read_text(encoding='utf-8').splitlines(True):
            if not line.strip():
                continue
            record = json.loads(line)
            episode = record.get('episode', record.get('episode_id'))
            (removed if episode is not None and int(episode) > checkpoint_episode
             else kept).append(line)
        if removed:
            orphan_dir.mkdir(parents=True, exist_ok=True)
            (orphan_dir / path.name).write_text(''.join(removed), encoding='utf-8')
            path.write_text(''.join(kept), encoding='utf-8')

    @classmethod
    def _prepare_resume_files(cls, output, checkpoint_episode):
        output = Path(output)
        orphan = output / 'resume_orphans' / f'after_episode_{checkpoint_episode:04d}'
        for directory, pattern in (
                (output / 'history_archive', 'episode_*.pt'),
                (output / 'trajectory' / 'episodes', 'episode_*.npz'),
                (output / 'checkpoints' / 'evaluation', 'episode_*.pt'),
                (output / 'checkpoints' / 'resumable', 'episode_*.pt')):
            if not directory.is_dir():
                continue
            for path in directory.glob(pattern):
                episode = int(path.stem.split('_')[-1])
                if episode > checkpoint_episode:
                    orphan.mkdir(parents=True, exist_ok=True)
                    target = orphan / path.relative_to(output)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(target))
        model_directory = output / 'model'
        if model_directory.is_dir():
            for path in model_directory.glob('*_*.pt'):
                episode = int(path.name.split('_', 1)[0])
                if episode > checkpoint_episode:
                    target = orphan / path.relative_to(output)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(target))
        trajectory_index = output / 'trajectory' / 'index.jsonl'
        cls._prune_jsonl(trajectory_index, checkpoint_episode, orphan / 'trajectory')
        for relative in (
                'metrics/records.jsonl', 'actions.jsonl',
                'intersection_metrics.jsonl', 'time_window_metrics.jsonl'):
            cls._prune_jsonl(output / relative, checkpoint_episode, orphan / 'logs')

    def _initialize_resume(self, pArgs):
        self.output_path = os.path.abspath(pArgs.resume_output)
        checkpoint = os.path.abspath(pArgs.resume_checkpoint)
        output = Path(self.output_path)
        checkpoint_path = Path(checkpoint)
        try:
            checkpoint_path.relative_to(output / 'checkpoints' / 'resumable')
        except ValueError as error:
            raise ValueError('Resume checkpoint must belong to the run output') from error
        payload = torch.load(checkpoint, map_location='cpu')
        if payload.get('checkpoint_type') != 'resumable':
            raise ValueError('Resume requires a resumable checkpoint')
        episode = int(payload['episode'])
        resolved_path = output / 'config' / 'resolved_config.yaml'
        with resolved_path.open(encoding='utf-8') as handle:
            self.config = yaml.safe_load(handle)
        self.config.pop('config_record', None)
        configured_episodes = int(self.config['trainer']['episodes'])
        if not 0 <= episode < configured_episodes:
            raise ValueError('Resume checkpoint episode is outside run budget')
        with (output / 'run_manifest.json').open(encoding='utf-8') as handle:
            run_manifest = json.load(handle)
        if payload.get('config_hash') != run_manifest.get('config_hash'):
            raise ValueError('Resume checkpoint config hash does not match run manifest')
        self.config['command']['output_path'] = self.output_path
        self.config['command']['resume_episode'] = episode
        self.config['command']['resume_checkpoint'] = checkpoint
        self.duplicate_config = {}
        config_dir = output / 'config'
        names = ('base.yml', f"{self.config['command']['agent']}.yml",
                 'simulator_source.cfg', 'experiment_overlay.yml')
        self.config_sources = {
            name: (config_dir / name).read_bytes() for name in names
            if (config_dir / name).is_file()}
        self._prepare_resume_files(output, episode)
        self.run_state = RunStateManager.resume(
            self.output_path, checkpoint, episode)
        self.config_registry()
        self.config_archive_path = str(config_dir)
        self.resume_checkpoint = checkpoint

    def config_registry(self):
        """Register configuration into the project-wide registries."""
        interface.Command_Setting_Interface(self.config)
        interface.Logger_param_Interface(self.config)
        interface.World_param_Interface(
            self.config, self.config_sources['simulator_source.cfg']
        )
        if self.config['model'].get('graphic', False):
            param = Registry.mapping['world_mapping']['setting'].param
            if self.config['command']['world'] in ['cityflow', 'sumo']:
                roadnet_path = param['dir'] + param['roadnetFile']
            else:
                roadnet_path = param['road_file_addr']
            interface.Graph_World_Interface(roadnet_path)
        interface.Logger_path_Interface(self.config)
        os.makedirs(Registry.mapping['logger_mapping']['path'].path, exist_ok=True)
        interface.Trainer_param_Interface(self.config)
        interface.ModelAgent_param_Interface(self.config)

    def run(self):
        try:
            logger = setup_logging(logging_level)
            self.trainer = Registry.mapping['trainer_mapping'][
                Registry.mapping['command_mapping']['setting'].param['task']
            ](logger)
            if self.resume:
                self.trainer.load_resumable_checkpoint(self.resume_checkpoint)
            model_settings = Registry.mapping['model_mapping']['setting'].param
            stage_checkpoint = model_settings.get('stage_checkpoint')
            if stage_checkpoint:
                self.trainer.load_shared_stage_checkpoint(
                    stage_checkpoint,
                    replay_policy=model_settings.get('replay_policy', 'clear'),
                    epsilon_mode=model_settings.get('epsilon_mode', 'reset_schedule'),
                    load_optimizer=model_settings.get('stage_load_optimizer', True),
                )
            if not self.resume:
                self.model_archive_path = archive_runtime_model(
                    self.config_archive_path, self.trainer,
                    Registry.mapping['command_mapping']['setting'].param['agent'])
            self.task = Registry.mapping['task_mapping'][
                Registry.mapping['command_mapping']['setting'].param['task']
            ](self.trainer)
            self.run_state.transition('运行中')
            start_time = time.time()
            self.task.run()
            logger.info(f"Total time taken: {time.time() - start_time}")
            for handler in logger.handlers:
                handler.flush()
            verify_config_archive(self.config_archive_path)
            if hasattr(self.trainer, 'structured_metrics'):
                self.trainer.structured_metrics.validate(require_records=True)
            self.run_state.transition('已完成', exit_code=0)
        except Exception as error:
            if self.run_state.status['status'] in {'已创建', '运行中'}:
                self.run_state.transition('失败', exit_code=1, error=error)
            raise


class EvaluationManifestRunner:
    """Reuse the registered trainer/world stack for manifest-driven reevaluation."""

    def __init__(self, pArgs):
        if not pArgs.evaluation_output:
            raise ValueError('--evaluation-output is required with --evaluation-manifest')
        self.args = pArgs
        _, self.collection = load_evaluation_collection_manifest(
            pArgs.evaluation_manifest
        )
        self.package = EvaluationPackageWriter(
            pArgs.evaluation_output, self.collection
        )

    @staticmethod
    def _source_config(controller):
        path = os.path.join(controller['run_dir'], 'config', 'resolved_config.yaml')
        with open(path, encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        config.pop('config_record', None)
        return config

    @staticmethod
    def _source_snapshots(controller):
        source_config_dir = os.path.join(controller['run_dir'], 'config')
        target_config_dir = os.path.join(
            controller.get('target_run_dir', controller['run_dir']), 'config'
        )
        snapshots = {}
        for name in ('base.yml', f"{controller['agent']}.yml"):
            path = os.path.join(source_config_dir, name)
            with open(path, 'rb') as handle:
                snapshots[name] = handle.read()
        with open(os.path.join(source_config_dir, 'simulator_source.cfg'), 'rb') as handle:
            snapshots['source_simulator_source.cfg'] = handle.read()
        with open(os.path.join(target_config_dir, 'simulator_source.cfg'), 'rb') as handle:
            snapshots['simulator_source.cfg'] = handle.read()
        return snapshots

    @staticmethod
    def _configure_registry(
        config, simulator_source, protected_world_fields=(),
    ):
        interface.Command_Setting_Interface(config)
        interface.Logger_param_Interface(config)
        interface.World_param_Interface(
            config, simulator_source,
            protected_world_fields=protected_world_fields,
        )
        if config['model'].get('graphic', False):
            param = Registry.mapping['world_mapping']['setting'].param
            if config['command']['world'] in ['cityflow', 'sumo']:
                roadnet_path = param['dir'] + param['roadnetFile']
            else:
                roadnet_path = param['road_file_addr']
            interface.Graph_World_Interface(roadnet_path)
        interface.Logger_path_Interface(config)
        os.makedirs(Registry.mapping['logger_mapping']['path'].path, exist_ok=True)
        interface.Trainer_param_Interface(config)
        interface.ModelAgent_param_Interface(config)

    @staticmethod
    def _attempt_logger(attempt_dir, name):
        logger = logging.getLogger(f'evaluation.{name}')
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging_level)
        logger_dir = os.path.join(attempt_dir, 'logger')
        os.makedirs(logger_dir, exist_ok=True)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(
            os.path.join(logger_dir, 'evaluation_BRF.log'), mode='w'
        )
        file_handler.setLevel(logging_level)
        logger.addHandler(file_handler)
        return logger

    def _run_attempt(self, controller, evaluation_seed):
        attempt_dir = self.package.attempt_dir(
            controller['controller_id'], evaluation_seed
        )
        config = self._source_config(controller)
        if controller.get('event_schedule') is not None:
            config['trainer']['event_schedule'] = controller['event_schedule']
            config['trainer'].pop('event_schedule_plan', None)
        command = config['command']
        command.update({
            'agent': controller['agent'],
            'network': controller.get('target_network', controller['network']),
            'seed': controller['training_seed'],
            'sumo_seed': evaluation_seed,
            'prefix': os.path.basename(attempt_dir),
            'output_path': attempt_dir,
            'world': 'sumo',
        })
        if controller['agent'] in {'batch_dqn', 'cql_dqn'}:
            command.update({'task': 'tsc', 'dataset': 'onfly'})
        config['model']['train_model'] = False
        config['model']['test_model'] = False
        config['model']['load_model'] = False
        if controller.get('evaluation_steps') is not None:
            evaluation_steps = int(controller['evaluation_steps'])
            if evaluation_steps <= 0:
                raise ValueError('evaluation_steps must be positive')
            config['trainer']['test_steps'] = evaluation_steps
        config['logger']['save_model'] = False
        snapshots = self._source_snapshots(controller)
        if self.collection['schema_version'] == 2:
            source_signal_config = config['world'].get('signal_config')
            config['world'] = compose_evaluation_world_config(
                config['world'], snapshots['simulator_source.cfg']
            )
            # signal_config is agent-required world metadata (phase_pairs,
            # lane_order, valid_acts) that never lives in simulator_source.cfg;
            # the compose step would otherwise drop it.  Same-network
            # evaluation only: network identity is validated below.
            if source_signal_config is not None:
                config['world']['signal_config'] = source_signal_config
            config['world']['saveReplay'] = False
            protected_world_fields = tuple(
                field for field in config['world']
                if field not in EVALUATION_SOURCE_WORLD_RUNTIME_FIELDS
            )
        else:
            config['world']['saveReplay'] = False
            protected_world_fields = ()
        self._configure_registry(
            config, snapshots['simulator_source.cfg'], protected_world_fields
        )
        traffic_identity = validate_cross_scene_traffic_identity(
            snapshots['source_simulator_source.cfg'],
            snapshots['simulator_source.cfg'],
            Registry.mapping['world_mapping']['setting'].config_path,
            controller.get('source_network', controller['network']),
            controller.get('target_network', controller['network']),
        )
        expected_vehicles = controller.get('expected_vehicle_count')
        if (
            expected_vehicles is not None
            and int(expected_vehicles)
            != traffic_identity['target']['expected_vehicle_count']
        ):
            raise ValueError(
                'Manifest expected_vehicle_count does not match target route: '
                f'{expected_vehicles} != '
                f'{traffic_identity["target"]["expected_vehicle_count"]}'
            )
        config_archive_path = archive_run_config(
            config, snapshots, Registry.mapping['world_mapping']['setting'].param
        )
        attempt_metadata = {
            'schema_version': 1,
            'controller_id': controller['controller_id'],
            'source_scene': controller.get('source_scene'),
            'source_network': controller.get(
                'source_network', controller['network']
            ),
            'target_scene': controller.get('target_scene'),
            'target_network': controller.get(
                'target_network', controller['network']
            ),
            'source_checkpoint_path': controller['checkpoint_path'],
            'source_checkpoint_sha256': controller['checkpoint_sha256'],
            'event_schedule': controller.get('event_schedule'),
            'checkpoint_role': controller.get('checkpoint_role', 'best'),
            'checkpoint_audit': controller.get('checkpoint_audit'),
            'evaluation_traffic_seed': evaluation_seed,
            'target_simulator_config_path': os.path.join(
                controller.get('target_run_dir', controller['run_dir']),
                'config', 'simulator_source.cfg',
            ),
            'effective_simulator_config_path': (
                Registry.mapping['world_mapping']['setting'].config_path
            ),
            'traffic_identity': traffic_identity,
            'runtime': None,
            'isolation_check': None,
        }
        self.package.write_attempt_metadata(attempt_dir, attempt_metadata)
        run_state = RunStateManager(config, config_archive_path)
        logger = self._attempt_logger(attempt_dir, controller['controller_id'])
        trainer = None
        try:
            trainer = Registry.mapping['trainer_mapping'][command['task']](logger)
            attempt_metadata['final_sumo_command'] = validate_sumo_command(
                trainer.world.sumo_cmd, traffic_identity['effective'],
                evaluation_seed,
            )
            self.package.write_attempt_metadata(attempt_dir, attempt_metadata)
            if controller['agent'] in {
                'dqn', 'shared_dqn', 'colight', 'sga_colight',
                'concat_colight', 'sga_flx_colight',
                'mplight', 'sga_mplight',
                'tarl_sensor', 'tarl_gat', 'tarl_concat',
                'tarl_attention', 'tarl_gating', 'tarl_selfattn', 'tarl_crossq',
            }:
                expected_type = (
                    'resumable'
                    if controller.get('checkpoint_role') == 'resumable'
                    else 'evaluation'
                )
                trainer.load_online_checkpoint(
                    controller['checkpoint_path'], expected_type=expected_type,
                )
                audit = controller.get('checkpoint_audit') or {}
                expected_hashes = audit.get('online_model_state_hashes')
                if expected_hashes:
                    loaded_hashes = [
                        hash_torch_state_dict(agent.model.state_dict())
                        for agent in trainer.agents
                    ]
                    if loaded_hashes != expected_hashes:
                        raise ValueError(
                            'Loaded online model hashes differ from '
                            'manifest checkpoint audit'
                        )
            elif controller['agent'] in {'batch_dqn', 'cql_dqn'}:
                payload = torch.load(
                    controller['checkpoint_path'], map_location='cpu'
                )
                if (
                    payload.get('schema_version') != 1
                    or payload.get('checkpoint_type') != 'evaluation'
                    or payload.get('training_mode') != 'pure_offline'
                    or payload.get('algorithm') != controller['agent']
                    or int(payload.get('training_update', -1))
                    != int(controller['checkpoint_episode'])
                ):
                    raise ValueError('Invalid Plan 2 offline checkpoint identity')
                if len(trainer.agents) != 1:
                    raise ValueError('Plan 2 evaluation expects exactly one agent')
                trainer.agents[0].model.load_state_dict(
                    payload['online_model_state_dict']
                )
                actual_model_hash = hash_torch_state_dict(
                    trainer.agents[0].model.state_dict()
                )
                expected_model_hash = controller['checkpoint_audit'][
                    'online_model_state_hash'
                ]
                if actual_model_hash != expected_model_hash:
                    raise ValueError(
                        'Loaded Plan 2 online model hash differs from manifest audit'
                    )
            archive_runtime_model(config_archive_path, trainer, controller['agent'])
            run_state.transition('运行中')
            context = {
                'controller_id': controller['controller_id'],
                'agent': controller['agent'],
                'network': controller['network'],
                'source_network': controller.get(
                    'source_network', controller['network']
                ),
                'target_network': controller.get(
                    'target_network', controller['network']
                ),
                'training_seed': controller['training_seed'],
                'evaluation_seed': evaluation_seed,
                'evaluation_traffic_seed': evaluation_seed,
                'evaluation_schema_version': self.collection['schema_version'],
                'checkpoint_episode': controller['checkpoint_episode'],
                'checkpoint_path': controller['checkpoint'],
                'checkpoint_sha256': controller['checkpoint_sha256'],
                'checkpoint_role': controller.get('checkpoint_role', 'best'),
                'source_policy': controller.get(
                    'source_policy', controller['agent']
                ),
                'reward_definition': (
                    'controller_reward_mean=agent_reward; '
                    'reward_network_mean=negative_queue_intersection_mean'
                ),
                'record_state_diagnostics': bool(
                    self.collection.get('record_state_diagnostics', False)
                ),
                'evaluation_record_type': (
                    'FINAL_CROSS_SCENE_EVALUATION'
                    if controller.get('checkpoint_role') == 'final'
                    else (
                        'RESUMABLE_CHECKPOINT_REEVALUATION'
                        if controller.get('checkpoint_role') == 'resumable'
                        else 'CHECKPOINT_REEVALUATION'
                    )
                ),
                'attempt_output_dir': attempt_dir,
                'training_stage': controller.get('training_stage'),
                'trained_until_scene': controller.get('trained_until_scene'),
                'evaluation_scene': controller.get('evaluation_scene'),
            }
            summary = trainer.evaluate_once(
                context, record_callback=self.package.append_record
            )
            # SUMO writes the final route/vehicle accounting when its owned
            # connection closes; close before validating the runtime log.
            trainer.world.close()
            attempt_metadata['runtime'] = validate_sumo_runtime_evidence(
                os.path.join(attempt_dir, 'sumo.log'),
                traffic_identity['target'], summary,
            )
            attempt_metadata['isolation_check'] = summary['isolation_check']
            self.package.write_attempt_metadata(attempt_dir, attempt_metadata)
            self.package.append_summary(summary)
            trainer.structured_metrics.validate(require_records=True)
            verify_config_archive(config_archive_path)
            run_state.transition('已完成', exit_code=0)
        except Exception as error:
            if run_state.status['status'] in {'已创建', '运行中'}:
                run_state.transition('失败', exit_code=1, error=error)
            raise
        finally:
            if trainer is not None and trainer.world is not None:
                trainer.world.close()
            for handler in logger.handlers:
                handler.flush()
                handler.close()
            logger.handlers.clear()

    def run(self):
        for controller in self.collection['controllers']:
            for evaluation_seed in controller.get(
                'evaluation_seeds', self.collection['evaluation_seeds']
            ):
                self._run_attempt(controller, evaluation_seed)
        self.package.finalize()
        print(f"Evaluation package completed: {self.package.output_dir}")


if __name__ == '__main__':
    if bool(args.evaluation_manifest) != bool(args.evaluation_output):
        parser.error('--evaluation-manifest and --evaluation-output must be used together')
    test = (
        EvaluationManifestRunner(args) if args.evaluation_manifest else Runner(args)
    )
    test.run()
