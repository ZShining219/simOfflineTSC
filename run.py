import task
import trainer
import agent
import dataset
from common.registry import Registry
from common import interface
from common.utils import *
from utils.logger import *
import time
import argparse
import logging
import yaml


# parseargs
parser = argparse.ArgumentParser(description='Run Experiment')
parser.add_argument('--thread_num', type=int, default=4, help='number of threads')  # used in cityflow
parser.add_argument('--ngpu', type=str, default="-1", help='gpu to be used')  # choose gpu card
parser.add_argument('--prefix', type=str, default='test', help="the number of prefix in this running process")
parser.add_argument('--seed', type=int, default=None, help="seed for pytorch backend")
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
        self.config, self.duplicate_config = build_config(pArgs)
        self.config_sources = capture_config_sources(self.config)
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
            self.model_archive_path = archive_runtime_model(
                self.config_archive_path,
                self.trainer,
                Registry.mapping['command_mapping']['setting'].param['agent'],
            )
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
        config_dir = os.path.join(controller['run_dir'], 'config')
        names = ('base.yml', f"{controller['agent']}.yml", 'simulator_source.cfg')
        snapshots = {}
        for name in names:
            path = os.path.join(config_dir, name)
            with open(path, 'rb') as handle:
                snapshots[name] = handle.read()
        return snapshots

    @staticmethod
    def _configure_registry(config, simulator_source):
        interface.Command_Setting_Interface(config)
        interface.Logger_param_Interface(config)
        interface.World_param_Interface(config, simulator_source)
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
        command = config['command']
        command.update({
            'agent': controller['agent'],
            'network': controller['network'],
            'seed': controller['training_seed'],
            'sumo_seed': evaluation_seed,
            'prefix': os.path.basename(attempt_dir),
            'output_path': attempt_dir,
            'world': 'sumo',
        })
        config['model']['train_model'] = False
        config['model']['test_model'] = False
        config['model']['load_model'] = False
        config['world']['saveReplay'] = False
        config['logger']['save_model'] = False
        snapshots = self._source_snapshots(controller)
        self._configure_registry(config, snapshots['simulator_source.cfg'])
        config_archive_path = archive_run_config(
            config, snapshots, Registry.mapping['world_mapping']['setting'].param
        )
        run_state = RunStateManager(config, config_archive_path)
        logger = self._attempt_logger(attempt_dir, controller['controller_id'])
        trainer = None
        try:
            trainer = Registry.mapping['trainer_mapping'][command['task']](logger)
            if controller['agent'] == 'dqn':
                trainer.load_evaluation_checkpoint(controller['checkpoint_path'])
            archive_runtime_model(config_archive_path, trainer, controller['agent'])
            run_state.transition('运行中')
            context = {
                'controller_id': controller['controller_id'],
                'agent': controller['agent'],
                'network': controller['network'],
                'training_seed': controller['training_seed'],
                'evaluation_seed': evaluation_seed,
                'checkpoint_episode': controller['checkpoint_episode'],
                'checkpoint_path': controller['checkpoint'],
                'checkpoint_sha256': controller['checkpoint_sha256'],
                'attempt_output_dir': attempt_dir,
            }
            summary = trainer.evaluate_once(
                context, record_callback=self.package.append_record
            )
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
            for evaluation_seed in self.collection['evaluation_seeds']:
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
