import os
import sys
import copy
import yaml
import logging
import json
import hashlib
import tempfile
from datetime import datetime
from json import JSONDecodeError

from common.registry import Registry


CONFIG_ARCHIVE_DIR = 'config'
RUN_SCHEMA_VERSION = 1
BASELINE_COMMIT = '73d860bb3924ec15c30433a8f8b7af17787baeff'
RUN_STATUSES = {'已创建', '运行中', '已完成', '失败'}


def _read_bytes(path):
    with open(path, 'rb') as file_handle:
        return file_handle.read()


def _atomic_write(path, content):
    """Write bytes to path without exposing a partially written file."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix='.tmp-', dir=directory)
    try:
        with os.fdopen(descriptor, 'wb') as file_handle:
            file_handle.write(content)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


def _write_json_atomic(path, value):
    content = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    _atomic_write(path, content.encode('utf-8'))


def _sanitize_error_message(error):
    message = ' '.join(str(error).split())[:1000]
    for value in os.environ.values():
        if len(value) >= 8 and value in message:
            message = message.replace(value, '[REDACTED]')
    return message


class RunStateManager:
    """Persist the immutable run identity and atomic lifecycle transitions."""

    def __init__(self, config, config_path):
        command = config['command']
        self.output_path = get_output_file_path(config)
        self.manifest_path = os.path.join(self.output_path, 'run_manifest.json')
        self.status_path = os.path.join(self.output_path, 'run_status.json')
        run_id = '/'.join((
            command['task'], f"{command['world']}_{command['agent']}",
            command['network'], command['prefix'],
        ))
        config_hash = hashlib.sha256(
            _read_bytes(os.path.join(config_path, 'resolved_config.yaml'))
        ).hexdigest()
        created_at = _utc_now()
        self.manifest = {
            'schema_version': RUN_SCHEMA_VERSION,
            'run_id': run_id,
            'task': command['task'],
            'agent': command['agent'],
            'world': command['world'],
            'network': command['network'],
            'prefix': command['prefix'],
            'training_seed': command['seed'],
            'sumo_seed_mode': 'fixed_default',
            'baseline_commit': BASELINE_COMMIT,
            'created_at_utc': created_at,
            'config_hash': config_hash,
        }
        self.status = {
            'schema_version': RUN_SCHEMA_VERSION,
            'run_id': run_id,
            'status': '已创建',
            'started_at_utc': None,
            'finished_at_utc': None,
            'exit_code': None,
            'error_type': None,
            'error_message': None,
        }
        _write_json_atomic(self.manifest_path, self.manifest)
        _write_json_atomic(self.status_path, self.status)

    def transition(self, status, exit_code=None, error=None):
        if status not in RUN_STATUSES:
            raise ValueError(f'Invalid run status: {status}')
        allowed = {
            '已创建': {'运行中', '失败'},
            '运行中': {'已完成', '失败'},
            '已完成': set(),
            '失败': set(),
        }
        current = self.status['status']
        if status not in allowed[current]:
            raise ValueError(f'Invalid run status transition: {current} -> {status}')
        if status == '运行中':
            if exit_code is not None or error is not None:
                raise ValueError('Running status cannot contain an exit code or error')
            self.status['started_at_utc'] = _utc_now()
        elif status == '已完成':
            if exit_code not in (None, 0) or error is not None:
                raise ValueError('Completed status requires exit_code=0 and no error')
            self.status['finished_at_utc'] = _utc_now()
            self.status['exit_code'] = 0
        elif status == '失败':
            if error is None or exit_code in (None, 0):
                raise ValueError('Failed status requires an error and non-zero exit code')
            self.status['finished_at_utc'] = _utc_now()
            self.status['exit_code'] = exit_code
            self.status['error_type'] = type(error).__name__
            self.status['error_message'] = _sanitize_error_message(error)
        self.status['status'] = status
        _write_json_atomic(self.status_path, self.status)


def verify_config_archive(config_path):
    """Verify every archived file against the persisted SHA-256 manifest."""
    manifest_path = os.path.join(config_path, 'config_hashes.json')
    try:
        manifest = json.loads(_read_bytes(manifest_path).decode('utf-8'))
    except (FileNotFoundError, JSONDecodeError, UnicodeDecodeError) as error:
        raise IOError(f'Invalid configuration hash manifest: {manifest_path}') from error
    if manifest.get('algorithm') != 'sha256' or not isinstance(manifest.get('files'), dict):
        raise IOError(f'Invalid configuration hash manifest schema: {manifest_path}')
    for name, expected_hash in manifest['files'].items():
        archived_path = os.path.join(config_path, name)
        try:
            actual_hash = hashlib.sha256(_read_bytes(archived_path)).hexdigest()
        except FileNotFoundError as error:
            raise IOError(f'Configuration archive file is missing: {name}') from error
        if actual_hash != expected_hash:
            raise IOError(
                f'Configuration archive verification failed for {name}: '
                f'expected {expected_hash}, got {actual_hash}'
            )
    return manifest['files']


def capture_config_sources(config):
    """Capture immutable source configuration before simulator registration."""
    command = config['command']
    task = command['task']
    agent = command['agent']
    network = command['network']
    source_paths = {
        'base.yml': os.path.join('configs', task, 'base.yml'),
        f'{agent}.yml': os.path.join('configs', task, f'{agent}.yml'),
        'simulator_source.cfg': os.path.join('configs', 'sim', f'{network}.cfg'),
    }
    missing = [path for path in source_paths.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            'Cannot archive experiment configuration; missing source file(s): '
            + ', '.join(missing)
        )
    return {name: _read_bytes(path) for name, path in source_paths.items()}


def reserve_run_output(config):
    """Reserve a new output directory and reject all pre-existing runs."""
    output_path = get_output_file_path(config)
    parent = os.path.dirname(output_path)
    os.makedirs(parent, exist_ok=True)
    try:
        os.mkdir(output_path)
    except FileExistsError as error:
        raise FileExistsError(
            f'Run output already exists: {output_path}. '
            'Use a new --prefix; existing logs, checkpoints, and configuration '
            'must not be overwritten or mixed with a new run.'
        ) from error
    return output_path


def resolve_simulator_config(config, source_content=None):
    """Create and resolve the simulator config inside the reserved run directory."""
    network = config['command']['network']
    source_path = os.path.join('configs', 'sim', f'{network}.cfg')
    if source_content is None:
        source_content = _read_bytes(source_path)
    resolved_path = os.path.join(
        get_output_file_path(config), CONFIG_ARCHIVE_DIR, 'simulator_resolved.cfg'
    )
    _atomic_write(resolved_path, source_content)
    other_world_settings = modify_config_file(resolved_path, config)
    return resolved_path, other_world_settings


def archive_run_config(config, source_snapshots, resolved_world):
    """Archive source and effective configuration before Trainer creation."""
    output_path = get_output_file_path(config)
    config_path = os.path.join(output_path, CONFIG_ARCHIVE_DIR)
    simulator_source_path = os.path.join(
        'configs', 'sim', f"{config['command']['network']}.cfg"
    )
    simulator_path = os.path.join(config_path, 'simulator_resolved.cfg')
    snapshots = dict(source_snapshots)
    snapshots['simulator_resolved.cfg'] = _read_bytes(simulator_path)

    resolved_config = copy.deepcopy(config)
    resolved_config['world'] = copy.deepcopy(resolved_world)
    resolved_config['config_record'] = {
        'created_at_utc': datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
        'sources': [
            os.path.join('configs', config['command']['task'], 'base.yml'),
            os.path.join(
                'configs', config['command']['task'],
                f"{config['command']['agent']}.yml"
            ),
            simulator_source_path,
        ],
    }
    snapshots['resolved_config.yaml'] = yaml.safe_dump(
        resolved_config, sort_keys=False, allow_unicode=True
    ).encode('utf-8')

    hashes = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in snapshots.items()
    }
    snapshots['config_hashes.json'] = (
        json.dumps({'algorithm': 'sha256', 'files': hashes}, indent=2, sort_keys=True)
        + '\n'
    ).encode('utf-8')

    for name, content in snapshots.items():
        _atomic_write(os.path.join(config_path, name), content)

    verify_config_archive(config_path)
    return config_path


def _json_value(value):
    """Convert common scalar configuration values to JSON-safe values."""
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _describe_torch_model(model):
    if model is None:
        return None

    import torch.nn as nn

    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    description = {
        'class': model.__class__.__name__,
        'module': model.__class__.__module__,
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        'trainable_parameter_count': sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        'linear_layers': [
            {
                'in_features': layer.in_features,
                'out_features': layer.out_features,
                'bias': layer.bias is not None,
            }
            for layer in linear_layers
        ],
    }
    if linear_layers:
        description.update({
            'input_dim': linear_layers[0].in_features,
            'hidden_layers': [layer.out_features for layer in linear_layers[:-1]],
            'output_dim': linear_layers[-1].out_features,
        })
    if hasattr(model, 'activation_name'):
        description['activation'] = _json_value(model.activation_name)
    return description


def _describe_optimizer(optimizer):
    if optimizer is None:
        return None
    param_groups = []
    for group in optimizer.param_groups:
        param_groups.append({
            key: _json_value(value)
            for key, value in group.items()
            if key != 'params'
        })
    return {
        'class': optimizer.__class__.__name__,
        'module': optimizer.__class__.__module__,
        'defaults': {
            key: _json_value(value)
            for key, value in optimizer.defaults.items()
        },
        'param_groups': param_groups,
    }


def _describe_loss(criterion):
    if criterion is None:
        return None
    description = {
        'class': criterion.__class__.__name__,
        'module': criterion.__class__.__module__,
    }
    if hasattr(criterion, 'reduction'):
        description['reduction'] = criterion.reduction
    return description


def _describe_agent(agent, rank):
    model = getattr(agent, 'model', None)
    target_model = getattr(agent, 'target_model', None)
    action_space = getattr(agent, 'action_space', None)
    description = {
        'rank': rank,
        'class': agent.__class__.__name__,
        'module': agent.__class__.__module__,
        'action_dim': getattr(action_space, 'n', None),
        'model': _describe_torch_model(model),
        'target_model': _describe_torch_model(target_model),
        'optimizer': _describe_optimizer(getattr(agent, 'optimizer', None)),
        'loss': _describe_loss(getattr(agent, 'criterion', None)),
    }
    if model is not None and target_model is not None:
        import torch

        model_state = model.state_dict()
        target_state = target_model.state_dict()
        description['target_matches_model_at_archive'] = (
            model_state.keys() == target_state.keys()
            and all(
                torch.equal(model_state[name], target_state[name])
                for name in model_state
            )
        )
    controller_parameters = {}
    for name in ('t_fixed', 't_min'):
        if hasattr(agent, name):
            controller_parameters[name] = _json_value(getattr(agent, name))
    if controller_parameters:
        description['controller_parameters'] = controller_parameters
    return description


def _refresh_config_hashes(config_path):
    files = {}
    for name in sorted(os.listdir(config_path)):
        path = os.path.join(config_path, name)
        if name == 'config_hashes.json' or not os.path.isfile(path):
            continue
        files[name] = hashlib.sha256(_read_bytes(path)).hexdigest()
    hash_content = (
        json.dumps({'algorithm': 'sha256', 'files': files}, indent=2, sort_keys=True)
        + '\n'
    ).encode('utf-8')
    _atomic_write(os.path.join(config_path, 'config_hashes.json'), hash_content)

    verify_config_archive(config_path)


def archive_runtime_model(config_path, trainer, agent_name):
    """Record the models/controllers actually created before task execution."""
    if trainer.agents is None:
        raise RuntimeError('Cannot archive runtime model before agents are created')
    description = {
        'schema_version': 1,
        'agent': agent_name,
        'agents': [
            _describe_agent(agent, rank)
            for rank, agent in enumerate(trainer.agents)
        ],
    }
    content = (json.dumps(description, indent=2, sort_keys=True) + '\n').encode('utf-8')
    _atomic_write(os.path.join(config_path, 'model_resolved.json'), content)
    _refresh_config_hashes(config_path)
    return os.path.join(config_path, 'model_resolved.json')


def modify_config_file(path, config):
    """
    load .cfg file at path and modify it according to the config parameters
    """
    assert(os.path.exists(path)), AssertionError(f"Simulator configuration at {path} not exists")
    param = config['world']
    logger_param = config['logger']

    if config['command']['world'] == 'cityflow':
        with open(path, 'r') as f:
            path_config = json.load(f)
        for k in path_config.keys():
            # modify config step1
            if param.get(k) is not None:
                path_config[k] = param.get(k)
        # modify config step2
        file_name = os.path.join(get_output_file_path(config),  logger_param['replay_dir'])
        if config['world']['dir'] in file_name:
            file_name = file_name.strip(f"{config['world']['dir']} + '\n'")
        path_config['roadnetLogFile'] = file_name + f"/{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.json"
        path_config['replayLogFile'] = file_name + f"/{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.txt"
        with open(path, 'w') as f:
            json.dump(path_config, f, indent=2)
        
    elif config['command']['world'] == 'sumo':
        with open(path, 'r') as f:
            path_config = json.load(f)
        # config step 1
        for k in path_config.keys():
            if param.get(k) is not None:
                path_config[k] = param.get(k)
        # config step 2
        #path_config['roadnetLogFile'] = file_name + f"/{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.json"
        #path_config['replayLogFile'] = file_name + f"/{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.txt"
        path_config['interval'] = param['interval']
        with open(path, 'w') as f:
            json.dump(path_config, f, indent=2)


    elif config['command']['world'] == 'openengine':
        # not in .json format
        with open(path, 'r') as f:
            contents = f.readlines()
        for idx, l in enumerate(contents):
            if '=' in l:
                lhs, _ = l.split('=')
                # TODO: check interval==10 here
                if lhs.strip() in param.keys() and lhs.strip() != 'interval':
                    rhs = ' ' + str(param[lhs.strip()]) + '\n'
                    contents[idx] = lhs + '=' + rhs
                # config step 2
                if lhs.strip() == 'max_time_epoch':
                    rhs = ' ' + str(config['trainer']['steps']) + '\n'
                    contents[idx] = lhs + '=' + rhs
            elif ':' in l:
                lhs, _ = l.split(':')
                if lhs.strip() == 'report_log_mode':
                    rhs = ' ' + str(param[lhs.strip()]) + '\n'
                    contents[idx] = lhs + ':' + rhs
                if lhs.strip() == 'report_log_addr':
                    file_name = get_output_file_path(config) + '/' +  logger_param['replay_dir'] 
                    path_config['roadnetLogFile'] = file_name + f"/{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.json"
                    rhs = ' ' + 'data/output_data/' + config['command']['task'] + '/'\
                        + f"{config['command']['world']}_{config['command']['agent']}_{config['command']['prefix']}"\
                            + '/' +  logger_param['replay_dir'] + '\n'
                    contents[idx] = lhs + ':' + rhs
        with open(path, 'w') as f:
            f.writelines(contents)
    else:
        raise NotImplementedError('Simulator environment not implemented')
    
    # config other world settings
    other_world_settings = dict()
    for k in param.keys():
        if k not in path_config.keys():
            other_world_settings[k] = param.get(k)
    return other_world_settings

def build_config(args):
    """
    process command line arguments and parameters stored in .yaml files.
    position args:
    -args: command line arguments take in from run.py
    """
    agent_name = os.path.join('./configs', args.task, f'{args.agent}.yml')
    config, duplicates_warning = load_config(agent_name)
    config.update({'command': args.__dict__})
    return config, duplicates_warning

def load_config(path, previous_includes=[]):
    """
    process individual .yaml file and eliminate duplicate parameters
    position args:
    -path: path of .yml file
    -previous_includes: list of .yml already processed
    """
    if path in previous_includes:
        raise ValueError(
            f"Cyclic configs include detected. {path} included in previous {previous_includes}"
        )
    previous_includes = previous_includes + [path]
    direct_config = yaml.load(open(path, "r"), Loader=yaml.Loader)
    # Load configs from included files.
    if "includes" in direct_config:
        includes = direct_config.pop("includes")
    else:
        includes = []
    if not isinstance(includes, list):
        raise AttributeError(
            "Includes must be a list, '{}' provided".format(type(includes))
        )
    config = {}
    duplicates_warning = {}
    # process config recursively
    for include in includes:
        include_config, inc_dup_warning = load_config(
            include, previous_includes
        )
        duplicates_warning.update(inc_dup_warning)
        config, duplicates = merge_dicts(config, include_config)
        duplicates_warning.update(duplicates)
    config, merge_dup_warning = merge_dicts(config, direct_config)
    duplicates_warning.update(merge_dup_warning)
    return config, duplicates_warning

def merge_dicts(dict1, dict2):
    """
    merge dict2 into dict1, and dict1 will not be overwrite by dict2
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)
    duplicates = {}

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k], duplicates_k = merge_dicts(dict1[k], dict2[k])
                if k not in duplicates.keys():
                    duplicates.update({k: duplicates_k})
            else:
                return_dict[k] = dict2[k]
                duplicates.update({k: v})
    return return_dict, duplicates

def load_config_dict(config_path, other_world_settings=None):
    """
    load .cfg file at config_path
    """
    try:
        with open(config_path, 'r') as f:
            path_config = json.load(f)
    except JSONDecodeError:
        with open(config_path, 'r') as f:
            contents = f.readlines()
            path_config = {}
            for l in contents:
                if ':' in l:
                    lhs, rhs = l.split(':')
                    try:
                        val = eval(rhs.strip().strip('\n'))
                    except NameError:
                        val = rhs.strip().strip('\n')
                    path_config.update({lhs.strip().strip('\n'): val})
                if '=' in l:
                    lhs, rhs = l.split('=')
                    try:
                        val = eval(rhs.strip().strip('\n'))
                    except NameError:
                        val = rhs.strip().strip('\n')
                    path_config.update({lhs.strip().strip('\n'): val})
    if other_world_settings is not None:
        path_config.update(other_world_settings)
    return path_config

def get_output_file_path(config):
    """"
    set output path
    """
    param = config['command']
    path = os.path.join(config['world']['dir'] , 'output_data', param['task'], 
        f"{param['world']}_{param['agent']}", param['network'], param['prefix'])
    return path


class SeverityLevelBetween(logging.Filter):
    def __init__(self, min_level, max_level):
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record):
        return self.min_level <= record.levelno < self.max_level

def setup_logging(level):
    root = logging.getLogger()

    # Perform setup only if logging has not been configured
    if not root.hasHandlers():
        root.setLevel(level)
        log_formatter = logging.Formatter(
            "%(asctime)s (%(levelname)s): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Send INFO to stdout
        handler_out = logging.StreamHandler(sys.stdout)
        handler_out.addFilter(
            SeverityLevelBetween(logging.INFO, logging.WARNING)
        )
        handler_out.setFormatter(log_formatter)
        root.addHandler(handler_out)

        # Send WARNING (and higher) to stderr
        handler_err = logging.StreamHandler(sys.stderr)
        handler_err.setLevel(logging.WARNING)
        handler_err.setFormatter(log_formatter)
        root.addHandler(handler_err)

        logger_dir = os.path.join(
            Registry.mapping['logger_mapping']['path'].path,
            Registry.mapping['logger_mapping']['setting'].param['log_dir'])
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)

        handler_file = logging.FileHandler(os.path.join(
            logger_dir,
            f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}_BRF.log"), mode='w'
        )
        handler_file.setLevel(level)  # TODO: SET LEVEL
        root.addHandler(handler_file)
    return root
