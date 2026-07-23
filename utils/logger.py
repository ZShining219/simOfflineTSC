import os
import sys
import copy
import csv
import re
import yaml
import numpy as np
import logging
import json
import hashlib
import tempfile
import random
from datetime import datetime
from json import JSONDecodeError

from common.registry import Registry


CONFIG_ARCHIVE_DIR = 'config'
RUN_SCHEMA_VERSION = 1
BASELINE_COMMIT = '73d860bb3924ec15c30433a8f8b7af17787baeff'
RUN_STATUSES = {'已创建', '运行中', '已完成', '失败'}
METRIC_RECORD_TYPES = {'TRAIN', 'EVALUATION', 'FINAL_EVALUATION'}
METRIC_FIELDS_V1 = (
    'schema_version', 'record_type', 'agent', 'network', 'training_seed',
    'episode', 'simulation_step', 'decision_step', 'global_decision_step',
    'gradient_updates', 'travel_time', 'reward_mean', 'reward_sum', 'queue',
    'delay', 'throughput', 'loss_mean', 'epsilon', 'wall_time_seconds',
)
METRIC_FIELDS = METRIC_FIELDS_V1 + (
    'real_delay', 'waiting_time', 'unfinished_vehicles',
    'action_distribution', 'phase_switches', 'phase_switch_frequency',
    'replay_size', 'replay_capacity', 'target_updates',
)
METRIC_FIELDS_V3 = METRIC_FIELDS + (
    'collected_transitions', 'sampled_transitions',
    'unique_sampled_transitions', 'unique_coverage', 'update_to_data_ratio',
    'sample_count_mean', 'sample_count_median', 'sample_count_p95',
    'sample_count_max', 'sampled_transition_mean_age',
)


def metric_fields_for_schema(schema_version):
    if schema_version == 1:
        return METRIC_FIELDS_V1
    if schema_version == 2:
        return METRIC_FIELDS
    if schema_version == 3:
        return METRIC_FIELDS_V3
    raise ValueError(f'Unsupported metric schema_version: {schema_version}')


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

    @classmethod
    def record_initialization_failure(cls, config, output_path, error, exit_code=1):
        """Record a failure after directory reservation but before config archival."""
        command = config['command']
        run_id = '/'.join((
            command['task'], f"{command['world']}_{command['agent']}",
            command['network'], command['prefix'],
        ))
        created_at = _utc_now()
        manifest = {
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
            'config_hash': None,
        }
        status = {
            'schema_version': RUN_SCHEMA_VERSION,
            'run_id': run_id,
            'status': '失败',
            'started_at_utc': None,
            'finished_at_utc': _utc_now(),
            'exit_code': exit_code,
            'error_type': type(error).__name__,
            'error_message': _sanitize_error_message(error),
        }
        _write_json_atomic(os.path.join(output_path, 'run_manifest.json'), manifest)
        _write_json_atomic(os.path.join(output_path, 'run_status.json'), status)
        return manifest, status

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
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def hash_torch_state_dict(state_dict):
    """Hash tensor content independently of torch serialization metadata."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().contiguous().cpu()
        fields = (name, str(tensor.dtype), json.dumps(list(tensor.shape)))
        for field in fields:
            encoded = field.encode('utf-8')
            digest.update(len(encoded).to_bytes(8, 'big'))
            digest.update(encoded)
        raw = tensor.numpy().tobytes(order='C')
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
    return digest.hexdigest()


def controlled_random_probe(count=8):
    """Read deterministic Python/NumPy samples and restore both RNG states."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        return {
            'python_random': [random.random() for _ in range(count)],
            'numpy_random': np.random.random(count).tolist(),
        }
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


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
        description['online_model_state_hash'] = hash_torch_state_dict(model_state)
        description['target_model_state_hash'] = hash_torch_state_dict(target_state)
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
        'reproducibility_probe': controlled_random_probe(),
        'agents': [
            _describe_agent(agent, rank)
            for rank, agent in enumerate(trainer.agents)
        ],
    }
    content = (json.dumps(description, indent=2, sort_keys=True) + '\n').encode('utf-8')
    _atomic_write(os.path.join(config_path, 'model_resolved.json'), content)
    _refresh_config_hashes(config_path)
    return os.path.join(config_path, 'model_resolved.json')


class StructuredMetricLogger:
    """Append schema-validated metric records as one complete JSON object per line."""

    def __init__(self, output_path):
        self.path = os.path.join(output_path, 'metrics', 'records.jsonl')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def append(self, record):
        self._validate_record(record)
        fields = metric_fields_for_schema(record['schema_version'])
        content = json.dumps(
            {field: _json_value(record[field]) for field in fields},
            ensure_ascii=False,
            separators=(',', ':'),
            allow_nan=False,
        ) + '\n'
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _validate_record(record):
        if not isinstance(record, dict) or 'schema_version' not in record:
            raise ValueError('Metric record is missing schema_version')
        fields = metric_fields_for_schema(record['schema_version'])
        missing = [field for field in fields if field not in record]
        extra = sorted(set(record) - set(fields))
        if missing or extra:
            raise ValueError(
                f'Invalid metric record fields; missing={missing}, extra={extra}'
            )
        if record['record_type'] not in METRIC_RECORD_TYPES:
            raise ValueError(f"Invalid metric record_type: {record['record_type']}")

    def validate(self, require_records=True):
        try:
            with open(self.path, encoding='utf-8') as handle:
                lines = handle.readlines()
        except FileNotFoundError as error:
            raise IOError(f'Structured metric log is missing: {self.path}') from error
        if require_records and not lines:
            raise IOError(f'Structured metric log has no records: {self.path}')
        for line_number, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
                self._validate_record(record)
            except (json.JSONDecodeError, ValueError) as error:
                raise IOError(
                    f'Invalid structured metric record at line {line_number}: {self.path}'
                ) from error
        return len(lines)


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
    if param.get('output_path'):
        return os.path.abspath(param['output_path'])
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


EVALUATION_REQUIRED_METRICS = (
    'reward', 'queue', 'delay', 'throughput', 'travel_time',
)
EVALUATION_SUMMARY_FIELDS = (
    'controller_id', 'agent', 'network', 'training_seed', 'evaluation_seed',
    'checkpoint_episode', 'checkpoint_path', 'checkpoint_sha256',
    'simulation_steps', 'decision_steps', 'travel_time', 'reward_mean',
    'queue', 'delay', 'real_delay', 'throughput', 'waiting_time',
    'unfinished_vehicles', 'wall_time_seconds',
)
EVALUATION_RECORD_FIELDS = (
    'schema_version', 'record_type', 'controller_id', 'agent', 'network',
    'training_seed', 'evaluation_seed', 'checkpoint_episode',
    'checkpoint_path', 'checkpoint_sha256', 'simulation_time_seconds',
    'decision_step', 'action_interval_seconds', 'actions', 'reward_agents',
    'controller_reward_agents',
    'reward_network_mean', 'reward_network_sum', 'queue_lanes',
    'queue_intersections', 'queue_network_mean', 'queue_network_sum',
    'delay_lanes', 'lane_vehicle_counts', 'delay_intersections',
    'delay_network_weighted_mean', 'throughput_interval',
    'throughput_cumulative',
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_evaluation_collection_manifest(path):
    """Validate and normalize an explicit best-checkpoint evaluation manifest."""
    manifest_path = os.path.abspath(path)
    with open(manifest_path, encoding='utf-8') as handle:
        payload = json.load(handle)
    required = {
        'schema_version', 'package_id', 'world', 'evaluation_seeds',
        'sampling_interval_seconds', 'smoothing_window_seconds', 'metrics',
        'controllers',
    }
    missing = sorted(required - set(payload)) if isinstance(payload, dict) else sorted(required)
    if missing:
        raise ValueError(f'Evaluation collection manifest missing fields: {missing}')
    if payload['schema_version'] != 1 or payload['world'] != 'sumo':
        raise ValueError('Evaluation collection manifest requires schema_version=1 and world=sumo')
    seeds = payload['evaluation_seeds']
    if not isinstance(seeds, list) or not seeds or any(
        not isinstance(seed, int) or seed < 0 for seed in seeds
    ) or len(set(seeds)) != len(seeds):
        raise ValueError('evaluation_seeds must be a non-empty unique integer list')
    if payload['sampling_interval_seconds'] <= 0:
        raise ValueError('sampling_interval_seconds must be positive')
    if payload['smoothing_window_seconds'] < payload['sampling_interval_seconds']:
        raise ValueError('smoothing_window_seconds cannot be shorter than sampling interval')
    missing_metrics = sorted(set(EVALUATION_REQUIRED_METRICS) - set(payload['metrics']))
    if missing_metrics:
        raise ValueError(f'Evaluation collection manifest missing metrics: {missing_metrics}')
    normalized = copy.deepcopy(payload)
    normalized['source_manifest'] = manifest_path
    seen_ids = set()
    normalized_controllers = []
    for controller in payload['controllers']:
        controller_required = {
            'controller_id', 'agent', 'network', 'training_seed',
            'run_dir', 'checkpoint',
        }
        missing_controller = sorted(controller_required - set(controller))
        if missing_controller:
            raise ValueError(
                f"Controller {controller.get('controller_id')} missing fields: "
                f'{missing_controller}'
            )
        controller_id = controller['controller_id']
        if not isinstance(controller_id, str) or not re.fullmatch(
            r'[A-Za-z0-9][A-Za-z0-9._-]*', controller_id
        ):
            raise ValueError(f'Unsafe controller_id: {controller_id!r}')
        if controller_id in seen_ids:
            raise ValueError(f'Duplicate controller_id: {controller_id}')
        seen_ids.add(controller_id)
        if controller['agent'] not in {'dqn', 'fixedtime', 'maxpressure'}:
            raise ValueError(f"Unsupported evaluation agent: {controller['agent']}")
        run_dir = os.path.abspath(os.path.expanduser(controller['run_dir']))
        with open(os.path.join(run_dir, 'run_manifest.json'), encoding='utf-8') as handle:
            run_manifest = json.load(handle)
        with open(os.path.join(run_dir, 'run_status.json'), encoding='utf-8') as handle:
            run_status = json.load(handle)
        if run_status.get('status') != '已完成' or run_status.get('exit_code') != 0:
            raise ValueError(f'Evaluation source run did not complete: {run_dir}')
        for field in ('agent', 'network', 'training_seed'):
            if run_manifest.get(field) != controller[field]:
                raise ValueError(
                    f'Controller {controller_id} {field} does not match source run'
                )
        verify_config_archive(os.path.join(run_dir, 'config'))
        checkpoint = controller['checkpoint']
        checkpoint_path = None
        checkpoint_episode = None
        checkpoint_sha256 = None
        if controller['agent'] == 'dqn':
            if not isinstance(checkpoint, str) or not checkpoint:
                raise ValueError(f'DQN controller {controller_id} requires checkpoint')
            checkpoint_path = (
                checkpoint if os.path.isabs(checkpoint)
                else os.path.join(run_dir, checkpoint)
            )
            checkpoint_path = os.path.abspath(checkpoint_path)
            with open(os.path.join(run_dir, 'evaluation', 'summary.json'), encoding='utf-8') as handle:
                evaluation_summary = json.load(handle)
            expected = os.path.abspath(os.path.join(run_dir, evaluation_summary['best_checkpoint']))
            if checkpoint_path != expected:
                raise ValueError(
                    f'DQN controller {controller_id} checkpoint is not the recorded best checkpoint'
                )
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(checkpoint_path)
            checkpoint_episode = int(evaluation_summary['best_episode'])
            checkpoint_sha256 = _sha256_file(checkpoint_path)
        elif checkpoint is not None:
            raise ValueError(f'Baseline controller {controller_id} checkpoint must be null')
        item = copy.deepcopy(controller)
        item.update({
            'run_dir': run_dir,
            'checkpoint_path': checkpoint_path,
            'checkpoint_episode': checkpoint_episode,
            'checkpoint_sha256': checkpoint_sha256,
        })
        normalized_controllers.append(item)
    normalized['controllers'] = normalized_controllers
    expected_controllers = payload.get('expected_controller_count')
    if expected_controllers is not None and expected_controllers != len(normalized_controllers):
        raise ValueError('Evaluation controller count does not match expected_controller_count')
    expected_episodes = payload.get('expected_episode_count')
    actual_episodes = len(normalized_controllers) * len(seeds)
    if expected_episodes is not None and expected_episodes != actual_episodes:
        raise ValueError('Evaluation episode count does not match expected_episode_count')
    normalized['expected_episode_count'] = actual_episodes
    return manifest_path, normalized


class EvaluationPackageWriter:
    """Crash-visible writer for one immutable decision-level evaluation package."""

    def __init__(self, output_dir, collection_manifest):
        self.output_dir = os.path.abspath(output_dir)
        try:
            os.makedirs(self.output_dir)
        except FileExistsError as error:
            raise FileExistsError(
                f'Evaluation output already exists: {self.output_dir}'
            ) from error
        self.records_path = os.path.join(self.output_dir, 'records.jsonl')
        self.summary_path = os.path.join(self.output_dir, 'summary.csv')
        self.collection_path = os.path.join(self.output_dir, 'collection_manifest.json')
        self.manifest_path = os.path.join(self.output_dir, 'manifest.json')
        self.attempts_dir = os.path.join(self.output_dir, 'attempts')
        os.makedirs(self.attempts_dir)
        self.collection_manifest = copy.deepcopy(collection_manifest)
        _write_json_atomic(self.collection_path, self.collection_manifest)
        self.summaries = []
        self.record_count = 0

    def attempt_dir(self, controller_id, evaluation_seed):
        path = os.path.join(
            self.attempts_dir, f'{controller_id}__eval_seed_{evaluation_seed}'
        )
        os.makedirs(path)
        return path

    def append_record(self, record):
        missing = [field for field in EVALUATION_RECORD_FIELDS if field not in record]
        extra = sorted(set(record) - set(EVALUATION_RECORD_FIELDS))
        if missing or extra or record.get('schema_version') != 1:
            raise ValueError(
                f'Invalid evaluation decision record; missing={missing}, extra={extra}'
            )
        content = json.dumps(
            {field: _json_value(record[field]) for field in EVALUATION_RECORD_FIELDS},
            ensure_ascii=False, separators=(',', ':'), allow_nan=False,
        ) + '\n'
        with open(self.records_path, 'a', encoding='utf-8') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        self.record_count += 1

    def append_summary(self, summary):
        missing = [field for field in EVALUATION_SUMMARY_FIELDS if field not in summary]
        if missing:
            raise ValueError(f'Evaluation summary missing fields: {missing}')
        self.summaries.append({field: _json_value(summary[field]) for field in EVALUATION_SUMMARY_FIELDS})
        descriptor, temporary_path = tempfile.mkstemp(prefix='.tmp-', dir=self.output_dir)
        try:
            with os.fdopen(descriptor, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=EVALUATION_SUMMARY_FIELDS)
                writer.writeheader()
                writer.writerows(self.summaries)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.summary_path)
        except Exception:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise

    def finalize(self):
        expected = self.collection_manifest['expected_episode_count']
        if len(self.summaries) != expected:
            raise ValueError(
                f'Evaluation package has {len(self.summaries)} episodes; expected {expected}'
            )
        manifest = {
            'schema_version': 1,
            'package_id': self.collection_manifest['package_id'],
            'status': 'completed',
            'created_at_utc': _utc_now(),
            'episode_count': len(self.summaries),
            'decision_record_count': self.record_count,
            'files': {
                name: {'sha256': _sha256_file(os.path.join(self.output_dir, name))}
                for name in ('collection_manifest.json', 'records.jsonl', 'summary.csv')
            },
        }
        _write_json_atomic(self.manifest_path, manifest)
        validate_evaluation_package(self.output_dir)
        return self.manifest_path


def validate_evaluation_package(output_dir):
    output_dir = os.path.abspath(output_dir)
    with open(os.path.join(output_dir, 'manifest.json'), encoding='utf-8') as handle:
        manifest = json.load(handle)
    if manifest.get('schema_version') != 1 or manifest.get('status') != 'completed':
        raise ValueError('Evaluation package is not a completed schema v1 package')
    for name, identity in manifest.get('files', {}).items():
        path = os.path.join(output_dir, name)
        if _sha256_file(path) != identity.get('sha256'):
            raise IOError(f'Evaluation package hash mismatch: {name}')
    with open(os.path.join(output_dir, 'collection_manifest.json'), encoding='utf-8') as handle:
        collection = json.load(handle)
    with open(os.path.join(output_dir, 'summary.csv'), newline='', encoding='utf-8') as handle:
        summaries = list(csv.DictReader(handle))
    expected = collection['expected_episode_count']
    if len(summaries) != expected or manifest['episode_count'] != expected:
        raise ValueError('Evaluation package episode count is incomplete')
    identities = {
        (row['controller_id'], int(row['evaluation_seed'])) for row in summaries
    }
    expected_identities = {
        (controller['controller_id'], seed)
        for controller in collection['controllers']
        for seed in collection['evaluation_seeds']
    }
    if identities != expected_identities:
        raise ValueError('Evaluation package summary identities are incomplete or duplicated')
    controller_by_id = {
        item['controller_id']: item for item in collection['controllers']
    }
    records_by_identity = {}
    record_count = 0
    with open(os.path.join(output_dir, 'records.jsonl'), encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise IOError(
                    f'Invalid evaluation record at line {line_number}'
                ) from error
            missing = [field for field in EVALUATION_RECORD_FIELDS if field not in record]
            extra = sorted(set(record) - set(EVALUATION_RECORD_FIELDS))
            if missing or extra or record.get('schema_version') != 1:
                raise ValueError(
                    f'Evaluation record {line_number} has invalid fields; '
                    f'missing={missing}, extra={extra}'
                )
            controller = controller_by_id.get(record['controller_id'])
            if controller is None or any(
                record[field] != controller[field]
                for field in ('agent', 'network', 'training_seed')
            ):
                raise ValueError(
                    f'Evaluation record {line_number} controller identity mismatch'
                )
            identity = (record['controller_id'], int(record['evaluation_seed']))
            if identity not in expected_identities:
                raise ValueError(
                    f'Evaluation record {line_number} has unexpected identity'
                )
            records_by_identity.setdefault(identity, []).append(record)
            record_count += 1
    if record_count != manifest['decision_record_count']:
        raise ValueError('Evaluation package decision record count mismatch')
    if set(records_by_identity) != expected_identities:
        raise ValueError('Evaluation package decision identities are incomplete')
    summaries_by_identity = {
        (row['controller_id'], int(row['evaluation_seed'])): row
        for row in summaries
    }
    for identity, records in records_by_identity.items():
        records.sort(key=lambda item: item['decision_step'])
        expected_steps = list(range(1, len(records) + 1))
        if [item['decision_step'] for item in records] != expected_steps:
            raise ValueError(f'Non-contiguous evaluation decision steps: {identity}')
        interval = collection['sampling_interval_seconds']
        if any(
            item['simulation_time_seconds'] != item['decision_step'] * interval
            or item['action_interval_seconds'] != interval
            for item in records
        ):
            raise ValueError(f'Invalid evaluation simulation-time continuity: {identity}')
        summary = summaries_by_identity[identity]
        if (
            int(summary['decision_steps']) != len(records)
            or int(summary['simulation_steps']) != int(records[-1]['simulation_time_seconds'])
        ):
            raise ValueError(f'Evaluation summary/decision count mismatch: {identity}')
    return {
        'manifest': manifest,
        'collection': collection,
        'summaries': summaries,
        'record_count': record_count,
    }
