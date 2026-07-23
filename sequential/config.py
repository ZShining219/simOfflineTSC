import copy
import os

import yaml


FORMAL_NETWORKS = (
    'sumohz1x1_config2', 'sumohz1x1',
    'sumohz1x1_config4', 'sumohz1x1_config3',
)
FORMAL_POLICIES = ('clear', 'fifo', 'fifo_matched_wait')


def load_sequential_config(path='configs/sequential/plan34.yml'):
    with open(path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    validate_sequential_config(config)
    return copy.deepcopy(config)


def validate_sequential_config(config):
    if config.get('schema_version') != 1:
        raise ValueError('Sequential config requires schema_version=1')
    orders = config.get('orders')
    if not isinstance(orders, dict) or set(orders) != {'O1', 'O2', 'O3', 'O4'}:
        raise ValueError('Sequential config must define O1..O4')
    for order_id, networks in orders.items():
        if len(networks) != 4 or set(networks) != set(FORMAL_NETWORKS):
            raise ValueError(f'{order_id} must contain each formal network once')
    if tuple(config.get('policies', ())) != FORMAL_POLICIES:
        raise ValueError('Sequential policies are not frozen')
    if config.get('training_seeds') != [0, 1, 2, 3, 4]:
        raise ValueError('Sequential training seeds must be 0..4')
    trainer = config.get('trainer', {})
    required = {
        'stage_episodes': [400, 100, 100, 100],
        'learning_start': 1000, 'buffer_size': 5000, 'batch_size': 64,
        'steps': 3600, 'action_interval': 10, 'target_update_interval': 10,
    }
    for key, expected in required.items():
        if trainer.get(key) != expected:
            raise ValueError(f'Invalid frozen trainer setting {key}')
    return config


def simulator_config_path(network):
    if network not in FORMAL_NETWORKS:
        raise ValueError(f'Unsupported sequential network: {network}')
    return os.path.join('configs', 'sim', f'{network}.cfg')
