"""SEQ8 open-protocol plan builder (schema_version=2).

Extends the frozen 4-scene formal protocol to arbitrary demand sequences:
orders may list any existing configs/sim/<name>.cfg network identity, may be
8 stages long, and may repeat networks (revisit semantics).  Children are
HA-SODQN-compatible entries so ``runtime.run_child`` executes them
unchanged; the offline leg uses archive_mode='SELF' (the run's own
stage<k transitions) or 'NONE' (CONT-FIFO).
"""

import copy
import os
import subprocess

import yaml

from .core import canonical_digest
from .io import atomic_json, read_json, sha256_file

SEQ8_SCHEMA_VERSION = 2
SEQ8_PLAN_SCHEMA_VERSION = 1
DEFAULT_INITIAL_CATALOG = (
    'data/output_data/ha_sodqn/engineering_20260726/initial_state_catalog.json')


def _git(command):
    return subprocess.check_output(['git', *command], text=True).strip()


def _repository_identity():
    return {
        'git_commit': _git(['rev-parse', 'HEAD']),
        'git_branch': _git(['branch', '--show-current']),
        'git_remote': _git(['remote', 'get-url', 'origin']),
    }


def load_seq8_config(path):
    with open(path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    validate_seq8_config(config)
    return copy.deepcopy(config)


def validate_seq8_config(config):
    if config.get('schema_version') != SEQ8_SCHEMA_VERSION:
        raise ValueError('SEQ8 config requires schema_version=2')
    orders = config.get('orders')
    if not isinstance(orders, dict) or not orders:
        raise ValueError('SEQ8 config must define named orders')
    stage_count = None
    for order_id, networks in orders.items():
        if not isinstance(networks, list) or not networks:
            raise ValueError(f'{order_id} must be a non-empty network list')
        for network in networks:
            path = os.path.join('configs', 'sim', f'{network}.cfg')
            if not os.path.exists(path):
                raise ValueError(f'{order_id} references unknown cfg {network}')
        if stage_count is None:
            stage_count = len(networks)
        elif len(networks) != stage_count:
            raise ValueError('SEQ8 orders must share one stage count')
    trainer = config.get('trainer', {})
    episodes = trainer.get('stage_episodes')
    if not isinstance(episodes, list) or len(episodes) != stage_count:
        raise ValueError('stage_episodes must match the order stage count')
    smoke = trainer.get('smoke_stage_episodes')
    if smoke is not None and len(smoke) != stage_count:
        raise ValueError('smoke_stage_episodes must match the order stage count')
    seeds = config.get('training_seeds')
    if not isinstance(seeds, list) or not seeds:
        raise ValueError('SEQ8 requires an explicit training_seeds list')
    conditions = config.get('conditions')
    if not isinstance(conditions, list) or not conditions:
        raise ValueError('SEQ8 requires an explicit conditions list')
    for condition in conditions:
        mode = str(condition.get('archive_mode', '')).upper()
        if mode == 'NONE':
            if condition.get('method') != 'CONT':
                raise ValueError('NONE condition must use method=CONT')
        elif mode == 'SELF':
            if condition.get('method') not in ('DHOA', 'RAND', 'COV', 'CQ',
                                               'CQA'):
                raise ValueError('SELF condition has unknown method')
            ratio = float(condition.get('offline_ratio', 0.0))
            if ratio not in (0.25, 0.5, 0.75):
                raise ValueError('SELF offline_ratio must be 0.25/0.50/0.75')
        else:
            raise ValueError(f'SEQ8 condition has bad archive_mode {mode!r}')
    return config


def _seed_initial_index(catalog_path):
    catalog = read_json(catalog_path)
    index = {}
    for entry in catalog['entries']:
        seed = int(entry['training_seed'])
        index.setdefault(seed, entry)
    return index


def _child(config, seed_initial_index, order_id, seed, condition,
           stage_episodes, experiment_stage):
    networks = config['orders'][order_id]
    initial = dict(seed_initial_index[int(seed)])
    # ep0 checkpoints are bitwise-equivalent across networks per seed
    # (same_seed_cross_network_equivalence in the source catalog); rebind the
    # provenance fields to this child's first network so the runtime binding
    # check stays meaningful for the seq8 order.
    initial['order_id'] = order_id
    initial['network'] = networks[0]
    initial['first_network'] = networks[0]
    initial['ordered_networks'] = list(networks)
    initial['initial_reuse_note'] = (
        'episode-0 init reused via same_seed_cross_network_equivalence')
    mode = str(condition['archive_mode']).upper()
    method = str(condition.get('method', 'CONT')).upper()
    ratio = float(condition.get('offline_ratio', 0.0))
    if mode == 'NONE':
        label = 'CONT'
    else:
        label = f'SELF-{method}-R{int(round(ratio * 100)):02d}'
    logical_id = f'{label}-{order_id}-SD{seed}'
    return {
        'logical_run_id': logical_id,
        'experiment_stage': experiment_stage,
        'condition': 'CONT_FIFO' if mode == 'NONE' else 'HA_SODQN_SELF',
        'policy': 'ha_sodqn',
        'algorithm_id': str(
            config.get('algorithm_id', 'independent_dqn')).lower(),
        'order_id': order_id,
        'training_seed': int(seed),
        'networks': list(networks),
        'stage_episodes': list(stage_episodes),
        'initial_state': initial,
        'initial_checkpoint': initial['checkpoint_path'],
        'initial_checkpoint_file_sha256': initial['checkpoint_file_sha256'],
        'archive_mode': mode,
        'method': method,
        'offline_ratio': ratio,
        'owp_capacity': int(config.get('owp_capacity', 5000)),
        'alignment_warmup_episodes': int(
            config.get('alignment_warmup_episodes', 10)),
        'trace_replay_samples': False,
        'estimated_output_bytes': 2 * 1024 ** 3,
        'interface': 'libsumo',
        'status': 'planned',
    }


def build_seq8_plan(output_path, config_path, experiment_stage='formal',
                    launch_authorized=None):
    config = load_seq8_config(config_path)
    catalog_path = os.path.abspath(config.get(
        'initial_state_catalog', DEFAULT_INITIAL_CATALOG))
    seed_initial_index = _seed_initial_index(catalog_path)
    trainer = config['trainer']
    stage_episodes = (
        trainer['smoke_stage_episodes']
        if experiment_stage == 'smoke' else trainer['stage_episodes']
    )
    children = []
    for order_id in config['orders']:
        for seed in config['training_seeds']:
            for condition in config['conditions']:
                children.append(_child(
                    config, seed_initial_index, order_id, seed, condition,
                    stage_episodes, experiment_stage))
    identities = [child['logical_run_id'] for child in children]
    if len(identities) != len(set(identities)):
        raise ValueError('SEQ8 plan contains duplicate logical identities')
    formal = experiment_stage != 'smoke'
    if launch_authorized is None:
        launch_authorized = not formal
    payload = {
        'schema_version': SEQ8_PLAN_SCHEMA_VERSION,
        'protocol_id': config['protocol_id'],
        'mode': 'formal' if formal else 'smoke',
        'experiment_stage': experiment_stage,
        'launch_authorized': bool(launch_authorized),
        'child_count': len(children),
        'max_child_concurrency': int(config.get('max_child_concurrency', 8)),
        'orders': config['orders'],
        'trainer': trainer,
        'model': config['model'],
        'rng_config': config.get('rng'),
        'analysis': config.get('analysis'),
        'config_path': os.path.abspath(config_path),
        'config_sha256': sha256_file(config_path),
        'initial_state_catalog': catalog_path,
        'initial_state_catalog_sha256': sha256_file(catalog_path),
        **_repository_identity(),
        'children': children,
    }
    payload['plan_digest'] = canonical_digest(payload)
    atomic_json(output_path, payload)
    return payload


def validate_seq8_plan(path):
    plan = read_json(path)
    if plan.get('schema_version') != SEQ8_PLAN_SCHEMA_VERSION:
        raise ValueError('Unsupported SEQ8 plan schema')
    recorded = plan.get('plan_digest')
    unsigned = dict(plan)
    unsigned.pop('plan_digest', None)
    if canonical_digest(unsigned) != recorded:
        raise ValueError('SEQ8 plan digest mismatch')
    if sha256_file(plan['config_path']) != plan['config_sha256']:
        raise ValueError('SEQ8 config changed after plan creation')
    if sha256_file(plan['initial_state_catalog']) != \
            plan['initial_state_catalog_sha256']:
        raise ValueError('SEQ8 initial-state catalog changed')
    if _git(['rev-parse', 'HEAD']) != plan['git_commit']:
        raise ValueError('Current checkout differs from SEQ8 frozen commit')
    children = plan.get('children', ())
    if len(children) != plan.get('child_count'):
        raise ValueError('SEQ8 child count mismatch')
    expected_budget = (
        plan['trainer']['smoke_stage_episodes']
        if plan['mode'] == 'smoke' else plan['trainer']['stage_episodes'])
    for child in children:
        if child['stage_episodes'] != expected_budget:
            raise ValueError('SEQ8 child training budget mismatch')
        if sha256_file(child['initial_checkpoint']) != \
                child['initial_checkpoint_file_sha256']:
            raise ValueError('SEQ8 initial checkpoint changed')
        mode = child.get('archive_mode')
        if mode == 'NONE':
            if child['method'] != 'CONT' or child['offline_ratio'] != 0.0:
                raise ValueError('Invalid CONT-FIFO identity')
        elif mode != 'SELF':
            raise ValueError(f'Invalid SEQ8 archive mode {mode!r}')
        if len(child.get('networks', ())) != len(expected_budget):
            raise ValueError('SEQ8 child stage count mismatch')
    return {'valid': True, 'child_count': len(children),
            'plan_digest': recorded}
