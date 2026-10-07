"""Design and audit the historical assets used by cross-algorithm HA runs.

The historical archive is an experimental input, not an opaque directory that
must happen to exist on a machine.  This module records the immutable asset
chain from the twenty Plan 1 behavior runs to the algorithm-specific episode-0
catalogs.  It deliberately does not create trajectory data; the source runs
must be produced by the normal online runner and then indexed by Plan 2.
"""

import os
import subprocess

import yaml

from .core import canonical_digest
from .io import atomic_json, read_json, sha256_file


ASSET_PROTOCOL = 'ha_cross_algorithm_assets_v1'
ASSET_SCHEMA_VERSION = 1
FORMAL_NETWORKS = (
    'sumohz1x1_config2', 'sumohz1x1',
    'sumohz1x1_config4', 'sumohz1x1_config3',
)
ALGORITHMS = ('independent_dqn', 'double_dqn', 'dueling_double_dqn')


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _abs(root, value):
    value = os.path.expanduser(str(value))
    return value if os.path.isabs(value) else os.path.abspath(os.path.join(root, value))


def load_historical_asset_config(path):
    path = os.path.abspath(path)
    with open(path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if config.get('schema_version') != ASSET_SCHEMA_VERSION:
        raise ValueError('Historical asset config requires schema_version=1')
    if config.get('protocol_id') != ASSET_PROTOCOL:
        raise ValueError('Unexpected historical asset protocol')
    if tuple(config.get('networks', ())) != FORMAL_NETWORKS:
        raise ValueError('Historical assets must cover the four frozen networks')
    if config.get('behavior_training_seeds') != [0, 1, 2, 3, 4]:
        raise ValueError('Historical asset behavior seeds must be 0..4')
    if config.get('source_episodes') != 100:
        raise ValueError('Cross-algorithm historical source runs must contain 100 episodes')
    if config.get('archive_episode_range') != [1, 100]:
        raise ValueError('HA archive must use Plan 1 episodes 1..100')
    paths = config.get('paths', {})
    required_paths = {
        'source_root', 'whitelist', 'dataset_root', 'archive_manifest',
        'initial_state_catalog', 'dueling_initial_state_catalog',
        'dueling_initial_root',
    }
    if set(paths) != required_paths:
        raise ValueError(
            f'Historical asset paths must be {sorted(required_paths)}'
        )
    return path, config


def _path_record(root, value):
    path = _abs(root, value)
    return {'path': path, 'exists': os.path.exists(path)}


def build_historical_asset_plan(output_path, config_path):
    config_path, config = load_historical_asset_config(config_path)
    # Repository-relative paths in experiment configs are resolved from the
    # checkout root, matching the existing sequential/offline CLIs.
    root = os.getcwd()
    paths = {
        key: _path_record(root, value)
        for key, value in config['paths'].items()
    }
    source_identities = [
        {'network': network, 'behavior_training_seed': seed}
        for network in config['networks']
        for seed in config['behavior_training_seeds']
    ]
    stages = [
        {
            'stage_id': 'A0_plan1_sources',
            'purpose': 'Generate immutable Plan 1 online DQN source runs',
            'identity_count': 20,
            'required_artifact': 'paths.source_root',
            'acceptance': [
                'each network/behavior seed appears exactly once',
                '100 episodes and episode-0 resumable checkpoint exist',
                'trajectory covers episodes 1..100 with frozen 16/8 semantics',
            ],
        },
        {
            'stage_id': 'A1_source_whitelist',
            'purpose': 'Freeze the approved 20 source identities and paths',
            'identity_count': 20,
            'required_artifact': 'paths.whitelist',
            'acceptance': ['CSV contains exactly run_path, network, behavior_training_seed'],
        },
        {
            'stage_id': 'A2_plan2_q1_index',
            'purpose': 'Build read-only Q1 dataset indexes over episodes 1..100',
            'identity_count': 4,
            'required_artifact': 'paths.dataset_root',
            'acceptance': ['one Q1 manifest per network', 'shard hashes and counts validate'],
        },
        {
            'stage_id': 'A3_behavior_seed_audit',
            'purpose': 'Determine whether same-seed trajectories are transition-identical',
            'identity_count': 1,
            'required_artifact': 'behavior_seed_audit_report',
            'acceptance': ['audit is deterministic and hash-addressed', 'result controls seed exclusion rule'],
        },
        {
            'stage_id': 'A4_hoa_archive_manifest',
            'purpose': 'Freeze the read-only HOA archive and visibility provenance',
            'identity_count': 1,
            'required_artifact': 'paths.archive_manifest',
            'acceptance': ['archive digest and source manifest hashes are recorded'],
        },
        {
            'stage_id': 'A5_initial_state_catalogs',
            'purpose': 'Provide algorithm-specific episode-0 resumable states',
            'identity_count': 20,
            'required_artifact': 'paths.initial_state_catalog',
            'acceptance': [
                'Independent and Double DQN share the audited frozen source state',
                'Dueling conversion preserves episode-0 Q values and has a new catalog hash',
            ],
        },
    ]
    payload = {
        'schema_version': ASSET_SCHEMA_VERSION,
        'protocol_id': ASSET_PROTOCOL,
        'config_path': config_path,
        'config_sha256': sha256_file(config_path),
        'git_commit': _git_commit(),
        'networks': list(config['networks']),
        'behavior_training_seeds': list(config['behavior_training_seeds']),
        'source_episodes': int(config['source_episodes']),
        'archive_episode_range': list(config['archive_episode_range']),
        'algorithms': list(ALGORITHMS),
        'source_identities': source_identities,
        'paths': paths,
        'stages': stages,
        'launch_authorized': False,
        'status': 'designed',
    }
    payload['plan_digest'] = canonical_digest(payload)
    atomic_json(output_path, payload)
    return payload


def validate_historical_asset_plan(path, require_existing=False):
    plan = read_json(path)
    if plan.get('schema_version') != ASSET_SCHEMA_VERSION or plan.get('protocol_id') != ASSET_PROTOCOL:
        raise ValueError('Unsupported historical asset plan')
    unsigned = dict(plan)
    recorded = unsigned.pop('plan_digest', None)
    if canonical_digest(unsigned) != recorded:
        raise ValueError('Historical asset plan digest mismatch')
    if not os.path.isfile(plan.get('config_path', '')):
        raise FileNotFoundError(plan.get('config_path', ''))
    if sha256_file(plan['config_path']) != plan.get('config_sha256'):
        raise ValueError('Historical asset config changed after plan creation')
    if _git_commit() != plan.get('git_commit'):
        raise ValueError('Current checkout differs from historical asset plan commit')
    if len(plan.get('source_identities', [])) != 20:
        raise ValueError('Historical asset plan must contain 20 source identities')
    expected = {
        (network, seed)
        for network in FORMAL_NETWORKS for seed in range(5)
    }
    actual = {
        (item.get('network'), int(item.get('behavior_training_seed')))
        for item in plan['source_identities']
    }
    if actual != expected:
        raise ValueError('Historical source identity matrix is incomplete')
    if [stage.get('stage_id') for stage in plan.get('stages', [])] != [
        'A0_plan1_sources', 'A1_source_whitelist', 'A2_plan2_q1_index',
        'A3_behavior_seed_audit', 'A4_hoa_archive_manifest',
        'A5_initial_state_catalogs',
    ]:
        raise ValueError('Historical asset stage chain is incomplete')
    paths = plan.get('paths', {})
    missing = []
    for key, record in paths.items():
        path_value = record.get('path') if isinstance(record, dict) else None
        if not path_value:
            raise ValueError(f'Historical asset path {key} is malformed')
        exists = os.path.exists(path_value)
        if require_existing and not exists:
            missing.append(path_value)
    if missing:
        raise FileNotFoundError(
            'Historical asset plan has missing required paths: ' + ', '.join(missing)
        )
    return {
        'valid': True,
        'plan_digest': recorded,
        'source_count': len(plan['source_identities']),
        'existing_path_count': sum(
            os.path.exists(item['path']) for item in paths.values()
        ),
        'missing_path_count': sum(
            not os.path.exists(item['path']) for item in paths.values()
        ),
        'require_existing': bool(require_existing),
    }
