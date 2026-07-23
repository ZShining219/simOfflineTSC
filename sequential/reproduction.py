import json
import os

import numpy as np

from .core import canonical_digest
from .io import atomic_json, sha256_file
from .parents import ParentSource, validate_parent_source


def plan1_trajectory_digest(run_path):
    trajectory_root = os.path.join(os.path.abspath(run_path), 'trajectory')
    index_path = os.path.join(trajectory_root, 'index.jsonl')
    entries = []
    with open(index_path, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    if [int(row['episode_id']) for row in entries] != list(range(1, 401)):
        raise ValueError('Plan 1 reproduction trajectory must contain episodes 1..400')
    episode_digests = []
    transition_count = 0
    for entry in entries:
        path = os.path.join(trajectory_root, entry['file'])
        if sha256_file(path) != entry['sha256']:
            raise ValueError(f'Plan 1 trajectory shard SHA-256 mismatch: {path}')
        with np.load(path, allow_pickle=False) as shard:
            count = len(shard['decision_step'])
            if count != int(entry['transition_count']):
                raise ValueError(f'Plan 1 trajectory count mismatch: {path}')
            semantic = {
                'stage_index': np.ones(count, dtype=np.int64),
                'local_episode': shard['episode_id'],
                'global_episode': shard['episode_id'],
                'decision_index': shard['decision_step'],
                'global_decision_step': shard['global_step'],
                'state': shard['state'], 'phase': shard['current_phase'],
                'action': shard['action'], 'reward': shard['reward'],
                'next_state': shard['next_state'], 'next_phase': shard['next_phase'],
                'terminated': shard['terminated'], 'truncated': shard['truncated'],
            }
            episode_digests.append({
                'episode': int(entry['episode_id']),
                'transition_count': count,
                'canonical_digest': canonical_digest(semantic),
            })
            transition_count += count
    return {
        'episode_count': len(entries), 'transition_count': transition_count,
        'episode_digests': episode_digests,
        'canonical_trajectory_digest': canonical_digest(episode_digests),
    }


def plan1_reproduction_signature(
    run_path, network, training_seed=0, require_plan1_formal=True,
):
    parent = validate_parent_source(ParentSource(
        os.path.abspath(run_path), network, int(training_seed),
    ), require_plan1_formal=require_plan1_formal)
    return {
        'run_path': os.path.abspath(run_path), 'network': network,
        'training_seed': int(training_seed),
        'trajectory': plan1_trajectory_digest(run_path),
        'online_parameter_digest': parent['digests']['online_parameter_digest'],
        'target_parameter_digest': parent['digests']['target_parameter_digest'],
        'optimizer_state_digest': parent['digests']['optimizer_state_digest'],
        'replay_content_digest': parent['digests']['replay_content_digest'],
        'replay_metadata_digest': parent['digests']['replay_metadata_digest'],
        'rng_state_digest': parent['digests']['rng_state_digest'],
        'epsilon': parent['epsilon'],
        'global_decision_step': parent['global_decision_step'],
        'gradient_updates': parent['gradient_updates'],
        'target_updates': parent['target_updates'],
        'next_target_sync_update': parent['next_target_sync_update'],
    }


def compare_plan1_reproduction(source_run, reproduction_run, network,
                               output_path=None, training_seed=0):
    source = plan1_reproduction_signature(source_run, network, training_seed)
    reproduction = plan1_reproduction_signature(
        reproduction_run, network, training_seed,
        require_plan1_formal=False,
    )
    fields = (
        'online_parameter_digest', 'target_parameter_digest',
        'optimizer_state_digest', 'replay_content_digest',
        'replay_metadata_digest', 'rng_state_digest', 'epsilon',
        'global_decision_step', 'gradient_updates', 'target_updates',
        'next_target_sync_update',
    )
    checks = {field: source[field] == reproduction[field] for field in fields}
    checks['canonical_trajectory_digest'] = (
        source['trajectory']['canonical_trajectory_digest']
        == reproduction['trajectory']['canonical_trajectory_digest']
    )
    checks['episode_trajectory_digests'] = (
        source['trajectory']['episode_digests']
        == reproduction['trajectory']['episode_digests']
    )
    report = {
        'schema_version': 1, 'network': network,
        'training_seed': int(training_seed), 'source': source,
        'reproduction': reproduction, 'checks': checks,
        'valid': all(checks.values()),
    }
    if output_path is not None:
        atomic_json(output_path, report)
    return report
