import json
import os
import re

import numpy as np

from .checkpoint import load_full_checkpoint
from .core import canonical_digest, canonical_transition_digest
from .io import read_json, sha256_file


def validate_hybrid_stage_visibility(pool, stage_index):
    """Ensure a hybrid pool exposes only completed historical stages."""
    visible = set(pool.visible_historical_stages())
    expected = set(range(1, int(stage_index)))
    if visible != expected:
        raise ValueError(
            f'hybrid stage leakage or missing pool: visible={sorted(visible)}, '
            f'expected={sorted(expected)}'
        )
    return {'valid': True, 'stage_index': int(stage_index),
            'visible_historical_stages': sorted(visible)}


TRAJECTORY_KEY = re.compile(r'^stage_(\d+):episode_(\d+):trajectory$')
EVALUATION_KEY = re.compile(
    r'^evaluation:stage_(\d+):local_(\d+):(.+)$'
)


def resolve_effective_attempt(path):
    path = os.path.abspath(path)
    logical_manifest = os.path.join(path, 'logical_run_manifest.json')
    if os.path.isfile(logical_manifest):
        lineage = read_json(logical_manifest)
        attempt_id = lineage.get('effective_attempt')
        if not attempt_id:
            raise ValueError(f'Logical run has no effective attempt: {path}')
        return os.path.join(path, 'attempts', attempt_id)
    if os.path.isfile(os.path.join(path, 'current_state.json')):
        return path
    raise FileNotFoundError(f'Cannot resolve Sequential attempt: {path}')


def _trajectory_records(path):
    with np.load(path, allow_pickle=False) as shard:
        count = len(shard['stage_index'])
        fields = (
            'stage_index', 'local_episode', 'global_episode', 'decision_index',
            'global_decision_step', 'state', 'phase', 'action', 'reward',
            'next_state', 'next_phase', 'terminated', 'truncated',
        )
        # Each compressed NPZ member is an independent stream. Materialize
        # every field once instead of repeatedly decompressing it per record.
        arrays = {field: shard[field] for field in fields}
        for index in range(count):
            record = {field: arrays[field][index] for field in fields}
            # The trainer stores reward as a zero-dimensional ndarray.  NPZ
            # indexing returns its scalar, so restore the canonical type.
            record['reward'] = np.array(record['reward'], copy=True)
            record['terminated'] = bool(record['terminated'])
            record['truncated'] = bool(record['truncated'])
            yield record


def validate_trajectory_marker(marker_path):
    marker = read_json(marker_path)
    shard_path = marker['trajectory_path']
    if not os.path.isfile(shard_path):
        raise FileNotFoundError(f'Trajectory shard is missing: {shard_path}')
    if sha256_file(shard_path) != marker['trajectory_sha256']:
        raise ValueError(f'Trajectory shard SHA-256 mismatch: {shard_path}')
    digests = [canonical_transition_digest(row) for row in _trajectory_records(shard_path)]
    if digests != marker['canonical_transition_digests']:
        raise ValueError(f'Canonical trajectory mismatch: {marker_path}')
    if len(digests) != int(marker['transition_count']):
        raise ValueError(f'Trajectory transition count mismatch: {marker_path}')
    return digests


def validate_evaluation_alias(alias_path):
    alias = read_json(alias_path)
    identity = alias['identity']
    committed_path = alias['physical_committed_path']
    committed = read_json(committed_path)
    physical_key = canonical_digest({
        'checkpoint_digest': identity['checkpoint_digest'],
        'evaluation_network': identity['evaluation_network'],
        'evaluation_protocol_digest': identity['evaluation_protocol_digest'],
    })
    if physical_key != alias['physical_key'] or physical_key != committed['physical_key']:
        raise ValueError(f'Evaluation physical identity mismatch: {alias_path}')
    for field in (
        'checkpoint_digest', 'evaluation_network', 'evaluation_protocol_digest'
    ):
        if committed[field] != identity[field]:
            raise ValueError(f'Evaluation committed {field} mismatch: {alias_path}')
    summary = read_json(committed['summary_path'])
    if summary['checkpoint_digest'] != identity['checkpoint_digest']:
        raise ValueError(f'Evaluation summary checkpoint mismatch: {alias_path}')
    with open(committed['decisions_path'], encoding='utf-8') as handle:
        decisions = [json.loads(line) for line in handle if line.strip()]
    if len(decisions) != int(summary['decision_steps']):
        raise ValueError(f'Evaluation decision count mismatch: {alias_path}')
    if [row['decision_index'] for row in decisions] != list(
        range(1, len(decisions) + 1)
    ):
        raise ValueError(f'Evaluation decision sequence is not contiguous: {alias_path}')
    return {'alias': alias, 'committed': committed, 'summary': summary}


def _operation_maps(state):
    trajectories = {}
    evaluations = {}
    diagnostics = {}
    for key, operation in state['completed_operations'].items():
        match = TRAJECTORY_KEY.match(key)
        if match:
            trajectories[(int(match.group(1)), int(match.group(2)))] = operation
            continue
        match = EVALUATION_KEY.match(key)
        if match:
            evaluations[(int(match.group(1)), int(match.group(2)), match.group(3))] = operation
            continue
        if key.endswith(':diagnostic'):
            diagnostics[key] = operation
    return trajectories, evaluations, diagnostics


def validate_attempt(path):
    attempt_dir = resolve_effective_attempt(path)
    state = read_json(os.path.join(attempt_dir, 'current_state.json'))
    child = read_json(os.path.join(attempt_dir, 'child_run_manifest.json'))
    completed = read_json(os.path.join(attempt_dir, 'completed.json'))
    if state['phase'] != 'COMPLETED' or completed['phase'] != 'COMPLETED':
        raise ValueError(f'Sequential attempt is not completed: {attempt_dir}')
    if state['logical_run_id'] != child['logical_run_id']:
        raise ValueError('Logical run identity differs between state and child manifest')
    trajectories, evaluations, diagnostics = _operation_maps(state)
    expected_episodes = {
        (stage, local)
        for stage, budget in enumerate(child['stage_episodes'], start=1)
        if stage > 1
        for local in range(1, int(budget) + 1)
    }
    if set(trajectories) != expected_episodes:
        raise ValueError('Committed child trajectory episode set is incomplete')
    trajectory_digests = []
    trajectory_refs = []
    for identity in sorted(trajectories):
        marker_path = trajectories[identity]['artifact']
        digests = validate_trajectory_marker(marker_path)
        trajectory_digests.extend(digests)
        trajectory_refs.append({
            'stage_index': identity[0], 'local_episode': identity[1],
            'marker_path': marker_path, 'transition_count': len(digests),
        })
    expected_evaluations = {
        (1, int(child['stage_episodes'][0]), network)
        for network in child['networks']
    }
    for stage, budget in enumerate(child['stage_episodes'], start=1):
        if stage == 1:
            continue
        training_network = child['networks'][stage - 1]
        expected_evaluations.update(
            (stage, local, training_network)
            for local in range(0, int(budget) + 1)
        )
        expected_evaluations.update(
            (stage, int(budget), network) for network in child['networks']
        )
    if set(evaluations) != expected_evaluations:
        missing = sorted(expected_evaluations - set(evaluations))
        extra = sorted(set(evaluations) - expected_evaluations)
        raise ValueError(f'Evaluation cell set mismatch; missing={missing}, extra={extra}')
    evaluation_cells = {}
    for identity in sorted(evaluations):
        validated = validate_evaluation_alias(evaluations[identity]['artifact'])
        alias_identity = validated['alias']['identity']
        expected_identity = {
            'stage_index': identity[0], 'local_episode': identity[1],
            'evaluation_network': identity[2],
        }
        if any(alias_identity[key] != value for key, value in expected_identity.items()):
            raise ValueError('Evaluation operation key and alias identity differ')
        evaluation_cells[identity] = validated
    if len(diagnostics) != len(expected_episodes):
        raise ValueError('Replay diagnostic episode set is incomplete')
    for operation in diagnostics.values():
        if not os.path.isfile(operation['artifact']):
            raise FileNotFoundError(operation['artifact'])
    stage_checkpoints = {}
    for stage in range(1, len(child['networks']) + 1):
        operation = state['completed_operations'].get(f'stage_{stage}:checkpoint')
        if operation is None:
            raise ValueError(f'Missing stage checkpoint operation: stage {stage}')
        stage_checkpoints[stage] = load_full_checkpoint(
            operation['artifact'], expected_type='stage_boundary'
        )
    final_checkpoint = stage_checkpoints[len(child['networks'])]
    final_digest = canonical_digest(final_checkpoint['agent_state'])
    if final_digest != final_checkpoint['canonical_state_digest']:
        raise ValueError('Final training state canonical digest mismatch')
    logical_view = {
        'total_episode_count': sum(int(value) for value in child['stage_episodes']),
        'parent': {
            'episode_count': int(child['stage_episodes'][0]),
            'trajectory_reference': child['parent_trajectory_reference'],
        },
        'child_episode_references': trajectory_refs,
    }
    return {
        'valid': True, 'attempt_dir': attempt_dir,
        'logical_run_id': child['logical_run_id'], 'policy': child['policy'],
        'child_manifest': child, 'state': state,
        'trajectory_digests': trajectory_digests,
        'trajectory_digest': canonical_digest(trajectory_digests),
        'evaluation_cells': evaluation_cells,
        'final_checkpoint': final_checkpoint,
        'final_training_state_digest': final_digest,
        'logical_episode_view': logical_view,
    }


def _evaluation_signature(validated):
    return {
        identity: {
            'physical_key': cell['alias']['physical_key'],
            'summary': {
                key: value for key, value in cell['summary'].items()
                if key != 'wall_time_seconds'
            },
        }
        for identity, cell in validated['evaluation_cells'].items()
    }


def compare_recovery_pair(control_path, fault_path):
    control = validate_attempt(control_path)
    fault = validate_attempt(fault_path)
    if control['child_manifest']['policy'] != fault['child_manifest']['policy']:
        raise ValueError('Recovery pair policies differ')
    control_agent = control['final_checkpoint']['agent_state']
    fault_agent = fault['final_checkpoint']['agent_state']
    checks = {
        'trajectory_equal': control['trajectory_digests'] == fault['trajectory_digests'],
        'evaluation_keys_equal': (
            set(control['evaluation_cells']) == set(fault['evaluation_cells'])
        ),
        'evaluation_physical_equal': (
            _evaluation_signature(control) == _evaluation_signature(fault)
        ),
        'final_training_state_equal': canonical_digest(control_agent) == canonical_digest(
            fault_agent
        ),
        'final_phase_control': control['state']['phase'],
        'final_phase_fault': fault['state']['phase'],
    }
    checks['valid'] = all(
        value is True for key, value in checks.items()
        if key.endswith('_equal')
    ) and checks['final_phase_control'] == checks['final_phase_fault'] == 'COMPLETED'
    return checks


def validate_experiment_outputs(manifest_path, output_root):
    plan = read_json(manifest_path)
    runs = []
    pairs = []
    by_policy_variant = {}
    for child in plan['children']:
        logical_root = os.path.join(output_root, child['logical_run_id'])
        validated = validate_attempt(logical_root)
        runs.append({
            'logical_run_id': validated['logical_run_id'],
            'attempt_dir': validated['attempt_dir'],
            'trajectory_digest': validated['trajectory_digest'],
            'final_training_state_digest': validated['final_training_state_digest'],
            'evaluation_cell_count': len(validated['evaluation_cells']),
            'logical_episode_view': validated['logical_episode_view'],
        })
        if child.get('variant'):
            by_policy_variant[(child['policy'], child['variant'])] = logical_root
    if plan.get('mode') == 'pilot':
        for policy in ('clear', 'fifo', 'fifo_matched_wait'):
            pairs.append({
                'policy': policy,
                **compare_recovery_pair(
                    by_policy_variant[(policy, 'control')],
                    by_policy_variant[(policy, 'fault')],
                ),
            })
    return {
        'valid': all(pair.get('valid', True) for pair in pairs),
        'mode': plan['mode'], 'run_count': len(runs),
        'runs': runs, 'recovery_pairs': pairs,
    }
