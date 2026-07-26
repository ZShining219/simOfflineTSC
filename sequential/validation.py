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
    lower_triangle = (
        child.get('protocol_id') == 'ha_sodqn_b100_v1'
        or child.get('condition') in {'M0', 'M1', 'M2', 'M3'}
    )
    first_stage = 1 if lower_triangle else 2
    expected_episodes = {
        (stage, local)
        for stage, budget in enumerate(child['stage_episodes'], start=1)
        if stage >= first_stage
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
    if lower_triangle:
        expected_evaluations = {
            (1, local, child['networks'][0])
            for local in range(1, int(child['stage_episodes'][0]) + 1)
        }
    else:
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
        visible_networks = (
            child['networks'][:stage]
            if lower_triangle
            else child['networks']
        )
        expected_evaluations.update(
            (stage, int(budget), network) for network in visible_networks
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
        'child_episode_references': trajectory_refs,
    }
    if child.get('protocol_id') == 'ha_sodqn_b100_v1':
        logical_view['initial_state'] = {
            'checkpoint_episode': 0,
            'checkpoint_path': child['initial_checkpoint'],
        }
    else:
        logical_view['parent'] = {
            'episode_count': int(child['stage_episodes'][0]),
            'trajectory_reference': child['parent_trajectory_reference'],
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


def validate_ha_attempt(path):
    validated = validate_attempt(path)
    child = validated['child_manifest']
    if child.get('protocol_id') != 'ha_sodqn_b100_v1':
        raise ValueError('Attempt is not an HA-SODQN run')
    diagnostics = []
    for key, operation in validated['state']['completed_operations'].items():
        if key.endswith(':diagnostic'):
            diagnostics.append(read_json(operation['artifact']))
    diagnostics.sort(key=lambda item: (item['stage_index'], item['local_episode']))
    expected_diagnostics = sum(int(value) for value in child['stage_episodes'])
    if len(diagnostics) != expected_diagnostics:
        raise ValueError('HA-SODQN diagnostic episode count mismatch')
    total_decisions = sum(
        item['transition_count']
        for item in validated['logical_episode_view']['child_episode_references']
    )
    expected_decisions = 360 * expected_diagnostics
    if total_decisions != expected_decisions:
        raise ValueError('HA-SODQN decision budget mismatch')
    final_state = validated['final_checkpoint']['agent_state']
    counters = final_state['counters']
    expected_updates = max(0, total_decisions - 1000)
    if int(counters['global_decision_step']) != total_decisions:
        raise ValueError('HA-SODQN global decision counter mismatch')
    if int(counters['gradient_updates']) != expected_updates:
        raise ValueError('HA-SODQN gradient update budget is unfair')
    if int(counters['target_updates']) != expected_updates // 10:
        raise ValueError('HA-SODQN target update budget mismatch')
    archive_mode = child['archive_mode']
    requested_ratio = float(child['offline_ratio'])
    expected_behavior_seeds = None
    if archive_mode != 'NONE':
        archive = read_json(child['archive_root_manifest'])
        expected_behavior_seeds = set(archive['behavior_training_seeds'])
        if archive['behavior_seed_rule'].get('exclude_matching_training_seed'):
            expected_behavior_seeds.discard(int(child['training_seed']))
    window_count = 0
    owp_digests = {}
    for diagnostic in diagnostics:
        stage = int(diagnostic['stage_index'])
        local_episode = int(diagnostic['local_episode'])
        current = child['networks'][stage - 1]
        allowed_offline = (
            set(child['networks'][:stage - 1])
            if archive_mode == 'P1C' else set(child['networks'])
        )
        if archive_mode == 'NONE':
            allowed_offline = set()
        visibility = diagnostic.get('visible_archive')
        if archive_mode == 'P1C':
            expected_visible = child['networks'][:stage - 1]
            if visibility is None and expected_visible:
                raise ValueError('P1C visible archive is missing')
            if visibility is not None and visibility['visibility']['visible_networks'] != expected_visible:
                raise ValueError('P1C stage visibility mismatch')
        digest = (diagnostic.get('owp_manifest') or {}).get('owp_digest')
        if digest is not None:
            previous = owp_digests.setdefault(stage, digest)
            if previous != digest:
                raise ValueError('OWP changed inside a stage')
        for window in diagnostic.get('sampling_windows', []):
            window_count += 1
            if int(window['online_count']) + int(window['offline_count']) != 64:
                raise ValueError('HA-SODQN mixed batch size is not 64')
            offline_count = int(window['offline_count'])
            actual_ratio = float(window['actual_offline_ratio'])
            if actual_ratio != offline_count / 64:
                raise ValueError('HA-SODQN actual offline ratio is inconsistent')
            offline_sources = {
                network for network, count in
                window.get('offline_sample_count_by_scene', {}).items()
                if int(count) > 0
            }
            if not offline_sources <= allowed_offline:
                raise ValueError(
                    f'P1C current/future leakage: stage={stage}, '
                    f'current={current}, sources={sorted(offline_sources)}'
                )
            fallback = (
                archive_mode == 'NONE'
                or (archive_mode == 'P1C' and stage == 1)
                or local_episode <= 10
            )
            if fallback and offline_count != 0:
                raise ValueError('HA-SODQN fallback/warm-up used offline samples')
            if not fallback:
                expected_offline = int(round(64 * requested_ratio))
                if offline_count != expected_offline:
                    raise ValueError('HA-SODQN offline quota mismatch')
            for key in ('loss_online', 'loss_total'):
                if window.get(key) is None or not np.isfinite(float(window[key])):
                    raise ValueError(f'HA-SODQN {key} is not finite')
            if offline_count and (
                window.get('loss_offline') is None
                or not np.isfinite(float(window['loss_offline']))
            ):
                raise ValueError('HA-SODQN offline loss is not finite')
        if expected_behavior_seeds is not None:
            used_seeds = {
                int(seed) for seed, count in
                diagnostic.get('offline_samples_by_behavior_seed', {}).items()
                if int(count) > 0
            }
            if not used_seeds <= expected_behavior_seeds:
                raise ValueError('HA-SODQN behavior-seed rule was violated')
            used_episodes = {
                int(episode) for episode, count in
                diagnostic.get('offline_samples_by_episode', {}).items()
                if int(count) > 0
            }
            if any(episode < 1 or episode > 100 for episode in used_episodes):
                raise ValueError('HA-SODQN sampled outside Plan 1 episodes 1-100')
    if window_count != expected_updates:
        raise ValueError('HA-SODQN diagnostic update count mismatch')
    attempt_dir = validated['attempt_dir']
    if archive_mode != 'NONE':
        for stage in range(1, len(child['networks']) + 1):
            visibility_path = os.path.join(
                attempt_dir, 'archive',
                f'stage_{stage:02d}_visibility_manifest.json',
            )
            if not os.path.isfile(visibility_path):
                raise FileNotFoundError(visibility_path)
            visibility_manifest = read_json(visibility_path)
            expected_visible = (
                child['networks'][:stage - 1]
                if archive_mode == 'P1C' else child['networks']
            )
            if visibility_manifest['visibility']['visible_networks'] != expected_visible:
                raise ValueError('Persisted archive visibility manifest mismatch')
            if stage in owp_digests:
                owp_path = os.path.join(
                    attempt_dir, 'archive', f'stage_{stage:02d}_owp_manifest.json',
                )
                if not os.path.isfile(owp_path):
                    raise FileNotFoundError(owp_path)
                if read_json(owp_path)['owp_digest'] != owp_digests[stage]:
                    raise ValueError('Persisted OWP manifest digest mismatch')
    return {
        **validated,
        'ha_audit': {
            'valid': True,
            'total_decisions': total_decisions,
            'gradient_updates': expected_updates,
            'target_updates': expected_updates // 10,
            'sampling_window_count': window_count,
            'owp_digest_by_stage': owp_digests,
            'p1c_leakage': False,
            'update_budget_fair': True,
        },
    }


def validate_ha_experiment(manifest_path, output_root):
    plan = read_json(manifest_path)
    runs = []
    for child in plan.get('children', []):
        validated = validate_ha_attempt(
            os.path.join(output_root, child['logical_run_id'])
        )
        runs.append({
            'logical_run_id': child['logical_run_id'],
            'attempt_dir': validated['attempt_dir'],
            'trajectory_digest': validated['trajectory_digest'],
            'final_training_state_digest': validated['final_training_state_digest'],
            'ha_audit': validated['ha_audit'],
        })
    return {'valid': True, 'run_count': len(runs), 'runs': runs}


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
