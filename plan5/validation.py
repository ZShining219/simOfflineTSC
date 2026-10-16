"""Fail-closed validators for Plan5 identities, checkpoints and outputs."""
import math
import os
from pathlib import Path

from sequential.core import canonical_digest
from sequential.io import read_json, sha256_file

from .checkpoint import load_checkpoint
from .config import ALGORITHMS, SCENES, SEEDS, TRANSITIONS
from .evaluator import formal_evaluation_schedule, full_checkpoint_schedule
from .manifest import validate_same_start
from .probe import validate_probe_manifest
from .schema import (
    JSON_REQUIRED_FIELDS, TABLE_SCHEMAS, validate_json_fields,
    validate_table_header,
)


def validate_sumo_command(command):
    command = list(command)
    forbidden = {'--seed', '--random'}
    present = sorted(forbidden & set(command))
    if present:
        raise ValueError(f'Plan5 SUMO command contains forbidden flags: {present}')
    if not command or os.path.basename(command[0]) not in {'sumo', 'sumo.exe'}:
        raise ValueError('Plan5 SUMO command does not resolve the sumo binary')
    return True


def validate_transition_row(row):
    transition = row.get('transition_id')
    if transition not in TRANSITIONS:
        raise ValueError('Unknown Plan5 transition identity')
    expected_source, expected_target = TRANSITIONS[transition]
    if row.get('source_scene') != expected_source \
            or row.get('target_scene') != expected_target:
        raise ValueError('Plan5 transition source/target mismatch')
    if row.get('algorithm_id') not in ALGORITHMS:
        raise ValueError('Unauthorized Plan5 algorithm')
    if int(row.get('training_seed', -1)) not in SEEDS:
        raise ValueError('Unauthorized Plan5 training seed')
    return True


def validate_launch_row(row):
    """Require every late-bound identity before reserving a formal attempt."""
    from .manifest import validate_run_manifest
    validate_run_manifest([row])
    if len(str(row.get('config_sha256', ''))) != 64 \
            or len(str(row.get('source_commit', ''))) != 40:
        raise ValueError('Plan5 launch source/config digest length is invalid')
    if row['algorithm_id'] == 'PPO':
        if not row.get('ppo_config_path') \
                or len(str(row.get('ppo_config_sha256', ''))) != 64:
            raise ValueError('Plan5 PPO launch row lacks frozen config binding')
    if row['run_type'] == 'TRANSITION':
        required = {
            'source_checkpoint_path', 'source_checkpoint_sha256',
            'model_state_digest', 'optimizer_state_digest',
            'algorithm_state_digest', 'rng_state_digest',
        }
        if any(not row.get(field) for field in required):
            raise ValueError('Plan5 transition launch row is not checkpoint-bound')
        validate_checkpoint_binding(row)
    return True


def validate_checkpoint_binding(row):
    path = row['source_checkpoint_path']
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if sha256_file(path) != row['source_checkpoint_sha256']:
        raise ValueError('Plan5 source checkpoint SHA mismatch')
    payload = load_checkpoint(path)
    identity = payload['identity']
    for field in ('algorithm_id', 'training_seed'):
        if identity.get(field) != row.get(field):
            raise ValueError(f'Plan5 source checkpoint {field} mismatch')
    expected_source = (
        row.get('scene') if row.get('run_type') == 'ANCHOR'
        else row.get('source_scene')
    )
    if identity.get('scene', identity.get('source_scene')) != expected_source:
        raise ValueError('Plan5 source checkpoint scene mismatch')
    return payload


def validate_same_start_matrix(rows):
    index = {
        (row['algorithm_id'], row['transition_id'], int(row['training_seed'])): row
        for row in rows if row.get('run_type') == 'TRANSITION'
    }
    present_algorithms = {row['algorithm_id'] for row in rows}
    if not present_algorithms or not present_algorithms <= set(ALGORITHMS):
        raise ValueError('Plan5 same-start matrix has invalid algorithms')
    expected = {
        (algorithm, transition, seed)
        for algorithm in present_algorithms
        for transition in TRANSITIONS
        for seed in SEEDS
    }
    if set(index) != expected:
        raise ValueError('Plan5 transition matrix is incomplete')
    for algorithm in present_algorithms:
        for seed in SEEDS:
            h34 = index.get((algorithm, 'H34', seed))
            l32 = index.get((algorithm, 'L32', seed))
            if h34 is None or l32 is None:
                raise ValueError('Plan5 same-start matrix is incomplete')
            validate_same_start(h34, l32)
    return True


def _finite(value, label):
    try:
        valid = math.isfinite(float(value))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f'Plan5 non-finite or missing metric: {label}')


def validate_formal_records(row, episode_summaries, evaluations,
                            probe_records):
    """Validate the complete in-memory matrix before a run may be completed."""
    summaries = list(episode_summaries)
    if [int(item.get('episode', -1)) for item in summaries] != list(
            range(1, 101)):
        raise ValueError('Plan5 formal training episode matrix is incomplete')
    previous_decisions = None
    for item in summaries:
        episode = int(item['episode'])
        required = {
            'decision_steps', 'mean_reward', 'reward_std', 'action_counts',
            'action_frequencies', 'model_input_summary', 'counters_end',
            'trajectory_digest',
        }
        if required - set(item):
            raise ValueError('Plan5 episode audit fields are incomplete')
        if int(item['decision_steps']) != 360:
            raise ValueError('Plan5 episode decision count is not 360')
        _finite(item['mean_reward'], f'episode {episode} mean_reward')
        _finite(item['reward_std'], f'episode {episode} reward_std')
        if item.get('loss_mean') is not None:
            _finite(item['loss_mean'], f'episode {episode} loss_mean')
        counts = list(item['action_counts'])
        frequencies = list(item['action_frequencies'])
        if len(counts) != 8 or any(
                int(value) != value or int(value) < 0 for value in counts
        ) or sum(int(value) for value in counts) != 360:
            raise ValueError('Plan5 episode action counts are invalid')
        if len(frequencies) != 8:
            raise ValueError('Plan5 episode action frequencies are invalid')
        for index, value in enumerate(frequencies):
            _finite(value, f'episode {episode} action_frequency_{index}')
            if not math.isclose(
                    float(value), int(counts[index]) / 360.0,
                    rel_tol=0.0, abs_tol=1e-15):
                raise ValueError('Plan5 action frequency/count mismatch')
        model = item['model_input_summary']
        expected_dim = 20 if row['algorithm_id'] == 'CTXDDQN' else 16
        if int(model.get('feature_dim', -1)) != expected_dim:
            raise ValueError('Plan5 model-input feature width mismatch')
        for statistic in ('mean', 'std', 'min', 'max'):
            values = list(model.get(statistic, ()))
            if len(values) != expected_dim:
                raise ValueError('Plan5 model-input summary is incomplete')
            for index, value in enumerate(values):
                _finite(
                    value,
                    f'episode {episode} model_input_{statistic}_{index}',
                )
        counters = item['counters_end']
        decisions = int(counters.get('global_decision_step', -1))
        if decisions < 360 or (
                previous_decisions is not None
                and decisions - previous_decisions != 360):
            raise ValueError('Plan5 global decision counter is discontinuous')
        previous_decisions = decisions
        if not item['trajectory_digest']:
            raise ValueError('Plan5 episode trajectory digest is empty')
        if row['algorithm_id'] in {'DDQN', 'CTXDDQN'}:
            for field in (
                    'gradient_updates', 'loss_mean', 'epsilon_end',
                    'replay_size_end'):
                if field not in item:
                    raise ValueError('Plan5 DQN episode metrics are incomplete')
            _finite(item['epsilon_end'], f'episode {episode} epsilon_end')
        elif 'minibatch_updates' not in item:
            raise ValueError('Plan5 PPO episode metrics are incomplete')

    if row['run_type'] in {'ANCHOR', 'CALIBRATION'}:
        schedule = formal_evaluation_schedule('ANCHOR', scene=row['scene'])
    else:
        schedule = formal_evaluation_schedule(
            'TRANSITION', source_scene=row['source_scene'],
            target_scene=row['target_scene'],
        )
    expected_cells = [
        (int(cell['episode']), cell['scene'], cell['role'])
        for cell in schedule
    ]
    actual_cells = [
        (int(item.get('episode', -1)), item.get('scene'), item.get('role'))
        for item in evaluations
    ]
    if actual_cells != expected_cells:
        raise ValueError('Plan5 frozen evaluation matrix is incomplete')
    for item in evaluations:
        summary = item.get('summary', {})
        if summary.get('network') != SCENES[item['scene']] \
                or int(summary.get('decision_steps', -1)) != 360 \
                or int(summary.get('simulation_duration_s', -1)) != 3600:
            raise ValueError('Plan5 frozen evaluation protocol mismatch')
        for metric in (
                'travel_time', 'throughput', 'mean_reward', 'mean_queue',
                'mean_delay'):
            _finite(
                summary.get(metric),
                f"evaluation {item['episode']}/{item['scene']} {metric}",
            )
        validate_sumo_command(summary.get('resolved_sumo_command', ()))

    if row['run_type'] == 'CALIBRATION':
        if probe_records:
            raise ValueError('PPO calibration must not emit probe records')
        expected_probe_keys = []
    else:
        scene = row.get('scene', row.get('source_scene'))
        episodes = (
            (0, 100) if row['run_type'] == 'ANCHOR'
            else full_checkpoint_schedule('TRANSITION')
        )
        expected_probe_keys = [
            (episode, scene, split)
            for episode in episodes for split in ('main', 'heldout')
        ]
    actual_probe_keys = [
        (int(item.get('episode', -1)), item.get('scene'), item.get('split'))
        for item in probe_records
    ]
    if actual_probe_keys != expected_probe_keys:
        raise ValueError('Plan5 fixed-probe record matrix is incomplete')
    for item in probe_records:
        expected_rows = 1440 if item['split'] == 'main' else 360
        if int(item.get('row_count', -1)) != expected_rows:
            raise ValueError('Plan5 fixed-probe row count mismatch')
        if not item.get('actions_digest') \
                or not item.get('policy_vectors_digest'):
            raise ValueError('Plan5 fixed-probe digest is empty')
    return {
        'episode_count': len(summaries),
        'evaluation_count': len(evaluations),
        'probe_record_count': len(probe_records),
        'finite_metrics': True,
        'late_metrics_present': all(
            any(
                int(item['episode']) == episode
                and item['role'] == 'current'
                for item in evaluations
            ) for episode in range(91, 101)
        ),
    }


def validate_formal_artifacts(row, episode_summaries, evaluations,
                              probe_records, checkpoint_index_path,
                              *, resume_from=None):
    """Validate hashes, isolation manifests, and resumable checkpoints."""
    report = validate_formal_records(
        row, episode_summaries, evaluations, probe_records,
    )
    for item in evaluations:
        path = Path(item['manifest_path'])
        if not path.is_file() or sha256_file(path) != item['manifest_sha256']:
            raise ValueError('Plan5 evaluation manifest hash/path mismatch')
        manifest = read_json(path)
        if not manifest.get('valid') or not all(
                manifest.get('checks', {}).values()):
            raise ValueError('Plan5 evaluation isolation proof is invalid')
    for item in probe_records:
        path = Path(item['path'])
        if not path.is_file() or sha256_file(path) != item['sha256']:
            raise ValueError('Plan5 probe result hash/path mismatch')

    index = read_json(checkpoint_index_path)
    entries = index.get('checkpoints', [])
    schedule_type = (
        'ANCHOR' if row['run_type'] == 'CALIBRATION' else row['run_type']
    )
    expected_keys = {
        ('evaluation_snapshot', episode) for episode in range(101)
    } | {
        ('full_resumable', episode)
        for episode in full_checkpoint_schedule(schedule_type)
    }
    actual_keys = {
        (item.get('checkpoint_type'), int(item.get('episode', -1)))
        for item in entries
    }
    if len(entries) != len(actual_keys) or actual_keys != expected_keys:
        raise ValueError('Plan5 checkpoint index matrix is incomplete')
    episode100 = None
    for item in entries:
        path = Path(item['path'])
        if not path.is_file() or sha256_file(path) != item['sha256']:
            raise ValueError('Plan5 checkpoint index hash/path mismatch')
        if item['checkpoint_type'] == 'full_resumable':
            payload = load_checkpoint(path, {
                'logical_run_id': row['logical_run_id'],
                'algorithm_id': row['algorithm_id'],
                'training_seed': int(row['training_seed']),
            })
            if int(payload['episode']) != int(item['episode']) \
                    or int(payload.get('extra_state', {}).get(
                        'completed_episode', -1)) != int(item['episode']):
                raise ValueError('Plan5 full checkpoint is not resumable')
            if int(item['episode']) == 100:
                episode100 = item
    if episode100 is None:
        raise ValueError('Plan5 episode100 full checkpoint is missing')
    if resume_from is not None:
        load_checkpoint(resume_from, {
            'logical_run_id': row['logical_run_id'],
            'algorithm_id': row['algorithm_id'],
            'training_seed': int(row['training_seed']),
        })
    return {
        **report,
        'checkpoint_count': len(entries),
        'checkpoint_valid': True,
        'episode100_resume_valid': True,
        'evaluation_isolation_valid': True,
        'fixed_probe_valid': True,
        'valid': True,
    }


def validate_output_package(root, *, allow_incomplete=False):
    root = Path(root)
    required_json = {
        'plan5_validation_report.json', 'plan5_resource_report.json',
        'canonical_reference_manifest.json', 'plan5_fixedtime_reference.json',
        'ppo_formal_config.json',
    }
    required_tables = set(TABLE_SCHEMAS)
    missing = sorted(
        name for name in required_json | required_tables
        if not (root / name).is_file()
    )
    if missing and not allow_incomplete:
        raise FileNotFoundError(f'Plan5 output package missing: {missing}')
    checked = []
    for name in sorted(set(TABLE_SCHEMAS) - set(missing)):
        validate_table_header(root / name, name)
        checked.append(name)
    json_checked = []
    for name in sorted((set(JSON_REQUIRED_FIELDS) - {'probe_manifest.json'}) - set(missing)):
        validate_json_fields(root / name, name)
        json_checked.append(name)
    probe_path = root / 'probe/plan5_fixed_probe_v1/probe_manifest.json'
    if probe_path.is_file():
        validate_probe_manifest(probe_path)
    elif not allow_incomplete:
        raise FileNotFoundError(probe_path)
    return {
        'valid': not missing, 'missing': missing,
        'tables_checked': checked, 'json_checked': json_checked,
    }


def _validate_hashed_file(path, expected_sha, label):
    path = Path(path)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError(f'Plan5 {label} hash/path mismatch')
    return path


def validate_fixedtime_evidence(reference_path, evidence_root):
    reference_path = Path(reference_path)
    payload = read_json(reference_path)
    if payload.get('protocol') != 'Plan5 common FixedTime evaluator' \
            or set(payload.get('scenes', {})) != set(SCENES) \
            or payload.get('digest') != canonical_digest(payload['scenes']):
        raise ValueError('Plan5 FixedTime reference identity is invalid')
    evidence_root = Path(evidence_root)
    results = {}
    for scene, frozen in payload['scenes'].items():
        if not frozen.get('repeated'):
            raise ValueError('Plan5 FixedTime repeatability flag is false')
        summaries = []
        paths = []
        for repeat in (1, 2):
            matches = sorted((evidence_root / scene / f'repeat_{repeat}').glob(
                'attempt_*_summary.json'
            ))
            if len(matches) != 1:
                raise ValueError('Plan5 FixedTime repeat evidence is incomplete')
            paths.append(matches[0])
            summaries.append(read_json(matches[0]))
        for summary in summaries:
            if summary.get('network') != SCENES[scene] \
                    or int(summary.get('decision_steps', -1)) != 360 \
                    or int(summary.get('simulation_duration_s', -1)) != 3600:
                raise ValueError('Plan5 FixedTime evaluator protocol mismatch')
            validate_sumo_command(summary.get('resolved_sumo_command', ()))
            for metric in (
                    'travel_time', 'throughput', 'mean_reward', 'mean_queue',
                    'mean_delay'):
                _finite(summary.get(metric), f'FixedTime {scene} {metric}')
        identities = {
            (summary['travel_time'], summary['trajectory_digest'])
            for summary in summaries
        }
        expected = {(frozen['travel_time'], frozen['trajectory_digest'])}
        if identities != expected:
            raise ValueError('Plan5 FixedTime repeated results changed')
        results[scene] = {
            'travel_time': frozen['travel_time'],
            'trajectory_digest': frozen['trajectory_digest'],
            'repeat_summaries': [str(path.resolve()) for path in paths],
            'repeat_summary_sha256': [sha256_file(path) for path in paths],
        }
    return {
        'valid': True, 'reference': str(reference_path.resolve()),
        'reference_sha256': sha256_file(reference_path), 'scenes': results,
    }


def validate_canonical_reference_evidence(path):
    payload = read_json(path)
    digest = payload.pop('manifest_sha256', None)
    if digest != canonical_digest(payload):
        raise ValueError('Plan5 canonical reference manifest digest mismatch')
    assets = payload.get('assets', [])
    expected = {
        (network, seed) for network in SCENES.values() for seed in SEEDS
    }
    actual = {
        (item.get('scene'), int(item.get('training_seed', -1)))
        for item in assets
    }
    if len(assets) != 20 or actual != expected:
        raise ValueError('Plan5 canonical reference matrix is incomplete')
    for item in assets:
        if item.get('availability') != 'available' \
                or int(item.get('episode', -1)) != 100:
            raise ValueError('Plan5 canonical reference asset is unavailable')
        _validate_hashed_file(
            item['checkpoint_path'], item['checkpoint_sha256'],
            'canonical reference checkpoint',
        )
        if item.get('validation_error') is not None \
                or not item.get('source_run_id') \
                or not item.get('state_digests'):
            raise ValueError('Plan5 canonical reference proof is incomplete')
    return {
        'valid': True, 'manifest': str(Path(path).resolve()),
        'manifest_sha256': sha256_file(path),
        'manifest_digest': digest, 'available_counts': {
            'available': 20, 'external': 0, 'missing': 0,
        },
    }


def validate_smoke_evidence(report_path, algorithm_id):
    report = read_json(report_path)
    expected_episodes = 1 if algorithm_id == 'PPO' else 3
    if report.get('algorithm_id') != algorithm_id \
            or int(report.get('training_seed', -1)) != 999 \
            or int(report.get('episodes', -1)) != expected_episodes \
            or int(report.get('decisions_per_episode', -1)) != 360 \
            or int(report.get('exit_code', -1)) != 0 \
            or not report.get('valid'):
        raise ValueError(f'Plan5 {algorithm_id} smoke identity is invalid')
    _validate_hashed_file(
        report['checkpoint_path'], report['checkpoint_sha256'],
        f'{algorithm_id} smoke checkpoint',
    )
    isolation = report.get('evaluation_isolation', {})
    if not isolation.get('valid') or not all(
            isolation.get('checks', {}).values()):
        raise ValueError(f'Plan5 {algorithm_id} smoke isolation failed')
    counters = report.get('counters', {})
    if algorithm_id in {'DDQN', 'CTXDDQN'}:
        expected_counters = {
            'global_decision_step': 1080,
            'gradient_updates': 80, 'target_updates': 8,
        }
    else:
        expected_counters = {
            'global_decision_step': 360, 'rollout_updates': 1,
        }
        ownership = report.get('parameter_ownership', {})
        if ownership.get('total_parameters') != 5833 \
                or sum(ownership.get(field, 0) for field in (
                    'shared_parameters', 'actor_only_parameters',
                    'critic_only_parameters')) != 5833:
            raise ValueError('Plan5 PPO parameter ownership is invalid')
    if counters != expected_counters:
        raise ValueError(f'Plan5 {algorithm_id} smoke counters mismatch')
    return {
        'valid': True, 'report': str(Path(report_path).resolve()),
        'report_sha256': sha256_file(report_path),
        'checkpoint_sha256': report['checkpoint_sha256'],
        'counters': counters, 'wall_time_seconds': report['wall_time_seconds'],
        'evaluation_isolation_checks': isolation['checks'],
        'invocation_command': report['invocation_command'],
    }


def validate_resume_evidence(report_path, algorithm_id):
    report = read_json(report_path)
    if report.get('algorithm_id') != algorithm_id \
            or int(report.get('training_seed', -1)) != 998 \
            or int(report.get('exit_code', -1)) != 0 \
            or not report.get('valid') \
            or not all(report.get(field) for field in (
                'state_equal', 'metrics_equal', 'counters_equal')) \
            or report.get('continuous_state_digest') != report.get(
                'resumed_state_digest'):
        raise ValueError(f'Plan5 {algorithm_id} resume equivalence failed')
    for execution in report.get('executions', {}).values():
        _validate_hashed_file(
            execution['checkpoint_path'], execution['checkpoint_sha256'],
            f'{algorithm_id} resume checkpoint',
        )
        validate_sumo_command(execution['resolved_sumo_command'])
    return {
        'valid': True, 'report': str(Path(report_path).resolve()),
        'report_sha256': sha256_file(report_path),
        'final_episode': report['final_episode'],
        'split_episode': report['split_episode'],
        'state_digest': report['continuous_state_digest'],
        'invocation_command': report['invocation_command'],
    }
