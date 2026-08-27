"""Run one four-stage arterial experiment with frozen evaluation gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arterial.experiment import (ARTERIAL_SCENES, SCENE_ORDERS,
                                 build_cross_scene_evaluation_manifest)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            'w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
        temporary = Path(handle.name)
    temporary.replace(path)


def canonical_hash(value):
    content = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':')
    ).encode('utf-8')
    return hashlib.sha256(content).hexdigest()


def load_frozen_hoa(path, expected_hash):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding='utf-8'))
    core = {key: value for key, value in payload.items()
            if key not in {'archive_hash', 'created_at', 'immutability_contract'}}
    actual_hash = canonical_hash(core)
    if payload.get('archive_hash') != actual_hash or actual_hash != expected_hash:
        raise ValueError('Frozen HOA archive hash mismatch')
    if (payload.get('archive_version') != 'arterial_1x6_hoa_v1'
            or payload.get('num_runs') != 20
            or payload.get('num_episodes') != 8000
            or payload.get('num_transitions') != 17280000):
        raise ValueError('Frozen HOA global contract mismatch')
    audit = json.loads(Path(payload['source_audit_path']).read_text(encoding='utf-8'))
    if canonical_hash(audit) != payload.get('source_audit_hash'):
        raise ValueError('Frozen HOA source audit hash mismatch')
    if audit.get('valid') is not True or audit.get('schema_consistent') is not True:
        raise ValueError('Frozen HOA source audit is not valid')
    scenes = {item['scene_id']: item for item in payload.get('scenes', [])}
    if set(scenes) != set(ARTERIAL_SCENES):
        raise ValueError('Frozen HOA scene set mismatch')
    for scene, item in scenes.items():
        scene_manifest = json.loads(
            Path(item['manifest_path']).read_text(encoding='utf-8'))
        scene_core = {key: value for key, value in scene_manifest.items()
                      if key != 'archive_hash'}
        if (canonical_hash(scene_core) != item['archive_hash']
                or scene_manifest.get('archive_hash') != item['archive_hash']
                or len(item.get('history_paths', [])) != 5
                or item.get('num_transitions') != 4320000):
            raise ValueError(f'Frozen HOA scene contract mismatch: {scene}')
        for history_path in item['history_paths']:
            if not (Path(history_path) / 'manifest.json').is_file():
                raise FileNotFoundError(history_path)
    return path, payload, scenes


def completed_run(path):
    status = Path(path) / 'run_status.json'
    if not status.is_file():
        return False
    payload = json.loads(status.read_text(encoding='utf-8'))
    return payload.get('status') == '已完成' and payload.get('exit_code') == 0


def completed_evaluation(path):
    manifest = Path(path) / 'manifest.json'
    if not manifest.is_file():
        return False
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    return (payload.get('status') == 'completed'
            and payload.get('episode_count') == 4
            and payload.get('decision_record_count') == 1440)


def latest_resumable(path):
    candidates = sorted(
        (Path(path) / 'checkpoints' / 'resumable').glob('episode_*.pt'),
        reverse=True)
    return candidates[0] if candidates else None


def run_command(command, cwd):
    subprocess.run(command, cwd=cwd, check=True)


def execute(spec_path):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    root = Path(spec['sequential_root']).resolve()
    status_path = root / 'sequential_status.json'
    status = {
        'schema_version': 1, 'run_id': spec['run_id'], 'status': 'RUNNING',
        'started_at': utc_now(), 'completed_stages': [],
        'hoa_archive_version': spec['hoa_archive_version'],
        'hoa_archive_hash': spec['hoa_archive_hash'],
    }
    if status_path.is_file():
        previous = json.loads(status_path.read_text(encoding='utf-8'))
        status['started_at'] = previous.get('started_at', status['started_at'])
    atomic_json(status_path, status)
    hoa_path, hoa, scene_indexes = load_frozen_hoa(
        spec['hoa_manifest_path'], spec['hoa_archive_hash'])
    if hoa['archive_version'] != spec['hoa_archive_version']:
        raise ValueError('Frozen HOA archive version mismatch')
    sampling_manifest = json.loads(
        Path(spec['hoa_sampling_index_path']).read_text(encoding='utf-8'))
    if (sampling_manifest.get('source_hoa_archive_hash') != hoa['archive_hash']
            or sampling_manifest.get('status') != 'completed'):
        raise ValueError('HOA sampling index identity mismatch')
    order = SCENE_ORDERS[spec['scene_order']]
    budget = int(spec['episode_budget_per_stage'])
    if budget <= 0:
        raise ValueError('Sequential stage budget must be positive')
    target_dirs = {scene: Path(path).resolve()
                   for scene, path in spec['evaluation_target_run_dirs'].items()}
    if set(target_dirs) != set(ARTERIAL_SCENES):
        raise ValueError('Evaluation target scene set mismatch')
    for scene, path in target_dirs.items():
        if not completed_run(path):
            raise ValueError(f'Evaluation target run is incomplete: {scene}')
    root.mkdir(parents=True, exist_ok=True)
    previous_checkpoint = None
    for stage_index, scene in enumerate(order):
        stage_number = stage_index + 1
        prefix = f"{spec['run_id']}_stage{stage_number}_{scene.replace('.', '')}"
        stage_output = Path(spec['stage_output_paths'][stage_index]).resolve()
        overlay = root / 'overlays' / f'stage_{stage_number}.yml'
        checkpoint = (stage_output / 'checkpoints' / 'resumable' /
                      f'episode_{budget:04d}.pt')
        if not completed_run(stage_output):
            history_paths = []
            if spec['offline_enabled']:
                visible = (list(ARTERIAL_SCENES) if spec['history_mode']
                           == 'non_causal_full_history_upper_bound'
                           else order[:stage_index])
                history_paths = [path for visible_scene in visible
                                 for path in scene_indexes[visible_scene]['history_paths']]
            command = [
                sys.executable, 'arterial_run.py', '--config', spec['config_path'],
                '--stage', str(stage_index), '--scene-order', spec['scene_order'],
                '--training-seed', str(spec['training_seed']),
                '--episode-budget', str(budget), '--offline-ratio',
                str(spec['offline_ratio']), '--epsilon-mode', spec['epsilon_mode'],
                '--prefix', prefix, '--system-profile-path',
                spec['system_profile_path'], '--system-profile-hash',
                spec['system_profile_hash'], '--hoa-manifest-path', str(hoa_path),
                '--hoa-archive-version', hoa['archive_version'],
                '--hoa-archive-hash', hoa['archive_hash'],
                '--hoa-sampling-index-path', spec['hoa_sampling_index_path'],
                '--output-overlay', str(overlay), '--execute',
            ]
            for history_path in history_paths:
                command += ['--history-path', history_path]
            if stage_index:
                if previous_checkpoint is None or not previous_checkpoint.is_file():
                    raise FileNotFoundError('Previous stage checkpoint is missing')
                command += ['--stage-checkpoint', str(previous_checkpoint)]
            resume = latest_resumable(stage_output)
            if resume is not None:
                command += ['--resume-output', str(stage_output),
                            '--resume-checkpoint', str(resume)]
            run_command(command, REPOSITORY_ROOT)
        if not completed_run(stage_output) or not checkpoint.is_file():
            raise ValueError(f'Stage {stage_number} completion evidence is invalid')
        evaluation_manifest = root / 'evaluation_manifests' / f'stage_{stage_number}.json'
        evaluation_output = root / 'frozen_evaluations' / f'stage_{stage_number}'
        if not completed_evaluation(evaluation_output):
            if evaluation_output.exists():
                failed_root = root / 'failed_evaluation_attempts'
                failed_root.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                shutil.move(
                    str(evaluation_output),
                    str(failed_root / f'stage_{stage_number}_{stamp}'))
            build_cross_scene_evaluation_manifest(
                evaluation_manifest, stage_output, target_dirs, checkpoint, budget,
                scene, stage_index, training_seed=spec['training_seed'],
                evaluation_seeds=(0,), evaluation_steps=3600)
            run_command([
                sys.executable, 'run.py', '--evaluation-manifest',
                str(evaluation_manifest), '--evaluation-output',
                str(evaluation_output),
            ], REPOSITORY_ROOT)
        if not completed_evaluation(evaluation_output):
            raise ValueError(f'Stage {stage_number} frozen evaluation is incomplete')
        previous_checkpoint = checkpoint
        status['completed_stages'] = list(range(1, stage_number + 1))
        atomic_json(status_path, status)
    status.update({'status': 'SUCCEEDED', 'completed_at': utc_now()})
    atomic_json(status_path, status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    try:
        execute(args.spec)
    except Exception as error:
        spec = json.loads(Path(args.spec).read_text(encoding='utf-8'))
        path = Path(spec['sequential_root']) / 'sequential_status.json'
        previous = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        previous.update({'status': 'FAILED', 'failed_at': utc_now(),
                         'error_type': type(error).__name__, 'error': str(error)})
        atomic_json(path, previous)
        raise


if __name__ == '__main__':
    main()
