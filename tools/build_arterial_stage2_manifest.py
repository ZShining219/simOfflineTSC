"""Build the frozen 21-run Stage 2 screening queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arterial.experiment import ARTERIAL_SCENES, SCENE_ORDERS, validate_experiment_config
from tools.build_arterial_stage0_manifest import THREAD_ENV
from tools.run_arterial_sequential import load_frozen_hoa

SEEDS = (1000, 1001, 1002)
METHODS = (
    ('continue_online_clear', False, 0.0, 'causal', True, 'reset_schedule'),
    ('continue_online_fifo', False, 0.0, 'causal', False, 'reset_schedule'),
    ('causal_semi_r25', True, 0.25, 'causal', True, 'reset_schedule'),
    ('causal_semi_r50', True, 0.50, 'causal', True, 'reset_schedule'),
    ('causal_semi_r75', True, 0.75, 'causal', True, 'reset_schedule'),
    ('non_causal_full_history_upper_bound_r50', True, 0.50,
     'non_causal_full_history_upper_bound', True, 'reset_schedule'),
    ('low_epsilon_continue_online_diagnostic', False, 0.0, 'causal', True,
     'fixed_low'),
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def completed_run(path):
    status = Path(path) / 'run_status.json'
    if not status.is_file():
        return False
    value = json.loads(status.read_text(encoding='utf-8'))
    return value.get('status') == '已完成' and value.get('exit_code') == 0


def build(args):
    output = Path(args.output).resolve()
    stage2_root = output.parent
    configs_dir = Path(args.config_dir).resolve()
    specs_dir = Path(args.spec_dir).resolve()
    hoa_path, hoa, _ = load_frozen_hoa(args.hoa_manifest, args.hoa_hash)
    sampling_path = Path(args.hoa_sampling_index).resolve()
    sampling = json.loads(sampling_path.read_text(encoding='utf-8'))
    if (sampling.get('status') != 'completed'
            or sampling.get('source_hoa_archive_hash') != hoa['archive_hash']
            or sampling.get('num_transitions') != 17280000
            or len(sampling.get('pools', [])) != 24):
        raise ValueError('Formal Stage 2 requires the complete frozen HOA sampling index')
    profile = Path(args.profile).resolve()
    profile_hash = sha256_file(profile)
    stage1 = json.loads(Path(args.stage1_state).read_text(encoding='utf-8'))
    tasks_by_scene = {}
    succeeded = [task for task in stage1['tasks'] if task.get('status') == 'SUCCEEDED']
    if len(succeeded) != 20:
        raise ValueError('Stage 2 requires exactly 20 successful Stage 1 runs')
    for scene in ARTERIAL_SCENES:
        candidates = sorted(
            (task for task in succeeded if task.get('scene') == scene),
            key=lambda task: int(task['training_seed']))
        if len(candidates) != 5 or [int(x['training_seed']) for x in candidates] != list(range(5)):
            raise ValueError(f'Stage 1 scene matrix is incomplete: {scene}')
        if not all(completed_run(item['output_path']) for item in candidates):
            raise ValueError(f'Stage 1 completion evidence is invalid: {scene}')
        tasks_by_scene[scene] = candidates[0]['output_path']
    base_path = Path(args.base_config).resolve()
    base = yaml.safe_load(base_path.read_text(encoding='utf-8'))
    git_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY_ROOT, text=True).strip()
    tasks = []
    for method, offline, ratio, history_mode, clear, epsilon in METHODS:
        config = dict(base)
        config.update({
            'scene_order': 'order_1', 'training_seed': 1000,
            'episode_budget': 200, 'offline_enabled': offline,
            'offline_ratio': ratio, 'history_access_mode': history_mode,
            'replay_clear_on_scene_switch': clear, 'epsilon_mode': epsilon,
        })
        validate_experiment_config({
            key: value for key, value in config.items()
            if key not in {'model', 'trainer', 'world', 'logger'}
        })
        config_path = configs_dir / f'{method}.yml'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(
            config, sort_keys=False, allow_unicode=True), encoding='utf-8')
        for seed in SEEDS:
            run_id = f'stage2_{method}_order1_seed{seed}'
            sequential_root = stage2_root / 'runs' / run_id
            stage_outputs = [
                str((REPOSITORY_ROOT / 'data/output_data/tsc/sumo_shared_dqn' /
                     ARTERIAL_SCENES[scene] /
                     f"{run_id}_stage{index}_{scene.replace('.', '')}").resolve())
                for index, scene in enumerate(SCENE_ORDERS['order_1'], start=1)
            ]
            spec = {
                'schema_version': 1, 'run_id': run_id, 'method': method,
                'scene_order': 'order_1', 'training_seed': seed,
                'episode_budget_per_stage': 200, 'offline_enabled': offline,
                'offline_ratio': ratio, 'history_mode': history_mode,
                'replay_clear_on_scene_switch': clear, 'epsilon_mode': epsilon,
                'config_path': str(config_path),
                'sequential_root': str(sequential_root),
                'stage_output_paths': stage_outputs,
                'evaluation_target_run_dirs': tasks_by_scene,
                'hoa_manifest_path': str(hoa_path),
                'hoa_archive_version': hoa['archive_version'],
                'hoa_archive_hash': hoa['archive_hash'],
                'hoa_sampling_index_path': str(sampling_path),
                'system_profile_path': str(profile),
                'system_profile_hash': profile_hash,
            }
            spec_path = specs_dir / f'{run_id}.json'
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(json.dumps(
                spec, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                encoding='utf-8')
            tasks.append({
                'run_id': run_id, 'plan': 'stage2_ratio_method_screening',
                'method': method, 'scene': 'sequential_4_stage',
                'scene_order': 'order_1', 'offline_ratio': ratio,
                'history_mode': history_mode, 'epsilon_mode': epsilon,
                'training_seed': seed, 'episode_budget': 200,
                'config_path': str(config_path),
                'output_path': str(sequential_root),
                'command': [sys.executable, 'tools/run_arterial_sequential.py',
                            '--spec', str(spec_path)],
                'retry_command': [sys.executable, 'tools/run_arterial_sequential.py',
                                  '--spec', str(spec_path)],
                'cwd': str(REPOSITORY_ROOT), 'env': THREAD_ENV,
                'completion': {
                    'path': str(sequential_root / 'sequential_status.json'),
                    'json_field': 'status', 'equals': 'SUCCEEDED',
                },
                'config_hash': sha256_file(config_path),
                'git_commit': git_commit, 'hoa_archive_hash': hoa['archive_hash'],
                'hoa_archive_version': hoa['archive_version'],
            })
    if len(tasks) != 21:
        raise AssertionError('Stage 2 matrix must contain 21 sequential runs')
    payload = {
        'schema_version': 1, 'plan': 'stage2_ratio_method_screening',
        'scene_order': 'order_1', 'sequential_training_seeds': list(SEEDS),
        'episode_budget_per_stage': 200, 'expected_tasks': 21,
        'expected_training_stages': 84, 'expected_frozen_evaluations': 336,
        'sumo_seed_policy': 'fixed simulator config seed 0; evaluation seed 0',
        'hoa_manifest_path': str(hoa_path),
        'hoa_archive_version': hoa['archive_version'],
        'hoa_archive_hash': hoa['archive_hash'],
        'system_profile_path': str(profile),
        'system_profile_hash': profile_hash, 'tasks': tasks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--config-dir', required=True)
    parser.add_argument('--spec-dir', required=True)
    parser.add_argument('--base-config', default='configs/arterial/sequential_online.yml')
    parser.add_argument('--hoa-manifest', default=
                        'artifacts/arterial_experiments/historical_archive/manifest.json')
    parser.add_argument('--hoa-hash', required=True)
    parser.add_argument('--hoa-sampling-index', default=
                        'artifacts/arterial_experiments/historical_archive/sampling_index/manifest.json')
    parser.add_argument('--stage1-state', default=
                        'artifacts/arterial_experiments/stage1/run_state/run_manifest.json')
    parser.add_argument('--profile', default='artifacts/system_profile/profile.json')
    args = parser.parse_args()
    print(build(args))


if __name__ == '__main__':
    main()
