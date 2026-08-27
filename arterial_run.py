"""Generate or execute shared-DQN arterial stage/evaluation commands."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from arterial.experiment import (ARTERIAL_SCENES, EvaluationMatrixWriter,
                                 stage_overlay, validate_experiment_config)


def command_for_stage(config, stage_index, overlay_path, prefix,
                      resume_output=None, resume_checkpoint=None):
    scene = config['scene_order'][stage_index]
    command = [
        sys.executable, 'run.py', '-t', 'tsc', '-a', 'shared_dqn',
        '-w', 'sumo', '-n', ARTERIAL_SCENES[scene], '--interface', 'libsumo',
        '--seed', str(config['training_seed']), '--prefix', prefix,
        '--experiment-config', str(overlay_path),
    ]
    if resume_output or resume_checkpoint:
        if not resume_output or not resume_checkpoint:
            raise ValueError('Resume requires both output and checkpoint paths')
        command += [
            '--resume-output', str(resume_output),
            '--resume-checkpoint', str(resume_checkpoint),
        ]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', type=int, required=True, help='zero-based stage')
    parser.add_argument('--training-seed', type=int)
    parser.add_argument('--episode-budget', type=int)
    parser.add_argument('--scene-order', choices=('order_1', 'order_2', 'order_3'))
    parser.add_argument(
        '--scene', choices=tuple(ARTERIAL_SCENES),
        help='run one explicitly selected scene (independent collector use)',
    )
    parser.add_argument('--offline-ratio', type=float)
    parser.add_argument('--system-profile-path')
    parser.add_argument('--system-profile-hash')
    parser.add_argument('--hoa-manifest-path')
    parser.add_argument('--hoa-archive-version')
    parser.add_argument('--hoa-archive-hash')
    parser.add_argument('--hoa-sampling-index-path')
    parser.add_argument('--epsilon-mode', choices=(
        'reset_schedule', 'continue_schedule', 'fixed_low'))
    parser.add_argument('--history-path', action='append', default=[])
    parser.add_argument('--stage-checkpoint')
    parser.add_argument('--prefix')
    parser.add_argument('--output-overlay', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume-output')
    parser.add_argument('--resume-checkpoint')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    overrides = {
        'training_seed': args.training_seed,
        'episode_budget': args.episode_budget,
        'scene_order': args.scene_order,
        'offline_ratio': args.offline_ratio,
        'epsilon_mode': args.epsilon_mode,
        'system_profile_path': args.system_profile_path,
        'system_profile_hash': args.system_profile_hash,
        'hoa_manifest_path': args.hoa_manifest_path,
        'hoa_archive_version': args.hoa_archive_version,
        'hoa_archive_hash': args.hoa_archive_hash,
        'hoa_sampling_index_path': args.hoa_sampling_index_path,
    }
    for key, value in overrides.items():
        if value is not None:
            raw[key] = value
    if args.scene is not None:
        if args.scene_order is not None:
            raise ValueError('--scene and --scene-order are mutually exclusive')
        if args.stage != 0:
            raise ValueError('--scene uses a single-scene order and requires --stage 0')
        raw['scene_order'] = [args.scene]
    # Example files may also carry direct run.py overlay sections.
    experiment = {key: value for key, value in raw.items()
                  if key not in {'model', 'trainer', 'world', 'logger'}}
    config = validate_experiment_config(experiment)
    generated = stage_overlay(
        config, args.stage, args.history_path, args.stage_checkpoint)
    overlay = {}
    for section in ('model', 'trainer', 'world', 'logger'):
        if section in raw:
            overlay[section] = dict(raw[section])
    for section, values in generated.items():
        if isinstance(values, dict):
            overlay.setdefault(section, {}).update(values)
        else:
            overlay[section] = values
    path = Path(args.output_overlay)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(overlay, sort_keys=False, allow_unicode=True),
                    encoding='utf-8')
    scene = config['scene_order'][args.stage]
    prefix = args.prefix or (
        f"arterial_{scene.replace('.', '')}_stage{args.stage}_"
        f"seed{config['training_seed']}")
    command = command_for_stage(
        config, args.stage, path, prefix, args.resume_output, args.resume_checkpoint)
    print(json.dumps({'overlay': str(path), 'command': command,
                      'run_metadata': overlay['run_metadata']},
                     ensure_ascii=False, indent=2))
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
