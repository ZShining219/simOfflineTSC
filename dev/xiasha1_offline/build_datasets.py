#!/usr/bin/env python3
"""Build the xiasha1 heterogeneous offline dataset index (E0-E5).

Scans completed ``xiasha1_online_v2`` runs (``xi2on_<arm>_<pack>_s<seed>``),
writes the run-list CSV (with ``arm`` column) and the explicit dataset-spec
JSON for the seven experiment pools x three pool replicas, then invokes
``python -m sequential.offline prepare-plan2`` to emit the read-only index.

Dataset root: ``data/output_data/offline_datasets/xiasha1/v1_hetero``.

Usage:
    python dev/xiasha1_offline/build_datasets.py --dry-run
    python dev/xiasha1_offline/build_datasets.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / 'data' / 'output_data' / 'tsc'
ARTIFACT_DIR = ROOT / 'artifacts' / 'xiasha1_offline_v2'
OUTPUT_ROOT = ROOT / 'data' / 'output_data' / 'offline_datasets' / 'xiasha1'
DATASET_ID = 'v2_hetero'
SOURCE_PREFIX = 'xi3re'  # xiasha1_online_v3 (geometry-fixed); v1 用 xi2on 已作废

# Scenario letters -> demand pack; controller letters -> arm.
NET = {
    'A': 'xiasha1_asym_ns',
    'B': 'xiasha1_dense_extreme',
    'C': 'xiasha1_fluct_burst',
    'D': 'xiasha1_asym_nonly',
}
ARM = {'a': 'd3qn', 'b': 'sac', 'c': 'frap', 'd': 'mplight'}
ARM_DIR = {
    'd3qn': 'sumo_dqn', 'sac': 'sumo_sac',
    'frap': 'sumo_frap', 'mplight': 'sumo_mplight',
}
ARM_AGENT = {
    'd3qn': 'dqn', 'sac': 'sac', 'frap': 'frap', 'mplight': 'mplight',
}
SEEDS = (0, 1, 2)
# Cells any experiment draws on: pack -> arms.
REQUIRED = {
    'A': {'a', 'b', 'c', 'd'},
    'B': {'a', 'c'},
    'C': {'a'},
    'D': {'a', 'd'},
}
# E1 same-arm replica pairs keyed by replica index.
E1_PAIRS = {0: (0, 1), 1: (1, 2), 2: (0, 2)}


def _scan_required_runs():
    rows = []
    missing = []
    for letter, arm_letters in REQUIRED.items():
        pack_net = NET[letter]
        pack = pack_net.split('_', 1)[1]
        for arm_letter in sorted(arm_letters):
            arm = ARM[arm_letter]
            for seed in SEEDS:
                run_dir = (
                    SOURCE_ROOT / ARM_DIR[arm] / pack_net
                    / f'{SOURCE_PREFIX}_{arm}_{pack}_s{seed}'
                )
                manifest_path = run_dir / 'run_manifest.json'
                if not manifest_path.is_file():
                    missing.append(str(run_dir.relative_to(ROOT)))
                    continue
                manifest = json.loads(manifest_path.read_text())
                problems = []
                if manifest.get('agent') != ARM_AGENT[arm]:
                    problems.append(f"agent={manifest.get('agent')}")
                if manifest.get('network') != pack_net:
                    problems.append(f"network={manifest.get('network')}")
                if manifest.get('training_seed') != seed:
                    problems.append(
                        f"training_seed={manifest.get('training_seed')}"
                    )
                if problems:
                    raise ValueError(
                        f'{run_dir.relative_to(ROOT)}: unexpected manifest '
                        f'fields: {"; ".join(problems)}'
                    )
                rows.append({
                    'run_path': str(run_dir),
                    'network': pack_net,
                    'behavior_training_seed': str(seed),
                    'arm': arm,
                })
    if missing:
        raise FileNotFoundError(
            'required source runs missing:\n  ' + '\n  '.join(missing)
        )
    return rows


def _spec(name, kind, eval_letter, networks, seeds, sampling,
          arms=None, cells=None):
    spec = {
        'name': name,
        'dataset_kind': kind,
        'evaluation_network': NET[eval_letter],
        'source_networks': [NET[letter] for letter in networks],
        'behavior_training_seeds': list(seeds),
        'episode_range': [1, 200],
        'sampling_strategy': sampling,
        'stage': 'full',
    }
    if arms is not None:
        spec['source_arms'] = [ARM[letter] for letter in arms]
    if cells is not None:
        spec['source_cells'] = [
            {'network': NET[net], 'arm': ARM[arm]}
            for net, arm in cells
        ]
    return spec


def build_specs():
    specs = []
    for replica in SEEDS:
        suffix = f'p{replica}'
        pair = E1_PAIRS[replica]
        specs.append(_spec(
            f'e0_{suffix}', 'single_scene', 'A', ['A'], [replica],
            'uniform_transition', arms=['a'],
        ))
        specs.append(_spec(
            f'e1_{suffix}', 'single_scene', 'A', ['A'], pair,
            'uniform_transition', arms=['a'],
        ))
        specs.append(_spec(
            f'e2_{suffix}', 'single_scene', 'A', ['A'], [replica],
            'uniform_transition', arms=['a', 'b'],
        ))
        specs.append(_spec(
            f'e3_{suffix}', 'single_scene', 'A', ['A'], [replica],
            'uniform_transition', arms=['a', 'b', 'c', 'd'],
        ))
        specs.append(_spec(
            f'e4_{suffix}', 'multi_scene', 'A', ['A', 'B', 'C'], [replica],
            'scene_balanced_batch', arms=['a'],
        ))
        specs.append(_spec(
            f'e4loo_{suffix}', 'leave_one_out', 'A', ['B', 'C', 'D'],
            [replica], 'scene_balanced_batch', arms=['a'],
        ))
        specs.append(_spec(
            f'e5_{suffix}', 'multi_scene', 'A', ['A', 'B', 'D'], [replica],
            'scene_balanced_batch',
            cells=[('A', 'a'), ('A', 'b'), ('B', 'c'), ('D', 'd')],
        ))
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    rows = _scan_required_runs()
    specs = build_specs()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    run_list_path = ARTIFACT_DIR / 'source_run_list.csv'
    specs_path = ARTIFACT_DIR / 'dataset_specs.json'

    print(f'source runs: {len(rows)} '
          f'(expected {sum(len(v) for v in REQUIRED.values()) * len(SEEDS)})')
    print(f'dataset specs: {len(specs)}')
    for spec in specs:
        cells = spec.get('source_cells')
        cell_desc = cells if cells else (
            f"arms={spec.get('source_arms')} x nets={spec['source_networks']}"
        )
        print(f"  {spec['name']:<10} {spec['dataset_kind']:<13} "
              f"eval={spec['evaluation_network']} seeds={spec['behavior_training_seeds']} "
              f"{cell_desc}")
    if args.dry_run:
        print('dry-run: nothing written')
        return

    with run_list_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['run_path', 'network', 'behavior_training_seed', 'arm'],
        )
        writer.writeheader()
        writer.writerows(rows)
    specs_path.write_text(json.dumps(specs, indent=1) + '\n')
    print(f'wrote {run_list_path.relative_to(ROOT)}')
    print(f'wrote {specs_path.relative_to(ROOT)}')

    command = [
        sys.executable, '-m', 'sequential.offline', 'prepare-plan2',
        '--run-list', str(run_list_path),
        '--output-root', str(OUTPUT_ROOT),
        '--dataset-id', DATASET_ID,
        '--source-root', str(SOURCE_ROOT),
        '--source-root-id', 'xiasha1_online_v3_root',
        '--expected-episodes', '200',
        '--no-plan1-formal',
        '--allowed-agents', 'dqn,sac,frap,mplight',
        '--relaxed-compatibility',
        '--dataset-specs', str(specs_path),
    ]
    print('running:', ' '.join(command))
    env = dict(os.environ)
    env.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    completed = subprocess.run(command, cwd=ROOT, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    dataset_root = OUTPUT_ROOT / DATASET_ID
    print(f'dataset root: {dataset_root.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
