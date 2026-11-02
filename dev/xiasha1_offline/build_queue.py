#!/usr/bin/env python3
"""Emit the xiasha1 heterogeneous offline run queue (E0-E5).

Reads ``artifacts/xiasha1_offline_v1/dataset_specs.json`` (produced by
``build_datasets.py``) and writes
``artifacts/xiasha1_offline_v1/run_queue.json`` with
7 experiments x 3 pool replicas x 2 algorithms x 3 offline seeds = 126 tasks.

Each task runs ``python -m sequential.offline train`` at the formal
144k-update budget with the 81-node evaluation overlay
(``configs/offline_tsc/xiasha1_v1.yml``) and the native backend.

Usage:
    python dev/xiasha1_offline/build_queue.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / 'artifacts' / 'xiasha1_offline_v2'
DATASET_ROOT = ROOT / 'data' / 'output_data' / 'offline_datasets' / 'xiasha1' / 'v2_hetero'
PYTHON = '/home/dev/miniforge3/envs/colight/bin/python'
EVAL_NETWORK = 'xiasha1_asym_ns'
OVERLAY = 'configs/offline_tsc/xiasha1_v1.yml'
EXPERIMENTS = ('e0', 'e1', 'e2', 'e3', 'e4', 'e4loo', 'e5')
REPLICAS = (0, 1, 2)
ALGORITHMS = ('batch_dqn', 'cql_dqn')
OFFLINE_SEEDS = (1000, 1001, 1002)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ARTIFACT_DIR / 'run_queue.json'))
    args = parser.parse_args()
    raise SystemExit('SCOPE_DISCARD: xiasha1_offline_v2 is closed; use python -m dev.hz1x1_offline.prepare')

    specs = json.loads(
        (ARTIFACT_DIR / 'dataset_specs.json').read_text(encoding='utf-8')
    )
    manifest_by_name = {
        spec['name']: DATASET_ROOT / 'datasets' / spec['name'] / 'manifest.json'
        for spec in specs
    }
    missing = [name for name, path in manifest_by_name.items()
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f'dataset manifests missing for specs: {missing}; '
            'run build_datasets.py first'
        )

    tasks = []
    for experiment in EXPERIMENTS:
        for replica in REPLICAS:
            spec_name = f'{experiment}_p{replica}'
            manifest = manifest_by_name[spec_name]
            for algorithm in ALGORITHMS:
                for seed in OFFLINE_SEEDS:
                    prefix = f'xi2off_{spec_name}_{algorithm}_o{seed}'
                    output_path = (
                        ROOT / 'data' / 'output_data' / 'offline_tsc'
                        / f'sumo_{algorithm}' / EVAL_NETWORK / prefix
                    )
                    tasks.append({
                        'run_id': prefix,
                        'experiment': experiment,
                        'pool_replica': replica,
                        'dataset_spec': spec_name,
                        'agent': algorithm,
                        'seed': seed,
                        'command': [
                            PYTHON, '-m', 'sequential.offline', 'train',
                            '-a', algorithm,
                            '-w', 'sumo',
                            '-n', EVAL_NETWORK,
                            '--dataset-manifest', str(manifest),
                            '--prefix', prefix,
                            '--seed', str(seed),
                            '--backend', 'native',
                            '--interface', 'libsumo',
                            '--experiment-config', OVERLAY,
                        ],
                        'cwd': str(ROOT),
                        'env': {
                            'LD_PRELOAD': '/home/dev/miniforge3/envs/colight/lib/libstdc++.so.6',
                            'CUDA_VISIBLE_DEVICES': '-1',
                            'OMP_NUM_THREADS': '1',
                            'MKL_NUM_THREADS': '1',
                            'OPENBLAS_NUM_THREADS': '1',
                            'NUMEXPR_NUM_THREADS': '1',
                            'VECLIB_MAXIMUM_THREADS': '1',
                        },
                        'output_path': str(output_path),
                        'completion': {
                            'path': str(output_path / 'run_status.json'),
                            'json_field': 'status',
                            'equals': '已完成',
                        },
                    })

    queue = {
        'schema_version': 1,
        'plan': 'xiasha1_offline_v2',
        'expected_tasks': len(tasks),
        'note': (
            'heterogeneous-history offline matrix E0-E5 on geometry-fixed '
            'v3 sources: fixed pools of complete xi3re runs (72k transitions '
            'each), uniform sampling, no quality filtering; 144k updates; '
            '81 evaluation nodes at 1800-update spacing; native backend; '
            'eval on xiasha1_asym_ns'
        ),
        'tasks': tasks,
        'dropped': [],
        'dropped_reason': None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(queue, indent=1, ensure_ascii=False) + '\n')
    print(f'wrote {out.relative_to(ROOT)} with {len(tasks)} tasks')


if __name__ == '__main__':
    main()
