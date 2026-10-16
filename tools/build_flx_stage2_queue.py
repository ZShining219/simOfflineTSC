#!/usr/bin/env python3
"""Emit the ATT-ENTITY-003 stage-2 run_queue.json (ENTITY-002 schema).

Matrix (budget: <=25 formal runs):
  random-E x {colight, sga_colight, sga_flx_colight} x seeds 7..47   = 15
  fixed-E  x sga_flx_colight                    x seeds 7..47       =  5
  (optional) N-train x sga_flx_colight          x seeds 7..47       =  5

All random-E cells share plan configs/events/plans/hz4x4_random_v1.yml
(sha256 in configs/events/plans/hz4x4_random_v1.manifest.json).
"""
import argparse
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PY = ('data/output_data/cross_algorithm/plan5_b100/'
      'environment/micromamba/envs/colight/bin/python3.10')
OUT = ROOT / 'data/output_data/tsc'
ENV = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
       'NUMEXPR_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
       'VECLIB_MAXIMUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '-1'}
SEEDS = [7, 17, 27, 37, 47]

# run prefix -> (agent, experiment config, output agent dir, distribution)
CELLS = [
    ('colight', 'colight_event_random_200',
     'configs/tsc/colight_event_random_200.yml', 'colight', 'random-E'),
    ('sga_colight', 'sga_colight_text_event_random_200',
     'configs/tsc/sga_colight_text_event_random_200.yml',
     'sga_colight', 'random-E'),
    ('sga_flx_colight', 'sga_flx_colight_text_event_random_200',
     'configs/tsc/sga_flx_colight_text_event_random_200.yml',
     'sga_flx_colight', 'random-E'),
    ('sga_flx_colight', 'sga_flx_colight_text_event_200',
     'configs/tsc/sga_flx_colight_text_event_200.yml',
     'sga_flx_colight', 'fixed-E'),
]
OPTIONAL = [
    ('sga_flx_colight', 'sga_flx_colight_text_normal_200',
     'configs/tsc/sga_flx_colight_text_normal_200.yml',
     'sga_flx_colight', 'normal-N'),
]


def task(agent, prefix, config, out_agent, dist, seed):
    run_id = f'{prefix}_s{seed}'
    return {
        'run_id': f'flx_stage2_v1/{run_id}',
        'command': [str(ROOT / PY), 'run.py', '-w', 'sumo', '-a', agent,
                    '-n', 'hz4x4', '--seed', str(seed), '--interface',
                    'libsumo', '--prefix', run_id,
                    '--experiment-config', config],
        'cwd': str(ROOT),
        'env': ENV,
        'priority': 0 if dist == 'random-E' else 1,
        'completion': {'path': str(OUT / f'sumo_{out_agent}/hz4x4/{run_id}'
                                   '/run_status.json'),
                       'json_field': 'status', 'equals': '已完成'},
        'experiment_id': 'ATT-ENTITY-003',
        'protocol': 'stage2-200ep',
        'agent': agent,
        'distribution': dist,
        'training_seed': seed,
        'config_path': config,
        'output_path': str(OUT / f'sumo_{out_agent}/hz4x4/{run_id}'),
        'retry_count': 0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--include-normal', action='store_true',
                    help='also queue the N-train negative diagnostic')
    ap.add_argument('--out', default=str(
        ROOT / 'artifacts/att_entity_003/run_queue_stage2.json'))
    args = ap.parse_args()
    cells = CELLS + (OPTIONAL if args.include_normal else [])
    tasks = [task(agent, prefix, config, out_agent, dist, seed)
             for agent, prefix, config, out_agent, dist in cells
             for seed in SEEDS]
    spec = {
        'schema_version': 1,
        'plan': 'flx_stage2_v1',
        'experiment_id': 'ATT-ENTITY-003',
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'seeds': SEEDS,
        'expected_tasks': len(tasks),
        'notes': ('stage-2: 3 agents x random-E x 5 seeds + sga_flx x '
                  'fixed-E x 5 seeds; shared plan hz4x4_random_v1'),
        'tasks': tasks,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    print(f'wrote {out}: {len(tasks)} tasks')


if __name__ == '__main__':
    main()
