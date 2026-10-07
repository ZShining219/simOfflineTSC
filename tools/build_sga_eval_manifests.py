#!/usr/bin/env python3
"""Build frozen-test evaluation collection manifests for ATT-ENTITY-002.

Emits one schema-v2 collection manifest per valid training run covering
checkpoints x event conditions x traffic seeds, plus an experiment_queue
manifest that runs each collection via `run.py --evaluation-manifest`.

Usage:
    python3 tools/build_sga_eval_manifests.py \
        --queue-state artifacts/sga_concat_colight_200_v1/run_state/run_manifest.json \
        --out-dir artifacts/sga_concat_colight_200_v1/eval_v1 \
        --checkpoints 150 200 --seeds 400007
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYBIN = 'data/output_data/cross_algorithm/plan5_b100/environment/micromamba/envs/colight/bin/python3.10'

CONDITIONS = {
    'none': 'configs/events/none.yml',
    'blockage': 'configs/events/eval/blockage.yml',
    'closure': 'configs/events/eval/closure.yml',
    'rain': 'configs/events/eval/rain.yml',
    'blockage_closure': 'configs/events/eval/blockage_closure.yml',
    'blockage_rain': 'configs/events/eval/blockage_rain.yml',
    'closure_rain': 'configs/events/eval/closure_rain.yml',
    'all': 'configs/events/eval/all.yml',
}
METRICS = ['reward', 'queue', 'delay', 'throughput', 'travel_time']


def agent_dir(agent):
    return {'colight': 'sumo_colight',
            'sga_colight': 'sumo_sga_colight',
            'concat_colight': 'sumo_concat_colight'}[agent]


def valid_runs(state_path):
    state = json.loads(Path(state_path).read_text())
    for task in state['tasks']:
        if task['status'] != 'SUCCEEDED':
            continue
        rid = task['run_id'].split('/')[-1]
        run_dir = Path(task['output_path'])
        manifest = json.loads((run_dir / 'run_manifest.json').read_text())
        yield {'run_id': rid, 'run_dir': run_dir,
               'agent': manifest['agent'],
               'training_seed': manifest['training_seed']}


def build(args):
    out_root = Path(args.out_dir).resolve()
    manifest_dir = out_root / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for run in valid_runs(args.queue_state):
        controllers = []
        for ep in args.checkpoints:
            ckpt = (run['run_dir'] / 'checkpoints' / 'resumable'
                    / f'episode_{ep:04d}.pt').resolve()
            ref = (run['run_dir'] / 'checkpoints' / 'evaluation'
                   / f'episode_{ep:04d}.pt').resolve()
            if not (ckpt.is_file() and ref.is_file()):
                raise FileNotFoundError(f"{run['run_id']} missing ep{ep} checkpoints")
            for cond, schedule in CONDITIONS.items():
                controllers.append({
                    'controller_id': f"{run['run_id']}_ep{ep}_{cond}",
                    'agent': run['agent'],
                    'network': 'hz4x4',
                    'source_network': 'hz4x4',
                    'target_network': 'hz4x4',
                    'training_seed': run['training_seed'],
                    'source_policy': run['agent'],
                    'run_dir': str(run['run_dir'].resolve()),
                    'target_run_dir': str(run['run_dir'].resolve()),
                    'checkpoint': str(ckpt),
                    'checkpoint_role': 'resumable',
                    'checkpoint_episode': ep,
                    'event_schedule': schedule,
                    'evaluation_steps': 3600,
                })
        package_id = f"eval_{run['run_id']}"
        collection = {
            'schema_version': 2,
            'package_id': package_id,
            'world': 'sumo',
            'evaluation_seeds': args.seeds,
            'sampling_interval_seconds': 10,
            'smoothing_window_seconds': 60,
            'metrics': METRICS,
            'record_state_diagnostics': True,
            'expected_controller_count': len(controllers),
            'expected_episode_count': len(controllers) * len(args.seeds),
            'controllers': controllers,
        }
        mpath = manifest_dir / f'{run["run_id"]}.json'
        mpath.write_text(json.dumps(collection, indent=2) + '\n')
        out_dir = out_root / 'packages' / run['run_id']
        tasks.append({
            'run_id': f'sga_eval_v1/{run["run_id"]}',
            'command': [str((ROOT / PYBIN).resolve()), 'run.py',
                        '--evaluation-manifest', str(mpath),
                        '--evaluation-output', str(out_dir)],
            'output_path': str(out_dir),
            'cwd': str(ROOT),
            'completion': {
                'path': str(out_dir / 'manifest.json'),
                'json_field': 'status',
                'equals': 'completed',
            },
        })
    queue = {
        'schema_version': 'experiment-queue-v1',
        'experiment_id': 'sga_concat_colight_200_v1_eval',
        'tasks': tasks,
    }
    qpath = out_root / 'run_queue.json'
    qpath.write_text(json.dumps(queue, indent=2) + '\n')
    print(f'{len(tasks)} tasks -> {qpath}')
    print(f'attempts per task: {len(args.checkpoints)} ckpt x '
          f'{len(CONDITIONS)} cond x {len(args.seeds)} seeds = '
          f'{len(args.checkpoints) * len(CONDITIONS) * len(args.seeds)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue-state', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--checkpoints', type=int, nargs='+', default=[150, 200])
    p.add_argument('--seeds', type=int, nargs='+', default=[400007])
    build(p.parse_args())
