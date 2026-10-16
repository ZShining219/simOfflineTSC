#!/usr/bin/env python3
"""Build frozen-test evaluation collections for ATT-ENTITY-003 stage-2 runs.

Same schema-v2 collection manifests as ATT-ENTITY-002, but:
  * adds sga_flx_colight -> sumo_sga_flx_colight output dir;
  * random-E runs are evaluated on the held-out randeval_v1 set
    (configs/events/eval_random/randeval_v1/, same distribution as the
    training plan, disjoint seed) plus none/all fixed anchors for
    cross-distribution comparability;
  * fixed-E runs keep the original 8-condition frozen library.

Usage:
    python3 tools/build_flx_eval_manifests.py \
        --queue-state artifacts/att_entity_003/run_state_stage2/run_manifest.json \
        --out-dir artifacts/att_entity_003/eval_v1 \
        --checkpoints 100 200 --seeds 400007
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYBIN = 'data/output_data/cross_algorithm/plan5_b100/environment/micromamba/envs/colight/bin/python3.10'

FIXED_CONDITIONS = {
    'none': 'configs/events/none.yml',
    'blockage': 'configs/events/eval/blockage.yml',
    'closure': 'configs/events/eval/closure.yml',
    'rain': 'configs/events/eval/rain.yml',
    'blockage_closure': 'configs/events/eval/blockage_closure.yml',
    'blockage_rain': 'configs/events/eval/blockage_rain.yml',
    'closure_rain': 'configs/events/eval/closure_rain.yml',
    'all': 'configs/events/eval/all.yml',
}
RANDOM_SET_MANIFEST = 'configs/events/eval_random/randeval_v1/manifest.json'
RANDOM_ANCHORS = {k: FIXED_CONDITIONS[k] for k in ('none', 'all')}
METRICS = ['reward', 'queue', 'delay', 'throughput', 'travel_time']


def agent_dir(agent):
    return {'colight': 'sumo_colight',
            'sga_colight': 'sumo_sga_colight',
            'concat_colight': 'sumo_concat_colight',
            'sga_flx_colight': 'sumo_sga_flx_colight'}[agent]


def random_conditions():
    manifest = json.loads((ROOT / RANDOM_SET_MANIFEST).read_text())
    conds = dict(manifest['conditions'])
    conds.update(RANDOM_ANCHORS)
    return conds


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
               'training_seed': manifest['training_seed'],
               'distribution': task.get('distribution', '')}


def build(args):
    out_root = Path(args.out_dir).resolve()
    manifest_dir = out_root / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rand_conds = random_conditions()
    tasks = []
    for run in valid_runs(args.queue_state):
        conds = rand_conds if run['distribution'] == 'random-E' \
            else FIXED_CONDITIONS
        controllers = []
        for ep in args.checkpoints:
            ckpt = (run['run_dir'] / 'checkpoints' / 'resumable'
                    / f'episode_{ep:04d}.pt').resolve()
            ref = (run['run_dir'] / 'checkpoints' / 'evaluation'
                   / f'episode_{ep:04d}.pt').resolve()
            if not (ckpt.is_file() and ref.is_file()):
                raise FileNotFoundError(
                    f"{run['run_id']} missing ep{ep} checkpoints")
            for cond, schedule in conds.items():
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
        collection = {
            'schema_version': 2,
            'package_id': f"eval_{run['run_id']}",
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
            'run_id': f'flx_eval_v1/{run["run_id"]}',
            'command': [str((ROOT / PYBIN).resolve()), 'run.py',
                        '--evaluation-manifest', str(mpath),
                        '--evaluation-output', str(out_dir)],
            'output_path': str(out_dir),
            'cwd': str(ROOT),
            'env': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                    'OPENBLAS_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '-1'},
            'completion': {'path': str(out_dir / 'manifest.json'),
                           'json_field': 'status', 'equals': 'completed'},
        })
    queue = {'schema_version': 'experiment-queue-v1',
             'experiment_id': 'flx_stage2_v1_eval', 'tasks': tasks}
    qpath = out_root / 'run_queue.json'
    qpath.write_text(json.dumps(queue, indent=2) + '\n')
    print(f'{len(tasks)} tasks -> {qpath}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue-state', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--checkpoints', type=int, nargs='+', default=[100, 200])
    p.add_argument('--seeds', type=int, nargs='+', default=[400007])
    build(p.parse_args())
