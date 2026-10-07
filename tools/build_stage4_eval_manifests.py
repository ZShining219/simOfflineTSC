#!/usr/bin/env python3
"""Build frozen-test evaluation collections for ATT-ENTITY-003 stage-4 runs.

Covers the stage-4 queue (B-fused extra seeds, B' meta-repr arm, MPLight)
plus the two stage-3 B pilot runs (align_trans_l0.1_s7/s17) so the B arm
reaches 5 paired seeds.  Same schema-v2 collection manifest; random-E runs
are evaluated on randeval_v1 + none/all anchors.

Usage:
    python3 tools/build_stage4_eval_manifests.py \
        --out-dir artifacts/att_entity_003/eval_stage4 \
        --checkpoints 100 200 --seeds 400007
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYBIN = 'data/output_data/cross_algorithm/plan5_b100/environment/micromamba/envs/colight/bin/python3.10'

RANDOM_SET_MANIFEST = 'configs/events/eval_random/randeval_v1/manifest.json'
RANDOM_ANCHORS = {
    'none': 'configs/events/none.yml',
    'all': 'configs/events/eval/all.yml',
}
METRICS = ['reward', 'queue', 'delay', 'throughput', 'travel_time']

AGENT_DIR = {
    'colight': 'sumo_colight',
    'sga_colight': 'sumo_sga_colight',
    'concat_colight': 'sumo_concat_colight',
    'sga_flx_colight': 'sumo_sga_flx_colight',
    'mplight': 'sumo_mplight',
}


def random_conditions():
    manifest = json.loads((ROOT / RANDOM_SET_MANIFEST).read_text())
    conds = dict(manifest['conditions'])
    conds.update(RANDOM_ANCHORS)
    return conds


def iter_runs(state_paths):
    seen = set()
    for state_path in state_paths:
        state = json.loads(Path(state_path).read_text())
        for task in state['tasks']:
            if task['status'] != 'SUCCEEDED':
                continue
            rid = task['run_id'].split('/')[-1]
            if rid in seen:
                continue
            seen.add(rid)
            run_dir = task.get('output_path')
            if run_dir is None:
                # Runner state drops the queue's output_path; reconstruct the
                # standard run dir from the -a agent and run prefix.
                try:
                    agent_name = task['command'][task['command'].index('-a') + 1]
                except (ValueError, IndexError):
                    continue
                run_dir = (ROOT / 'data' / 'output_data' / 'tsc'
                           / f'sumo_{agent_name}' / 'hz4x4' / rid)
            run_dir = Path(run_dir)
            if not (run_dir / 'run_manifest.json').is_file():
                continue
            manifest = json.loads((run_dir / 'run_manifest.json').read_text())
            yield {'run_id': rid, 'run_dir': run_dir,
                   'agent': manifest['agent'],
                   'training_seed': manifest['training_seed'],
                   'distribution': task.get('distribution', '')}


def build(args):
    out_root = Path(args.out_dir).resolve()
    manifest_dir = out_root / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    conds = random_conditions()
    tasks = []
    for run in iter_runs(args.queue_state):
        controllers = []
        for ep in args.checkpoints:
            resumable = (run['run_dir'] / 'checkpoints' / 'resumable'
                         / f'episode_{ep:04d}.pt').resolve()
            ref = (run['run_dir'] / 'checkpoints' / 'evaluation'
                   / f'episode_{ep:04d}.pt').resolve()
            if resumable.is_file() and ref.is_file():
                ckpt, role = resumable, 'resumable'
            elif run['agent'] in {'mplight', 'sga_mplight'} and ref.is_file():
                # PFRL-based MPLight never produces resumable checkpoints
                # (no target_model); use its audited evaluation checkpoint
                # converted from the legacy model/*.pt snapshot.
                ckpt, role = ref, 'best'
            else:
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
                    'checkpoint_role': role,
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
            'run_id': f'stage4_eval_v1/{run["run_id"]}',
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
             'experiment_id': 'stage4_v1_eval', 'tasks': tasks}
    qpath = out_root / 'run_queue.json'
    qpath.write_text(json.dumps(queue, indent=2) + '\n')
    print(f'{len(tasks)} tasks -> {qpath}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue-state', nargs='+', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--checkpoints', type=int, nargs='+', default=[100, 200])
    p.add_argument('--seeds', type=int, nargs='+', default=[400007])
    build(p.parse_args())
