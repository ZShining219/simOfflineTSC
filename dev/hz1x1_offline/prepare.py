"""Prepare hz1x1 history indexes, queue and ledger; never execute training.

Run in the colight environment: python -m dev.hz1x1_offline.prepare all
Individual faces: datasets, queue, register, check. Parameters live in YAML.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from data.dataset.offline_trajectory_dataset import (
    OFFLINE_DATASET_BUILDER_VERSION, OfflineTrajectoryDataset,
    prepare_plan2_datasets, validate_offline_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path('configs/offline_tsc/hz1x1_v1.yml')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def build_specs(cfg):
    specs = []
    for replica, seed in enumerate(cfg['pool_seeds']):
        for name, condition in cfg['conditions'].items():
            cells = [{'network': cfg['scenes'][scene]['network'], 'arm': arm}
                     for scene, arm in condition['cells']]
            seeds = cfg['e1_seed_pairs'][replica] if condition.get('paired_seeds') else [seed]
            specs.append({
                'name': f'{name}_p{replica}', 'dataset_kind': condition['kind'],
                'evaluation_network': cfg['scenes'][cfg['target']]['network'],
                'source_networks': sorted({cell['network'] for cell in cells}),
                'source_cells': cells, 'behavior_training_seeds': seeds,
                'episode_range': [1, cfg['episodes']], 'stage': 'full',
                'sampling_strategy': 'uniform_transition',
            })
    return specs


def source_rows(cfg):
    rows = []
    approved = {row['run_id'] for row in (
        json.loads(line) for line in (ROOT / 'ledger/runs.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()) if row['campaign'] == cfg['source_campaign'] and row['status'] == 'DONE'}
    for scene in cfg['scenes'].values():
        for arm, identity in cfg['arms'].items():
            for seed in cfg['pool_seeds']:
                run_id = f"{cfg['source_prefix']}_{arm}_{scene['pack']}_s{seed}"
                if run_id not in approved:
                    raise ValueError(f'Source not DONE in the approved campaign ledger: {run_id}')
                run = ROOT / 'data/output_data/tsc' / identity['directory'] / scene['network'] / run_id
                manifest = read_json(run / 'run_manifest.json')
                if (manifest['agent'], manifest['network'], manifest['training_seed']) != (
                    identity['agent'], scene['network'], seed
                ):
                    raise ValueError(f'Source identity mismatch: {run_id}')
                rows.append({'run_path': str(run), 'network': scene['network'],
                             'arm': arm, 'behavior_training_seed': seed})
    return rows


def build_datasets(cfg, artifacts, dataset):
    if dataset.exists():
        manifest = read_json(dataset / 'manifest.json')
        if manifest['dataset_builder_version'] != OFFLINE_DATASET_BUILDER_VERSION:
            raise ValueError('Existing dataset needs a new dataset_id; do not overwrite')
        raise FileExistsError(f'Dataset already exists: {dataset.relative_to(ROOT)}; use check')
    rows, specs = source_rows(cfg), build_specs(cfg)
    artifacts.mkdir(parents=True, exist_ok=True)
    run_list = artifacts / 'source_run_list.csv'
    with run_list.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['run_path', 'network', 'arm', 'behavior_training_seed'])
        writer.writeheader()
        writer.writerows(rows)
    write_json(artifacts / 'dataset_specs.json', specs)
    prepare_plan2_datasets(
        str(run_list), str(dataset.parent), cfg['dataset_id'],
        source_root=str(ROOT / 'data/output_data/tsc'), source_root_id=cfg['source_campaign'],
        expected_episodes=cfg['episodes'], require_plan1_formal=False,
        allowed_agents=tuple(identity['agent'] for identity in cfg['arms'].values()),
        relaxed_compatibility=True, dataset_specs=specs,
        reward_alignment=cfg['reward_alignment'],
    )
    root = read_json(dataset / 'manifest.json')
    if root['expected_decisions_per_episode'] != cfg['decisions_per_episode']:
        raise ValueError('Source decision count differs from campaign design')
    print(f"datasets: {len(rows)} sources, {len(specs)} pools, {root['raw_transition_count']} transitions")


def build_queue(cfg, artifacts, dataset):
    target = cfg['scenes'][cfg['target']]['network']
    tasks = []
    for spec in build_specs(cfg):
        manifest = dataset / 'datasets' / spec['name'] / 'manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError(str(manifest.relative_to(ROOT)))
        for algorithm in cfg['algorithms']:
            for seed in cfg['offline_seeds']:
                run_id = f"{cfg['run_prefix']}_{spec['name']}_{algorithm}_o{seed}"
                output = ROOT / 'data/output_data/offline_tsc' / f'sumo_{algorithm}' / target / run_id
                tasks.append({
                    'run_id': run_id, 'dataset_spec': spec['name'],
                    'experiment': spec['name'].rsplit('_p', 1)[0],
                    'pool_replica': int(spec['name'].rsplit('_p', 1)[1]),
                    'agent': algorithm, 'seed': seed,
                    'command': [sys.executable, '-m', 'sequential.offline', 'train',
                                '-a', algorithm, '-w', 'sumo', '-n', target,
                                '--dataset-manifest', str(manifest.relative_to(ROOT)),
                                '--prefix', run_id, '--seed', str(seed),
                                '--backend', 'native', '--interface', 'libsumo',
                                '--experiment-config', str(CONFIG)],
                    'cwd': str(ROOT), 'output_path': str(output),
                    'env': {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1',
                            'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                            'NUMEXPR_NUM_THREADS': '1',
                            'LD_PRELOAD': str(Path(sys.prefix) / 'lib/libstdc++.so.6')},
                    'completion': {'path': str(output / 'run_status.json'),
                                   'json_field': 'status', 'equals': '已完成'},
                })
    path = artifacts / 'run_queue.json'
    if path.exists() and read_json(path) != {
        'schema_version': 1, 'plan': cfg['plan_id'], 'expected_tasks': len(tasks),
        'tasks': tasks, 'dropped': [], 'dropped_reason': None,
    }:
        raise FileExistsError('Queue already exists with different content')
    write_json(path, {'schema_version': 1, 'plan': cfg['plan_id'], 'expected_tasks': len(tasks),
                      'tasks': tasks, 'dropped': [], 'dropped_reason': None})
    print(f'queue: {len(tasks)} tasks prepared; no dispatch')


def config_hash(task):
    paths = [CONFIG, Path('configs/offline_tsc/base.yml'),
             Path(f"configs/offline_tsc/{task['agent']}.yml"),
             Path(task['command'][task['command'].index('--dataset-manifest') + 1]),
             Path(f"configs/sim/{task['command'][task['command'].index('-n') + 1]}.cfg")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        digest.update(b'\0')
        digest.update((ROOT / path).read_bytes())
    return digest.hexdigest()


def register(cfg, artifacts):
    ledger = ROOT / 'ledger/runs.jsonl'
    original = ledger.read_text(encoding='utf-8').splitlines(keepends=True)
    existing = {json.loads(line)['run_id'] for line in original if line.strip()}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    changed, fresh = 0, []
    # Status transitions retain the row identity and the original registration fields.
    for index, line in enumerate(original):
        row = json.loads(line)
        if (row['campaign'] == cfg['name'] and row['status'] == 'REGISTERED'
                and row.get('run_id', '').startswith('hz1off_')):
            if row.get('started') or row.get('tmux'):
                raise ValueError(f"Old hz1x1 task may have started; inspect before replacing: {row['run_id']}")
            row.update(status='SCOPE_DISCARD', abort_reason='scope_discard')
            row['notes'] += '; SCOPE_DISCARD 2026-10-23：未派发三控制器矩阵由 MPLight 四控制器修订替代；原队列与 v1_hetero 数据集留档，新任务使用 mplight_r1 身份。'
            original[index] = json.dumps(row, ensure_ascii=False) + '\n'
            changed += 1
        if row['campaign'] == 'xiasha1_offline_v2' and row['status'] != 'SCOPE_DISCARD':
            if row['status'] != 'REGISTERED' or row.get('started'):
                raise ValueError('xiasha v2 state changed; inspect before discarding')
            row.update(status='SCOPE_DISCARD', abort_reason='scope_discard')
            row['notes'] += '; SCOPE_DISCARD 2026-10-23：用户裁定 xiasha 数据全量作废；v3 源亦不用于正式离线实验；126 任务未派发，转 hz1x1_offline_v1，旧队列留档禁用。'
            original[index] = json.dumps(row, ensure_ascii=False) + '\n'
            changed += 1
    for task in read_json(artifacts / 'run_queue.json')['tasks']:
        if task['run_id'] in existing:
            continue
        fresh.append({
            'run_id': task['run_id'], 'line': 'pure_offline', 'campaign': cfg['name'],
            'arm': task['agent'], 'alias_of': None,
            'net': cfg['scenes'][cfg['target']]['network'], 'seed': task['seed'],
            'episodes_total': 144000, 'episodes_increment': 144000, 'resume_from': None,
            'ngpu': -1, 'status': 'REGISTERED', 'plan_id': cfg['plan_id'],
            'code_commit': commit, 'dirty': True,
            'queue_file': str((artifacts / 'run_queue.json').relative_to(ROOT)),
            'run_dir': str(Path(task['output_path']).relative_to(ROOT)),
            'tmux': None, 'started': None, 'eval_mode': 'offline_periodic_eval',
            'behavior_source': f"hz1x1/{cfg['dataset_id']}/{task['dataset_spec']}",
            'metrics': None, 'abort_reason': None, 'error': None, 'evidence_lost': False,
            'notes': 'REGISTERED 2026-10-23：仅准备未派发；完整 200x360 源 run，'
                     '均匀 transition 采样，无质量筛选；d3qn /12、FRAP/SAC/MPLight 原值，'
                     '源 shard 不变；terminated 或 truncated 关闭 bootstrap；144k updates/81 eval nodes。',
            'run_kind': 'train', 'config_sha': config_hash(task),
            'sumo_seed_mode': 'fixed_default', 'sumo_seed': None,
            'eval_seed': task['seed'] + 20000, 'condition': task['dataset_spec'],
            'superseded_by': None, 'runtime_env': 'colight',
        })
    temporary = ledger.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(original) + ''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                                   for row in fresh), encoding='utf-8')
    temporary.replace(ledger)
    for old_queue_path, reason in (
        (ROOT / 'artifacts/xiasha1_offline_v2/run_queue.json',
         'SCOPE_DISCARD 2026-10-23: xiasha sources withdrawn; use hz1x1_offline_v1'),
        (ROOT / 'artifacts/hz1x1_offline_v1/run_queue.json',
         'SCOPE_DISCARD 2026-10-23: three-controller queue replaced by mplight_r1'),
    ):
        if old_queue_path.exists():
            queue = read_json(old_queue_path)
            queue['dropped'] = [task['run_id'] for task in queue['tasks']]
            queue['dropped_reason'] = reason
            write_json(old_queue_path, queue)
    print(f'ledger: discarded={changed}; hz1x1 MPLight revision registered={len(fresh)}')


def check(cfg, artifacts, dataset):
    report = []
    for spec in build_specs(cfg):
        path = dataset / 'datasets' / spec['name'] / 'manifest.json'
        result = validate_offline_dataset(str(path))
        data = OfflineTrajectoryDataset(str(path), seed=11000, verify_hashes=False)
        expected_cells = len(spec['source_cells']) * len(spec['behavior_training_seeds'])
        assert result['transition_count'] == expected_cells * cfg['episodes'] * cfg['decisions_per_episode']
        batch = data.sample_batch(64)
        assert batch.observations.shape == (64, data.observation_dim)
        assert data.action_dim == 8 and data.observation_dim == 16
        assert all(row['transform']['training']['scale'] == 1.0 for row in data.manifest['reward_view']['sources'].values())
        report.append({'spec': spec['name'], 'transitions': result['transition_count'],
                       'statistics': data.statistics()})
        print(f"checked {spec['name']}: {result['transition_count']}", flush=True)
    write_json(artifacts / 'preflight_validation.json', {'training_launched': False, 'pools': report})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('face', choices=('datasets', 'queue', 'register', 'check', 'all'))
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = yaml.safe_load((ROOT / CONFIG).read_text(encoding='utf-8'))['offline_campaign']
    artifacts = ROOT / 'artifacts' / cfg['name'] / cfg['revision']
    dataset = ROOT / 'data/output_data/offline_datasets/hz1x1' / cfg['dataset_id']
    if args.face in ('datasets', 'all'):
        build_datasets(cfg, artifacts, dataset)
    if args.face in ('queue', 'all'):
        build_queue(cfg, artifacts, dataset)
    if args.face in ('check', 'all'):
        check(cfg, artifacts, dataset)
    if args.face in ('register', 'all'):
        register(cfg, artifacts)


if __name__ == '__main__':
    main()
