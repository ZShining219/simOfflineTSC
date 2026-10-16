"""Build the cl_hz1x1 semi-offline queue manifest + per-run overlays.

Same protocol as run_queue_cl1 (orders/seeds/stage budget/eval cells) but
each stage archives its transitions and stages >1 train with
``offline_ratio=0.25`` sampled causally from all *previous* stage archives
(``history_access_mode=causal``).  Online replay is cleared on every scene
switch, so the only cross-scene data path is the history archive — this is
the "25% semi-offline" arm.  Single-task baselines are shared with cl1
(offline_ratio=0 is the same code path), so none are regenerated here.

Outputs (all under gitignored artifacts/cl_hz1x1/):
  overlays/<run_id>.yml   per-run experiment overlay consumed by run.py
  run_queue_so1.json      QueueRunner manifest (depends_on chains)
  ledger_rows_so.jsonl    REGISTERED rows to append into ledger/runs.jsonl
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts' / 'cl_hz1x1'
OVERLAYS = ART / 'overlays'
PY = sys.executable
LD_PRELOAD = str(ROOT / 'data' / 'output_data' / 'cross_algorithm' /
                 'plan5_b100' / 'environment' / 'micromamba' / 'envs' /
                 'colight' / 'lib' / 'libstdc++.so.6')
ENV = {
    'LD_PRELOAD': LD_PRELOAD,
    'CUDA_VISIBLE_DEVICES': '-1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
}

NETS = {
    's1': 'sumohz1x1',
    'c4': 'sumohz1x1_config4',
    'c2': 'sumohz1x1_config2',
    'c3': 'sumohz1x1_config3',
}
NET_TAG = {v: k for k, v in NETS.items()}
ORDERS = {
    'o2': [NETS['s1'], NETS['c4'], NETS['c2'], NETS['c3']],
    'o3': [NETS['c2'], NETS['c3'], NETS['s1'], NETS['c4']],
}
SO_BASE = {
    'offline_sampling_strategy': 'scene_intersection_balanced',
    'history_access_mode': 'causal',
    'collect_history': True,
}
ARMS = {
    'soddqn': {'agent': 'dqn',
               'model_extra': {'double_dqn': True, **SO_BASE}},
    'soduel': {'agent': 'dqn',
               'model_extra': {'double_dqn': True, 'dueling': True, **SO_BASE}},
    'sofrap': {'agent': 'frap', 'model_extra': {**SO_BASE}},
}
SEEDS = [0, 1, 2]
EVAL_SEEDS = [101, 102, 103]
STAGE_EPISODES = 100
EVAL_EPISODES_AT = [0, 50, 100]
OFFLINE_RATIO = 0.25

OUT = ROOT / 'data' / 'output_data' / 'tsc'


def out_dir(agent, net, prefix):
    return OUT / f'sumo_{agent}' / net / prefix


def overlay_path(run_id):
    return OVERLAYS / f'{run_id}.yml'


def yaml_value(v):
    if isinstance(v, list):
        return '[' + ', '.join(
            f"'{x}'" if isinstance(x, str) else str(x) for x in v) + ']'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    return str(v)


def write_overlay(run_id, model_extra, trainer_extra):
    lines = []
    if model_extra:
        lines.append('model:')
        for k, v in model_extra.items():
            lines.append(f'  {k}: {yaml_value(v)}')
        lines.append('')
    lines.append('trainer:')
    for k, v in trainer_extra.items():
        lines.append(f'  {k}: {yaml_value(v)}')
    path = overlay_path(run_id)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def train_overlay(run_id, arm, stage_idx, nets, history_paths, prev_ckpt):
    me = dict(ARMS[arm]['model_extra'])
    me['scene_id'] = nets[stage_idx]
    me['roadnet_id'] = 'hz1x1'
    me['training_stage'] = stage_idx
    me['scene_order'] = list(nets)
    # Stage 1 has no visible history: keep the archive writer on but run
    # purely online (offline_ratio=0), matching the shared_dqn convention.
    me['offline_ratio'] = 0.0 if stage_idx == 0 else OFFLINE_RATIO
    if history_paths:
        me['history_paths'] = history_paths
    if stage_idx > 0:
        me['stage_checkpoint'] = str(prev_ckpt)
        me['replay_policy'] = 'clear'
        me['epsilon_mode'] = 'reset_schedule'
        me['stage_load_optimizer'] = 'true'
    te = {
        'episodes': STAGE_EPISODES,
        'evaluation_episodes': EVAL_EPISODES_AT,
        'resumable_checkpoint_episodes': EVAL_EPISODES_AT,
        'test_when_train': False,
    }
    return write_overlay(run_id, me, te)


def eval_overlay(run_id, arm, ckpt):
    me = dict(ARMS[arm]['model_extra'])
    # Eval cells never train: offline keys are dropped and history is not
    # collected, matching the cl1 frozen-checkpoint evaluation protocol.
    for key in SO_BASE:
        me.pop(key, None)
    me.update({
        'stage_checkpoint': str(ckpt),
        'replay_policy': 'clear',
        'epsilon_mode': 'fixed_low',
        'stage_load_optimizer': 'true',
    })
    te = {
        'episodes': 0,
        'evaluation_episodes': [0],
        'resumable_checkpoint_episodes': [0],
        'test_when_train': False,
    }
    return write_overlay(run_id, me, te)


def cmd(agent, net, prefix, overlay, seed, sumo_seed):
    return [
        PY, 'run.py', '-w', 'sumo', '-a', agent, '-n', net,
        '--seed', str(seed), '--sumo_seed', str(sumo_seed),
        '--interface', 'libsumo', '--prefix', prefix,
        '--experiment-config', str(overlay),
    ]


def completion(prefix_out):
    return {
        'path': str(prefix_out / 'run_status.json'),
        'json_field': 'status',
        'equals': '已完成',
    }


def main():
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    tasks = []
    ledger = []

    def add_ledger(run_id, arm, net, seed, kind, stage, episodes):
        ledger.append({
            'run_id': run_id, 'line': 'cl_hz1x1', 'campaign': 'cl_hz1x1',
            'arm': f'{arm}_so', 'alias_of': None,
            'net': net, 'seed': seed, 'episodes_total': episodes,
            'episodes_increment': episodes, 'resume_from': None, 'ngpu': -1,
            'status': 'REGISTERED', 'plan_id': 'cl_hz1x1_so_v1',
            'code_commit': None, 'dirty': None,
            'queue_file': 'artifacts/cl_hz1x1/run_queue_so1.json',
            'run_dir': None, 'tmux': None, 'started': None,
            'eval_mode': 'frozen_stage_ckpt' if kind == 'eval' else None,
            'behavior_source': None, 'metrics': {}, 'abort_reason': None,
            'error': None, 'evidence_lost': False,
            'notes': f'REGISTERED: cl_hz1x1 semi-offline {kind} '
                     f'order_stage={stage} offline_ratio={OFFLINE_RATIO}',
            'run_kind': kind, 'config_sha': None, 'sumo_seed_mode': None,
            'sumo_seed': None, 'eval_seed': None,
            'condition': 'semi_offline',
            'superseded_by': None, 'runtime_env': 'colight',
            'end_time': None,
        })

    for order_name, nets in ORDERS.items():
        for seed in SEEDS:
            for arm, spec in ARMS.items():
                agent = spec['agent']
                prev_ckpt = None
                prev_run_id = None
                history_paths = []
                for st, net in enumerate(nets):
                    rid = f'so1_{arm}_{order_name}_sd{seed}_st{st + 1}'
                    odir = out_dir(agent, net, rid)
                    ov = train_overlay(
                        rid, arm, st, nets, history_paths, prev_ckpt)
                    ckpt = (odir / 'checkpoints' / 'resumable' /
                            f'episode_{STAGE_EPISODES:04d}.pt')
                    archive_manifest = odir / 'history_archive' / 'manifest.json'
                    task = {
                        'run_id': rid,
                        'command': cmd(agent, net, rid, ov, seed, 0),
                        'cwd': str(ROOT), 'env': dict(ENV),
                        'output_path': str(odir),
                        'completion': completion(odir),
                        'checkpoint_path': str(ckpt),
                        'campaign': 'cl_hz1x1', 'arm': f'{arm}_so',
                        'order': order_name, 'stage': st + 1,
                        'scene': net, 'training_seed': seed,
                        'replay_condition': 'semi_offline',
                        'episode_budget': STAGE_EPISODES,
                    }
                    requires = []
                    if prev_run_id:
                        task['depends_on'] = [prev_run_id]
                        requires.append({'path': str(prev_ckpt)})
                        requires.extend(
                            {'path': str(p / 'manifest.json')}
                            for p in history_paths)
                    if requires:
                        task['requires'] = requires
                    tasks.append(task)
                    add_ledger(rid, arm, net, seed, 'train',
                               f'{order_name}/st{st + 1}', STAGE_EPISODES)
                    for enet in nets[:st + 1]:
                        for es in EVAL_SEEDS:
                            eid = (f'so1_{arm}_{order_name}_sd{seed}'
                                   f'_ev{st + 1}_{NET_TAG[enet]}_es{es}')
                            eodir = out_dir(agent, enet, eid)
                            eov = eval_overlay(eid, arm, ckpt)
                            tasks.append({
                                'run_id': eid,
                                'command': cmd(agent, enet, eid, eov,
                                               seed, es),
                                'cwd': str(ROOT), 'env': dict(ENV),
                                'output_path': str(eodir),
                                'completion': completion(eodir),
                                'depends_on': [rid],
                                'requires': [{'path': str(ckpt)}],
                                'campaign': 'cl_hz1x1',
                                'arm': f'{arm}_so',
                                'order': order_name,
                                'stage': st + 1, 'scene': enet,
                                'training_seed': seed,
                                'eval_sumo_seed': es,
                                'checkpoint_path': str(ckpt),
                                'run_kind': 'eval',
                            })
                            add_ledger(eid, arm, enet, seed, 'eval',
                                       f'{order_name}/ev{st + 1}', 0)
                    history_paths.append(odir / 'history_archive')
                    prev_ckpt, prev_run_id = ckpt, rid

    manifest = {
        'schema_version': 1,
        'plan': 'cl_hz1x1_semi_offline_v1',
        'expected_tasks': len(tasks),
        'orders': ORDERS,
        'arms': list(ARMS),
        'offline_ratio': OFFLINE_RATIO,
        'history_access_mode': 'causal',
        'replay_condition': 'clear_online_buffer',
        'training_seeds': SEEDS,
        'eval_sumo_seeds': EVAL_SEEDS,
        'stage_episodes': STAGE_EPISODES,
        'sumo_seed_policy': 'train=0 fixed; eval=101/102/103',
        'tasks': tasks,
    }
    (ART / 'run_queue_so1.json').write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    (ART / 'ledger_rows_so.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in ledger) + '\n',
        encoding='utf-8')
    n_train = sum(1 for t in tasks if t.get('run_kind') != 'eval')
    print(f'tasks={len(tasks)} train={n_train} '
          f'eval={len(tasks) - n_train} ledger_rows={len(ledger)}')


if __name__ == '__main__':
    main()
