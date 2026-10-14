"""Build the cl_hz1x1 Q-anchor semi-offline queue manifest + per-run overlays.

Extension of build_queue_so.py: same causal 25% semi-offline protocol
(orders/seeds/stage budget/eval cells/cleared online replay), but each
stage>1 additionally loads frozen stage-end teachers for every visible
history scene via ``teacher_paths`` and adds output-anchor losses on the
offline part of every minibatch:

    loss = TD(online ∪ offline)
         + anchor_lambda_policy * KL(softmax(Q_teacher/T) ‖ softmax(Q_theta/T))
         + anchor_lambda_value  * Huber(Q_theta − Q_teacher)   [non-taken dims]

implemented in agent/dqn.py::DQNAgent and agent/frap.py::FRAP_DQNAgent.
Teachers are the previous stages' episode-end online_model checkpoints —
the same files already referenced by ``stage_checkpoint``.

Phases (argv[1], default 'screen'):
  smoke   single soddqnp chain, o2, seed0, 10 episodes/stage — mechanics check
  screen  ddqn o2 seeds0-2: λ_π ∈ {1,10} plus one (λ_π=1, λ_q=0.1) pv probe —
          picks the anchor weight before the main matrix
  main    {ddqn,duel,frap} × {+policy, +policy+value} × o2 × seeds0-2,
          λ fixed by the screen outcome (edit LAMBDAS below)

Outputs (all under gitignored artifacts/cl_hz1x1/):
  overlays/<run_id>.yml          per-run experiment overlay consumed by run.py
  run_queue_soa_<phase>.json     QueueRunner manifest (depends_on chains)
  ledger_rows_soa_<phase>.jsonl  REGISTERED rows to append into ledger/runs.jsonl
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts' / 'cl_hz1x1'
OVERLAYS = ART / 'overlays'
PY = '/data/users/zfh/workspace/envs/colight/bin/python'
LD_PRELOAD = ('/data/users/zfh/workspace/projects/simOfflineTSC/data/output_data/'
              'cross_algorithm/plan5_b100/environment/micromamba/envs/colight/'
              'lib/libstdc++.so.6')
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
# Anchor keys must never reach eval cells (same convention as SO_BASE).
ANCHOR_KEYS = (
    'anchor_lambda_policy', 'anchor_lambda_value', 'anchor_temperature',
    'teacher_paths',
)

# Filled after the screen phase picks the weights; screen arms carry their
# own explicit lambdas.
LAMBDA_POLICY_MAIN = 1.0
LAMBDA_VALUE_MAIN = 0.1


def anchor_p(lam):
    return {'anchor_lambda_policy': lam, 'anchor_temperature': 1.0}


def anchor_pv(lam_p, lam_v):
    return {**anchor_p(lam_p), 'anchor_lambda_value': lam_v}


ARMS_SCREEN = {
    'soddqnp_l1': {'agent': 'dqn', 'model_extra': {
        'double_dqn': True, **SO_BASE, **anchor_p(1.0)}},
    'soddqnp_l10': {'agent': 'dqn', 'model_extra': {
        'double_dqn': True, **SO_BASE, **anchor_p(10.0)}},
    'soddqnpv_probe': {'agent': 'dqn', 'model_extra': {
        'double_dqn': True, **SO_BASE, **anchor_pv(1.0, 0.1)}},
}
ARMS_MAIN = {
    'soddqnp': {'agent': 'dqn', 'model_extra': {
        'double_dqn': True, **SO_BASE, **anchor_p(LAMBDA_POLICY_MAIN)}},
    'soduelp': {'agent': 'dqn', 'model_extra': {
        'double_dqn': True, 'dueling': True,
        **SO_BASE, **anchor_p(LAMBDA_POLICY_MAIN)}},
    'sofrapp': {'agent': 'frap', 'model_extra': {
        **SO_BASE, **anchor_p(LAMBDA_POLICY_MAIN)}},
    'soddqnpv': {'agent': 'dqn', 'model_extra': {
        'double_dqn': True, **SO_BASE,
        **anchor_pv(LAMBDA_POLICY_MAIN, LAMBDA_VALUE_MAIN)}},
    'soduelpv': {'agent': 'dqn', 'model_extra': {
        'double_dqn': True, 'dueling': True,
        **SO_BASE, **anchor_pv(LAMBDA_POLICY_MAIN, LAMBDA_VALUE_MAIN)}},
    'sofrappv': {'agent': 'frap', 'model_extra': {
        **SO_BASE, **anchor_pv(LAMBDA_POLICY_MAIN, LAMBDA_VALUE_MAIN)}},
}
ARMS_SMOKE = {
    'soddqnp': {'agent': 'dqn', 'model_extra': {
        'double_dqn': True, **SO_BASE, **anchor_p(1.0)}},
}
SEEDS = [0, 1, 2]
EVAL_SEEDS = [101, 102, 103]
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
            if isinstance(v, dict):
                lines.append(f'  {k}:')
                for sub_k, sub_v in v.items():
                    lines.append(f'    {sub_k}: {sub_v}')
            else:
                lines.append(f'  {k}: {yaml_value(v)}')
        lines.append('')
    lines.append('trainer:')
    for k, v in trainer_extra.items():
        lines.append(f'  {k}: {yaml_value(v)}')
    path = overlay_path(run_id)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def train_overlay(run_id, model_extra, stage_idx, nets, history_paths,
                  teacher_ckpts, prev_ckpt, stage_episodes, eval_at):
    me = dict(model_extra)
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
        if 'anchor_lambda_policy' in me or 'anchor_lambda_value' in me:
            me['teacher_paths'] = {
                scene: str(teacher_ckpts[scene]) for scene in nets[:stage_idx]}
    te = {
        'episodes': stage_episodes,
        'evaluation_episodes': eval_at,
        'resumable_checkpoint_episodes': eval_at,
        'test_when_train': False,
    }
    return write_overlay(run_id, me, te)


def eval_overlay(run_id, model_extra, ckpt):
    me = dict(model_extra)
    # Eval cells never train: offline and anchor keys are dropped, matching
    # the cl1/so1 frozen-checkpoint evaluation protocol.
    for key in (*SO_BASE, *ANCHOR_KEYS):
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


def main(phase, arms, orders, seeds, eval_seeds, stage_episodes, eval_at,
         prefix):
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    tasks = []
    ledger = []

    def add_ledger(run_id, arm, net, seed, kind, stage, episodes):
        ledger.append({
            'run_id': run_id, 'line': 'cl_hz1x1', 'campaign': 'cl_hz1x1',
            'arm': f'{arm}_soa', 'alias_of': None,
            'net': net, 'seed': seed, 'episodes_total': episodes,
            'episodes_increment': episodes, 'resume_from': None, 'ngpu': -1,
            'status': 'REGISTERED', 'plan_id': f'cl_hz1x1_anchor_{phase}',
            'code_commit': None, 'dirty': None,
            'queue_file': f'artifacts/cl_hz1x1/run_queue_soa_{phase}.json',
            'run_dir': None, 'tmux': None, 'started': None,
            'eval_mode': 'frozen_stage_ckpt' if kind == 'eval' else None,
            'behavior_source': None, 'metrics': {}, 'abort_reason': None,
            'error': None, 'evidence_lost': False,
            'notes': f'REGISTERED: cl_hz1x1 anchor {phase} {kind} '
                     f'order_stage={stage} offline_ratio={OFFLINE_RATIO}',
            'run_kind': kind, 'config_sha': None, 'sumo_seed_mode': None,
            'sumo_seed': None, 'eval_seed': None,
            'condition': 'semi_offline_anchor',
            'superseded_by': None, 'runtime_env': 'colight',
            'end_time': None,
        })

    for order_name, nets in orders.items():
        for seed in seeds:
            for arm, spec in arms.items():
                agent = spec['agent']
                prev_ckpt = None
                prev_run_id = None
                history_paths = []
                teacher_ckpts = {}
                for st, net in enumerate(nets):
                    rid = f'{prefix}_{arm}_{order_name}_sd{seed}_st{st + 1}'
                    odir = out_dir(agent, net, rid)
                    ov = train_overlay(
                        rid, spec['model_extra'], st, nets, history_paths,
                        teacher_ckpts, prev_ckpt, stage_episodes, eval_at)
                    ckpt = (odir / 'checkpoints' / 'resumable' /
                            f'episode_{stage_episodes:04d}.pt')
                    archive_manifest = odir / 'history_archive' / 'manifest.json'
                    task = {
                        'run_id': rid,
                        'command': cmd(agent, net, rid, ov, seed, 0),
                        'cwd': str(ROOT), 'env': dict(ENV),
                        'output_path': str(odir),
                        'completion': completion(odir),
                        'checkpoint_path': str(ckpt),
                        'campaign': 'cl_hz1x1', 'arm': f'{arm}_soa',
                        'order': order_name, 'stage': st + 1,
                        'scene': net, 'training_seed': seed,
                        'replay_condition': 'semi_offline_anchor',
                        'episode_budget': stage_episodes,
                    }
                    requires = []
                    if prev_run_id:
                        task['depends_on'] = [prev_run_id]
                        requires.append({'path': str(prev_ckpt)})
                        requires.extend(
                            {'path': str(p / 'manifest.json')}
                            for p in history_paths)
                        requires.extend(
                            {'path': str(p)} for p in teacher_ckpts.values())
                    if requires:
                        task['requires'] = requires
                    tasks.append(task)
                    add_ledger(rid, arm, net, seed, 'train',
                               f'{order_name}/st{st + 1}', stage_episodes)
                    for enet in nets[:st + 1]:
                        for es in eval_seeds:
                            eid = (f'{prefix}_{arm}_{order_name}_sd{seed}'
                                   f'_ev{st + 1}_{NET_TAG[enet]}_es{es}')
                            eodir = out_dir(agent, enet, eid)
                            eov = eval_overlay(eid, spec['model_extra'], ckpt)
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
                                'arm': f'{arm}_soa',
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
                    teacher_ckpts[net] = ckpt
                    prev_ckpt, prev_run_id = ckpt, rid

    manifest = {
        'schema_version': 1,
        'plan': f'cl_hz1x1_anchor_{phase}',
        'expected_tasks': len(tasks),
        'orders': orders,
        'arms': list(arms),
        'offline_ratio': OFFLINE_RATIO,
        'history_access_mode': 'causal',
        'replay_condition': 'clear_online_buffer',
        'anchor_loss': ('policy KL on softmax(Q/T) + optional masked value '
                        'Huber vs per-scene frozen stage-end teachers'),
        'training_seeds': seeds,
        'eval_sumo_seeds': eval_seeds,
        'stage_episodes': stage_episodes,
        'sumo_seed_policy': 'train=0 fixed; eval=101/102/103',
        'tasks': tasks,
    }
    queue_path = ART / f'run_queue_soa_{phase}.json'
    queue_path.write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    (ART / f'ledger_rows_soa_{phase}.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in ledger) + '\n',
        encoding='utf-8')
    n_train = sum(1 for t in tasks if t.get('run_kind') != 'eval')
    print(f'phase={phase} tasks={len(tasks)} train={n_train} '
          f'eval={len(tasks) - n_train} ledger_rows={len(ledger)}')


if __name__ == '__main__':
    phase = sys.argv[1] if len(sys.argv) > 1 else 'screen'
    if phase == 'smoke':
        main('smoke', ARMS_SMOKE, {'o2': ORDERS['o2']}, [0], [101],
             stage_episodes=10, eval_at=[0, 10], prefix='soamk')
    elif phase == 'screen':
        main('screen', ARMS_SCREEN, {'o2': ORDERS['o2']}, SEEDS, EVAL_SEEDS,
             stage_episodes=100, eval_at=[0, 50, 100], prefix='soas')
    elif phase == 'main':
        main('main', ARMS_MAIN, {'o2': ORDERS['o2']}, SEEDS, EVAL_SEEDS,
             stage_episodes=100, eval_at=[0, 50, 100], prefix='soa1')
    else:
        raise SystemExit(f'unknown phase: {phase}')
