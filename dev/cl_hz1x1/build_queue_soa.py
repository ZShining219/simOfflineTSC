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
  main_o3 order-shift replication: {+policy} arms only × o3 × seeds0-2
          λ fixed by the screen outcome (edit LAMBDAS below)
  rho     dose-response: {replay, +P} × ρ∈{0.125,0.5} × o2 × seeds0-2
  vdqn    vanilla DQN pair (double_dqn=false): {replay, +P} × o2 × seeds0-2
          (ρ=0.25 cells come from so1/soa1)

Outputs (all under gitignored artifacts/cl_hz1x1/):
  overlays/<run_id>.yml          per-run experiment overlay consumed by run.py
  run_queue_soa_<phase>.json     QueueRunner manifest (depends_on chains)
  ledger_rows_soa_<phase>.jsonl  REGISTERED rows to append into ledger/runs.jsonl
"""
import json
import sys
from pathlib import Path

import yaml

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
# Rho dose-response supplement: replay-only and +P at rho in {0.125, 0.5}.
# Combined with the r=0.25 cells from so1/soa1 this yields a 3x2 grid per
# backbone.  Arm names carry the ratio so run_ids stay unique.
ARMS_RHO = {}
for bb, agent, extra in (('soddqn', 'dqn', {'double_dqn': True}),
                         ('soduel', 'dqn', {'double_dqn': True,
                                            'dueling': True}),
                         ('sofrap', 'frap', {})):
    for tag, rho in (('r12', 0.125), ('r50', 0.5)):
        ARMS_RHO[f'{bb}_{tag}'] = {'agent': agent, 'model_extra': {
            **extra, **SO_BASE, 'offline_ratio': rho}}
        ARMS_RHO[f'{bb}p_{tag}'] = {'agent': agent, 'model_extra': {
            **extra, **SO_BASE, 'offline_ratio': rho,
            **anchor_p(LAMBDA_POLICY_MAIN)}}
# Vanilla (non-double) DQN pair: the whole campaign used double_dqn=True,
# so the plain DQN cell needs both its replay baseline and +P arm.
ARMS_VDQN = {
    'sodqn': {'agent': 'dqn', 'model_extra': {
        'double_dqn': False, **SO_BASE}},
    'sodqnp': {'agent': 'dqn', 'model_extra': {
        'double_dqn': False, **SO_BASE, **anchor_p(LAMBDA_POLICY_MAIN)}},
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
    # Per-arm ratio override (rho dose-response arms); stage 1 always online.
    me['offline_ratio'] = 0.0 if stage_idx == 0 else me.get(
        'offline_ratio', OFFLINE_RATIO)
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
    for key in (*SO_BASE, *ANCHOR_KEYS, 'offline_ratio'):
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


def snap_overlay(run_id, model_extra, snap):
    """Eval-only cell loading a bare model/{e}_0.pt snapshot (no resumable)."""
    me = dict(model_extra)
    for key in (*SO_BASE, *ANCHOR_KEYS, 'offline_ratio'):
        me.pop(key, None)
    me.update({
        'model_snapshot': str(snap),
        'epsilon_mode': 'fixed_low',
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

    def add_ledger(run_id, arm, net, seed, kind, stage, episodes, rho):
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
                     f'order_stage={stage} offline_ratio={rho}',
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
                rho = spec['model_extra'].get('offline_ratio', OFFLINE_RATIO)
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
                               f'{order_name}/st{st + 1}', stage_episodes,
                               0.0 if st == 0 else rho)
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
                                       f'{order_name}/ev{st + 1}', 0, rho)
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
        'arm_offline_ratio': {
            a: s['model_extra'].get('offline_ratio', OFFLINE_RATIO)
            for a, s in arms.items()},
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


def main_vbase(stage_episodes=100, eval_at=(0, 50, 100)):
    """Vanilla-DQN single-scene baselines (cl1_* naming convention).

    matrix_report.json baselines are keyed '{backbone}_{tag}_sd{seed}' read
    from ``cl1_{arm}_base_{tag}_sd{seed}`` runs; every existing backbone had a
    base run except plain DQN, so emit the 4-scene x 3-seed grid here.
    """
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    tasks, ledger = [], []
    for tag, net in NETS.items():
        for seed in SEEDS:
            rid = f'cl1_dqn_base_{tag}_sd{seed}'
            ov = write_overlay(rid, {'double_dqn': False}, {
                'episodes': stage_episodes,
                'evaluation_episodes': list(eval_at),
                'resumable_checkpoint_episodes': list(eval_at),
                'test_when_train': False})
            odir = out_dir('dqn', net, rid)
            tasks.append({
                'run_id': rid,
                'command': cmd('dqn', net, rid, ov, seed, 0),
                'cwd': str(ROOT), 'env': dict(ENV),
                'output_path': str(odir),
                'completion': completion(odir),
                'campaign': 'cl_hz1x1', 'arm': 'dqn_base',
                'scene': net, 'training_seed': seed,
                'run_kind': 'train',
                'episode_budget': stage_episodes,
            })
            ledger.append({
                'run_id': rid, 'line': 'cl_hz1x1', 'campaign': 'cl_hz1x1',
                'arm': 'dqn_base', 'alias_of': None, 'net': net, 'seed': seed,
                'episodes_total': stage_episodes,
                'episodes_increment': stage_episodes, 'resume_from': None,
                'ngpu': -1, 'status': 'REGISTERED',
                'plan_id': 'cl_hz1x1_anchor_vdqnb', 'code_commit': None,
                'dirty': None,
                'queue_file': 'artifacts/cl_hz1x1/run_queue_soa_vdqnb.json',
                'run_dir': None, 'tmux': None, 'started': None,
                'eval_mode': None, 'behavior_source': None, 'metrics': {},
                'abort_reason': None, 'error': None, 'evidence_lost': False,
                'notes': 'REGISTERED: cl_hz1x1 vanilla dqn single-scene '
                         'baseline (mirror of cl1_{arm}_base runs)',
                'run_kind': 'train', 'config_sha': None,
                'sumo_seed_mode': None, 'sumo_seed': None, 'eval_seed': None,
                'condition': 'single_scene_baseline', 'superseded_by': None,
                'runtime_env': 'colight', 'end_time': None,
            })
    manifest = {
        'schema_version': 1,
        'plan': 'cl_hz1x1_anchor_vdqnb',
        'expected_tasks': len(tasks),
        'note': 'vanilla DQN (double_dqn=false) single-scene baselines, '
                'cl1_* naming for harvest_matrix baseline lookup',
        'training_seeds': SEEDS,
        'stage_episodes': stage_episodes,
        'tasks': tasks,
    }
    (ART / 'run_queue_soa_vdqnb.json').write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    (ART / 'ledger_rows_soa_vdqnb.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in ledger) + '\n',
        encoding='utf-8')
    print(f'phase=vdqnb tasks={len(tasks)} ledger_rows={len(ledger)}')


SNAP_EPS = (5, 10, 15, 20)
# Convergence-time probe for the strongest measured entry shock:
# order o3 stage 3 (c3 -> s1). Each cell loads model/{e}_0.pt of a finished
# stage-3 run via model_snapshot (target-net snapshot for dqn-family,
# online-net for frap) and runs the standard frozen-eval protocol.
SNAP_CELLS = {
    'ddqn': ('dqn', {'double_dqn': True},
             {'clear': 'cl1_ddqn_clear_o3', 'semi': 'so1_soddqn_o3',
              'p': 'soa3_soddqnp_o3'}),
    'duel': ('dqn', {'double_dqn': True, 'dueling': True},
             {'clear': 'cl1_duel_clear_o3', 'semi': 'so1_soduel_o3',
              'p': 'soa3_soduelp_o3'}),
    'frap': ('frap', {},
             {'clear': 'cl1_frap_clear_o3', 'semi': 'so1_sofrap_o3',
              'p': 'soa3_sofrapp_o3'}),
}


def main_snap(eval_seeds=EVAL_SEEDS):
    """Snapshot-policy eval cells at ep5/10/15/20 of o3 st3 (new scene s1)."""
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    tasks, ledger, missing = [], [], []
    net = NETS['s1']
    for bb, (agent, mextra, prefixes) in SNAP_CELLS.items():
        for cond, prefix in prefixes.items():
            for seed in SEEDS:
                src = out_dir(agent, net, f'{prefix}_sd{seed}_st3')
                for ep in SNAP_EPS:
                    snap = src / 'model' / f'{ep}_0.pt'
                    if not snap.exists():
                        missing.append(str(snap))
                        continue
                    for es in eval_seeds:
                        rid = (f'snap3_{bb}_{cond}_o3_sd{seed}'
                               f'_e{ep}_es{es}')
                        eov = snap_overlay(rid, mextra, snap)
                        tasks.append({
                            'run_id': rid,
                            'command': cmd(agent, net, rid, eov, seed, es),
                            'cwd': str(ROOT), 'env': dict(ENV),
                            'output_path': str(out_dir(agent, net, rid)),
                            'completion': completion(
                                out_dir(agent, net, rid)),
                            'requires': [{'path': str(snap)}],
                            'campaign': 'cl_hz1x1',
                            'arm': f'snap3_{bb}_{cond}',
                            'order': 'o3', 'stage': 3, 'scene': net,
                            'training_seed': seed, 'eval_sumo_seed': es,
                            'checkpoint_path': str(snap),
                            'run_kind': 'eval',
                            'snapshot_episode': ep,
                        })
                        ledger.append({
                            'run_id': rid, 'line': 'cl_hz1x1',
                            'campaign': 'cl_hz1x1',
                            'arm': f'snap3_{bb}_{cond}_soa',
                            'alias_of': None, 'net': net, 'seed': seed,
                            'episodes_total': 0, 'episodes_increment': 0,
                            'resume_from': None, 'ngpu': -1,
                            'status': 'REGISTERED',
                            'plan_id': 'cl_hz1x1_snap3_convergence',
                            'code_commit': None, 'dirty': None,
                            'queue_file': 'artifacts/cl_hz1x1/'
                                          'run_queue_soa_snap3.json',
                            'run_dir': None, 'tmux': None, 'started': None,
                            'eval_mode': 'frozen_snapshot',
                            'behavior_source': None, 'metrics': {},
                            'abort_reason': None, 'error': None,
                            'evidence_lost': False,
                            'notes': f'REGISTERED: cl_hz1x1 snap3 '
                                     f'{bb}/{cond} o3/st3 snapshot ep{ep} '
                                     f'(target-net snapshot for dqn, '
                                     f'online-net for frap)',
                            'run_kind': 'eval', 'config_sha': None,
                            'sumo_seed_mode': None, 'sumo_seed': None,
                            'eval_seed': es,
                            'condition': 'semi_offline_anchor',
                            'superseded_by': None,
                            'runtime_env': 'colight', 'end_time': None,
                        })
    manifest = {
        'schema_version': 1,
        'plan': 'cl_hz1x1_snap3_convergence',
        'expected_tasks': len(tasks),
        'note': ('o3 st3 (c3->s1) snapshot-policy evals at ep5/10/15/20; '
                 'dqn-family model/{e}_0.pt holds target_model weights, '
                 'frap holds online weights — convergent-time estimates are '
                 'therefore conservative for dqn-family arms'),
        'episodes': list(SNAP_EPS),
        'training_seeds': SEEDS, 'eval_sumo_seeds': list(eval_seeds),
        'tasks': tasks,
    }
    (ART / 'run_queue_soa_snap3.json').write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    (ART / 'ledger_rows_soa_snap3.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in ledger) + '\n',
        encoding='utf-8')
    print(f'phase=snap3 tasks={len(tasks)} ledger_rows={len(ledger)} '
          f'missing={len(missing)}')
    for m in missing:
        print('  missing:', m)


SNAP3O_EPS = [0, 5, 10, 15, 20, 50, 100]
# Online-fidelity rerun of o3 st3: resumable checkpoints at the dense grid
# hold TRUE online weights (model/{e}_0.pt only stores the target net for
# dqn-family).  RNG restoration via stage_checkpoint reproduces the original
# trajectory bit-exactly (verified: 25/25 train records identical).
# frap is excluded: the eval-manifest loader whitelist has no frap branch,
# and its shock cell is negligible anyway (settles by ep5).
SNAP3O_CELLS = {
    'ddqn': ('dqn', {'clear': 'cl1_ddqn_clear_o3',
                    'semi': 'so1_soddqn_o3',
                    'p': 'soa3_soddqnp_o3'}),
    'duel': ('dqn', {'clear': 'cl1_duel_clear_o3',
                    'semi': 'so1_soduel_o3',
                    'p': 'soa3_soduelp_o3'}),
}


def main_snap3o(seeds=SEEDS):
    """Stage-3 reruns with dense resumable grid for online-net snapshot evals."""
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    tasks, ledger, missing = [], [], []
    for bb, (agent, prefixes) in SNAP3O_CELLS.items():
        for cond, prefix in prefixes.items():
            for seed in seeds:
                src_rid = f'{prefix}_sd{seed}_st3'
                src_overlay = overlay_path(src_rid)
                if not src_overlay.exists():
                    missing.append(str(src_overlay))
                    continue
                src = yaml.safe_load(
                    src_overlay.read_text(encoding='utf-8'))
                rid = f'snap3o_{bb}_{cond}_o3_sd{seed}_st3'
                odir = out_dir(agent, NETS['s1'], rid)
                me = dict(src.get('model') or {})
                te = {
                    'episodes': 100,
                    'evaluation_episodes': SNAP3O_EPS,
                    'resumable_checkpoint_episodes': SNAP3O_EPS,
                    'test_when_train': False,
                }
                ov = write_overlay(rid, me, te)
                tasks.append({
                    'run_id': rid,
                    'command': cmd(agent, NETS['s1'], rid, ov, seed, 0),
                    'cwd': str(ROOT), 'env': dict(ENV),
                    'output_path': str(odir),
                    'completion': completion(odir),
                    'requires': [{'path': me['stage_checkpoint']}],
                    'campaign': 'cl_hz1x1',
                    'arm': f'snap3o_{bb}_{cond}',
                    'order': 'o3', 'stage': 3, 'scene': NETS['s1'],
                    'training_seed': seed,
                    'replay_condition': 'semi_offline_anchor',
                    'episode_budget': 100,
                })
                ledger.append({
                    'run_id': rid, 'line': 'cl_hz1x1',
                    'campaign': 'cl_hz1x1', 'arm': f'snap3o_{bb}_{cond}_soa',
                    'alias_of': src_rid, 'net': NETS['s1'], 'seed': seed,
                    'episodes_total': 100, 'episodes_increment': 100,
                    'resume_from': me['stage_checkpoint'], 'ngpu': -1,
                    'status': 'REGISTERED',
                    'plan_id': 'cl_hz1x1_snap3o_online',
                    'code_commit': None, 'dirty': None,
                    'queue_file': 'artifacts/cl_hz1x1/'
                                  'run_queue_soa_snap3o.json',
                    'run_dir': None, 'tmux': None, 'started': None,
                    'eval_mode': None, 'behavior_source': None,
                    'metrics': {}, 'abort_reason': None, 'error': None,
                    'evidence_lost': False,
                    'notes': f'REGISTERED: cl_hz1x1 snap3o online-fidelity '
                             f'rerun of {src_rid} with resumable grid '
                             f'{SNAP3O_EPS} (RNG-restored, trajectory '
                             f'identical)',
                    'run_kind': 'train', 'config_sha': None,
                    'sumo_seed_mode': None, 'sumo_seed': 0,
                    'eval_seed': None,
                    'condition': 'semi_offline_anchor',
                    'superseded_by': None, 'runtime_env': 'colight',
                    'end_time': None,
                })
    manifest = {
        'schema_version': 1,
        'plan': 'cl_hz1x1_snap3o_online',
        'expected_tasks': len(tasks),
        'note': ('o3 st3 reruns with resumable ckpts at ep0/5/10/15/20/50/100 '
                 'so frozen evals measure the ONLINE net; deterministic '
                 'reproduction via RNG-restoring stage_checkpoint'),
        'episodes': SNAP3O_EPS,
        'training_seeds': list(seeds),
        'tasks': tasks,
    }
    (ART / 'run_queue_soa_snap3o.json').write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    (ART / 'ledger_rows_soa_snap3o.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in ledger) + '\n',
        encoding='utf-8')
    print(f'phase=snap3o tasks={len(tasks)} ledger_rows={len(ledger)} '
          f'missing={len(missing)}')
    for m in missing:
        print('  missing:', m)


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
    elif phase == 'main_o3':
        # Order-shift replication of the winning (+P) arms only.
        p_arms = {k: v for k, v in ARMS_MAIN.items() if not k.endswith('pv')}
        main('main_o3', p_arms, {'o3': ORDERS['o3']}, SEEDS, EVAL_SEEDS,
             stage_episodes=100, eval_at=[0, 50, 100], prefix='soa3')
    elif phase == 'rho':
        # rho dose-response: {replay, +P} x rho in {0.125, 0.5} x o2.
        main('rho', ARMS_RHO, {'o2': ORDERS['o2']}, SEEDS, EVAL_SEEDS,
             stage_episodes=100, eval_at=[0, 50, 100], prefix='soar')
    elif phase == 'vdqn':
        # Vanilla DQN pair (replay-only baseline + +P), o2 only.
        main('vdqn', ARMS_VDQN, {'o2': ORDERS['o2']}, SEEDS, EVAL_SEEDS,
             stage_episodes=100, eval_at=[0, 50, 100], prefix='soav')
    elif phase == 'vdqnb':
        # Vanilla DQN single-scene baselines for ΔATT denominators.
        main_vbase()
    elif phase == 'snap3':
        # Snapshot-policy convergence probe on the o3 st3 shock cell.
        main_snap()
    elif phase == 'snap3o':
        # Online-fidelity reruns of o3 st3 with dense resumable grid.
        main_snap3o()
    else:
        raise SystemExit(f'unknown phase: {phase}')
