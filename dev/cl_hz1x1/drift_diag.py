#!/usr/bin/env python3
"""Historical policy/value drift diagnostic for the cl_hz1x1 campaign.

Frozen, inference-only diagnosis.  Measures whether shared-parameter updates
during later-stage training drift the value network on *historical* states,
and whether the drift lines up with closed-loop Travel-Time degradation.

Subcommands
-----------
pool
    Build a per-scene reference state pool via frozen greedy rollouts of a
    stage-end baseline model on held-out SUMO seeds.  For scenes whose two
    replay arms already diverged (c4/c2) the pool is merged from both arms'
    baselines and each record keeps a `src` label.  The same pool is used for
    every checkpoint of every arm.
measure
    Load `online_model_state_dict` from resumable checkpoints (no training,
    no writes) and forward Q on the shared pool.  Per scene x arm x seed x
    checkpoint reports:
      q_l1           mean |Q - Q_base|               (raw)
      qc_l1          same on state-centred Q (remove per-state offset)
      flip           P(argmax Q != argmax Q_base)    (cumulative)
      flip_step      P(argmax != previous ckpt in the same arm chain)
      gap            mean(Q_top1 - Q_top2);  gap_delta vs baseline
    plus a supplementary `archive` dataset: same metrics on a fixed sample of
    so-arm history_archive states (distribution-complement probe).
    Cross-arm rows: greedy-action disagreement between clear/so checkpoints
    on the identical pool.
mkqueue
    Emit a governed run_queue JSON + ledger rows for the missing frozen TT
    evaluation cells (mid-stage checkpoints on historical scenes).
report
    Harvest TT from eval dirs (existing ev* cells + drift cells), join with
    drift rows, emit JSON summary + figures:
      drift matrix (stage x scene), drift-vs-progress curves,
      drift-vs-TT correspondence scatter.

Run inside the `colight` env with the run_queue LD_PRELOAD.
"""
import argparse
import copy
import glob
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import agent  # noqa: F401  register agent classes into Registry

NETNAME = {'s1': 'sumohz1x1', 'c4': 'sumohz1x1_config4',
           'c2': 'sumohz1x1_config2', 'c3': 'sumohz1x1_config3'}
NETTAG = {v: k for k, v in NETNAME.items()}
ORDERS = {'o2': ['s1', 'c4', 'c2', 'c3'], 'o3': ['c2', 'c3', 's1', 'c4']}
SEEDS = [0, 1, 2]
EVAL_SEEDS = [101, 102, 103]
POOL_SEEDS = [301, 302, 303, 304, 305]
POOL_TARGET = 1500
EPS = [0, 50, 100]
OUT = ROOT / 'data' / 'output_data' / 'tsc'
ART = ROOT / 'artifacts' / 'cl_hz1x1'

# arm tag -> (agent_dir, train run prefix pattern {o},{sd},{st})
ARMS = {
    'ddqn_clear': ('sumo_dqn', 'cl1_ddqn_clear_{o}_sd{sd}_st{st}'),
    'duel_clear': ('sumo_dqn', 'cl1_duel_clear_{o}_sd{sd}_st{st}'),
    'soddqn':     ('sumo_dqn', 'so1_soddqn_{o}_sd{sd}_st{st}'),
    'soduel':     ('sumo_dqn', 'so1_soduel_{o}_sd{sd}_st{st}'),
}
PY = '/data/users/zfh/workspace/envs/colight/bin/python'
LD_PRELOAD = ('/data/users/zfh/workspace/projects/simOfflineTSC/data/output_data/'
              'cross_algorithm/plan5_b100/environment/micromamba/envs/colight/'
              'lib/libstdc++.so.6')


def run_dir(agent_dir, scene_key, prefix):
    return OUT / agent_dir / NETNAME[scene_key] / prefix


def ckpt_path(agent_dir, scene_key, prefix, ep):
    return (run_dir(agent_dir, scene_key, prefix)
            / 'checkpoints' / 'resumable' / f'episode_{ep:04d}.pt')


# ---------------------------------------------------------------- model utils
def build_model(sd):
    """Rebuild Q net from a state dict (mirrors dev/cl_hz1x1/grad_diag.py)."""
    from agent.dqn import DQNNet, DuelingDQNNet
    keys = set(sd)
    assert 'lane_embedding.weight' not in keys, 'frap not in scope'
    sd0 = next(iter(sd.values()))
    in_dim, out_dim = sd0.shape[1], list(sd.values())[-1].shape[0]
    if 'feature.0.weight' in keys:
        return 'dueling', DuelingDQNNet(in_dim, out_dim)
    return 'mlp', DQNNet(in_dim, out_dim)


def load_q(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    arch, model = build_model(ck['agents'][0]['online_model_state_dict'])
    model.load_state_dict(ck['agents'][0]['online_model_state_dict'])
    model.eval()
    return arch, model


def features(obs, phase):
    """obs (N,8) + phase (N,) int -> (N,16) one-hot model input (one_hot=True)."""
    n = len(obs)
    ph = np.zeros((n, 8), dtype=np.float32)
    ph[range(n), np.asarray(phase, dtype=int)] = 1.0
    return np.concatenate([np.asarray(obs, dtype=np.float32), ph], axis=1)


def q_forward(model, feat):
    with torch.no_grad():
        return model(torch.tensor(feat, dtype=torch.float32),
                     train=False).numpy()


def drift_metrics(q, qb, qprev=None):
    dq = np.abs(q - qb)
    cen = lambda x: x - x.mean(axis=1, keepdims=True)
    dqc = np.abs(cen(q) - cen(qb))
    part = np.partition(-q, 1, axis=1)
    gap = float(np.mean(-part[:, 0] + part[:, 1]))
    part_b = np.partition(-qb, 1, axis=1)
    gap_b = float(np.mean(-part_b[:, 0] + part_b[:, 1]))
    out = {
        'q_l1': float(dq.mean()),
        'q_l1_rel': float(dq.mean() / (np.abs(qb).mean() + 1e-9)),
        'qc_l1': float(dqc.mean()),
        'qc_l1_rel': float(dqc.mean() / (np.abs(cen(qb)).mean() + 1e-9)),
        'flip': float(np.mean(q.argmax(1) != qb.argmax(1))),
        'gap': gap,
        'gap_delta': gap - gap_b,
        'q_abs': float(np.abs(q).mean()),
    }
    if qprev is not None:
        out['flip_step'] = float(np.mean(q.argmax(1) != qprev.argmax(1)))
        out['q_l1_step'] = float(np.abs(q - qprev).mean())
    return out


# ------------------------------------------------------------- pool building
def _registry_stack(net, prefix, sumo_seed):
    """Minimal run.py-equivalent registry stack for a frozen rollout."""
    from common import interface
    from common.registry import Registry
    from utils.logger import build_config

    args = SimpleNamespace(
        task='tsc', agent='dqn', world='sumo', network=net, dataset='onfly',
        prefix=prefix, seed=0, sumo_seed=sumo_seed, thread_num=4, ngpu='-1',
        debug=False, interface='libsumo', delay_type='apx', episodes=None,
        evaluation_manifest=None, evaluation_output=None,
        experiment_config=None, resume_output=None, resume_checkpoint=None)
    config, _ = build_config(args)
    interface.Command_Setting_Interface(config)
    interface.Logger_param_Interface(config)
    interface.World_param_Interface(config)
    interface.Logger_path_Interface(config)
    os.makedirs(Registry.mapping['logger_mapping']['path'].path,
                exist_ok=True)
    interface.Trainer_param_Interface(config)
    interface.ModelAgent_param_Interface(config)
    return config


def build_stack(net, prefix, sumo_seed):
    """world + agent + env + metric, mirroring TSCTrainer construction."""
    from common.registry import Registry
    from common.metrics import Metrics
    from environment import TSCEnv

    config = _registry_stack(net, prefix, sumo_seed)
    cmd = Registry.mapping['command_mapping']['setting'].param
    world = Registry.mapping['world_mapping'][cmd['world']](
        Registry.mapping['world_mapping']['setting'].config_path,
        cmd['thread_num'], interface=cmd['interface'])
    agents = [Registry.mapping['model_mapping'][cmd['agent']](world, 0)]
    metric = Metrics(['rewards', 'queue', 'delay'],
                     ['delay', 'real avg travel time', 'throughput'],
                     world, agents)
    return world, agents, TSCEnv(world, agents, metric), config


def collect_pool(scene, ckpt, src_tag, out_npz, seeds=POOL_SEEDS,
                 episodes_per_seed=1, target=POOL_TARGET):
    net = NETNAME[scene]
    world, agents, env, config = build_stack(
        net, f'driftpool_{scene}_{src_tag}', seeds[0])
    arch, model = load_q(ckpt)
    agents[0].model.load_state_dict(model.state_dict())
    ag = agents[0]
    interval = config['trainer']['action_interval']
    test_steps = config['trainer']['test_steps']
    obs_l, ph_l, meta_l = [], [], []
    for seed in seeds:
        for ep in range(episodes_per_seed):
            world.set_episode_seed(seed)
            obs = env.reset()
            ag.reset()
            steps = 0
            while steps < test_steps:
                phases = np.stack([a.get_phase() for a in agents])
                acts = np.stack([
                    a.get_action(obs[i], phases[i], test=True)
                    for i, a in enumerate(agents)])
                obs_l.append(np.asarray(obs[0][0], dtype=np.float32))
                ph_l.append(int(np.asarray(ag.get_phase()).reshape(-1)[0]))
                meta_l.append((src_tag, seed, ep, steps))
                for _ in range(min(interval, test_steps - steps)):
                    obs, rews, dones, _ = env.step(acts.flatten())
                    steps += 1
                    if all(dones):
                        break
                if all(dones):
                    break
    obs, ph = np.stack(obs_l), np.asarray(ph_l)
    if len(obs) > target:
        sel = np.random.RandomState(0).choice(len(obs), target, replace=False)
        sel.sort()
        obs, ph = obs[sel], ph[sel]
        meta_l = [meta_l[i] for i in sel]
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, obs=obs, phase=ph,
                        meta=np.asarray(meta_l, dtype=object),
                        src_tag=src_tag, scene=scene, ckpt=str(ckpt))
    print(f'pool {out_npz}: {len(obs)} states from {src_tag} '
          f'(seeds={seeds})', flush=True)


def merge_pools(parts, out_npz, target=POOL_TARGET):
    obs, ph, meta = [], [], []
    for p in parts:
        z = np.load(p, allow_pickle=True)
        obs.append(z['obs']); ph.append(z['phase'])
        meta += list(z['meta'])
    obs, ph = np.concatenate(obs), np.concatenate(ph)
    if len(obs) > target:
        sel = np.random.RandomState(0).choice(len(obs), target, replace=False)
        sel.sort()
        obs, ph = obs[sel], ph[sel]
        meta = [meta[i] for i in sel]
    np.savez_compressed(out_npz, obs=obs, phase=ph,
                        meta=np.asarray(meta, dtype=object))
    print(f'merged pool {out_npz}: {len(obs)} states '
          f'({len(parts)} sources)', flush=True)


# -------------------------------------------------------------------- measure
def chain_jobs(arm, order, hist_scene, upto_stage):
    """Checkpoints where hist_scene is already history.

    Returns list of (label, stage, ep, path).  Baseline = the stage that
    trained hist_scene (ep100) plus every later checkpoint ep{0,50,100}.
    """
    scenes = ORDERS[order]
    base_st = scenes.index(hist_scene) + 1
    agent_dir, pat = ARMS[arm]
    jobs = []
    for st in range(base_st + 1, upto_stage + 1):
        for ep in EPS:
            prefix = pat.format(o=order, sd='{sd}', st=st)
            jobs.append((st, ep, agent_dir, scenes[st - 1], prefix))
    return base_st, jobs


def measure_scene(arm, order, seed, hist_scene, upto_stage, pool_npz,
                  out_rows, archive_dirs=None):
    scenes = ORDERS[order]
    base_st = scenes.index(hist_scene) + 1
    agent_dir, pat = ARMS[arm]
    base_prefix = pat.format(o=order, sd=seed, st=base_st)
    base_ck = ckpt_path(agent_dir, scenes[base_st - 1], base_prefix, 100)

    z = np.load(pool_npz, allow_pickle=True)
    feat = features(z['obs'], z['phase'])
    _, mbase = load_q(base_ck)
    qb = q_forward(mbase, feat)
    qprev = None

    _, jobs = chain_jobs(arm, order, hist_scene, upto_stage)
    for st, ep, adir, scene_key, pat_sd in jobs:
        ck = ckpt_path(adir, scene_key, pat_sd.format(sd=seed), ep)
        if not ck.exists():
            print(f'  !missing {ck}', flush=True)
            continue
        _, m = load_q(ck)
        q = q_forward(m, feat)
        row = {'dataset': 'pool', 'scene': hist_scene, 'arm': arm,
               'order': order, 'seed': seed,
               'stage': st, 'ep': ep, 'ckpt': str(ck.relative_to(ROOT)),
               'base_ckpt': str(base_ck.relative_to(ROOT))}
        row.update(drift_metrics(q, qb, qprev))
        qprev = q
        out_rows.append(row)
        print(f"  {arm} sd{seed} {hist_scene} st{st}ep{ep}: "
              f"flip={row['flip']:.3f} q_l1={row['q_l1']:.3f} "
              f"gap={row['gap']:.3f}", flush=True)

    if archive_dirs:
        arecs = []
        for d in archive_dirs:
            for shard in sorted(Path(d).glob('episode_*.pt'))[:25]:
                arecs.extend(torch.load(shard, map_location='cpu',
                                        weights_only=False))
        if arecs:
            rng = np.random.RandomState(0)
            sel = rng.choice(len(arecs), min(2000, len(arecs)),
                             replace=False)
            aobs = np.stack([np.asarray(arecs[i].state) for i in sel])
            aph = np.asarray([int(np.asarray(arecs[i].phase).reshape(-1)[0])
                              for i in sel])
            afeat = features(aobs, aph)
            aqb = q_forward(mbase, afeat)
            aprev = None
            for st, ep, adir, scene_key, pat_sd in chain_jobs(
                    arm, order, hist_scene, upto_stage)[1]:
                ck = ckpt_path(adir, scene_key, pat_sd.format(sd=seed), ep)
                if not ck.exists():
                    continue
                _, m = load_q(ck)
                q = q_forward(m, afeat)
                row = {'dataset': 'archive', 'scene': hist_scene, 'arm': arm,
                       'order': order, 'seed': seed, 'stage': st, 'ep': ep,
                       'ckpt': str(ck.relative_to(ROOT)),
                       'base_ckpt': str(base_ck.relative_to(ROOT))}
                row.update(drift_metrics(q, aqb, aprev))
                aprev = q
                out_rows.append(row)


def cross_arm_rows(rows_out, arms, order, seed, hist_scene, upto_stage,
                   pool_npz):
    """Greedy-action disagreement between arms on the identical pool."""
    z = np.load(pool_npz, allow_pickle=True)
    feat = features(z['obs'], z['phase'])
    scenes = ORDERS[order]
    base_st = scenes.index(hist_scene) + 1
    qs = {}
    for arm in arms:
        agent_dir, pat = ARMS[arm]
        for st in range(base_st + 1, upto_stage + 1):
            for ep in EPS:
                ck = ckpt_path(agent_dir, scenes[st - 1],
                               pat.format(o=order, sd=seed, st=st), ep)
                if not ck.exists():
                    continue
                _, m = load_q(ck)
                qs[(arm, st, ep)] = q_forward(m, feat)
    for (a, st, ep), qa in qs.items():
        for (b, st2, ep2), qbv in list(qs.items()):
            if st2 == st and ep2 == ep and a < b:
                rows_out.append({
                    'dataset': 'pool', 'scene': hist_scene, 'arm': f'{a}|{b}',
                    'order': order, 'seed': seed, 'stage': st, 'ep': ep,
                    'flip': float(np.mean(qa.argmax(1) != qbv.argmax(1))),
                    'q_l1': float(np.abs(qa - qbv).mean())})


# -------------------------------------------------------------------- mkqueue
def eval_overlay_yaml(arm, ckpt):
    backbone = 'duel' if 'duel' in arm else 'ddqn'
    me = {'double_dqn': True, 'dueling': backbone == 'duel'}
    lines = ['model:', f"  double_dqn: {str(me['double_dqn'])}",
             f"  dueling: {str(me['dueling'])}",
             f'  stage_checkpoint: {ckpt}',
             '  replay_policy: clear', '  epsilon_mode: fixed_low',
             '  stage_load_optimizer: true', '', 'trainer:',
             '  episodes: 0', '  evaluation_episodes: [0]',
             '  resumable_checkpoint_episodes: [0]',
             '  test_when_train: false']
    return '\n'.join(lines) + '\n'


def mkqueue(hist_scene, arms, order, stages_eps, out_queue, out_ledger):
    """Frozen eval cells: hist_scene @ (stage,ep) x arm x seed x eseed."""
    odir_ov = ART / 'overlays_drift'
    odir_ov.mkdir(parents=True, exist_ok=True)
    tasks, ledger = [], []
    scenes = ORDERS[order]
    for arm in arms:
        agent_dir, pat = ARMS[arm]
        for sd in SEEDS:
            for st, ep in stages_eps:
                ck = ckpt_path(agent_dir, scenes[st - 1],
                               pat.format(o=order, sd=sd, st=st), ep)
                assert ck.exists(), f'missing {ck}'
                for es in EVAL_SEEDS:
                    rid = (f'drift_{hist_scene}_{arm}_{order}_sd{sd}'
                           f'_st{st}ep{ep:03d}_es{es}')
                    ov = odir_ov / f'{rid}.yml'
                    ov.write_text(eval_overlay_yaml(arm, ck),
                                  encoding='utf-8')
                    odir = run_dir(agent_dir, hist_scene, rid)
                    tasks.append({
                        'run_id': rid,
                        'command': [PY, 'run.py', '-w', 'sumo', '-a', 'dqn',
                                    '-n', NETNAME[hist_scene],
                                    '--seed', str(sd),
                                    '--sumo_seed', str(es),
                                    '--interface', 'libsumo',
                                    '--prefix', rid,
                                    '--experiment-config', str(ov)],
                        'cwd': str(ROOT), 'env': {'LD_PRELOAD': LD_PRELOAD},
                        'output_path': str(odir),
                        'completion': {'path': str(odir / 'run_status.json'),
                                       'json_field': 'status',
                                       'equals': '已完成'},
                        'requires': [{'path': str(ck)}],
                        'campaign': 'cl_hz1x1', 'arm': arm, 'order': order,
                        'stage': st, 'scene': NETNAME[hist_scene],
                        'training_seed': sd, 'eval_sumo_seed': es,
                        'checkpoint_path': str(ck), 'run_kind': 'eval'})
                    ledger.append({
                        'run_id': rid, 'line': 'cl_hz1x1',
                        'campaign': 'cl_hz1x1', 'arm': arm, 'alias_of': None,
                        'net': NETNAME[hist_scene], 'seed': sd,
                        'episodes_total': 0, 'episodes_increment': 0,
                        'resume_from': None, 'ngpu': -1, 'status': 'REGISTERED',
                        'plan_id': 'cl_hz1x1_drift',
                        'code_commit': None, 'dirty': None,
                        'queue_file': str(Path(out_queue).resolve()
                                          .relative_to(ROOT)),
                        'run_dir': str(odir.relative_to(ROOT)),
                        'tmux': None, 'started': None,
                        'eval_mode': 'frozen_stage_ckpt',
                        'behavior_source': None, 'metrics': {},
                        'abort_reason': None, 'error': None,
                        'evidence_lost': False,
                        'notes': ('REGISTERED: cl_hz1x1 drift-diag eval '
                                  f'{hist_scene}@st{st}ep{ep}'),
                        'run_kind': 'eval', 'config_sha': None,
                        'sumo_seed_mode': None, 'sumo_seed': es,
                        'eval_seed': es, 'condition': 'drift_diag',
                        'superseded_by': None, 'runtime_env': 'colight',
                        'end_time': None})
    Path(out_queue).write_text(json.dumps({'tasks': tasks}, indent=1))
    with open(out_ledger, 'w') as f:
        for r in ledger:
            f.write(json.dumps(r) + '\n')
    print(f'{len(tasks)} eval tasks -> {out_queue}; '
          f'ledger -> {out_ledger}', flush=True)


# ------------------------------------------------------------------- harvest
def harvest_tt(hist_scene, arms, order):
    """Collect per-cell TT: existing ev{st}_{scene} cells + drift cells."""
    rows = []
    scenes = ORDERS[order]
    for arm in arms:
        agent_dir, pat = ARMS[arm]
        for sd in SEEDS:
            for st in range(1, 5):
                for es in EVAL_SEEDS:
                    # existing campaign eval cell (stage-end ckpt)
                    rid = pat.format(o=order, sd=sd, st=st)
                    eid = (rid.rsplit('_st', 1)[0]
                           + f'_ev{st}_{hist_scene}_es{es}')
                    d = run_dir(agent_dir, hist_scene, eid)
                    sj = d / 'evaluation' / 'summary.json'
                    if sj.exists():
                        tt = json.loads(sj.read_text())['evaluations'][0][
                            'travel_time']
                        rows.append({'scene': hist_scene, 'arm': arm,
                                     'seed': sd, 'stage': st, 'ep': 100,
                                     'es': es, 'tt': tt, 'src': 'campaign'})
                    # drift-diag cells
                    for ep in EPS:
                        rid = (f'drift_{hist_scene}_{arm}_{order}_sd{sd}'
                               f'_st{st}ep{ep:03d}_es{es}')
                        sj = (run_dir(agent_dir, hist_scene, rid)
                              / 'evaluation' / 'summary.json')
                        if sj.exists():
                            tt = json.loads(sj.read_text())['evaluations'][0][
                                'travel_time']
                            rows.append({'scene': hist_scene, 'arm': arm,
                                         'seed': sd, 'stage': st, 'ep': ep,
                                         'es': es, 'tt': tt, 'src': 'drift'})
    return rows


def cmd_report(rows_json, tt_json, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = json.loads(Path(rows_json).read_text())
    tt = json.loads(Path(tt_json).read_text())
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pool = [r for r in rows if r['dataset'] == 'pool' and '|' not in r['arm']]
    base_stage = {sc: ORDERS['o2'].index(sc) + 1 for sc in
                  {r['scene'] for r in pool}}
    xorder = [(r['stage'], r['ep']) for r in pool]
    xs = sorted(set(xorder))
    xmap = {x: i + 1 for i, x in enumerate(xs)}
    for sc, bs in base_stage.items():
        xmap[(bs, 100)] = 0
    xlab = ['base'] + [f'st{s}e{e}' for s, e in xs]
    arms = sorted({r['arm'] for r in pool})
    scenes = sorted({r['scene'] for r in pool})
    ttm = {}
    for t in tt:
        ttm.setdefault((t['scene'], t['arm'], t['seed'], t['stage'], t['ep']),
                       []).append(t['tt'])

    # fig1: drift curves + TT overlay per arm x scene (seed-mean)
    for sc in scenes:
        fig, axes = plt.subplots(1, len(arms), figsize=(6 * len(arms), 4),
                                 sharey=False)
        if len(arms) == 1:
            axes = [axes]
        for ax, arm in zip(axes, arms):
            for seed in SEEDS:
                sub = [r for r in pool if r['arm'] == arm
                       and r['scene'] == sc and r['seed'] == seed]
                sub.sort(key=lambda r: xmap[(r['stage'], r['ep'])])
                ax.plot([0] + [xmap[(r['stage'], r['ep'])] for r in sub],
                        [0] + [r['flip'] for r in sub], 'o-', ms=3,
                        label=f'sd{seed} flip')
            ax2 = ax.twinx()
            ax2.set_ylabel('travel time (s)')
            for seed in SEEDS:
                tt_pts = sorted(
                    (xmap[(st, ep)], np.mean(v))
                    for (s, a, sd, st, ep), v in ttm.items()
                    if s == sc and a == arm and sd == seed)
                if tt_pts:
                    ax2.plot([p[0] for p in tt_pts],
                             [p[1] for p in tt_pts], 'k^--', ms=4,
                             alpha=0.35 + 0.3 * seed,
                             label=f'TT sd{seed} (right)')
            ax.set_xticks(range(len(xs) + 1))
            ax.set_xticklabels(xlab, rotation=45, fontsize=7)
            ax.set_title(f'{sc} | {arm}')
            ax.set_ylabel('greedy flip vs own base')
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(outdir / f'fig_drift_tt_{sc}.png', dpi=150)
        plt.close(fig)

    # fig2: drift-vs-TT correspondence (per scene)
    for sc in scenes:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, arm in zip(axes, [a for a in arms if 'clear' in a or True][:2]):
            for seed in SEEDS:
                sub = [r for r in pool if r['arm'] == arm
                       and r['scene'] == sc and r['seed'] == seed]
                xs2, ys2 = [], []
                for r in sub:
                    k = (sc, arm, seed, r['stage'], r['ep'])
                    if k in ttm:
                        xs2.append(r['flip'])
                        ys2.append(np.mean(ttm[k]))
                ax.scatter(xs2, ys2, s=18, label=f'sd{seed}', alpha=.7)
            ax.set_xlabel('greedy flip vs own base')
            ax.set_ylabel('TT on old scene (s)')
            ax.set_title(f'{sc} | {arm}')
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(outdir / f'fig_drift_vs_tt_{sc}.png', dpi=150)
        plt.close(fig)

    print('figures ->', outdir)


# ----------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('pool')
    p.add_argument('--scene', required=True, choices=list(NETNAME))
    p.add_argument('--ckpt', required=True)
    p.add_argument('--src', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', default=','.join(map(str, POOL_SEEDS)))
    p.add_argument('--eps', type=int, default=1)
    p.add_argument('--target', type=int, default=POOL_TARGET)

    p = sub.add_parser('merge')
    p.add_argument('--parts', nargs='+', required=True)
    p.add_argument('--out', required=True)

    p = sub.add_parser('measure')
    p.add_argument('--scene', required=True, choices=list(NETNAME))
    p.add_argument('--arms', default='ddqn_clear,soddqn')
    p.add_argument('--order', default='o2')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--pool', required=True)
    p.add_argument('--archive-glob', default=None,
                   help='glob per {sd}: archive dirs of the scene stage')
    p.add_argument('--upto', type=int, default=4)
    p.add_argument('--out', required=True)

    p = sub.add_parser('mkqueue')
    p.add_argument('--scene', required=True, choices=list(NETNAME))
    p.add_argument('--arms', default='ddqn_clear,soddqn')
    p.add_argument('--order', default='o2')
    p.add_argument('--spec', default='2:0,2:50,3:0,3:50,4:0,4:50',
                   help='stage:ep list')
    p.add_argument('--queue', required=True)
    p.add_argument('--ledger', required=True)

    p = sub.add_parser('harvest')
    p.add_argument('--scene', required=True, choices=list(NETNAME))
    p.add_argument('--arms', default='ddqn_clear,soddqn')
    p.add_argument('--order', default='o2')
    p.add_argument('--out', required=True)

    p = sub.add_parser('report')
    p.add_argument('--rows', required=True)
    p.add_argument('--tt', required=True)
    p.add_argument('--outdir', required=True)

    args = ap.parse_args()

    if args.cmd == 'pool':
        collect_pool(args.scene, args.ckpt, args.src, args.out,
                     seeds=[int(s) for s in args.seeds.split(',')],
                     episodes_per_seed=args.eps, target=args.target)
    elif args.cmd == 'merge':
        merge_pools(args.parts, args.out)
    elif args.cmd == 'measure':
        rows = []
        arms = args.arms.split(',')
        for arm in arms:
            for sd in [int(s) for s in args.seeds.split(',')]:
                adirs = None
                if args.archive_glob:
                    adirs = glob.glob(args.archive_glob.format(sd=sd))
                pool = args.pool.format(sd=sd) if '{sd}' in args.pool \
                    else args.pool
                measure_scene(arm, args.order, sd, args.scene, args.upto,
                              pool, rows, adirs)
        for sd in [int(s) for s in args.seeds.split(',')]:
            if len(arms) > 1:
                pool = args.pool.format(sd=sd) if '{sd}' in args.pool \
                    else args.pool
                cross_arm_rows(rows, arms, args.order, sd, args.scene,
                               args.upto, pool)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print('written', args.out, len(rows), 'rows')
    elif args.cmd == 'mkqueue':
        spec = [(int(s.split(':')[0]), int(s.split(':')[1]))
                for s in args.spec.split(',')]
        mkqueue(args.scene, args.arms.split(','), args.order, spec,
                Path(args.queue), Path(args.ledger))
    elif args.cmd == 'harvest':
        rows = harvest_tt(args.scene, args.arms.split(','), args.order)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print('written', args.out, len(rows), 'cells')
    elif args.cmd == 'report':
        cmd_report(args.rows, args.tt, args.outdir)


if __name__ == '__main__':
    main()
