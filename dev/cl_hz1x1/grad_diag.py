#!/usr/bin/env python3
"""Gradient-coordination diagnostic for the cl_hz1x1 semi-offline campaign.

For a stage boundary (model checkpoint entering scene_k), we measure whether
TD gradients from new-scene experience and historical (old-scene) experience
conflict, and whether a real optimizer step on new data harms the old TD loss:

  cos(g_new, g_old)            batch-level gradient cosine
  cos(g_new, g_mix)/g_old      how the actual 48+16 semi-offline batch rotates
  sign-conflict fraction       share of params where g_new*g_old < 0
                               (plain and |g_old|-magnitude-weighted)
  ||g_old|| / ||g_new||        gradient norm ratio
  delta TD_old / TD_new        mean |TD error| change on held-out eval sets
                               after ONE real clipped optimizer step on
                               (a) pure new batch, (b) 48+16 mixed batch
  cosine noise floor           cos between two independent same-source draws

Everything is offline: models are rebuilt from resumable checkpoints
(``agents[0].online_model_state_dict`` etc.) and transitions come from the
``history_archive`` shards written by the so1 runs. For clear-arm checkpoints
the SAME archives are used as new/old data probes (identical schema), which
isolates the trained-weight contribution to gradient conflict.

Outputs: appends JSON rows to --out (one row per boundary x checkpoint).

Usage (colight env, LD_PRELOAD libstdc++ as in run_queue env):
  python dev/cl_hz1x1/grad_diag.py --out artifacts/cl_hz1x1/grad_diag.json
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from agent.dqn import DQNNet, DuelingDQNNet
from agent.frap import FRAP

ROOT = Path(__file__).resolve().parents[2]
NETDIR = {'dqn': 'sumo_dqn', 'frap': 'sumo_frap'}
NETNAME = {'s1': 'sumohz1x1', 'c4': 'sumohz1x1_config4',
           'c2': 'sumohz1x1_config2', 'c3': 'sumohz1x1_config3'}
ORDERS = {'o2': ['s1', 'c4', 'c2', 'c3'], 'o3': ['c2', 'c3', 's1', 'c4']}
PHASE_PAIRS = [[2, 6], [0, 4], [3, 7], [1, 5], [6, 7], [2, 3], [4, 5], [0, 1]]

GAMMA, LR, CLIP, BATCH, OFFLINE_N = 0.95, 1e-3, 5.0, 64, 16
EVAL_NEW, EVAL_OLD_PER_SCENE, DRAWS = 512, 256, 32


def run_dir(agent_dir, scene_key, prefix):
    return ROOT / 'data/output_data/tsc' / agent_dir / NETNAME[scene_key] / prefix


def load_records(archive_dir, max_shards=25):
    recs = []
    for shard in sorted(Path(archive_dir).glob('episode_*.pt'))[:max_shards]:
        recs.extend(torch.load(shard, map_location='cpu', weights_only=False))
    return recs


def build_batch(records, idx, arch):
    """Replicate agent._batchwise feature construction for archived records."""
    obs_t, obs_tp, rew, act, ph_t, ph_tp = [], [], [], [], [], []
    for i in idx:
        r = records[i]
        obs_t.append(r.state)
        obs_tp.append(r.next_state)
        rew.append(r.reward)
        act.append(r.action)
        ph_t.append(int(r.phase[0]) if hasattr(r.phase, '__len__') else int(r.phase))
        ph_tp.append(int(r.next_phase[0]) if hasattr(r.next_phase, '__len__')
                     else int(r.next_phase))
    obs_t = np.stack(obs_t)
    obs_tp = np.stack(obs_tp)
    if arch == 'frap':
        ft = np.concatenate([np.asarray(ph_t).reshape(-1, 1), obs_t], axis=1)
        ftp = np.concatenate([np.asarray(ph_tp).reshape(-1, 1), obs_tp], axis=1)
    else:
        pt = np.zeros((len(idx), 8), dtype=np.float32)
        ptp = np.zeros((len(idx), 8), dtype=np.float32)
        pt[range(len(idx)), ph_t] = 1.0
        ptp[range(len(idx)), ph_tp] = 1.0
        ft = np.concatenate([obs_t, pt], axis=1)
        ftp = np.concatenate([obs_tp, ptp], axis=1)
    return (torch.tensor(ft, dtype=torch.float32),
            torch.tensor(ftp, dtype=torch.float32),
            torch.tensor(rew, dtype=torch.float32),
            torch.tensor(act, dtype=torch.long))


def comp_mask(phase_pairs):
    m = []
    for i, a in enumerate(phase_pairs):
        z = np.zeros(len(phase_pairs) - 1, dtype=np.int64)
        c = 0
        for j, b in enumerate(phase_pairs):
            if i == j:
                continue
            if len(set(a + b)) == 3:
                z[c] = 1
            c += 1
        m.append(z)
    return torch.from_numpy(np.asarray(m))


class _AgentConf:
    param = {'demand_shape': 1, 'one_hot': False}


def build_model(sd):
    keys = set(sd)
    if 'lane_embedding.weight' in keys:
        m = FRAP(_AgentConf(), len(PHASE_PAIRS), PHASE_PAIRS,
                 comp_mask(PHASE_PAIRS))
        return 'frap', m
    sd0 = next(iter(sd.values()))
    in_dim = sd0.shape[1]
    out_dim = list(sd.values())[-1].shape[0]
    if 'feature.0.weight' in keys:
        return 'dueling', DuelingDQNNet(in_dim, out_dim)
    return 'mlp', DQNNet(in_dim, out_dim)


def flat_grad(loss, model):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()])


def td_loss_batch(model, target_model, batch, double_dqn, need_grad):
    b_t, b_tp, rewards, actions = batch
    if double_dqn:
        na = model(b_tp, train=False).argmax(dim=1, keepdim=True)
        out = target_model(b_tp, train=False).gather(1, na).squeeze(1)
        target = rewards + GAMMA * out
    else:
        out = target_model(b_tp, train=False)
        target = rewards + GAMMA * torch.max(out, dim=1)[0]
    target_f = model(b_t, train=False).clone()
    for i, a in enumerate(actions):
        target_f[i][a] = target[i]
    if need_grad:
        return nn.MSELoss()(model(b_t, train=True), target_f)
    with torch.no_grad():
        pred = model(b_t, train=True)
    return nn.MSELoss()(pred, target_f)


def td_abs_err(model, target_model, batch, double_dqn):
    b_t, b_tp, rewards, actions = batch
    with torch.no_grad():
        if double_dqn:
            na = model(b_tp, train=False).argmax(dim=1, keepdim=True)
            out = target_model(b_tp, train=False).gather(1, na).squeeze(1)
            target = rewards + GAMMA * out
        else:
            out = target_model(b_tp, train=False)
            target = rewards + GAMMA * torch.max(out, dim=1)[0]
        q = model(b_t, train=True).gather(1, actions.reshape(-1, 1)).squeeze(1)
    return (target - q).abs().mean().item()


def one_step(model, target_model, opt_state, opt_kind, batch, double_dqn):
    """Clone model+optimizer, take ONE real clipped step, return clone."""
    m2 = copy.deepcopy(model)
    t2 = copy.deepcopy(target_model)
    if opt_kind == 'adam':
        opt = torch.optim.Adam(m2.parameters(), lr=LR, eps=1e-7)
    else:
        opt = torch.optim.RMSprop(m2.parameters(), lr=LR, alpha=0.9,
                                  centered=False, eps=1e-7)
    if opt_state is not None:
        try:
            opt.load_state_dict(copy.deepcopy(opt_state))
        except Exception:
            pass
    loss = td_loss_batch(m2, t2, batch, double_dqn, need_grad=True)
    opt.zero_grad()
    loss.backward()
    clip_grad_norm_(m2.parameters(), CLIP)
    opt.step()
    return m2, t2


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def diagnose(ckpt_path, new_recs, old_recs_by_scene, label, draws=DRAWS):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ast = ck['agents'][0]
    arch, model = build_model(ast['online_model_state_dict'])
    model.load_state_dict(ast['online_model_state_dict'])
    _, target_model = build_model(ast['target_model_state_dict'])
    target_model.load_state_dict(ast['target_model_state_dict'])
    opt_state = ast.get('optimizer_state_dict')
    opt_kind = 'adam' if arch == 'frap' else 'rmsprop'
    double_dqn = arch != 'frap'

    # eval sets (fixed)
    rng = random.Random(0)
    ev_new = build_batch(new_recs, rng.sample(range(len(new_recs)),
                                            min(EVAL_NEW, len(new_recs))), arch)
    ev_old_scene = {}
    for sc, recs in old_recs_by_scene.items():
        ev_old_scene[sc] = build_batch(
            recs, rng.sample(range(len(recs)), min(EVAL_OLD_PER_SCENE,
                                                  len(recs))), arch)
    ev_old_all = build_batch(
        [r for recs in old_recs_by_scene.values() for r in recs],
        rng.sample(range(sum(len(v) for v in old_recs_by_scene.values())),
                   EVAL_OLD_PER_SCENE * len(old_recs_by_scene)), arch)

    rows = []
    for draw in range(draws):
        d_rng = random.Random(1000 + draw)
        ni = d_rng.sample(range(len(new_recs)), BATCH)
        per_old = BATCH // max(1, len(old_recs_by_scene))
        oi = {sc: d_rng.sample(range(len(r)), min(per_old, len(r)))
              for sc, r in old_recs_by_scene.items()}
        mix_oi = {sc: d_rng.sample(range(len(r)),
                                 min(OFFLINE_N // max(1, len(old_recs_by_scene)),
                                     len(r)))
                  for sc, r in old_recs_by_scene.items()}

        b_new = build_batch(new_recs, ni, arch)
        # old batch: scene-balanced pooled
        old_flat = []
        for sc, idxs in oi.items():
            old_flat += [(sc, i) for i in idxs]
        recs_old = [old_recs_by_scene[sc][i] for sc, i in old_flat]
        b_old = build_batch(recs_old, list(range(len(recs_old))), arch)
        recs_mix = ([new_recs[i] for i in d_rng.sample(range(len(new_recs)),
                                                     BATCH - OFFLINE_N)]
                    + [old_recs_by_scene[sc][i]
                       for sc, idxs in mix_oi.items() for i in idxs])
        b_mix = build_batch(recs_mix, list(range(len(recs_mix))), arch)

        g_new = flat_grad(td_loss_batch(model, target_model, b_new,
                                      double_dqn, True), model)
        g_old = flat_grad(td_loss_batch(model, target_model, b_old,
                                      double_dqn, True), model)
        g_mix = flat_grad(td_loss_batch(model, target_model, b_mix,
                                      double_dqn, True), model)
        # noise floor: second independent new draw
        ni2 = d_rng.sample(range(len(new_recs)), BATCH)
        g_new2 = flat_grad(td_loss_batch(
            model, target_model, build_batch(new_recs, ni2, arch),
            double_dqn, True), model)

        sign_conflict = float(((g_new * g_old) < 0).float().mean())
        w = g_old.abs()
        sign_conflict_w = float((w[(g_new * g_old) < 0].sum()
                                 / (w.sum() + 1e-12)))

        td_old0 = td_abs_err(model, target_model, ev_old_all, double_dqn)
        td_new0 = td_abs_err(model, target_model, ev_new, double_dqn)

        m_a, _ = one_step(model, target_model, opt_state, opt_kind,
                          b_new, double_dqn)
        d_old_new = td_abs_err(m_a, target_model, ev_old_all,
                               double_dqn) - td_old0
        d_new_new = td_abs_err(m_a, target_model, ev_new,
                               double_dqn) - td_new0
        m_b, _ = one_step(model, target_model, opt_state, opt_kind,
                          b_mix, double_dqn)
        d_old_mix = td_abs_err(m_b, target_model, ev_old_all,
                               double_dqn) - td_old0
        d_new_mix = td_abs_err(m_b, target_model, ev_new,
                               double_dqn) - td_new0

        rows.append(dict(
            cos_new_old=cosine(g_new, g_old),
            cos_new_mix=cosine(g_new, g_mix),
            cos_old_mix=cosine(g_old, g_mix),
            cos_new_new=cosine(g_new, g_new2),
            sign_conflict=sign_conflict,
            sign_conflict_w=sign_conflict_w,
            norm_ratio=float(g_old.norm() / (g_new.norm() + 1e-12)),
            td_old=td_old0, td_new=td_new0,
            dTD_old_newstep=d_old_new, dTD_new_newstep=d_new_new,
            dTD_old_mixstep=d_old_mix, dTD_new_mixstep=d_new_mix,
        ))

    agg = {'label': label, 'arch': arch, 'draws': draws,
           'ckpt': str(ckpt_path.relative_to(ROOT)),
           'metrics': {}}
    for k in rows[0]:
        v = [r[k] for r in rows]
        agg['metrics'][k] = {'mean': float(np.mean(v)),
                             'std': float(np.std(v))}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='artifacts/cl_hz1x1/grad_diag.json')
    ap.add_argument('--draws', type=int, default=DRAWS)
    args = ap.parse_args()

    jobs = []
    # (label, agent_dir, arm_prefix, order, seed, enter_stage)
    # so arm counterpart uses its own ckpt; clear arm uses cl1 ckpt with the
    # SAME so archives as data probes.
    for (tag, agent_dir, arm, order, seed, st) in [
            ('so', 'sumo_dqn', 'soddqn', 'o2', 0, 2),
            ('so', 'sumo_dqn', 'soddqn', 'o2', 0, 3),
            ('so', 'sumo_dqn', 'soddqn', 'o2', 0, 4),
            ('so', 'sumo_dqn', 'soddqn', 'o3', 0, 2),
            ('so', 'sumo_dqn', 'soduel', 'o2', 1, 4),
            ('so', 'sumo_frap', 'sofrap', 'o2', 0, 4),
            ('clear', 'sumo_dqn', 'ddqn_clear', 'o2', 0, 2),
            ('clear', 'sumo_dqn', 'ddqn_clear', 'o2', 0, 4),
            ('clear', 'sumo_dqn', 'ddqn_clear', 'o3', 0, 2),
            ('clear', 'sumo_dqn', 'duel_clear', 'o2', 1, 4),
            ('clear', 'sumo_frap', 'frap_clear', 'o2', 0, 4),
    ]:
        scenes = ORDERS[order]
        new_scene = scenes[st - 1]
        old_scenes = scenes[:st - 1]
        batch_pref = 'so1' if tag == 'so' else 'cl1'
        prev_prefix = f'{batch_pref}_{arm}_{order}_sd{seed}_st{st - 1}'
        so_arm = {'ddqn_clear': 'soddqn', 'duel_clear': 'soduel',
                  'frap_clear': 'sofrap'}.get(arm, arm)
        cur_prefix = f'so1_{so_arm}_{order}_sd{seed}'
        ckpt = (run_dir(agent_dir if agent_dir != 'sumo_frap' else 'sumo_frap',
                        scenes[st - 2], prev_prefix)
                / 'checkpoints/resumable/episode_0100.pt')
        new_arch = run_dir(agent_dir, new_scene,
                           f'{cur_prefix}_st{st}') / 'history_archive'
        old_archs = {sc: run_dir(agent_dir, sc,
                                 f'{cur_prefix}_st{i + 1}') / 'history_archive'
                     for i, sc in enumerate(old_scenes)}
        jobs.append(dict(label=f'{tag}:{arm}_{order}_sd{seed}_st{st}',
                         ckpt=ckpt, new=new_arch, old=old_archs))

    results = []
    for j in jobs:
        for p in [j['ckpt'], j['new'], *j['old'].values()]:
            assert Path(p).exists(), f'missing {p}'
        new_recs = load_records(j['new'])
        old_recs = {sc: load_records(p) for sc, p in j['old'].items()}
        print(f"== {j['label']}  new={len(new_recs)} "
              f"old={ {k: len(v) for k, v in old_recs.items()} }", flush=True)
        res = diagnose(j['ckpt'], new_recs, old_recs, j['label'],
                       draws=args.draws)
        results.append(res)
        for k, v in res['metrics'].items():
            print(f"   {k:22s} {v['mean']:+.4f} ± {v['std']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print('written', args.out)


if __name__ == '__main__':
    main()
