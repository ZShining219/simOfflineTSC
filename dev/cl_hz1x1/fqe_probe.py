#!/usr/bin/env python3
"""Offline policy-value estimation feasibility probe (frozen, inference-only).

Question: can FQE fitted on *independent historical trajectories* rank six
frozen candidate policies (s1 baseline; Clear/SO stage-4 ep{0,50,100}) well
enough to separate Clear's collapse from SO's retention on s1?

Six policies per training seed (three seeds):
    base        shared st1 ep100 model (identical across arms - verified)
    clear_ep*   cl1_ddqn_clear o2 st4 resumable episode_{0,50,100}
    so_ep*      so1_soddqn o2 st4 resumable episode_{0,50,100}

Ground truth per policy: closed-loop travel_time + reward_sum from the
frozen eval cells (existing ev* + drift_* runs, eval seeds 101-103).

FQE: Q_{k+1} <- argmin E_D[(r + gamma*Q_k(s',pi(s')) - Q(s,a))^2],
semi-gradient on a fixed dataset of s1 transitions from the so-arm
history_archive (st1 shards; last record of each shard marked terminal).
Dataset coverage variants probe the coverage-error relation:
    all / early(ep1-50) / late(ep51-100) / n4k / explor(off-greedy only)

Outputs JSONL rows + printed per-seed ranking tables and Spearman stats.

Usage (colight env + LD_PRELOAD):
  python dev/cl_hz1x1/fqe_probe.py --out artifacts/cl_hz1x1/fqe_s1.json
  python dev/cl_hz1x1/fqe_probe.py --mc --out artifacts/cl_hz1x1/mc_s1.json
      # --mc: closed-loop rollouts on eval seeds 101-103 collecting
      # per-decision rewards -> true gamma-discounted Monte-Carlo return
      # J_gamma = sum_t gamma^t r_t; also validates reward_sum vs official.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.stats import spearmanr

GAMMA = 0.95
FQE_ITERS, FQE_EPOCHS, FQE_LR = 60, 4, 1e-3
EVAL_SEEDS = [101, 102, 103]
OUT = ROOT / 'data' / 'output_data' / 'tsc'


def build_model(sd):
    from agent.dqn import DQNNet
    sd0 = next(iter(sd.values()))
    return DQNNet(sd0.shape[1], list(sd.values())[-1].shape[0])


def load_online(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = build_model(ck['agents'][0]['online_model_state_dict'])
    m.load_state_dict(ck['agents'][0]['online_model_state_dict'])
    m.eval()
    return m


def features(obs, phase):
    n = len(obs)
    ph = np.zeros((n, 8), dtype=np.float32)
    ph[range(n), np.asarray(phase, dtype=int)] = 1.0
    return np.concatenate([np.asarray(obs, dtype=np.float32), ph], axis=1)


def policies(sd):
    base = (OUT / 'sumo_dqn/sumohz1x1'
            / f'cl1_ddqn_clear_o2_sd{sd}_st1/checkpoints/resumable'
            / 'episode_0100.pt')
    pol = {'base': base}
    for tag, ep in [('ep0', 0), ('ep50', 50), ('ep100', 100)]:
        pol[f'clear_{tag}'] = (
            OUT / 'sumo_dqn/sumohz1x1_config3'
            / f'cl1_ddqn_clear_o2_sd{sd}_st4/checkpoints/resumable'
            / f'episode_{ep:04d}.pt')
        pol[f'so_{tag}'] = (
            OUT / 'sumo_dqn/sumohz1x1_config3'
            / f'so1_soddqn_o2_sd{sd}_st4/checkpoints/resumable'
            / f'episode_{ep:04d}.pt')
    return {k: load_online(v) for k, v in pol.items()}


def true_returns(sd):
    """Closed-loop reward_sum & TT per policy (eval seeds 101-103)."""
    cells = {}
    cells['base'] = [OUT / 'sumo_dqn/sumohz1x1'
                     / f'cl1_ddqn_clear_o2_sd{sd}_ev1_s1_es{es}'
                     for es in EVAL_SEEDS]
    for tag, ep in [('ep0', 0), ('ep50', 50)]:
        cells[f'clear_{tag}'] = [
            OUT / 'sumo_dqn/sumohz1x1'
            / f'drift_s1_ddqn_clear_o2_sd{sd}_st4ep{ep:03d}_es{es}'
            for es in EVAL_SEEDS]
        cells[f'so_{tag}'] = [
            OUT / 'sumo_dqn/sumohz1x1'
            / f'drift_s1_soddqn_o2_sd{sd}_st4ep{ep:03d}_es{es}'
            for es in EVAL_SEEDS]
    cells['clear_ep100'] = [OUT / 'sumo_dqn/sumohz1x1'
                            / f'cl1_ddqn_clear_o2_sd{sd}_ev4_s1_es{es}'
                            for es in EVAL_SEEDS]
    cells['so_ep100'] = [OUT / 'sumo_dqn/sumohz1x1'
                         / f'so1_soddqn_o2_sd{sd}_ev4_s1_es{es}'
                         for es in EVAL_SEEDS]
    out = {}
    for name, dirs in cells.items():
        rws, tts = [], []
        for d in dirs:
            rl = d / 'metrics' / 'records.jsonl'
            rec = [json.loads(l) for l in open(rl)
                   if 'FINAL_EVALUATION' in l]
            rws.append(rec[-1]['reward_sum'])
            tts.append(rec[-1]['travel_time'])
        out[name] = {'reward_sum': rws, 'tt': tts,
                     'reward_mean': float(np.mean(rws)),
                     'tt_mean': float(np.mean(tts))}
    return out


def load_transitions(sd, variant, base_model):
    """s1 transitions from so-arm st1 archive, coverage variants."""
    d = (OUT / 'sumo_dqn/sumohz1x1'
         / f'so1_soddqn_o2_sd{sd}_st1/history_archive')
    shards = sorted(Path(d).glob('episode_*.pt'))
    if variant == 'early':
        shards = shards[:50]
    elif variant == 'late':
        shards = shards[50:]
    obs, ph, act, rew, nobs, nph, done = [], [], [], [], [], [], []
    for sh in shards:
        recs = torch.load(sh, map_location='cpu', weights_only=False)
        for i, r in enumerate(recs):
            obs.append(np.asarray(r.state, dtype=np.float32))
            ph.append(int(np.asarray(r.phase).reshape(-1)[0]))
            act.append(int(r.action))
            rew.append(float(r.reward))
            nobs.append(np.asarray(r.next_state, dtype=np.float32))
            nph.append(int(np.asarray(r.next_phase).reshape(-1)[0]))
            done.append(bool(r.terminated) or bool(r.truncated))
        done[-1] = True  # shard end == episode horizon
    obs, ph, act, rew = map(np.asarray, (obs, ph, act, rew))
    nobs, nph, done = np.asarray(nobs), np.asarray(nph), np.asarray(done)
    if variant == 'explor':
        keep = act != greedy_np(base_model, features(obs, ph))
        obs, ph, act, rew = obs[keep], ph[keep], act[keep], rew[keep]
        nobs, nph, done = nobs[keep], nph[keep], done[keep]
    elif variant == 'n4k':
        sel = np.random.RandomState(0).choice(
            len(obs), min(4000, len(obs)), replace=False)
        obs, ph, act, rew = obs[sel], ph[sel], act[sel], rew[sel]
        nobs, nph, done = nobs[sel], nph[sel], done[sel]
    return {'s': features(obs, ph), 'a': act.astype(np.int64),
            'r': rew.astype(np.float32), 's2': features(nobs, nph),
            'd': done.astype(np.float32), 'n': len(obs)}


def greedy_np(model, feat):
    with torch.no_grad():
        return model(torch.tensor(feat, dtype=torch.float32),
                     train=False).argmax(1).numpy()


def fqe(ds, pi_model, rng_seed=0):
    """Semi-gradient FQE; returns V_hat = mean_s Q(s, pi(s)) on dataset."""
    from agent.dqn import DQNNet
    torch.manual_seed(rng_seed)
    q = DQNNet(ds['s'].shape[1], 8)
    opt = torch.optim.Adam(q.parameters(), lr=FQE_LR)
    s = torch.tensor(ds['s']); a = torch.tensor(ds['a']).reshape(-1, 1)
    r = torch.tensor(ds['r']); s2 = torch.tensor(ds['s2'])
    d = torch.tensor(ds['d'])
    with torch.no_grad():
        na = pi_model(s2, train=False).argmax(1, keepdim=True)
    for _ in range(FQE_ITERS):
        with torch.no_grad():
            tgt = r + GAMMA * (1 - d) * q(s2).gather(1, na).squeeze(1)
        for _ in range(FQE_EPOCHS):
            loss = torch.nn.functional.mse_loss(
                q(s).gather(1, a).squeeze(1), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        v = q(s).gather(1, pi_model(s, train=False)
                        .argmax(1, keepdim=True)).mean().item()
    return v


def mc_rollout(sd, eval_seed, pm):
    """Single closed-loop episode: per-decision rewards -> J_gamma/rsum.

    Mirrors the trainer eval loop: action=argmax(test=True), per-decision
    reward = mean over action_interval sim-step rewards.  Runs ONE policy
    per call; the caller spawns a fresh subprocess per rollout because
    repeated libsumo start/close in one process is unstable.
    """
    sys.path.insert(0, str(ROOT / 'dev' / 'cl_hz1x1'))
    import drift_diag as D
    world, agents, env, config = D.build_stack(
        'sumohz1x1', f'fqemc_sd{sd}_es{eval_seed}', eval_seed)
    interval = config['trainer']['action_interval']
    test_steps = config['trainer']['test_steps']
    agents[0].model.load_state_dict(pm.state_dict())
    ag = agents[0]
    rs = []
    world.set_episode_seed(eval_seed)
    obs = env.reset()
    ag.reset()
    steps = 0
    dones = [False]
    while steps < test_steps and not all(dones):
        phases = np.stack([a.get_phase() for a in agents])
        acts = np.stack([
            a.get_action(obs[i], phases[i], test=True)
            for i, a in enumerate(agents)])
        rlist = []
        for _ in range(min(interval, test_steps - steps)):
            obs, rews, dones, _ = env.step(acts.flatten())
            steps += 1
            rlist.append(np.asarray(rews).reshape(-1)[0])
            if all(dones):
                break
        rs.append(float(np.mean(rlist)))
    gam = np.power(GAMMA, np.arange(len(rs)))
    return {'r': rs, 'j_gamma': float(np.dot(gam, rs)),
            'reward_sum': float(np.sum(rs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='artifacts/cl_hz1x1/fqe_s1.json')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--variants', default='all,early,late,n4k,explor')
    ap.add_argument('--iters', type=int, default=FQE_ITERS)
    ap.add_argument('--mc', action='store_true',
                    help='one closed-loop rollout; needs --policy/--es/--sd')
    ap.add_argument('--policy', default=None)
    ap.add_argument('--es', type=int, default=None)
    ap.add_argument('--sd', type=int, default=None)
    args = ap.parse_args()

    if args.mc:
        # single-rollout mode: print one JSON line for the driver loop
        pol = policies(args.sd)[args.policy]
        v = mc_rollout(args.sd, args.es, pol)
        print('MCROW ' + json.dumps(
            {'seed': args.sd, 'eval_seed': args.es, 'policy': args.policy,
             'j_gamma': v['j_gamma'], 'reward_sum': v['reward_sum'],
             'n_steps': len(v['r'])}))
        return

    rows = []
    for sd in [int(x) for x in args.seeds.split(',')]:
        pol = policies(sd)
        truth = true_returns(sd)
        base_m = list(pol.values())[0]
        for var in args.variants.split(','):
            ds = load_transitions(sd, var, base_m)
            for name, pm in pol.items():
                v = fqe(ds, pm, rng_seed=sd)
                rows.append({'seed': sd, 'variant': var, 'policy': name,
                             'v_hat': v, 'n': ds['n'],
                             'true_reward': truth[name]['reward_mean'],
                             'true_tt': truth[name]['tt_mean']})
                print(f"sd{sd} {var:6s} {name:12s} V={v:9.2f} "
                      f"true_r={truth[name]['reward_mean']:9.1f} "
                      f"tt={truth[name]['tt_mean']:6.1f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))

    # ---- ranking diagnostics ----
    print('\n== Spearman(V_hat, true_reward) & separation per seed/coverage ==')
    good = ['base', 'clear_ep0', 'so_ep0', 'so_ep50', 'so_ep100']
    bad = ['clear_ep50', 'clear_ep100']
    for sd in [int(x) for x in args.seeds.split(',')]:
        for var in args.variants.split(','):
            sub = [r for r in rows if r['seed'] == sd
                   and r['variant'] == var]
            rho = spearmanr([r['v_hat'] for r in sub],
                            [r['true_reward'] for r in sub]).statistic
            gmin = min(r['v_hat'] for r in sub if r['policy'] in good)
            bmax = max(r['v_hat'] for r in sub if r['policy'] in bad)
            ok = 'SEP' if bmax < gmin else 'MIX'
            print(f"sd{sd} {var:7s} n={sub[0]['n']:5d} rho={rho:+.3f} "
                  f"bad_max={bmax:9.2f} good_min={gmin:9.2f} -> {ok}")
    print('\nwritten', args.out, len(rows), 'rows')


if __name__ == '__main__':
    main()
