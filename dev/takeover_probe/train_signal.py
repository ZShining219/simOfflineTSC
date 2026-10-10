"""Phase-level Double-DQN trainer for the scene-guided signal agent.

Single-layer counterpart of ``train_router``: episodes are full S2 runs
with a sampled congestion (surge) injection schedule; each ``interval``
-second decision tick yields one (s, a, r, s') transition whose gradient
flows through the scene encoder — the condensed representation is shaped
by control value directly, with no controller-selection layer in between.

Auxiliary loss (binding alignment): while an event report is active the
cross-attention lane weights are encouraged toward the report's bound
lanes — same as the router trainer, skipped for state-only arms.

Usage:
    python -m dev.takeover_probe.train_signal --variant sga \
        --episodes 100 --out output_data/dev_takeover_probe/sig_sga
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dev.takeover_probe.probe import (  # noqa: E402
    build_controllers, build_world, _bind_all,
)
from dev.takeover_probe.dispatch_state import FeatureBuilder  # noqa: E402
from dev.takeover_probe.signal_model import (  # noqa: E402
    STATE_ONLY, SceneResidualPolicy, SceneSignalAgent,
)
from dev.takeover_probe.signal_runtime import (  # noqa: E402
    run_signal_episode,
)
from dev.takeover_probe.train_router import (  # noqa: E402
    EDGE_TARGETS, TRAIN_DURS, TRAIN_EDGES, TRAIN_ONSETS, TRAIN_PERIODS,
)
from tools.sumo_live_broadcast.injects import parse_injects  # noqa: E402


def sample_injects(rng, allow_none_p=0.25):
    """Single congestion event on a random injectable edge.

    Reports are pure-phenomenon ('道路拥堵') — no objective flag is ever
    set (the broadcast names a phenomenon, not a governance directive).
    """
    if rng.random() < allow_none_p:
        return []
    edge = TRAIN_EDGES[int(rng.integers(len(TRAIN_EDGES)))]
    onset = int(TRAIN_ONSETS[rng.integers(len(TRAIN_ONSETS))])
    dur = int(TRAIN_DURS[rng.integers(len(TRAIN_DURS))])
    period = int(TRAIN_PERIODS[rng.integers(len(TRAIN_PERIODS))])
    return parse_injects(
        [f'surge:{edge}:{EDGE_TARGETS[edge]}:{onset}:{dur}:{period}'])


def train(args):
    import torch

    rng = np.random.default_rng(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    world = build_world(args.scene, args.traffic_seed_base, Path(
        args.runtime_dir or '/tmp/takeover_signal'), out / 'logger',
        {'t_fixed': args.t_fixed, 't_min': args.t_min})
    ranks = list(range(len(world.intersection_ids)))
    n_phases = len(world.intersections[0].phases)

    fb0 = FeatureBuilder(
        world, [], report_delay=args.report_delay, layout='identity',
        phys=args.phys, include_forecast=not args.no_forecast,
        phase_age=True)
    state_only = args.variant in STATE_ONLY
    residual_mode = args.residual_dqn is not None
    if residual_mode and args.variant not in ('sga', 'attn'):
        raise ValueError(
            f'--residual-dqn needs a binding variant (sga/attn), '
            f'got {args.variant!r}')
    if residual_mode:
        # guided补课 form: frozen base DQN supplies q_base per tick;
        # the scene module only learns a bounded gated correction.
        # BC is pointless here — the base already drives.
        agent = SceneResidualPolicy(
            lane_dim=fb0.lane_tokens(0.0).shape[1],
            text_dim=fb0.text_dim, global_dim=fb0.global_dim,
            n_phases=n_phases, d_model=args.d_model,
            variant=args.variant, residual_pool=args.residual_pool,
            geom=fb0.model_geom(), n_edges=fb0.n_edges,
            edge_idx=fb0.lane_edge_idx,
            dose_frac=args.res_dose, bind_tau=args.res_bind_tau)
        target = SceneResidualPolicy(
            lane_dim=fb0.lane_tokens(0.0).shape[1],
            text_dim=fb0.text_dim, global_dim=fb0.global_dim,
            n_phases=n_phases, d_model=args.d_model,
            variant=args.variant, residual_pool=args.residual_pool,
            geom=fb0.model_geom(), n_edges=fb0.n_edges,
            edge_idx=fb0.lane_edge_idx,
            dose_frac=args.res_dose, bind_tau=args.res_bind_tau)
    else:
        agent = SceneSignalAgent(
            lane_dim=fb0.lane_tokens(0.0).shape[1],
            text_dim=fb0.text_dim, global_dim=fb0.global_dim,
            n_phases=n_phases, d_model=args.d_model,
            variant=args.variant, residual_pool=args.residual_pool,
            geom=fb0.model_geom() if not state_only else None,
            n_lanes=fb0.n_lanes)
        target = SceneSignalAgent(
            lane_dim=fb0.lane_tokens(0.0).shape[1],
            text_dim=fb0.text_dim, global_dim=fb0.global_dim,
            n_phases=n_phases, d_model=args.d_model,
            variant=args.variant, residual_pool=args.residual_pool,
            geom=fb0.model_geom() if not state_only else None,
            n_lanes=fb0.n_lanes)
    target.load_state_dict(agent.state_dict())
    target.eval()
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    replay = deque(maxlen=args.buffer)

    meta = dict(
        lane_dim=fb0.lane_tokens(0.0).shape[1], text_dim=fb0.text_dim,
        global_dim=fb0.global_dim, n_lanes=fb0.n_lanes,
        d_model=args.d_model, residual_pool=args.residual_pool,
        **fb0.feature_meta())
    if residual_mode:
        meta.update(kind='residual',
                    residual_dqn=str(Path(args.residual_dqn).name),
                    res_dose=args.res_dose,
                    res_bind_tau=args.res_bind_tau)

    # ---- phase 0: teacher demonstrations + BC warm-start ---------------
    if args.teacher_eps > 0 and not residual_mode:
        demos = []
        for tep in range(args.teacher_eps):
            injects = sample_injects(rng, allow_none_p=args.allow_none_p)
            world.close()
            world = build_world(
                args.scene, args.traffic_seed_base - 1 - tep,
                Path(args.runtime_dir or '/tmp/takeover_signal'),
                out / 'logger',
                {'t_fixed': args.t_fixed, 't_min': args.t_min,
                 'min_green_vehicle': 3, 'max_red_vehicle': 6})
            ctrls = build_controllers()
            _bind_all(world, ctrls, ranks, needed={args.teacher})
            fb_t = FeatureBuilder(
                world, injects, report_delay=args.report_delay,
                layout='identity', phys=args.phys,
                include_forecast=not args.no_forecast, phase_age=True)
            transitions, _ = run_signal_episode(
                world, fb_t, ranks, injects=injects,
                controller=ctrls[args.teacher], interval=args.interval,
                episode_seconds=args.episode_seconds, record_snaps=True,
                switch_penalty=args.switch_penalty)
            replay.extend(transitions)
            demos += [(t['s'], t['a']) for t in transitions]
        bc_loss = bc_updates(agent, opt, demos, args, rng)
        target.load_state_dict(agent.state_dict())
        print(json.dumps({'teacher': args.teacher,
                          'demo_pairs': len(demos),
                          'bc_loss': round(bc_loss, 4)}), flush=True)
        if args.episodes <= 0:
            agent.save(out / 'agent.pt', meta={**meta, 'episode': 0,
                                               'bc_only': True})
    if residual_mode and args.episodes <= 0:
        agent.save(out / 'agent.pt',
                   meta={**meta, 'episode': 0, 'untrained': True})

    log = []
    for ep in range(1, args.episodes + 1):
        injects = sample_injects(rng, allow_none_p=args.allow_none_p)
        # fresh SUMO world per episode -> independent traffic seed
        world.close()
        world = build_world(args.scene, args.traffic_seed_base + ep,
                            Path(args.runtime_dir or '/tmp/takeover_signal'),
                            out / 'logger',
                            {'t_fixed': args.t_fixed, 't_min': args.t_min,
                             'min_green_vehicle': 3,
                             'max_red_vehicle': 6})
        ranks = list(range(len(world.intersection_ids)))
        fb = FeatureBuilder(
            world, injects, report_delay=args.report_delay,
            layout='identity', phys=args.phys,
            include_forecast=not args.no_forecast, phase_age=True)
        q_base_fn = None
        if residual_mode:
            from dev.takeover_probe.scene_mp import bound_dqn_agent
            from tools.sumo_live_broadcast.controllers import (
                inference_q_values,
            )
            dqn = bound_dqn_agent(world, args.residual_dqn,
                                  rank=ranks[0])
            q_base_fn = lambda dqn=dqn: inference_q_values(dqn)
        eps = max(args.eps_final,
                  args.eps_start - (args.eps_start - args.eps_final)
                  * ep / max(1, args.eps_decay_episodes))
        transitions, records = run_signal_episode(
            world, fb, ranks, injects=injects, agent=agent,
            interval=args.interval, episode_seconds=args.episode_seconds,
            epsilon=eps, rng=rng, switch_penalty=args.switch_penalty,
            q_base_fn=q_base_fn)
        replay.extend(transitions)
        if residual_mode:
            stats = update_residual(agent, target, opt, replay, args, rng)
        else:
            stats = update(agent, target, opt, replay, args, rng)
        if ep % args.target_sync == 0:
            target.load_state_dict(agent.state_dict())
        ret = float(sum(t['r'] for t in transitions))
        mean_q = float(np.mean([r['total_queue'] for r in records])) \
            if records else 0.0
        ev_edge = (injects[0]['edge_from'] if injects else '')
        ev_lanes = set(fb.edge_to_laneidx.get(ev_edge, []))
        ev_q = float(sum(
            sum(rec['lane_queue'].get(fb.lanes[i], 0) for i in ev_lanes)
            for rec in records)) if ev_lanes else 0.0
        row = {
            'ep': ep, 'eps': round(eps, 3), 'return': round(ret, 3),
            'mean_queue': round(mean_q, 2), 'ev_queue_int': round(ev_q, 1),
            'event': (f"{injects[0]['kind']}:{ev_edge}" if injects
                      else 'none'),
            **stats,
        }
        log.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if ep % args.ckpt_every == 0 or ep == args.episodes:
            agent.save(out / 'agent.pt', meta={**meta, 'episode': ep})
    (out / 'train_log.json').write_text(
        json.dumps(log, ensure_ascii=False, indent=1))


def bc_updates(agent, opt, demos, args, rng):
    """Warm-start: imitate the teacher's phase choices (CE over Q logits)."""
    import torch
    from dev.takeover_probe.dispatch_model import collate
    if not demos:
        return 0.0
    losses = []
    for _ in range(args.bc_updates):
        idx = rng.integers(0, len(demos), size=args.batch)
        s = collate([demos[i][0] for i in idx])
        a = torch.as_tensor([demos[i][1] for i in idx])
        q, _ = agent.q_values(s)
        loss = torch.nn.functional.cross_entropy(q, a)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(agent.parameters(),
                                           args.grad_clip)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return float(np.mean(losses))


def update_residual(policy, target, opt, replay, args, rng):
    """Double-DQN TD on the residual policy's biased Q surface.

    q_base enters as stored per-tick values ('qb'/'qbp') — the frozen
    base stays outside the graph; gradients reach only the encoder and
    the zero-init delta head.  ``res_reg`` optionally penalises the
    normalised intervention magnitude (|Q-Q_base|/cap), a sparsity
    prior for "engage only where it pays".
    """
    import torch
    from dev.takeover_probe.dispatch_model import collate
    if len(replay) < args.batch * 2:
        return {'loss': 0.0, 'aux': 0.0, 'gate': 0.0}
    losses, auxs, gates = [], [], []
    idxs = rng.integers(0, len(replay),
                        size=(args.updates_per_ep, args.batch))
    rlist = list(replay)
    for idx_row in idxs:
        batch_tr = [rlist[i] for i in idx_row]
        s = collate([t['s'] for t in batch_tr])
        sp = collate([t['sp'] for t in batch_tr])
        qb = torch.as_tensor(
            np.stack([t['qb'] for t in batch_tr]), dtype=torch.float32)
        qbp = torch.as_tensor(
            np.stack([t['qbp'] for t in batch_tr]), dtype=torch.float32)
        a = torch.as_tensor([t['a'] for t in batch_tr])
        r = torch.as_tensor([t['r'] for t in batch_tr],
                            dtype=torch.float32)
        done = torch.as_tensor([t['done'] for t in batch_tr],
                               dtype=torch.float32)
        q, w = policy(s, qb)
        with torch.no_grad():
            a_star = policy(sp, qbp)[0].argmax(dim=-1)
            q_next = target(sp, qbp)[0]
            td = r + args.gamma * (1 - done) * q_next.gather(
                1, a_star.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(
            q.gather(1, a.unsqueeze(1)).squeeze(1), td)
        aux = torch.zeros(())
        terms = []
        for i, t in enumerate(batch_tr):
            tgt = t['s'].get('event_lane_idx') or []
            if tgt:
                p = w[i, torch.as_tensor(tgt)].sum()
                terms.append(-torch.log(p.clamp_min(1e-6)))
        if terms:
            aux = torch.stack(terms).mean() * args.aux_weight
        reg = torch.zeros(())
        if args.res_reg > 0:
            spread = (qb.max(dim=-1).values
                      - qb.min(dim=-1).values).clamp_min(1e-6)
            cap = args.res_dose * spread.unsqueeze(-1)
            reg = args.res_reg * ((q - qb) / cap).pow(2).mean()
        (loss + aux + reg).backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           args.grad_clip)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        auxs.append(float(aux))
        gates.append(float(policy.bind_gate(w).mean()))
    return {'loss': round(float(np.mean(losses)), 4),
            'aux': round(float(np.mean(auxs)), 4),
            'gate': round(float(np.mean(gates)), 4)}


def update(agent, target, opt, replay, args, rng):
    import torch
    from dev.takeover_probe.dispatch_model import collate
    if len(replay) < args.batch * 2:
        return {'loss': 0.0, 'aux': 0.0}
    losses, auxs = [], []
    idxs = rng.integers(0, len(replay),
                        size=(args.updates_per_ep, args.batch))
    rlist = list(replay)
    for idx_row in idxs:
        batch_tr = [rlist[i] for i in idx_row]
        s = collate([t['s'] for t in batch_tr])
        sp = collate([t['sp'] for t in batch_tr])
        a = torch.as_tensor([t['a'] for t in batch_tr])
        r = torch.as_tensor([t['r'] for t in batch_tr],
                            dtype=torch.float32)
        done = torch.as_tensor([t['done'] for t in batch_tr],
                               dtype=torch.float32)
        q, w = agent.q_values(s)
        with torch.no_grad():
            a_star = agent.q_values(sp)[0].argmax(dim=-1)
            q_next = target.q_values(sp)[0]
            td = r + args.gamma * (1 - done) * q_next.gather(
                1, a_star.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(
            q.gather(1, a.unsqueeze(1)).squeeze(1), td)
        aux = torch.zeros(())
        if agent.variant in ('attn', 'sga'):
            terms = []
            for i, t in enumerate(batch_tr):
                tgt = t['s'].get('event_lane_idx') or []
                if tgt:
                    p = w[i, torch.as_tensor(tgt)].sum()
                    terms.append(-torch.log(p.clamp_min(1e-6)))
            if terms:
                aux = torch.stack(terms).mean() * args.aux_weight
        (loss + aux).backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(agent.parameters(),
                                           args.grad_clip)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        auxs.append(float(aux))
    return {'loss': round(float(np.mean(losses)), 4),
            'aux': round(float(np.mean(auxs)), 4)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--variant', default='sga',
                   choices=['attn', 'sga', 'selfattn', 'concat',
                            'selfattn_st', 'concat_st', 'mlp_st'])
    p.add_argument('--phys', choices=['full', 'queue'], default='full')
    p.add_argument('--no-forecast', action='store_true',
                   help='zero the est_remaining text slot (report = '
                        'phenomenon only, duration is a prediction target)')
    p.add_argument('--report-delay', type=float, default=0.0,
                   help='seconds between event onset and the broadcast '
                        'reaching the model (0 = immediate report)')
    p.add_argument('--residual-pool', choices=['mean', 'mms'],
                   default='mean')
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--episode-seconds', type=int, default=3600)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--allow-none-p', type=float, default=0.25)
    p.add_argument('--residual-dqn', default=None,
                   help='PATH to frozen DQN weights — train a '
                        'SceneResidualPolicy (bounded gated ΔQ on the '
                        'live Q surface) instead of a from-scratch '
                        'agent; variant must be sga or attn')
    p.add_argument('--res-dose', type=float, default=0.35,
                   help='residual tilt budget as a fraction of the live '
                        'Q spread (same cap as the adapter)')
    p.add_argument('--res-bind-tau', type=float, default=0.4,
                   help='edge bind mass below this -> gate exactly 0 -> '
                        'policy ≡ frozen base')
    p.add_argument('--res-reg', type=float, default=0.0,
                   help='weight of the normalised-intervention penalty '
                        '(engage-only-where-it-pays sparsity)')
    p.add_argument('--teacher', default='maxpressure',
                   help='teacher controller for demonstration episodes '
                        '(0 episodes = BC off)')
    p.add_argument('--teacher-eps', type=int, default=0)
    p.add_argument('--bc-updates', type=int, default=500)
    p.add_argument('--switch-penalty', type=float, default=0.0,
                   help='reward cost per commanded phase change — the '
                        'yellow/all-red transition is real capacity loss; '
                        'without it greedy policies collapse to toggling')
    p.add_argument('--eps-start', type=float, default=1.0)
    p.add_argument('--eps-final', type=float, default=0.05)
    p.add_argument('--eps-decay-episodes', type=int, default=60)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--buffer', type=int, default=50000)
    p.add_argument('--updates-per-ep', type=int, default=60)
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--grad-clip', type=float, default=10.0)
    p.add_argument('--aux-weight', type=float, default=0.05)
    p.add_argument('--target-sync', type=int, default=5)
    p.add_argument('--ckpt-every', type=int, default=25)
    p.add_argument('--rng-seed', type=int, default=0)
    p.add_argument('--traffic-seed-base', type=int, default=100)
    p.add_argument('--t-fixed', type=int, default=30)
    p.add_argument('--t-min', type=int, default=10)
    p.add_argument('--runtime-dir', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
