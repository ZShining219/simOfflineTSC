"""P1 pilot trainer: SMDP Double-DQN over controllers for event dispatch.

Episodes = full S2 episodes with a sampled surge-injection schedule (or
none).  Each 120 s window yields one transition; the router's TD gradient
flows through the scene-guided attention so the scene representation is
shaped by control value, not by reconstruction.

Auxiliary loss (attention alignment): while an event report is active the
cross-attention lane weights are encouraged toward the report's bound
lanes — small weight, skipped for the concat baseline (no attention).

Usage:
    python -m dev.takeover_probe._router.train_router --variant attn \
        --episodes 40 --out output_data/dev_takeover_probe/router_attn
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dev.takeover_probe.probe import (  # noqa: E402
    build_controllers, build_world, _bind_all,
)
from dev.takeover_probe.dispatch_state import FeatureBuilder  # noqa: E402
from dev.takeover_probe._router.router_runtime import (  # noqa: E402
    run_router_episode,
)
from tools.sumo_live_broadcast.injects import parse_injects  # noqa: E402

TRAIN_EDGES = ['road_0_1_0', 'road_1_0_1', 'road_2_1_2', 'road_1_2_3']
EDGE_TARGETS = {
    'road_0_1_0': 'road_1_1_0', 'road_1_0_1': 'road_1_1_2',
    'road_2_1_2': 'road_1_2_1', 'road_1_2_3': 'road_1_3_2',
}
TRAIN_PERIODS = [2, 4]
TRAIN_DURS = [300, 600, 900]
TRAIN_ONSETS = [300, 600, 900, 1200]


def sample_injects(rng, allow_none_p=0.25, p_multi=0.0,
                   periods=None, durs=None, onsets=None,
                   p_throttle=0.0, throttle_speeds=(2.0, 5.0)):
    """Random surge schedule from the training grid; ~25% no-event.

    When an event exists the declared governance objective is drawn
    50/50 (global / focus).  With probability ``p_multi`` a second
    concurrent surge on a different edge is injected as a physical
    distractor; the broadcast still reports the focus-declared event.
    The objective is invisible to the traffic state — the router can
    only read it off the broadcast text.
    """
    periods = periods or TRAIN_PERIODS
    durs = durs or TRAIN_DURS
    onsets = onsets or TRAIN_ONSETS
    if rng.random() < allow_none_p:
        return []
    edge = TRAIN_EDGES[int(rng.integers(len(TRAIN_EDGES)))]
    obj = 'focus' if rng.random() < 0.5 else 'global'
    onset = int(onsets[rng.integers(len(onsets))])
    dur = int(durs[rng.integers(len(durs))])
    if rng.random() < p_throttle:
        # capacity-cut event (demand-side dual of surge): speed-cap the
        # whole edge, connectivity preserved — a different physical
        # mechanism, so 'kind' is a real semantic channel
        spd = float(throttle_speeds[rng.integers(len(throttle_speeds))])
        inj = parse_injects(
            [f'throttle:{edge}:{onset}:{dur}:{spd}:{obj}'])[0]
        return [inj]
    specs = [(f"surge:{edge}:{EDGE_TARGETS[edge]}:{onset}:{dur}:"
              f"{int(periods[rng.integers(len(periods))])}:"
              f"{obj}")]
    if rng.random() < p_multi:
        others = [e for e in TRAIN_EDGES if e != edge]
        edge2 = others[int(rng.integers(len(others)))]
        onset2 = max(0, onset + int(rng.integers(-1, 2)) * 150)
        obj2 = ('global' if obj == 'focus'
                else ('focus' if rng.random() < 0.3 else 'global'))
        specs.append(
            f"surge:{edge2}:{EDGE_TARGETS[edge2]}:{onset2}:{dur}:"
            f"{int(periods[rng.integers(len(periods))])}:"
            f"{obj2}")
    return parse_injects(specs)


def priority_lanes_for(fb, injects):
    """Lane names scored when the declared objective is 'focus'."""
    focus = [i for i in injects if i.get('objective') == 'focus']
    if not focus:
        return None
    idx = []
    for inj in focus:
        edge = inj.get('edge_from') or inj.get('edge') or ''
        idx += fb.edge_to_laneidx.get(edge, [])
    return [fb.lanes[i] for i in sorted(set(idx))] or None


def train(args):
    import torch
    from dev.takeover_probe.dispatch_model import (
        DispatchRouter, collate,
    )

    rng = np.random.default_rng(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    extra_models = []
    for spec in args.extra_model:
        name, path = spec.split('=', 1)
        extra_models.append((name, path))
    controllers = build_controllers(
        dqn_s1_path=args.dqn_s1, extra_models=extra_models)
    ctrl_ids = [c for c in args.controllers.split(',') if c]
    controllers = {k: v for k, v in controllers.items() if k in ctrl_ids}
    missing = set(ctrl_ids) - set(controllers)
    if missing:
        raise ValueError(f'controllers not registered: {sorted(missing)}')

    # probe world first for feature dims (lanes differ per scene)
    world = build_world(args.scene, args.traffic_seed_base, Path(
        args.runtime_dir or '/tmp/takeover_probe'), out / 'logger',
        {'t_fixed': args.t_fixed, 't_min': args.t_min})
    ranks = list(range(len(world.intersection_ids)))
    fb0 = FeatureBuilder(
        world, [], report_delay=args.report_delay, layout=args.layout,
        include_event_flag=True if args.event_flag else None)
    dims = dict(
        lane_dim=fb0.lane_tokens(0.0).shape[1],
        text_dim=fb0.text_dim, global_dim=fb0.global_dim,
        geom=fb0.model_geom())
    if args.variant in ('attn', 'sga'):
        print(f'layout={fb0.layout} flag={fb0.include_event_flag} '
              f'lane_dim={dims["lane_dim"]} geom={dims["geom"]}')
    if args.init_ckpt:
        router, payload = DispatchRouter.load(args.init_ckpt)
        for cid in ctrl_ids:
            if cid not in router.controller_ids:
                router.add_controller(cid, init=args.new_arm_init)
                print(f'added controller {cid} (init={args.new_arm_init})')
        if args.freeze_encoder:
            for p in router.encoder.parameters():
                p.requires_grad_(False)
        if args.freeze_head:
            for p in router.head.parameters():
                p.requires_grad_(False)
        if args.freeze_old_emb:
            router.freeze_rows_except_last()
        print('finetune: encoder_frozen=%s head_frozen=%s '
              'old_emb_frozen=%s arms=%s'
              % (args.freeze_encoder, args.freeze_head,
                 args.freeze_old_emb, router.controller_ids))
    else:
        router = DispatchRouter(
            **dims, controller_ids=ctrl_ids, d_model=args.d_model,
            variant=args.variant, residual_pool=args.residual_pool)
    target = DispatchRouter(
        **dims, controller_ids=list(router.controller_ids),
        d_model=args.d_model, variant=args.variant,
        residual_pool=args.residual_pool)
    target.load_state_dict(router.state_dict())
    target.eval()
    opt = torch.optim.Adam(
        [p for p in router.parameters() if p.requires_grad], lr=args.lr)
    replay = deque(maxlen=20000)

    log_path = out / 'train_log.jsonl'
    pending_pair = None    # (sumo_seed, injects) forced next episode
    with log_path.open('w', encoding='utf-8') as log:
        for ep in range(args.episodes):
            if pending_pair is not None:
                sumo_seed, injects = pending_pair
                pending_pair = None
            else:
                injects = sample_injects(
                    rng, allow_none_p=args.p_none, p_multi=args.p_multi,
                    periods=args.periods, durs=args.durs,
                    onsets=args.onsets, p_throttle=args.p_throttle,
                    throttle_speeds=args.throttle_speeds)
                sumo_seed = args.traffic_seed_base + ep
                # objective contrast: replay the same seed+event under
                # the *opposite* declared objective — identical physics,
                # only the broadcast text differs -> the only channel
                # that can separate the two TD traces.
                if (len(injects) == 1 and rng.random() < args.pair_prob
                        and injects[0].get('objective')):
                    other = dict(injects[0])
                    other['objective'] = (
                        'focus' if other['objective'] == 'global'
                        else 'global')
                    pending_pair = (sumo_seed, [dict(other)])
            # fresh SUMO world per episode -> independent traffic seed
            world.close()
            world = build_world(args.scene, sumo_seed, Path(
                args.runtime_dir or '/tmp/takeover_probe'),
                out / 'logger',
                {'t_fixed': args.t_fixed, 't_min': args.t_min,
                 'min_green_vehicle': 3, 'max_red_vehicle': 6})
            ranks = list(range(len(world.intersection_ids)))
            _bind_all(world, controllers, ranks, needed=set(ctrl_ids))
            fb = FeatureBuilder(
                world, injects, report_delay=args.report_delay,
                layout=args.layout,
                include_event_flag=True if args.event_flag else None)
            eps = max(args.eps_end,
                      args.eps_start
                      - ep * (args.eps_start - args.eps_end)
                      / max(1, args.episodes - 1))
            pri_lanes = priority_lanes_for(fb, injects)
            transitions, records, windows = run_router_episode(
                world, controllers, fb, router, ranks,
                injects=injects, interval=args.interval,
                window_decisions=args.window_decisions,
                episode_seconds=args.episode_seconds,
                epsilon=eps, rng=rng,
                priority_lanes=pri_lanes)
            replay.extend(transitions)
            # aux alignment decays to zero over the first aux_decay_frac of
            # training: early binding guidance, then pure Q optimisation.
            aux_w = args.aux_weight
            if args.aux_decay_frac > 0:
                aux_w *= max(0.0, 1.0 - ep
                             / (args.aux_decay_frac * args.episodes))
            stats = update(router, target, opt, replay, args, rng,
                           aux_weight=aux_w)
            target.load_state_dict(router.state_dict())
            sel = {}
            for w in windows:
                sel[w['controller']] = sel.get(w['controller'], 0) + 1
            ev = {'episode': ep, 'sumo_seed': sumo_seed,
                  'inject': ((injects[0].get('edge_from')
                              or injects[0].get('edge'))
                             if injects else 'none'),
                  'objective': (injects[0].get('objective', 'global')
                                if injects else 'none'),
                  'epsilon': round(eps, 3),
                  'windows': len(windows),
                  'mean_window_reward': round(float(np.mean(
                      [w['reward'] for w in windows])), 3),
                  'throughput': records[-1]['throughput'] if records else 0,
                  'selection': sel, **stats}
            log.write(json.dumps(ev, ensure_ascii=False) + '\n')
            log.flush()
            print(json.dumps(ev, ensure_ascii=False))
    meta = dict(dims, d_model=args.d_model, scene=args.scene,
                episodes=args.episodes,
                residual_pool=args.residual_pool,
                feature_layout=fb0.layout,
                event_flag=fb0.include_event_flag,
                dqn_s1=str(args.dqn_s1))
    router.save(out / f'router_{args.variant}.pt', meta=meta)
    print(f'saved {out}/router_{args.variant}.pt')


def update(router, target, opt, replay, args, rng, aux_weight=None):
    import torch
    from dev.takeover_probe.dispatch_model import collate
    stats = {}
    if len(replay) < args.batch * 2:
        return {'loss': 0.0, 'aux': 0.0, 'attn_grad': 0.0}
    losses, auxs, grads = [], [], []
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
        q, w = router.q_values(s)
        with torch.no_grad():
            q_next_online, _ = router.q_values(sp)
            a_star = q_next_online.argmax(dim=-1)
            q_next_target, _ = target.q_values(sp)
            td = r + args.gamma * (1 - done) * q_next_target.gather(
                1, a_star.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(
            q.gather(1, a.unsqueeze(1)).squeeze(1), td)
        aux = torch.zeros(())
        if router.variant in ('attn', 'sga'):
            terms = []
            for i, t in enumerate(batch_tr):
                tgt = t['s'].get('event_lane_idx') or []
                if tgt:
                    p = w[i, torch.as_tensor(tgt)].sum()
                    terms.append(-torch.log(p.clamp_min(1e-6)))
            if terms:
                aux = torch.stack(terms).mean() * (
                    args.aux_weight if aux_weight is None else aux_weight)
        total = loss + aux
        opt.zero_grad()
        total.backward()
        gn = sum(p.grad.norm().item() for p in router.parameters()
                 if p.grad is not None)
        opt.step()
        losses.append(loss.item())
        auxs.append(float(aux))
        grads.append(gn)
    return {'loss': round(float(np.mean(losses)), 4),
            'aux': round(float(np.mean(auxs)), 4),
            'attn_grad': round(float(np.mean(grads)), 4)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--variant', choices=['attn', 'sga', 'selfattn',
                                          'concat'], default='attn')
    p.add_argument('--residual-pool', choices=['mean', 'mms'],
                   default='mean',
                   help='attn encoder only: residual traffic summary = '
                        'lane mean, or mean+max+std over physical '
                        'features (queue/count/wait)')
    p.add_argument('--layout', choices=['identity', 'flag'],
                   default='identity',
                   help="featurization layout: 'identity' gives lane "
                        "tokens an edge-id one-hot and encodes the "
                        "report edge in the same space (structural "
                        "binding, text is the only location source); "
                        "'flag' reproduces the legacy event_flag layout")
    p.add_argument('--event-flag', action='store_true',
                   help='identity layout only: append the report-derived '
                        'event flag after the lane identity dims '
                        '(redundant-channel ablation arm)')
    p.add_argument('--episodes', type=int, default=40)
    p.add_argument('--controllers', default='dqn_s1,maxpressure')
    p.add_argument('--init-ckpt', default=None,
                   help='load pretrained router; ids in --controllers not '
                        'present are appended via add_controller')
    p.add_argument('--new-arm-init', choices=['clone', 'zero'],
                   default='clone')
    p.add_argument('--freeze-encoder', action='store_true',
                   help='finetune: keep the scene encoder (attention) fixed')
    p.add_argument('--freeze-head', action='store_true')
    p.add_argument('--freeze-old-emb', action='store_true',
                   help='finetune: only the newest ctrl_emb row trains')
    p.add_argument('--dqn-s1', default=None)
    p.add_argument('--extra-model', action='append', default=[],
                   help='register snapshot controller NAME=CKPT_PATH '
                        '(repeatable; also list the name in --controllers)')
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--updates-per-ep', type=int, default=60)
    p.add_argument('--eps-start', type=float, default=0.25)
    p.add_argument('--eps-end', type=float, default=0.05)
    p.add_argument('--aux-weight', type=float, default=0.2)
    p.add_argument('--aux-decay-frac', type=float, default=0.0,
                   help='if >0, linearly decay aux weight to zero over the '
                        'first this-fraction of episodes')
    p.add_argument('--periods', type=lambda s: [int(x) for x in s.split(',')],
                   default=None,
                   help='override TRAIN_PERIODS grid, e.g. "2" biases '
                        'sampling toward severe surges (calibration runs)')
    p.add_argument('--durs', type=lambda s: [int(x) for x in s.split(',')],
                   default=None, help='override TRAIN_DURS grid')
    p.add_argument('--onsets', type=lambda s: [int(x) for x in s.split(',')],
                   default=None, help='override TRAIN_ONSETS grid')
    p.add_argument('--p-throttle', type=float, default=0.0,
                   help='prob. the episode event is a throttle '
                        '(capacity cut, connectivity preserved) instead '
                        'of a surge (demand spike); mixed-kind training')
    p.add_argument('--throttle-speeds',
                   type=lambda s: [float(x) for x in s.split(',')],
                   default=[2.0, 5.0],
                   help='throttle max-speed grid (m/s)')
    p.add_argument('--p-none', type=float, default=0.25)
    p.add_argument('--p-multi', type=float, default=0.0,
                   help='prob. of adding a second concurrent surge event '
                        'on a different edge as a physical distractor')
    p.add_argument('--pair-prob', type=float, default=0.0,
                   help='prob. of replaying the same event with the '
                        'opposite declared objective in the next episode')
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--window-decisions', type=int, default=12)
    p.add_argument('--episode-seconds', type=int, default=3600)
    p.add_argument('--report-delay', type=float, default=30.0)
    p.add_argument('--t-fixed', type=int, default=30)
    p.add_argument('--t-min', type=int, default=10)
    p.add_argument('--rng-seed', type=int, default=0)
    p.add_argument('--traffic-seed-base', type=int, default=3000)
    p.add_argument('--runtime-dir', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.dqn_s1 is None:
        from dev.takeover_probe.probe import DEFAULT_DQN_S1
        args.dqn_s1 = str(DEFAULT_DQN_S1)
    train(args)


if __name__ == '__main__':
    main()
