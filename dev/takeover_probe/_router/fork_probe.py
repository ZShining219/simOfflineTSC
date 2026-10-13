"""Diagnostic fork: replay an episode with a forced controller at chosen
windows, then compare the realized queue trajectory against the router's
free choice.

Same SUMO seed + identical decisions before the fork time => identical
physical state at fork; deterministic sim => any later divergence is
attributable to the forced window's controller choice (plus knock-on
window choices the router makes afterward from the forked state).

Usage:
    python -m dev.takeover_probe._router.fork_probe \
        --router <ckpt> --extra-model dhoa=<ckpt> \
        --event 'surge:road_2_1_2:road_1_2_1:900:750:3' --seed 5000 \
        --fork 2160=maxpressure --fork 2280=maxpressure \
        --out output_data/dev_takeover_probe/fork_dqn1680
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dev.takeover_probe.probe import (  # noqa: E402
    build_controllers, build_world, _bind_all, DEFAULT_DQN_S1,
)
from dev.takeover_probe.dispatch_state import (  # noqa: E402
    FeatureBuilder, fb_kwargs_from_meta,
)
from dev.takeover_probe.dispatch_model import DispatchRouter  # noqa: E402
from dev.takeover_probe._router.router_runtime import (  # noqa: E402
    run_router_episode,
)
from hitl.live.injects import parse_injects  # noqa: E402


def run_one(args, fork_map):
    """One replay; fork_map {} = free routing."""
    world = build_world(
        args.scene, args.seed, Path(args.runtime_dir or '/tmp/fork_probe'),
        Path(args.out) / 'logger',
        {'t_fixed': 30, 't_min': 10,
         'min_green_vehicle': 3, 'max_red_vehicle': 6})
    try:
        ranks = list(range(len(world.intersection_ids)))
        extra = [tuple(m.split('=', 1)) for m in args.extra_model]
        controllers = build_controllers(
            dqn_s1_path=args.dqn_s1, extra_models=extra)
        router, payload = DispatchRouter.load(args.router)
        _bind_all(world, controllers, ranks,
                  needed=set(router.controller_ids))
        injects = parse_injects([args.event]) if args.event else []
        fb = FeatureBuilder(
            world, injects, report_delay=args.report_delay,
            **fb_kwargs_from_meta(payload.get('meta')))
        ctrl = {k: controllers[k] for k in router.controller_ids}
        _, records, windows = run_router_episode(
            world, ctrl, fb, router, ranks,
            injects=injects, interval=args.interval,
            window_decisions=args.window_decisions,
            episode_seconds=args.episode_seconds, epsilon=0.0,
            collect=False, force_choices=fork_map)
    finally:
        world.close()
    return records, windows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--router', required=True)
    p.add_argument('--extra-model', action='append', default=[])
    p.add_argument('--event', default=None,
                   help='surge:FROM:TO:START:DUR:PERIOD')
    p.add_argument('--seed', type=int, default=5000)
    p.add_argument('--fork', action='append', default=[],
                   help='T=controller or T0:T1:STEP=controller, '
                        'repeat to force several windows')
    p.add_argument('--scene', default='S2')
    p.add_argument('--dqn-s1', default=None)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--window-decisions', type=int, default=12)
    p.add_argument('--episode-seconds', type=int, default=3600)
    p.add_argument('--report-delay', type=float, default=30.0)
    p.add_argument('--runtime-dir', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.dqn_s1 is None:
        args.dqn_s1 = str(DEFAULT_DQN_S1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fork_map = {}
    for spec in args.fork:
        lhs, c = spec.split('=')
        if ':' in lhs:
            t0, t1, st = (float(x) for x in lhs.split(':'))
            t = t0
            while t <= t1 + 1e-6:
                fork_map[t] = c
                t += st
        else:
            fork_map[float(lhs)] = c

    rec_b, win_b = run_one(args, {})
    rec_f, win_f = run_one(args, fork_map)

    for tag, recs, wins in (('base', rec_b, win_b), ('fork', rec_f, win_f)):
        (out / f'records_{tag}.jsonl').write_text(
            '\n'.join(json.dumps(r) for r in recs) + '\n')
        (out / f'windows_{tag}.jsonl').write_text('\n'.join(
            json.dumps({k: w.get(k) for k in (
                't', 'controller', 'switched', 'reward', 'text',
                'q_values', 'attn_top_lane')}) for w in wins) + '\n')

    qb = np.array([r['total_queue'] for r in rec_b])
    qf = np.array([r['total_queue'] for r in rec_f])
    tb = np.array([r['t'] for r in rec_b])
    t0 = min(fork_map) if fork_map else 0
    post = tb >= t0
    print(json.dumps({
        'fork': fork_map, 'seed': args.seed, 'event': args.event,
        'qi_base': float(qb.sum()), 'qi_fork': float(qf.sum()),
        'qi_post_base': float(qb[post].sum()),
        'qi_post_fork': float(qf[post].sum()),
        'delta_post': float(qf[post].sum() - qb[post].sum()),
        'base_windows': [
            {'t': w['t'], 'ctrl': w['controller'],
             'q': w.get('q_values')} for w in win_b],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
