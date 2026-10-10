"""Counterfactual winner scan: does the best controller flip with
text-visible event parameters (duration / intensity)?

Two surge events (edge,onset,period,dur_short) vs (edge,onset,period,
dur_long) are *physically identical* for all sim time < onset+dur_short —
every window boundary in that span sees the same observable state; only
the broadcast text (est_remaining_s / severity) differs.  If the
whole-episode counterfactual winner (lowest queue integral of the fixed
arms) differs between pair members, those mid-event windows are points
where text is the only way to pick right -> the substrate for proving
text necessity.

Usage:
    python -m dev.takeover_probe._router.scan_pairs --out <dir>
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dev.takeover_probe.probe import (  # noqa: E402
    build_controllers, build_world, _bind_all, DEFAULT_DQN_S1,
)
from dev.takeover_probe.dispatch_state import FeatureBuilder  # noqa: E402
from dev.takeover_probe._router.router_runtime import (  # noqa: E402
    run_router_episode,
)
from dev.takeover_probe._router.eval_router import eval_metrics  # noqa: E402
from tools.sumo_live_broadcast.injects import parse_injects  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--edges', default='road_0_1_0,road_1_0_1,road_2_1_2,road_1_2_3')
    p.add_argument('--onset', type=int, default=600)
    p.add_argument('--periods', default='2,3,6')
    p.add_argument('--durs', default='300,450,750,1050')
    p.add_argument('--seeds', default='7000,7001')
    p.add_argument('--dqn-s1', default=str(DEFAULT_DQN_S1))
    p.add_argument('--episode-seconds', type=int, default=3600)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--window-decisions', type=int, default=12)
    p.add_argument('--runtime-dir', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    targets = {
        'road_0_1_0': 'road_1_1_0', 'road_1_0_1': 'road_1_1_2',
        'road_2_1_2': 'road_1_2_1', 'road_1_2_3': 'road_1_3_2',
    }
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runtime = Path(args.runtime_dir or '/tmp/takeover_probe')

    edges = [e for e in args.edges.split(',') if e]
    periods = [int(x) for x in args.periods.split(',')]
    durs = [int(x) for x in args.durs.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]

    rows, world = [], None
    try:
        for edge, per, dur, seed in itertools.product(
                edges, periods, durs, seeds):
            spec = f'surge:{edge}:{targets[edge]}:{args.onset}:{dur}:{per}'
            if world is not None:
                world.close()
            world = build_world(
                args.scene, seed, runtime, out / 'logger',
                {'t_fixed': 30, 't_min': 10})
            ranks = list(range(len(world.intersection_ids)))
            controllers = build_controllers(
                dqn_s1_path=args.dqn_s1, extra_models=())
            injects = parse_injects([spec])
            fb = FeatureBuilder(world, injects)
            for arm in ('dqn_s1', 'maxpressure'):
                world.reset()
                _bind_all(world, controllers, ranks, needed={arm})
                _, records, windows = run_router_episode(
                    world, controllers, fb, None, ranks,
                    injects=injects, interval=args.interval,
                    window_decisions=args.window_decisions,
                    episode_seconds=args.episode_seconds,
                    fixed_controller=arm, collect=False)
                m = eval_metrics(
                    records, windows, world, event_edge=edge)
                row = {'edge': edge, 'period': per, 'dur': dur,
                       'seed': seed, 'arm': arm, **m}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        if world is not None:
            world.close()

    (out / 'scan_rows.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)
        + '\n', encoding='utf-8')

    # winner matrix: for each (edge, period, seed, dur) the winning arm
    cell = {}
    for r in rows:
        key = (r['edge'], r['period'], r['seed'], r['dur'])
        cell.setdefault(key, {})[r['arm']] = r['queue_integral']
    winners = {}
    for (edge, per, seed, dur), q in cell.items():
        winners.setdefault((edge, per, seed), {})[dur] = (
            min(q, key=q.get), q)
    print('\n=== winner matrix (queue_integral -> winner) ===')
    flips = []
    for (edge, per, seed), durs_map in sorted(winners.items()):
        durs_sorted = sorted(durs_map)
        seq = [durs_map[d][0] for d in durs_sorted]
        line = '  '.join(
            f'd{d}:{durs_map[d][0][:3]}({durs_map[d][1]})'[:24]
            for d in durs_sorted)
        print(f'{edge} p{per} s{seed} | {line}')
        if len(set(seq)) > 1:
            flips.append((edge, per, seed))
    print(f'\nflip cells (winner changes with duration): {len(flips)}')
    for f in flips:
        print('  FLIP', f)
    (out / 'scan_flips.json').write_text(json.dumps({
        'flips': [
            {'edge': e, 'period': p, 'seed': s} for e, p, s in flips],
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
