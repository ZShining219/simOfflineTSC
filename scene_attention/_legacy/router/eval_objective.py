"""Paired objective-conditioning evaluation — the text-necessity test.

For each cell (edge, period, dur, seed) the *same* physical episode is
run twice with identical traffic and identical surge injection; only the
broadcast-declared governance objective differs:

    member G: surge:E:T:D:P:global   (reward = global congestion)
    member F: surge:E:T:D:P:focus    (reward = incident-approach queues)

The objective is invisible to traffic state at every timestep — text is
the only channel carrying it.  If the counterfactual best fixed arm
differs between members (an "objective flip"), a router that reads the
text can track the flip; a blind router cannot.

Arms per member:
  fixed_dqn_s1 / fixed_mp            counterfactual winners
  router                             correct text (as broadcast)
  router_wrong_obj                   text claims the *other* objective
  router_no_text                     text removed entirely

Metrics: score under the declared objective (global -> queue_integral,
focus -> event-lane queue integral), oracle regret vs best fixed arm,
and whether the router's dominant choice matches the member winner.

Usage:
    python -m scene_attention._legacy.router.eval_objective \
        --router data/output_data/dev_takeover_probe/routerX/router_attn.pt \
        --out <dir>
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scene_attention.probe.takeover import (  # noqa: E402
    build_controllers, build_world, _bind_all, DEFAULT_DQN_S1,
)
from scene_attention.dispatch_state import (  # noqa: E402
    FeatureBuilder, fb_kwargs_from_meta,
)
from scene_attention._legacy.router.router_runtime import (  # noqa: E402
    run_router_episode,
)
from scene_attention._legacy.router.eval_router import eval_metrics  # noqa: E402
from scene_attention._legacy.router.train_router import (  # noqa: E402
    EDGE_TARGETS, priority_lanes_for,
)
from hitl.live.injects import parse_injects  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--router', required=True)
    p.add_argument('--dqn-s1', default=str(DEFAULT_DQN_S1))
    p.add_argument('--cells', default=(
        'road_1_0_1:6:300,road_1_0_1:6:450,road_1_0_1:6:750,'
        'road_0_1_0:6:300,road_0_1_0:6:450,'
        'road_2_1_2:6:300,road_2_1_2:6:450'),
        help='edge:period:dur list')
    p.add_argument('--onset', type=int, default=600)
    p.add_argument('--seeds', default='9100,9101,9102')
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--window-decisions', type=int, default=12)
    p.add_argument('--episode-seconds', type=int, default=3600)
    p.add_argument('--report-delay', type=float, default=30.0)
    p.add_argument('--runtime-dir', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    from scene_attention.dispatch_model import DispatchRouter

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    router, payload = DispatchRouter.load(args.router)
    fb_kw = fb_kwargs_from_meta(payload.get('meta'))

    cells = []
    for tok in args.cells.split(','):
        e, per, dur = tok.split(':')
        cells.append((tuple(e.split('+')), int(per), int(dur)))
    seeds = [int(s) for s in args.seeds.split(',')]
    runtime = Path(args.runtime_dir or '/tmp/takeover_probe')

    rows, world = [], None
    try:
        for (edges, per, dur), seed in itertools.product(cells, seeds):
            if len(edges) == 1:
                # single-event cell: members = declared objective on/off
                members = [('global', None), ('focus', edges[0])]
            else:
                # multi-event cell: both surges injected; the member
                # decides which edge the broadcast prioritises (focus)
                members = [(f'focus:{e}', e) for e in edges]
            for objective, focus_edge in members:
                specs = [
                    f'surge:{e}:{EDGE_TARGETS[e]}:{args.onset}'
                    f':{dur}:{per}:'
                    f'{"focus" if e == focus_edge else "global"}'
                    for e in edges]
                if world is not None:
                    world.close()
                world = build_world(
                    args.scene, seed, runtime, out / 'logger',
                    {'t_fixed': 30, 't_min': 10})
                ranks = list(range(len(world.intersection_ids)))
                controllers = build_controllers(
                    dqn_s1_path=args.dqn_s1, extra_models=())
                arms = [('fixed_dqn_s1', None, 'dqn_s1', None),
                        ('fixed_mp', None, 'maxpressure', None),
                        ('router', router, None, None),
                        ('router_wrong_obj', router, None,
                         {'objective': (
                             'global' if objective != 'global'
                             else 'focus')}),
                        ('router_no_text', router, None,
                         {'kind': 'none'})]
                for arm_name, rt, fixed, t_over in arms:
                    world.reset()
                    bind_ids = (
                        set(rt.controller_ids) if rt is not None
                        else {fixed})
                    _bind_all(world, controllers, ranks, needed=bind_ids)
                    injects = parse_injects(specs)
                    fb = FeatureBuilder(
                        world, injects, report_delay=args.report_delay,
                        **fb_kw)
                    pri = priority_lanes_for(fb, injects)
                    ctrl = (
                        {k: controllers[k] for k in rt.controller_ids}
                        if rt is not None else controllers)
                    _, records, windows = run_router_episode(
                        world, ctrl, fb, rt, ranks,
                        injects=injects, interval=args.interval,
                        window_decisions=args.window_decisions,
                        episode_seconds=args.episode_seconds,
                        epsilon=0.0, fixed_controller=fixed,
                        text_override=t_over, priority_lanes=pri,
                        collect=False)
                    m = eval_metrics(records, windows, world,
                                     event_edge=focus_edge or edges[0])
                    row = {'cell': f'{"+".join(edges)}:p{per}:d{dur}',
                           'edge': edges[0], 'period': per, 'dur': dur,
                           'seed': seed, 'objective': objective,
                           'focus_edge': focus_edge, 'arm': arm_name,
                           **m}
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        if world is not None:
            world.close()

    (out / 'obj_rows.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)
        + '\n', encoding='utf-8')

    # ---- analysis ------------------------------------------------
    def score(r):
        # focus members are scored on the focus event's lane queues;
        # the global member on total queue integral.
        return (r['event_lane_queue_integral']
                if r['objective'] != 'global' else r['queue_integral'])

    cells_agg = {}
    for r in rows:
        cells_agg.setdefault(
            (r['cell'], r['seed'], r['objective']), {})[r['arm']] = r
    flips, report = 0, []
    for (cell, seed, obj), arms in sorted(cells_agg.items()):
        if 'fixed_dqn_s1' not in arms or 'fixed_mp' not in arms:
            continue
        wd = 'dqn_s1' if score(arms['fixed_dqn_s1']) < score(
            arms['fixed_mp']) else 'maxpressure'
        report.append({
            'cell': cell, 'seed': seed, 'objective': obj,
            'winner': wd,
            'scores': {a: round(score(r), 1) for a, r in arms.items()},
        })
    # flip detection: same (cell,seed), winners differ across objective
    by_pair = {}
    for e in report:
        by_pair.setdefault((e['cell'], e['seed']), {})[
            e['objective']] = e
    for (cell, seed), mem in sorted(by_pair.items()):
        if 'global' in mem and 'focus' in mem and (
                mem['global']['winner'] != mem['focus']['winner']):
            flips += 1
    out_rows = {
        'n_flip_cells': flips,
        'n_pairs': len(by_pair),
        'per_member': report,
    }
    (out / 'obj_summary.json').write_text(
        json.dumps(out_rows, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print('\n=== member winners ===')
    for e in report:
        print(json.dumps(e, ensure_ascii=False))
    print(f'\nflip pairs: {flips}/{len(by_pair)}')

    # oracle regret on flip pairs only
    print('\n=== oracle regret on flip pairs (score - best_fixed) ===')
    for (cell, seed), mem in sorted(by_pair.items()):
        if 'global' not in mem or 'focus' not in mem:
            continue
        if mem['global']['winner'] == mem['focus']['winner']:
            continue
        for obj in ('global', 'focus'):
            arms = cells_agg[(cell, seed, obj)]
            best = min(score(arms['fixed_dqn_s1']),
                       score(arms['fixed_mp']))
            line = {'cell': cell, 'seed': seed, 'objective': obj}
            for a in ('router', 'router_wrong_obj', 'router_no_text',
                      'fixed_dqn_s1', 'fixed_mp'):
                if a in arms:
                    line[a] = round(score(arms[a]) - best, 1)
            print(json.dumps(line, ensure_ascii=False))


if __name__ == '__main__':
    main()
