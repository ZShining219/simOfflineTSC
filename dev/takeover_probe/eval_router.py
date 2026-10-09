"""P2/P3 frozen evaluation for the dispatch pilot.

P2 arms (same test conditions, all frozen):
  fixed_dqn_s1   DQN-S1 controls whole episode
  fixed_mp       MaxPressure controls whole episode
  router_concat  SMDP router, no-attention encoder (arm C)
  router_attn    Scene-Guided Attention router (arm D)

P3 attention ablation (frozen attn router):
  text=correct | no_text | wrong_location

Test conditions are held-out (edge x period x duration x onset combos
absent from the training grid) on test traffic seeds — nothing here
updates weights.

Usage:
    python -m dev.takeover_probe.eval_router \
        --attn-ckpt <p> --concat-ckpt <p> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dev.takeover_probe.probe import (  # noqa: E402
    build_controllers, build_world, _bind_all, DEFAULT_DQN_S1,
)
from dev.takeover_probe.dispatch_state import FeatureBuilder  # noqa: E402
from dev.takeover_probe.router_runtime import (  # noqa: E402
    run_router_episode,
)
from tools.sumo_live_broadcast.injects import parse_injects  # noqa: E402

# held-out combos: (edge, period, dur, onset) absent from train grid
TEST_EVENTS = [
    ('road_0_1_0', 'road_1_1_0', 3, 450, 600),
    ('road_1_0_1', 'road_1_1_2', 3, 450, 300),
    ('road_2_1_2', 'road_1_2_1', 3, 750, 900),
    (None, None, None, None, None),          # no-event control cell
]


def lane_edge(lane):
    return lane.rsplit('_', 1)[0]


def eval_metrics(records, windows, world, event_edge=None):
    final = records[-1] if records else {}
    qsum = sum(r['total_queue'] for r in records)
    tt = world.get_vehicles() if world is not None else 0.0
    lanes = sorted({l for r in records for l in r['lane_queue']})
    peak_per_lane = {
        l: max(r['lane_queue'].get(l, 0) for r in records)
        for l in lanes}
    ev = {
        'throughput': int(final.get('throughput', 0)),
        'mean_travel_time': round(float(tt), 2),
        'queue_integral': int(qsum),
        'max_lane_queue': int(max(peak_per_lane.values() or [0])),
        'max_other_lane_queue': int(max(
            v for l, v in peak_per_lane.items()
            if not event_edge or lane_edge(l) != event_edge
        ) or 0),
        'windows': len(windows),
        'switches': sum(1 for w in windows if w.get('switched')),
        'controller_share': {},
    }
    for w in windows:
        c = w['controller']
        ev['controller_share'][c] = ev['controller_share'].get(c, 0) + 1
    if event_edge:
        ev_lanes = [l for l in lanes if lane_edge(l) == event_edge]
        evq = [
            max(r['lane_queue'].get(l, 0) for l in ev_lanes)
            for r in records]
        ev['event_lane_peak'] = max(evq) if evq else 0
        ev['event_lane_queue_integral'] = int(sum(evq))
    return ev


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--attn-ckpt', default=None)
    p.add_argument('--concat-ckpt', default=None)
    p.add_argument('--extra', action='append', default=[],
                   help='extra router arm as NAME=CKPT (repeatable)')
    p.add_argument('--dqn-s1', default=str(DEFAULT_DQN_S1))
    p.add_argument('--extra-model', action='append', default=[],
                   help='register snapshot controller NAME=CKPT_PATH '
                        '(repeatable; needed when a router arm references '
                        'a non-builtin controller)')
    p.add_argument('--test-seed-base', type=int, default=5000)
    p.add_argument('--n-seeds', type=int, default=3)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--window-decisions', type=int, default=12)
    p.add_argument('--episode-seconds', type=int, default=3600)
    p.add_argument('--report-delay', type=float, default=30.0)
    p.add_argument('--p3', action='store_true',
                   help='also run no_text / wrong_location ablations')
    p.add_argument('--runtime-dir', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    from dev.takeover_probe.dispatch_model import DispatchRouter

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    routers = {}
    for name, ck in [('router_attn', args.attn_ckpt),
                     ('router_concat', args.concat_ckpt)]:
        if ck:
            routers[name], _ = DispatchRouter.load(ck)
    for spec_kv in args.extra:
        name, ck = spec_kv.split('=', 1)
        routers[name], _ = DispatchRouter.load(ck)

    rows = []
    world = None
    runtime = Path(args.runtime_dir or '/tmp/takeover_probe')
    try:
        for ev_idx, (edge, to_edge, per, dur, onset) in enumerate(
                TEST_EVENTS):
            spec = (
                f'surge:{edge}:{to_edge}:{onset}:{dur}:{per}'
                if edge else None)
            for seed_i in range(args.n_seeds):
                sumo_seed = args.test_seed_base + ev_idx * 100 + seed_i
                injects = parse_injects([spec]) if spec else []
                if world is not None:
                    world.close()
                world = build_world(
                    args.scene, sumo_seed, runtime, out / 'logger',
                    {'t_fixed': 30, 't_min': 10,
                     'min_green_vehicle': 3, 'max_red_vehicle': 6})
                ranks = list(range(len(world.intersection_ids)))
                extra_models = []
                for mspec in args.extra_model:
                    mname, mpath = mspec.split('=', 1)
                    extra_models.append((mname, mpath))
                controllers = build_controllers(
                    dqn_s1_path=args.dqn_s1, extra_models=extra_models)
                edges_all = sorted(
                    {l.rsplit('_', 1)[0]
                     for l in (world.get_info('lane_waiting_count')
                               or {})})
                wrong_edge = next(
                    e for e in edges_all if e != edge) if edge else None
                arms = [('fixed_dqn_s1', None, 'dqn_s1', None),
                        ('fixed_mp', None, 'maxpressure', None)]
                for name, router in routers.items():
                    arms.append((name, router, None, None))
                if args.p3 and 'router_attn' in routers:
                    arms.append((
                        'attn_no_text', routers['router_attn'], None,
                        {'kind': 'none'}))
                    if wrong_edge:
                        arms.append((
                            'attn_wrong_loc', routers['router_attn'],
                            None, {'edge': wrong_edge}))
                for arm_name, router, fixed, t_over in arms:
                    world.reset()
                    bind_ids = (
                        set(router.controller_ids) if router is not None
                        else {fixed})
                    _bind_all(world, controllers, ranks, needed=bind_ids)
                    injects_i = parse_injects([spec]) if spec else []
                    fb = FeatureBuilder(
                        world, injects_i,
                        report_delay=args.report_delay)
                    ctrl = (
                        {k: controllers[k] for k in router.controller_ids}
                        if router is not None else controllers)
                    transitions, records, windows = run_router_episode(
                        world, ctrl, fb, router, ranks,
                        injects=injects_i, interval=args.interval,
                        window_decisions=args.window_decisions,
                        episode_seconds=args.episode_seconds,
                        epsilon=0.0, fixed_controller=fixed,
                        text_override=t_over, collect=False)
                    m = eval_metrics(
                        records, windows, world,
                        event_edge=edge)
                    rows.append({
                        'event': spec or 'none',
                        'seed': sumo_seed, 'arm': arm_name, **m})
                    print(json.dumps(rows[-1], ensure_ascii=False),
                          flush=True)
                    (out / f'records_{arm_name}_e{ev_idx}_s{seed_i}'
                     '.jsonl').write_text(
                        '\n'.join(json.dumps(r) for r in records) + '\n',
                        encoding='utf-8')
                    if windows:
                        (out / f'windows_{arm_name}_e{ev_idx}_s{seed_i}'
                         '.jsonl').write_text(
                            '\n'.join(json.dumps({
                                't': w['t'], 'controller': w['controller'],
                                'switched': w['switched'],
                                'reward': w.get('reward'),
                                'text': w['text'],
                                'q_values': w.get('q_values'),
                                'attn_top_lane': w.get('attn_top_lane'),
                                'attn_weights': w.get('attn_weights'),
                                'lane_names': w.get('lane_names'),
                            }, ensure_ascii=False) for w in windows)
                            + '\n', encoding='utf-8')
    finally:
        if world is not None:
            world.close()
    (out / 'eval_rows.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)
        + '\n', encoding='utf-8')
    # aggregate
    agg = {}
    for r in rows:
        key = (r['event'], r['arm'])
        agg.setdefault(key, []).append(r)
    print('\n=== summary (mean over seeds) ===')
    summary = []
    for (event, arm), rs in sorted(agg.items()):
        row = {'event': event, 'arm': arm}
        for m in ('throughput', 'mean_travel_time', 'queue_integral',
                  'max_lane_queue', 'switches', 'event_lane_peak',
                  'event_lane_queue_integral'):
            vals = [r[m] for r in rs if m in r]
            if vals:
                row[m] = round(float(np.mean(vals)), 2)
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False))
    (out / 'eval_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')


if __name__ == '__main__':
    main()
