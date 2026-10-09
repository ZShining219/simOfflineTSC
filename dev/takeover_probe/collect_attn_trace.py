"""Collect full attention-weight traces for the heatmap figure.

Runs one event cell under the frozen scene-guided router with three text
conditions (correct / wrong-location / no-text) and dumps per-window
attention weight vectors.  Output feeds ``make_figs.fig7``.

Usage:
    python -m dev.takeover_probe.collect_attn_trace \
        --ckpt output_data/dev_takeover_probe/router_attn_obj3/router_attn.pt \
        --out output_data/dev_takeover_probe/figs/attn_trace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dev.takeover_probe.probe import (  # noqa: E402
    build_world, build_controllers, _bind_all, DEFAULT_DQN_S1)
from dev.takeover_probe.dispatch_state import FeatureBuilder  # noqa: E402
from dev.takeover_probe.router_runtime import run_router_episode  # noqa: E402
from tools.sumo_live_broadcast.injects import parse_injects  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scene', default='S2')
    ap.add_argument('--event',
                    default='surge:road_2_1_2:road_1_2_1:900:750:3')
    ap.add_argument('--seed', type=int, default=5200)
    ap.add_argument('--report-delay', type=float, default=30.0)
    ap.add_argument('--dqn-s1', default=str(DEFAULT_DQN_S1))
    args = ap.parse_args()

    from dev.takeover_probe.dispatch_model import DispatchRouter

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    router, _ = DispatchRouter.load(args.ckpt)
    injects = parse_injects([args.event])
    true_edge = args.event.split(':')[1]

    world = build_world(args.scene, args.seed,
                        Path('/tmp/takeover_probe'), out / 'logger',
                        {'t_fixed': 30, 't_min': 10})
    try:
        ranks = list(range(len(world.intersection_ids)))
        controllers = build_controllers(
            dqn_s1_path=args.dqn_s1, extra_models=())
        edges_all = sorted({l.rsplit('_', 1)[0] for l in
                            (world.get_info('lane_waiting_count') or {})})
        wrong_edge = next(e for e in edges_all if e != true_edge)
        arms = [
            ('correct', None),
            ('wrong_loc', {'edge': wrong_edge}),
            ('no_text', {'kind': 'none'}),
        ]
        for name, t_over in arms:
            world.reset()
            _bind_all(world, controllers, ranks,
                      needed=set(router.controller_ids))
            fb = FeatureBuilder(world, injects,
                                report_delay=args.report_delay)
            ctrl = {k: controllers[k] for k in router.controller_ids}
            _, _, windows = run_router_episode(
                world, ctrl, fb, router, ranks,
                injects=parse_injects([args.event]),
                epsilon=0.0, text_override=t_over, collect=False)
            path = out / f'attn_trace_{name}.jsonl'
            with path.open('w') as f:
                for w in windows:
                    f.write(json.dumps({
                        't': w['t'], 'controller': w['controller'],
                        'text': w['text'],
                        'attn_weights': w.get('attn_weights'),
                        'lane_names': w.get('lane_names'),
                        'true_edge': true_edge,
                        'reported_edge': (true_edge if t_over is None
                                          else t_over.get('edge')),
                    }) + '\n')
            print('wrote', path)
    finally:
        world.close()


if __name__ == '__main__':
    main()
