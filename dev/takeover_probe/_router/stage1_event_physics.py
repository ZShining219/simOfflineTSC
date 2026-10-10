"""Stage-1 event-physics check: do different event *kinds* produce
different physical perturbations, and does controller choice matter
under each?

Runs, all on the same SUMO seed with a neutral FixedTime prefix:
  none            clean baseline
  close1          lane_close on ONE lane of road_2_1_2 (edge stays
                  passable via the sibling lane — the design question)
  close2          BOTH lanes of road_2_1_2 closed (impassable check:
                  if demand can't clear, signals can't help)
  surge           comparable-duration surge on the same edge
For each event run also compare fixed_mp / fixed_dqn_s1 on the same
seed to see whether arm identity changes the outcome (i.e. whether the
event creates a control problem at all).

Usage:
    python -m dev.takeover_probe.stage1_event_physics --seed 5200 \
        --out output_data/dev_takeover_probe/stage1_physics
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dev.takeover_probe.probe import (  # noqa: E402
    build_controllers, build_world, _bind_all, run_episode,
    DEFAULT_DQN_S1,
)
from tools.sumo_live_broadcast.injects import parse_injects  # noqa: E402


CASES = {
    'none': [],
    'throttle_severe': ['throttle:road_2_1_2:600:600:2'],
    'throttle_mild': ['throttle:road_2_1_2:600:600:8'],
    'surge': ['surge:road_2_1_2:road_1_2_1:600:600:3'],
}
EDGE_LANES = ['road_2_1_2_0', 'road_2_1_2_1']


def summarize(records, t_on=600, t_end=1200):
    """Phase integrals: during-event queue on the target edge's lanes,
    global queue, throughput at end."""
    seg = [r for r in records if t_on <= r['t'] < t_end + 600]
    edge_q = sum(
        sum(r['lane_queue'].get(l, 0) for l in EDGE_LANES) for r in seg)
    lane0_q = sum(r['lane_queue'].get(EDGE_LANES[0], 0) for r in seg)
    lane1_q = sum(r['lane_queue'].get(EDGE_LANES[1], 0) for r in seg)
    tot_q = sum(r['total_queue'] for r in seg)
    occ0 = [r['lane_count'].get(EDGE_LANES[0], 0) for r in seg]
    occ1 = [r['lane_count'].get(EDGE_LANES[1], 0) for r in seg]
    worst = {}
    for r in seg:
        for l, q in r['lane_queue'].items():
            worst[l] = max(worst.get(l, 0), q)
    return {
        'qi_total': tot_q, 'qi_edge': edge_q,
        'qi_lane0': lane0_q, 'qi_lane1': lane1_q,
        'throughput_end': records[-1]['throughput'],
        'edge_occupancy_mean': round(
            (sum(occ0) + sum(occ1)) / max(1, len(occ0)), 2),
        'max_edge_q': max(
            sum(r['lane_queue'].get(l, 0) for l in EDGE_LANES)
            for r in seg),
        'top3_queue_lanes': sorted(
            worst.items(), key=lambda kv: -kv[1])[:3],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--seed', type=int, default=5200)
    p.add_argument('--episode-seconds', type=int, default=2400)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    world = build_world(
        args.scene, args.seed, Path('/tmp/takeover_probe'),
        out / 'logger', {'t_fixed': 30, 't_min': 10,
                         'min_green_vehicle': 3, 'max_red_vehicle': 6})
    rows = []
    try:
        ranks = list(range(len(world.intersection_ids)))
        controllers = build_controllers(DEFAULT_DQN_S1)
        needed = {'fixedtime', 'maxpressure', 'dqn_s1'}
        _bind_all(world, controllers, ranks, needed=needed)
        for case, specs in CASES.items():
            for ctrl in (['fixedtime'] if case == 'none'
                         else ['fixedtime', 'maxpressure', 'dqn_s1']):
                world.reset()
                _bind_all(world, controllers, ranks, needed=needed)
                injects = parse_injects(specs)
                records, _ = run_episode(
                    world, controllers, ranks,
                    prefix_id=ctrl, takeover_id=None, fork_step=0,
                    interval=args.interval,
                    episode_seconds=args.episode_seconds,
                    injects=injects)
                row = {'case': case, 'ctrl': ctrl, **summarize(records)}
                rows.append(row)
                (out / f'records_{case}_{ctrl}.jsonl').write_text(
                    '\n'.join(json.dumps(r) for r in records) + '\n')
                print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        world.close()
    (out / 'summary.json').write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
