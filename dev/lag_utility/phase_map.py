#!/usr/bin/env python3
"""Dump per-node action(phase)->served lanelinks map for hz4x4.

Replicates world/world_sumo.py's phase construction offline:
- `generate_valid_phase`: step the sim 500s, collect distinct RYG states per
  traffic light in first-seen order, drop states containing 'y' and all-r/s.
- `Intersection.phase_available_lanelinks`: for each green phase index p,
  every controlled link whose state char is 'G' or 's' contributes
  (from_lane, to_lane).

Output JSON:
    {"nodes": {node_id: {"intersection_index": i,
                         "phases": [[[from_lane, to_lane], ...], ...],
                         "phase_states": ["GrGr...", ...],
                         "in_lane_order": [lane_id, ...]}},
     "edges": {edge_id: {"from": junction, "to": junction,
                         "lanes": [lane_id, ...]}}}

`in_lane_order` replicates LaneVehicleGenerator(in_only=True) lane ordering:
in_roads sorted by _sort_roads key (direction asc via _get_direction on the
incoming lane shape tail), lanes per road sorted by index ascending
(world.RIGHT=True -> ascending), flattened.  This is the feature order of
ob_traffic / queue lane dims in the dumps.

Usage:
    python3 dev/lag_utility/phase_map.py \
        --sim-config configs/sim/hz4x4.cfg --out dev/lag_utility/hz4x4_phase_map.json
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import libsumo
import sumolib


def natural(value):
    return tuple(int(part) if part.isdigit() else part
                 for part in re.split(r'(\d+)', str(value)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.sim_config).read_text())
    cmd = [sumolib.checkBinary('sumo'), '-c',
           os.path.join(ROOT / cfg['dir'], cfg['combined_file']),
           '--no-warnings', 'true']
    libsumo.start(cmd)

    ids = tuple(sorted(libsumo.trafficlight.getIDList(), key=natural))
    # generate_valid_phase: observe RYG states over a 500s warmup rollout
    seen = {i: [] for i in ids}
    for _ in range(500):
        for i in ids:
            st = libsumo.trafficlight.getRedYellowGreenState(i)
            if st not in seen[i]:
                seen[i].append(st)
        libsumo.simulationStep()
    green = {}
    for i in ids:
        green[i] = [s for s in seen[i]
                    if 'y' not in s and s.count('r') + s.count('s') != len(s)]
    links = {i: libsumo.trafficlight.getControlledLinks(i) for i in ids}
    libsumo.close()

    net = sumolib.net.readNet(str(ROOT / cfg['dir'] / cfg['roadnetFile']))
    edges = {}
    for e in net.getEdges():
        eid = e.getID()
        if eid.startswith(':'):
            continue
        edges[eid] = {'from': e.getFromNode().getID(),
                      'to': e.getToNode().getID(),
                      'lanes': [l.getID() for l in e.getLanes()]}

    out = {}
    for idx, i in enumerate(ids):
        phases = []
        for st in green[i]:
            served = []
            for n, ch in enumerate(st):
                if ch not in ('G', 's'):
                    continue
                if n >= len(links[i]) or not links[i][n]:
                    continue
                fr, to = links[i][n][0][0], links[i][n][0][1]
                served.append([fr, to])
            phases.append(served)
        out[i] = {'intersection_index': idx, 'phases': phases,
                  'phase_states': green[i]}

    # Reproduce LaneVehicleGenerator(in_only=True) feature ordering offline:
    # world_sumo.Intersection collects in-roads by first appearance in
    # getControlledLinks, sorts them by _get_direction(lane_shape, out=False)
    # (atan2(dx, dy) on the shape's last segment, normalized to [0, 2pi)),
    # then lists each road's lanes sorted by lane index ascending.
    def road_dir(shape):
        dx = shape[-2][0] - shape[-1][0]
        dy = shape[-2][1] - shape[-1][1]
        a = math.atan2(dx, dy)
        return a if a >= 0 else a + 2 * math.pi

    for i in ids:
        road_lanes, road_dirs = {}, {}
        for link in links[i]:
            if not link:
                continue
            fr = link[0][0]
            road = fr[:-2]
            road_lanes.setdefault(road, [])
            if fr not in road_lanes[road]:
                road_lanes[road].append(fr)
                road_dirs[road] = road_dir(net.getLane(fr).getShape())
        in_roads = sorted(road_lanes, key=lambda r: road_dirs[r])
        in_lane_order = [l for r in in_roads
                         for l in sorted(road_lanes[r],
                                         key=lambda x: int(x.rsplit('_', 1)[1]))]
        out[i]['in_lane_order'] = in_lane_order
    Path(args.out).write_text(json.dumps({'nodes': out, 'edges': edges}, indent=1))
    n_phase = {i: len(v['phases']) for i, v in out.items()}
    print(f'wrote {args.out}; nodes={len(out)}; edges={len(edges)}; '
          f'phases/node={sorted(set(n_phase.values()))}')


if __name__ == '__main__':
    main()
