#!/usr/bin/env python3
"""Micro-probe: WHERE do vehicles bound for a closed edge actually wait?

Reproduces ep0 of hz4x4_random_v1 (single closure road_1_2_3, t 665-954,
sumo seed 9000) with default signals, and every 10s logs, at the feeder
intersection_1_2 and downstream intersection_1_1:
  - vehicles standing on junction-INTERNAL edges (':<node>_*')  [invisible to
    lane features — box blockage]
  - vehicles standing on the dead approach lanes (links into the closed edge)
  - lane_count on the closed edge's lanes (downstream in-lanes)
  - for internal-edge standers: whether the vehicle's route wants the closed
    edge next.

Usage:
  python3 dev/lag_utility/probe_junction_block.py --seed 9000 \
      --out data/output_data/analysis/att_entity_004/lag_utility/ep0_probe.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import libsumo
import sumolib

EDGE = 'road_1_2_3'
BEGIN, END = 665, 954
FEEDER, DOWN = 'intersection_1_2', 'intersection_1_1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=9000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--apply-closure', type=int, default=1,
                    help='0 = no-event control')
    args = ap.parse_args()

    cfg = json.loads((ROOT / 'configs/sim/hz4x4.cfg').read_text())
    cmd = [sumolib.checkBinary('sumo'), '-c',
           str(ROOT / cfg['dir'] / cfg['combined_file']),
           '--seed', str(args.seed), '--no-warnings', 'true',
           '--time-to-teleport', '-1', '--ignore-route-errors', 'true']
    libsumo.start(cmd)
    sumolib_net = sumolib.net.readNet(
        str(ROOT / cfg['dir'] / cfg['roadnetFile']))

    lanes_E = [f'{EDGE}_{i}' for i in range(3)]
    closed = False
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open('w')
    for step in range(1100):
        libsumo.simulationStep()
        now = libsumo.simulation.getTime()
        if args.apply_closure and now >= BEGIN and not closed:
            for l in lanes_E:
                libsumo.lane.setDisallowed(l, ['all'])
            closed = True
        if closed and now >= END:
            for l in lanes_E:
                libsumo.lane.setDisallowed(l, [])
            closed = False
        if now < BEGIN - 60 or now > END + 120 or int(now) % 10:
            continue
        rec = {'t': now, 'closed': closed}
        for node, tag in ((FEEDER, 'feeder'), (DOWN, 'down')):
            internal = [e for e in libsumo.edge.getIDList()
                        if e.startswith(f':{node}')]
            stand, stand_want_e = [], []
            for e in internal:
                for v in libsumo.edge.getLastStepVehicleIDs(e):
                    if libsumo.vehicle.getSpeed(v) > 0.3:
                        continue
                    stand.append(v)
                    try:
                        rte = libsumo.vehicle.getRoute(v)
                        idx = libsumo.vehicle.getRouteIndex(v)
                        nxt = rte[idx + 1] if idx + 1 < len(rte) else None
                    except Exception:
                        nxt = None
                    if nxt == EDGE:
                        stand_want_e.append(v)
            rec[f'{tag}_internal_stopped'] = stand
            rec[f'{tag}_internal_want_e'] = stand_want_e
            # halted queue on all incoming lanes of the node
            qh = 0
            for l in sumolib_net.getNode(node).getIncoming():
                for ln in l.getLanes():
                    qh += libsumo.lane.getLastStepHaltingNumber(ln.getID())
            rec[f'{tag}_queue'] = qh
        # vehicles wanting E next, wherever they currently are
        want_e_now = []
        for v in libsumo.vehicle.getIDList():
            try:
                rte = libsumo.vehicle.getRoute(v)
                idx = libsumo.vehicle.getRouteIndex(v)
                if idx + 1 < len(rte) and rte[idx + 1] == EDGE:
                    want_e_now.append((v, libsumo.vehicle.getLaneID(v),
                                       round(libsumo.vehicle.getSpeed(v), 1)))
            except Exception:
                pass
        rec['want_e_elsewhere'] = want_e_now[:20]
        rec['n_want_e'] = len(want_e_now)
        fh.write(json.dumps(rec) + '\n')
        fh.flush()
    libsumo.close()
    print('wrote', out)


if __name__ == '__main__':
    main()
