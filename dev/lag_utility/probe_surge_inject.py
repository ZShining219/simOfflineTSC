#!/usr/bin/env python3
"""Micro-probe: end-to-end demand_surge validation without the trainer.

Binds SumoEventRuntime directly to libsumo on the hz4x4 eval surge schedule
(hz_surge_01: road_0_1_0 -> road_2_1_0, 600 vph, t 900-1200) and verifies:

  1. bind() succeeds (findRoute + persistent route added);
  2. vehicles are injected deterministically during the active window
     (audit().surge_stats inserted/failed counts, expected ~end-start over
     period = 300 * 600/3600 = 50 vehicles);
  3. public reports render and ReportGrounder parses them (strict);
  4. injected vehicles actually enter the network (vehicle id list samples).

Usage:
  python3 dev/lag_utility/probe_surge_inject.py --seed 9000
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import libsumo
import sumolib

from world.sumo_events.schema import Event, Schedule
from world.sumo_events.runtime import SumoEventRuntime
from world.sumo_events.grounding import ReportGrounder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=9000)
    ap.add_argument('--begin', type=float, default=900)
    ap.add_argument('--end', type=float, default=1200)
    ap.add_argument('--rate', type=float, default=600)
    ap.add_argument('--origin', default='road_0_1_0')
    ap.add_argument('--dest', default='road_2_1_0')
    args = ap.parse_args()

    cfg = json.loads((ROOT / 'configs/sim/hz4x4.cfg').read_text())
    net_file = str(ROOT / cfg['dir'] / cfg['roadnetFile'])
    cmd = [sumolib.checkBinary('sumo'), '-c',
           str(ROOT / cfg['dir'] / cfg['combined_file']),
           '--seed', str(args.seed), '--no-warnings', 'true',
           '--time-to-teleport', '-1', '--ignore-route-errors', 'true']
    libsumo.start(cmd)

    event = Event(event_id='hz_surge_01', kind='demand_surge',
                  begin=args.begin, end=args.end,
                  edge_id=args.origin, to_edge_id=args.dest,
                  rate_vph=args.rate)
    runtime = SumoEventRuntime(net_file, Schedule((event,)))
    runtime.bind(libsumo)
    print('[bind] ok — route for',
          runtime._surge['hz_surge_01'])

    grounder = ReportGrounder(runtime.network_catalog())
    seen_active = seen_cleared = False
    expected = int((args.end - args.begin) / (3600.0 / args.rate))
    for _ in range(int(args.end) + 400):
        libsumo.simulationStep()
        runtime.synchronize()
        now = libsumo.simulation.getTime()
        for report in runtime.reports():
            if report.status == 'active' and not seen_active:
                seen_active = True
                g = grounder.map_report(report)
                print(f'[t={now:.0f} active] {report.text}')
                print(f'         grounded kind={g.event_kind} scope={g.scope} '
                      f'mentions={[(m.role, m.entity_id) for m in g.mentions]}')
            if report.status == 'cleared' and not seen_cleared:
                seen_cleared = True
                g = grounder.map_report(report)
                print(f'[t={now:.0f} cleared] {report.text}')
                print(f'         grounded kind={g.event_kind} scope={g.scope} '
                      f'mentions={[(m.role, m.entity_id) for m in g.mentions]}')
        if seen_cleared:
            break
    audit = runtime.audit()
    stats = audit['surge_stats']['hz_surge_01']
    print(f"[audit] expected≈{expected} inserted={stats['inserted']} "
          f"failed={stats['failed']}")
    print(f"[audit] states={audit['states']} operators={audit['operators']['demand_surge'][:60]}...")
    live = [v for v in libsumo.vehicle.getIDList() if v.startswith('__tsc_surge_')]
    print(f'[live] {len(live)} surge vehicles still in network, e.g. {live[:3]}')
    libsumo.close()


if __name__ == '__main__':
    main()
