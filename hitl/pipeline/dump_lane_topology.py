#!/usr/bin/env python
"""dump_lane_topology.py — dump per-network lane topology for offline scoring.

Boots one headless SUMO world (same construction as the broadcast engine /
sequential runtime), then records:

    in_lanes                 world.in_lanes order (== state[:L] lane order)
    lane_length_m            per-lane length in metres
    lane_capacity_veh        length / jam spacing (7.5 m default)
    phase_served_lanes       per green action index -> in-lane indices served
    min_green_s              min green phase duration in the program

Usage:
    python hitl/pipeline/dump_lane_topology.py \
        --sim-config configs/sim/sumohz1x1.cfg \
        --out hitl/topology/topology_sumohz1x1.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

JAM_SPACING_M = 7.5  # ~5.5m vehicle + 2m gap at standstill


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim-config', required=True)
    parser.add_argument('--interface', default='libsumo')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    from common import interface as registry_interface
    from common.registry import Registry
    import world.world_sumo as world_sumo

    registry_interface.Command_Setting_Interface(
        {'command': {'sumo_seed': None}})
    Registry.mapping['logger_mapping']['path'].path = '/tmp/hitl_topology'
    world = world_sumo.World(
        str(Path(args.sim_config).resolve()), 1, interface=args.interface)
    try:
        world.reset()
        inter = world.intersections[0]
        in_lanes = list(world.in_lanes)
        index = {lane: i for i, lane in enumerate(in_lanes)}
        lengths = {
            lane: float(world.eng.lane.getLength(lane)) for lane in in_lanes
        }
        # green phases only (inter.phases indexes into green_phases)
        served = []
        for p in range(len(inter.green_phases)):
            lanes = sorted(
                index[l] for l in inter.phase_available_startlanes[p]
                if l in index)
            served.append(lanes)
        durations = [
            float(ph.duration) for ph in
            world.eng.trafficlight.getAllProgramLogics(inter.id)[0].phases
        ]
        payload = {
            'schema_version': 1,
            'simulator_config': os.path.abspath(args.sim_config),
            'network': world.map,
            'intersection_id': inter.id,
            'in_lanes': in_lanes,
            'lane_length_m': lengths,
            'jam_spacing_m': JAM_SPACING_M,
            'lane_capacity_veh': {
                lane: lengths[lane] / JAM_SPACING_M for lane in in_lanes
            },
            'phase_served_lanes': served,
            'green_phase_count': len(inter.green_phases),
            'phase_durations_s': durations,
            'min_phase_s': min(durations),
        }
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print(json.dumps({k: payload[k] for k in (
            'network', 'intersection_id', 'in_lanes', 'green_phase_count',
            'min_phase_s')}, indent=1))
    finally:
        world.close()


if __name__ == '__main__':
    main()
