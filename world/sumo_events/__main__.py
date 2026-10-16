"""Reproducible integration validation using the existing hz4x4 World/agent.

Run from the repository root in the colight environment::

    python -m world.sumo_events --seeds 7 17 --repeats 2 --seconds 300 \
        --output data/output_data/sumo_events/hz4x4_validation

The output is validation evidence, not a trained policy or research result.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from common.registry import Registry
from world.world_sumo import World
from . import install_events, load_schedule
from .trex_blockage import PREFIX


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(args):
    # Lazy import keeps the event runtime independent of RL/PyTorch packages.
    from agent.fixedtime import FixedTimeAgent

    schedule = load_schedule(args.schedule)
    if args.seconds <= 0 or args.phase_seconds <= 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Positive durations and distinct seeds are required')
    if args.seconds < max((e.end for e in schedule.events), default=0) + 1:
        raise ValueError('--seconds must include at least one step after every event ends')
    if args.repeats < 2:
        raise ValueError('Use at least two repeats to verify reset/replay')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in args.seeds:
        run_dir = output / f'seed_{seed}'
        run_dir.mkdir()
        Registry.mapping['command_mapping']['setting'] = SimpleNamespace(param={'sumo_seed': seed})
        Registry.mapping['model_mapping']['setting'] = SimpleNamespace(param={'t_fixed': args.phase_seconds})
        Registry.mapping['logger_mapping']['path'] = SimpleNamespace(path=str(run_dir))
        world = World(args.sim_config, interface=args.interface)
        world.sumo_cmd += ['--no-step-log', 'true']
        runtime = install_events(world, schedule)
        due_vehicles = {e.get('id') for _, e in ET.iterparse(world.route)
                        if e.tag == 'vehicle' and float(e.get('depart')) < args.seconds}
        rows = []
        try:
            for repeat in range(args.repeats):
                world.reset()
                agents = [FixedTimeAgent(world, rank) for rank in range(len(world.intersections))]
                world._update_infos()
                raw = runtime._engine
                initial_speed = {lane: raw.lane.getMaxSpeed(lane) for lane in raw.lane.getIDList()}
                # Schedules beginning at zero may already affect speeds/permissions.
                initial_speed.update(runtime._base_speed)
                trace_hash = hashlib.sha256()
                departed, arrived = set(), set()
                collision_count = teleport_count = 0
                sample_rows = []
                for _ in range(args.seconds):
                    actions = [agent.get_action(agent.get_ob(), agent.get_phase()) for agent in agents]
                    world.step(actions)
                    now = world.get_current_time()
                    reports = runtime.reports()
                    collision_count += raw.simulation.getCollidingVehiclesNumber()
                    teleport_count += raw.simulation.getStartingTeleportNumber()
                    departed.update(world.last_entered_vehicle_ids)
                    arrived.update(world.last_exited_vehicle_ids)
                    traffic = tuple(world.eng.vehicle.getIDList())
                    if any(v.startswith(PREFIX) for v in traffic + tuple(world.inside_vehicles) + tuple(world.vehicle_trajectory)):
                        raise AssertionError('Obstacle leaked into traffic observations or trip accounting')
                    if departed - arrived != set(traffic):
                        raise AssertionError('Departed traffic disappeared without arrival')
                    positions = [(v, raw.vehicle.getLaneID(v), round(raw.vehicle.getLanePosition(v), 8),
                                  round(raw.vehicle.getSpeed(v), 8)) for v in sorted(traffic)]
                    trace_hash.update(json.dumps([now, actions, positions,
                                                  [asdict(r) for r in reports]], sort_keys=True).encode())
                    for intersection in world.intersections:
                        for lane in intersection.full_observation.values():
                            if any(v['name'].startswith(PREFIX) for v in lane['vehicles']):
                                raise AssertionError('Obstacle leaked into lane features/rewards')
                    if any(now == e.begin or now == e.end for e in schedule.events):
                        sample_rows.append({'time': now, 'reports': [asdict(r) for r in reports],
                                            'active_obstacles': dict(runtime._obstacles),
                                            'closed_lanes': sorted(runtime._closed), 'rain_factor': runtime._factor})
                evidence = runtime.audit()
                expected_transitions = {(e.event_id, s, t) for e in schedule.events
                                        for s, t in [('active', e.begin), ('cleared', e.end)]}
                actual_transitions = {(r['event_id'], r['status'], r['updated_at']) for r in evidence['transitions']}
                if actual_transitions != expected_transitions:
                    raise AssertionError('Physical/report transitions do not match schedule times')
                if collision_count or teleport_count:
                    raise AssertionError('Collisions or teleports during event validation')
                if runtime._obstacles or runtime._closed or runtime._factor != 1:
                    raise AssertionError('Event effects did not clear')
                for lane, speed in initial_speed.items():
                    if abs(raw.lane.getMaxSpeed(lane) - speed) > 1e-7:
                        raise AssertionError('Original speed not restored')
                for lane, permissions in runtime._base_permissions.items():
                    if set(raw.lane.getDisallowed(lane)) != set(permissions):
                        raise AssertionError('Original permissions not restored')
                pending = set(raw.simulation.getPendingVehicles())
                if due_vehicles - departed - pending:
                    raise AssertionError('Scheduled demand was dropped instead of departing/waiting')
                row = {
                    'seed': seed, 'repeat': repeat, 'simulator': 'SUMO', 'interface': args.interface,
                    'agent': 'FixedTimeAgent', 'network': world.net, 'seconds': args.seconds,
                    'phase_seconds': args.phase_seconds, 'sumo_command': list(world.sumo_cmd),
                    'trace_sha256': trace_hash.hexdigest(), 'departed': len(departed), 'arrived': len(arrived),
                    'pending': len(pending), 'scheduled_due': len(due_vehicles),
                    'collision_count': collision_count, 'teleport_count': teleport_count,
                    'validated_baseline_routes': runtime.validated_routes,
                    'events': evidence, 'boundary_samples': sample_rows,
                }
                rows.append(row)
                (run_dir / f'repeat_{repeat}.json').write_text(json.dumps(row, indent=2) + '\n')
                print(json.dumps({k: row[k] for k in ('seed', 'repeat', 'seconds', 'departed', 'arrived', 'trace_sha256')}), flush=True)
            if len({row['trace_sha256'] for row in rows}) != 1:
                raise AssertionError('Reset/replay with the same seed produced a different traffic/text trace')
            results.extend(rows)
        finally:
            world.close()
    module_files = sorted(Path(__file__).parent.glob('*.py'))
    files = module_files + [Path(__file__).parent / 'LICENSE.T-REX', Path(args.schedule),
                            Path(args.sim_config), Path('world/world_sumo.py'), Path('tests/test_sumo_events.py')]
    summary = {
        'status': 'passed', 'scope': 'engineering_integration_validation',
        'invocation': getattr(sys, 'orig_argv', [sys.executable] + sys.argv),
        'settings': vars(args), 'files_sha256': {str(p): _digest(p) for p in files},
        'runs': [{k: r[k] for k in ('seed', 'repeat', 'seconds', 'departed', 'arrived', 'trace_sha256')}
                 for r in results],
        'checks': ['all_three_physical_events', 'same_time_reports', 'restored_speed_and_permissions',
                   'no_obstacle_in_traffic_metrics', 'no_dropped_due_demand', 'no_collisions_or_teleports',
                   'same_seed_reset_replay'],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sim-config', default='configs/sim/hz4x4.cfg')
    parser.add_argument('--schedule', default='configs/events/hz4x4.yml')
    parser.add_argument('--interface', choices=('libsumo', 'traci'), default='libsumo')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--phase-seconds', type=int, default=20)
    parser.add_argument('--output', required=True, help='New evidence directory (refuses overwriting)')
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()
