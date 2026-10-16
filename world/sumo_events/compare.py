"""Paired quantitative comparison of the three event operators on hz4x4.

Run in colight from the repository root:
    python -m world.sumo_events.compare --output data/output_data/sumo_events/comparison

Five conditions share routes, seed, horizon, controller and SUMO options.
Metrics exclude artificial obstacle vehicles. Completed-trip travel time is
reported with throughput and unfinished vehicles because it is censored by the
episode horizon. These are mechanism checks, not evidence of NLP effectiveness.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from common.registry import Registry
from world.world_sumo import World
from . import Schedule, install_events, load_schedule
from .trex_blockage import PREFIX


CONDITIONS = ('normal', 'lane_blockage', 'road_closure', 'global_rain', 'combined')
LABELS = {'normal': '无事件', 'lane_blockage': '局部车道阻塞', 'road_closure': '整段道路封闭',
          'global_rain': '全局降雨', 'combined': '三类事件组合'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_window(rows):
    """All rows are one second. Speed is vehicle-time weighted, not mean-of-means."""
    total_vehicle_seconds = sum(row['vehicles'] for row in rows)
    return {
        'mean_queue_vehicles': statistics.mean(row['queue_vehicles'] for row in rows),
        'queued_vehicle_seconds': sum(row['queue_vehicles'] for row in rows),
        'vehicle_seconds': total_vehicle_seconds,
        'mean_speed_mps': (sum(row['sum_speed_mps'] for row in rows) / total_vehicle_seconds
                           if total_vehicle_seconds else None),
        'target_lane_queue': statistics.mean(row['target_lane_queue'] for row in rows),
        'closure_area_queue': statistics.mean(row['closure_area_queue'] for row in rows),
        'closure_road_entries': sum(row['closure_road_entries'] for row in rows),
    }


def run_condition(args, schedule, seed, condition, run_dir):
    from agent.fixedtime import FixedTimeAgent

    selected = tuple(e for e in schedule.events if condition == 'combined' or e.kind == condition)
    Registry.mapping['command_mapping']['setting'] = SimpleNamespace(param={'sumo_seed': seed})
    Registry.mapping['model_mapping']['setting'] = SimpleNamespace(param={'t_fixed': args.phase_seconds})
    Registry.mapping['logger_mapping']['path'] = SimpleNamespace(path=str(run_dir))
    world = World(args.sim_config, interface='libsumo')
    try:
        world.sumo_cmd += ['--no-step-log', 'true']
        runtime = install_events(world, Schedule(selected))
        world.reset()
        agents = [FixedTimeAgent(world, rank) for rank in range(len(world.intersections))]
        world._update_infos()
        catalog = runtime.network_catalog()
        by_kind = {e.kind: e for e in schedule.events}
        block, closure, rain = (by_kind[k] for k in ('lane_blockage', 'road_closure', 'global_rain'))
        closure_lanes = sorted(lane for lane, data in catalog.items() if data['edge_id'] == closure.edge_id)
        upstream_junction = catalog[closure_lanes[0]]['from_junction']
        closure_area = {lane for lane, data in catalog.items()
                        if not data['internal'] and (data['edge_id'] == closure.edge_id
                                                   or data['to_junction'] == upstream_junction)}
        rain_lanes = sorted(lane for lane, data in catalog.items() if data['motor_vehicle_lane'])
        raw = runtime._engine
        normal_speeds = {lane: raw.lane.getMaxSpeed(lane) for lane in rain_lanes}
        normal_permissions = {lane: raw.lane.getDisallowed(lane) for lane in closure_lanes}
        due = {e.get('id') for _, e in ET.iterparse(world.route)
               if e.tag == 'vehicle' and float(e.get('depart')) < args.seconds}
        departed, arrived, previous_road_vehicles = set(), set(), set()
        collisions = teleports = report_mismatches = 0
        action_hash, pre_event_hash = hashlib.sha256(), hashlib.sha256()
        rows, physical_checks = [], []
        first_event = min(e.begin for e in schedule.events)
        boundaries = {e.begin for e in schedule.events} | {e.end for e in schedule.events}
        actual_changes = {e.event_id: [] for e in selected}
        previous_actual = {e.kind: False for e in schedule.events}
        for _ in range(args.seconds):
            actions = [a.get_action(a.get_ob(), a.get_phase()) for a in agents]
            action_hash.update(json.dumps(actions).encode())
            world.step(actions)
            now = world.get_current_time()
            vehicles = world.eng.vehicle.getIDList()
            speeds = {v: raw.vehicle.getSpeed(v) for v in vehicles}
            vehicle_lanes = {v: raw.vehicle.getLaneID(v) for v in vehicles}
            queue = [v for v in vehicles if speeds[v] < .1]
            road_vehicles = {v for v in vehicles if vehicle_lanes[v] in closure_lanes}
            new_entries = len(road_vehicles - previous_road_vehicles)
            previous_road_vehicles = road_vehicles
            departed.update(world.last_entered_vehicle_ids)
            arrived.update(world.last_exited_vehicle_ids)
            collisions += raw.simulation.getCollidingVehiclesNumber()
            teleports += raw.simulation.getStartingTeleportNumber()
            if departed - arrived != set(vehicles) or any(v.startswith(PREFIX) for v in vehicles):
                raise AssertionError('Traffic accounting mismatch or obstacle leakage')
            reports = {report.event_id: report for report in runtime.reports()}
            obstacle_ids = [v for v in raw.vehicle.getIDList() if v.startswith(PREFIX)]
            closed_count = sum('passenger' in raw.lane.getDisallowed(lane) for lane in closure_lanes)
            speed_ratio = raw.lane.getMaxSpeed(rain_lanes[0]) / normal_speeds[rain_lanes[0]]
            actual = {'lane_blockage': bool(obstacle_ids), 'road_closure': closed_count == len(closure_lanes),
                      'global_rain': speed_ratio < 1 - 1e-8}
            report_active = {e.kind: e.event_id in reports and reports[e.event_id].status == 'active'
                             for e in schedule.events}
            report_mismatches += sum(actual[k] != report_active[k] for k in actual)
            for event in selected:
                if actual[event.kind] != previous_actual[event.kind]:
                    actual_changes[event.event_id].append((now, 'active' if actual[event.kind] else 'cleared'))
            previous_actual = actual
            row = {'time': now, 'vehicles': len(vehicles), 'queue_vehicles': len(queue),
                   'sum_speed_mps': sum(speeds.values()),
                   'mean_speed_mps': sum(speeds.values()) / len(vehicles) if vehicles else 0.,
                   'target_lane_queue': sum(vehicle_lanes[v] == block.lane_id for v in queue),
                   'closure_area_queue': sum(vehicle_lanes[v] in closure_area for v in queue),
                   'closure_road_entries': new_entries, 'arrived': len(arrived),
                   'obstacle_count': len(obstacle_ids), 'closed_lane_count': closed_count,
                   'rain_speed_ratio': speed_ratio,
                   'block_report_active': int(report_active['lane_blockage']),
                   'closure_report_active': int(report_active['road_closure']),
                   'rain_report_active': int(report_active['global_rain'])}
            rows.append(row)
            if now < first_event:
                pre_event_hash.update(json.dumps(row, sort_keys=True).encode())
            if now in boundaries or now == args.seconds:
                expected_ratio = rain.speed_factor if actual['global_rain'] else 1.
                wrong_speeds = sum(abs(raw.lane.getMaxSpeed(lane) / base - expected_ratio) > 1e-7
                                   for lane, base in normal_speeds.items())
                if wrong_speeds:
                    raise AssertionError('Network-wide speed scope mismatch')
                physical_checks.append({'time': now, 'checked_rain_lanes': len(rain_lanes),
                                        'wrong_speed_lanes': wrong_speeds, 'actual': dict(actual),
                                        'reports': [asdict(r) for r in reports.values()]})
        pending = set(raw.simulation.getPendingVehicles())
        if due - departed - pending or collisions or teleports or report_mismatches:
            raise AssertionError('Dropped demand, collisions, teleports or report mismatch')
        for lane, allowed in normal_permissions.items():
            if set(raw.lane.getDisallowed(lane)) != set(allowed):
                raise AssertionError('Road permissions did not restore')
        audit = runtime.audit()
        report_changes = {(r['event_id'], r['status']): r['updated_at'] for r in audit['transitions']}
        lags = [report_changes[(event_id, state)] - time for event_id, changes in actual_changes.items()
                for time, state in changes]
        if len(lags) != 2 * len(selected):
            raise AssertionError('Missing event transition')
        begin, end = schedule.events[0].begin, schedule.events[0].end
        # A row at t describes the simulation interval (t-1, t]. The physical
        # intervention at t is already visible in reports, but affects motion
        # during the NEXT interval. Avoid mixing these two time conventions.
        active_rows = [r for r in rows if begin < r['time'] <= end]
        recovery_rows = [r for r in rows if end < r['time'] <= min(end + 600, args.seconds)]
        summary = {'condition': condition, 'seed': seed, 'seconds': args.seconds,
                   'sumo_command': list(world.sumo_cmd), 'sumo_version': raw.getVersion(),
                   'event_schedule': Schedule(selected).to_dict(), 'event_audit': audit,
                   'whole_episode': summarize_window(rows), 'event_window': summarize_window(active_rows),
                   'recovery_600s': summarize_window(recovery_rows),
                   'completed_trip_time_s': statistics.mean(world.vehicles.values()) if world.vehicles else None,
                   'departed': len(departed), 'arrived': len(arrived), 'running': len(vehicles), 'pending': len(pending),
                   'action_sha256': action_hash.hexdigest(), 'pre_event_trace_sha256': pre_event_hash.hexdigest(),
                   'collision_count': collisions, 'teleport_count': teleports,
                   'report_physical_mismatches': report_mismatches,
                   'report_checks': 3 * args.seconds, 'report_transition_lags_s': lags,
                   'physical_boundary_checks': physical_checks}
        write_csv(run_dir / 'timeline.csv', rows)
        (run_dir / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'seed': seed, 'condition': condition, 'arrived': len(arrived),
                          'event_queue': summary['event_window']['mean_queue_vehicles'],
                          'event_speed': summary['event_window']['mean_speed_mps']}, ensure_ascii=False), flush=True)
        return summary
    finally:
        world.close()


def aggregate(results):
    """Seed-wise paired differences; descriptive ranges, no significance claims."""
    metrics = {
        'event_queue': lambda r: r['event_window']['mean_queue_vehicles'],
        'event_speed': lambda r: r['event_window']['mean_speed_mps'],
        'target_lane_queue': lambda r: r['event_window']['target_lane_queue'],
        'closure_area_queue': lambda r: r['event_window']['closure_area_queue'],
        'closure_road_entries': lambda r: r['event_window']['closure_road_entries'],
        'completed_trip_time': lambda r: r['completed_trip_time_s'],
        'arrived': lambda r: r['arrived'], 'running': lambda r: r['running'],
        'whole_queue': lambda r: r['whole_episode']['mean_queue_vehicles'],
    }
    baseline = {r['seed']: r for r in results if r['condition'] == 'normal'}
    records = []
    for condition in CONDITIONS:
        selected = sorted((r for r in results if r['condition'] == condition), key=lambda r: r['seed'])
        for name, extract in metrics.items():
            values = [extract(r) for r in selected]
            deltas = [extract(r) - extract(baseline[r['seed']]) for r in selected]
            records.append({'condition': condition, 'metric': name, 'n_seeds': len(values),
                            'mean': statistics.mean(values), 'minimum': min(values), 'maximum': max(values),
                            'paired_delta_mean': statistics.mean(deltas),
                            'paired_delta_min': min(deltas), 'paired_delta_max': max(deltas),
                            'values': values, 'paired_deltas': deltas})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sim-config', default='configs/sim/hz4x4.cfg')
    parser.add_argument('--schedule', default='configs/events/hz4x4_comparison.yml')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--phase-seconds', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-only', action='store_true', help='Regenerate figures from existing data')
    args = parser.parse_args()
    if args.render_only:
        from .comparison_plots import render
        render(args.output)
        return
    schedule = load_schedule(args.schedule)
    if (len(schedule.events) != 3 or {e.kind for e in schedule.events} != set(CONDITIONS[1:4])
            or len({(e.begin, e.end) for e in schedule.events}) != 1):
        raise ValueError('Comparison requires one event per kind with a shared time window')
    if min(e.begin for e in schedule.events) <= 0 or args.seconds <= max(e.end for e in schedule.events):
        raise ValueError('Include a pre-event period and a post-event recovery period')
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds) or args.phase_seconds <= 0:
        raise ValueError('Use at least two distinct seeds and positive phase duration')
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.sim_config).read_text())
    sources = list(Path(__file__).parent.glob('*.py')) + [Path(args.schedule), Path(args.sim_config),
              Path('world/world_sumo.py'), Path('agent/fixedtime.py'),
              Path(config['dir']) / config['roadnetFile'], Path(config['dir']) / config['flowFile']]
    manifest = {'invocation': getattr(sys, 'orig_argv', [sys.executable] + sys.argv),
                'settings': dict(vars(args), output=str(args.output)), 'schedule': schedule.to_dict(),
                'scope': 'controlled_event_mechanism_comparison',
                'source_sha256': {str(p): digest(p) for p in sources},
                'metric_definitions': {
                    'queue': 'all ordinary running vehicles with speed < 0.1 m/s; per-second network counts',
                    'speed': 'sum of ordinary vehicle speeds divided by ordinary vehicle-seconds in the window',
                    'trip_time': 'mean actual-departure-to-arrival time of completed trips only (horizon-censored)',
                    'active_window': '(begin, end] for traffic samples; [begin, end) for physical/report state',
                    'closure_area': 'closed edge plus all external incoming edges of its upstream junction',
                    'error_bands': 'range across seeds, not a confidence interval',
                }}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    results = []
    for seed in args.seeds:
        for condition in CONDITIONS:
            run_dir = args.output / f'seed_{seed}' / condition
            run_dir.mkdir(parents=True)
            results.append(run_condition(args, schedule, seed, condition, run_dir))
        paired = [r for r in results if r['seed'] == seed]
        if len({r['action_sha256'] for r in paired}) != 1 or len({r['pre_event_trace_sha256'] for r in paired}) != 1:
            raise AssertionError('Conditions differ in controller actions or pre-event traffic')
    aggregated = aggregate(results)
    summary = {'status': 'completed', 'runs': len(results), 'seeds': args.seeds,
               'conditions': list(CONDITIONS), 'metrics': aggregated,
               'event_window': [schedule.events[0].begin, schedule.events[0].end],
               'all_actions_matched_within_seed': True, 'all_pre_event_traffic_matched_within_seed': True,
               'max_report_lag_s': max(abs(lag) for r in results for lag in r['report_transition_lags_s']),
               'report_physical_mismatches': sum(r['report_physical_mismatches'] for r in results),
               'collision_count': sum(r['collision_count'] for r in results),
               'teleport_count': sum(r['teleport_count'] for r in results)}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    write_csv(args.output / 'comparison.csv', [{k: v for k, v in r.items() if isinstance(v, (str, int, float))}
                                              for r in aggregated])
    from .comparison_plots import render
    render(args.output)


if __name__ == '__main__':
    main()
