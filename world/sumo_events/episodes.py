"""Versioned per-episode plans over the existing fixed-schedule event adapter.

Each reset installs a fresh immutable runtime on a closed World. The frozen
v1 runtime/schedule is never mutated while bound to a simulation.
"""
from bisect import bisect_left
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from common.paper_experiment import checked_json, write_json
from .integration import install_events
from .schema import Event, Schedule


class EpisodeEvents:
    def __init__(self, world, plan_path, plan_hash, output_path):
        self.world = world
        self.plan = checked_json(plan_path, plan_hash)
        self.plan_hash = plan_hash
        self.root = Path(output_path)
        self.cursor = 0
        self.selection = None
        self.current = None
        self.stream = None
        self.last = None
        self.callback = None
        self._base_reset, self._base_step_sim, self._base_close = world.reset, world.step_sim, world.close
        original_step = world.step
        vehicles = ET.parse(world.route).getroot().findall('vehicle')
        departures = [float(v.get('depart')) for v in vehicles]
        self.due_ids = {v.get('id') for v in vehicles if float(v.get('depart')) < self.plan['seconds']}
        self.departures = sorted(departures)
        self.demand = bisect_left(self.departures, self.plan['seconds'])
        if not self.demand:
            raise ValueError('Episode plan has no scheduled demand')

        def step(actions=None):
            result = original_step(actions)
            self.sample(actions)
            return result

        world.reset = self.reset
        world.step = step

    def select(self, row, role):
        self.selection = (row, role)

    def reset(self):
        self.close_stream()
        self.world.close()
        if self.selection is None:
            if self.cursor >= len(self.plan['rows']):
                raise ValueError('Training reset exceeded the frozen episode budget')
            row, role = self.plan['rows'][self.cursor], 'train'
            self.cursor += 1
        else:
            row, role = self.selection
            self.selection = None
        self.row, self.role = row, role
        command = list(self.world.sumo_cmd)
        if '--seed' in command:
            command[command.index('--seed') + 1] = str(row['sumo_seed'])
        else:
            command += ['--seed', str(row['sumo_seed'])]
        self.world.sumo_cmd = command
        # Restore only the hooks owned by install_events; keep signal-control
        # reset and the ordinary World implementation as the reusable base.
        self.world.reset = self._base_reset
        self.world.step_sim = self._base_step_sim
        self.world.close = self._base_close
        self.world.__dict__.pop('_sumo_event_runtime', None)
        schedule = Schedule(tuple(Event(**e) for e in row['schedule']['events']), row['schedule']['schema_version'])
        self.current = install_events(self.world, schedule)
        from .safe_placement import install_safe_placement, LEGACY
        install_safe_placement(self.current, self.plan.get('event_execution', {}).get('placement', LEGACY))
        if row.get('clearance_replay') is not None:
            from .safe_placement import install_clearance_replay
            install_clearance_replay(self.current, row['clearance_replay'])
        catalog = self.current.network_catalog()
        self._lane_edges = {lane: info['edge_id'] for lane, info in catalog.items()}
        self._previous_lanes = {}
        self.block_lanes = {e.lane_id for e in schedule.events if e.kind == 'lane_blockage'}
        closed_edges = {e.edge_id for e in schedule.events if e.kind == 'road_closure'}
        upstream = {d['from_junction'] for d in catalog.values() if d['edge_id'] in closed_edges}
        self.closure_area = {lane for lane, d in catalog.items() if not d['internal'] and
                             (d['edge_id'] in closed_edges or d['to_junction'] in upstream)}
        installed_reset = self.world.reset
        self.world.reset = self.reset
        result = installed_reset()
        from .safe_placement import install_removal_view
        install_removal_view(self.world, self.current)
        self.departed, self.arrived = set(), set()
        self.vehicle_seconds = self.queue_seconds = self.removed_vehicle_seconds = 0
        self.last = None
        self.finished = False
        self.vehicle_type = None
        self.trace = hashlib.sha256()
        self.prefixes = {}
        self.prefixes_before = {'0': hashlib.sha256().hexdigest()}
        self.pre_event_trace = hashlib.sha256()
        self.first_event = min([e.begin for e in schedule.events] +
                              [v['time'] for v in row.get('clearance_replay', [])], default=float('inf'))
        self.timeline = self.root / 'environment' / role / (row['case_id'] + '.csv')
        self.timeline.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.timeline.open('w', newline='')
        self.writer = None
        return result

    def sample(self, actions):
        if self.current is None or self.finished:
            return
        world, raw = self.world, self.current._engine
        now = world.get_current_time()
        if self.last is not None and now == self.last['simulation_time']:
            return
        if raw.simulation.getCollidingVehiclesNumber() or raw.simulation.getStartingTeleportNumber():
            raise RuntimeError('Episode physical gate: collision or teleport')
        self.departed.update(world.last_entered_vehicle_ids)
        self.arrived.update(world.last_exited_vehicle_ids)
        removed = {v['vehicle'] for v in getattr(self.current, 'removed_vehicles', [])}
        vehicles = sorted(world.eng.vehicle.getIDList())
        if (removed & self.arrived or not removed <= self.departed
                or self.departed - self.arrived - removed != set(vehicles)):
            raise RuntimeError('Episode physical gate: vehicle conservation failed')
        pending = bisect_left(self.departures, now) - len(self.departed)
        if pending < 0:
            raise RuntimeError('Episode demand schedule disagrees with actual departures')
        if vehicles and self.vehicle_type is None:
            self.vehicle_type = raw.vehicle.getTypeID(vehicles[0])
            if self.vehicle_type != 'pkw':
                raise RuntimeError('Frozen pkw vehicle type was not applied')
        positions = [(v, raw.vehicle.getLaneID(v), raw.vehicle.getLanePosition(v), raw.vehicle.getSpeed(v))
                     for v in vehicles]
        lane_queues = Counter(p[1] for p in positions if p[3] < .1)
        queue = sum(lane_queues.values())
        self.vehicle_seconds += len(vehicles) + pending
        # Cleared traffic remains unfinished demand through the horizon. This
        # prevents forced injection from improving J merely by deleting cars.
        self.removed_vehicle_seconds += len(removed)
        self.queue_seconds += queue
        reports = self.current.reports()
        active = sum(r.status == 'active' for r in reports)
        expected = sum(e.begin <= now < e.end for e in self.current.schedule.events)
        if active != expected:
            raise RuntimeError('Episode report timing gate failed')
        signals = [raw.trafficlight.getRedYellowGreenState(i) for i in world.intersection_ids]
        snapshot = json.dumps([now, positions, signals], separators=(',', ':')).encode()
        self.trace.update(snapshot)
        if now % self.plan['action_interval'] == 0:
            self.prefixes[str(int(now))] = self.trace.hexdigest()
        if (now + 1) % self.plan['action_interval'] == 0:
            self.prefixes_before[str(int(now + 1))] = self.trace.hexdigest()
        if now < self.first_event:
            self.pre_event_trace.update(snapshot)
        self.last = {'simulation_time': now, 'running': len(vehicles), 'pending_due': pending,
                     'sum_speed_mps': sum(p[3] for p in positions),
                     'mean_speed_mps': sum(p[3] for p in positions) / len(vehicles) if vehicles else 0.,
                     'queue_vehicles': queue, 'arrived': len(self.arrived), 'departed': len(self.departed),
                     'active_reports': active, 'forced_removed': len(removed),
                     'system_vehicle_seconds': self.vehicle_seconds + self.removed_vehicle_seconds,
                     'system_time_per_vehicle': (self.vehicle_seconds + self.removed_vehicle_seconds) / self.demand,
                     'physical_system_time_per_vehicle': self.vehicle_seconds / self.demand,
                     'removed_vehicle_seconds': self.removed_vehicle_seconds,
                     'queue_vehicle_seconds': self.queue_seconds,
                     'blockage_lane_queue': sum(lane_queues[l] for l in self.block_lanes),
                     'closure_area_queue': sum(lane_queues[l] for l in self.closure_area),
                     # Normal controls retain lane counts for later paired
                     # local-region analysis without rerunning the simulation.
                     'lane_queues_json': json.dumps(lane_queues, sort_keys=True, separators=(',', ':'))}
        if self.plan.get('record_lane_flow'):
            current = {v: lane for v, lane, _, _ in positions}
            entries, exits = Counter(), Counter()
            for v, lane in current.items():
                old = self._previous_lanes.get(v)
                if old is None or self._lane_edges.get(old) != self._lane_edges.get(lane):
                    entries[lane] += 1
            for v, lane in self._previous_lanes.items():
                new = current.get(v)
                if v not in removed and (new is None or self._lane_edges.get(new) != self._lane_edges.get(lane)):
                    exits[lane] += 1
            self.last['lane_entries_json'] = json.dumps(entries, sort_keys=True, separators=(',', ':'))
            self.last['lane_exits_json'] = json.dumps(exits, sort_keys=True, separators=(',', ':'))
            self._previous_lanes = current
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(self.last))
            self.writer.writeheader()
        self.writer.writerow(self.last)
        if now % 10 == 0:
            self.stream.flush()
            if self.callback:
                self.callback(self.last)

    def finish(self):
        if self.last is None or self.last['simulation_time'] != self.plan['seconds']:
            raise RuntimeError('Episode ended before its frozen horizon')
        if self.finished:
            return self.result
        if any(r.status != 'cleared' for r in self.current.reports()):
            raise RuntimeError('Episode ended before every event cleared')
        if self.due_ids - self.departed - set(self.current._engine.simulation.getPendingVehicles()):
            raise RuntimeError('Scheduled demand was dropped instead of waiting for insertion')
        self.close_stream()
        self.result = dict(self.last, role=self.role, case_id=self.row['case_id'],
                           kind=self.row['kind'], sumo_seed=self.row['sumo_seed'],
                           target_split=self.row.get('target_split', 'train'),
                           local_regions={'blockage_lanes': sorted(self.block_lanes),
                                          'closure_area_lanes': sorted(self.closure_area)},
                           scheduled_demand=self.demand, mean_queue=self.queue_seconds / self.plan['seconds'],
                           completed_trip_time=self.world.get_average_travel_time(),
                           schedule=self.current.schedule.to_dict(), events=self.current.audit(),
                           vehicle_type=self.vehicle_type, sumo_command=list(self.world.sumo_cmd),
                           sumo_version=list(self.current._engine.getVersion()),
                           trace_sha256=self.trace.hexdigest(), pre_event_trace_sha256=self.pre_event_trace.hexdigest(),
                           trace_prefixes=self.prefixes,
                           trace_prefixes_before=self.prefixes_before,
                           pre_event_trace_boundary='strictly_before_onset',
                           system_time_accounting='running+pending_due+forced_removed_until_horizon',
                           timeline=str(self.timeline))
        if 'severity' in self.row:
            self.result['severity'] = self.row['severity']
        if 'clearance_replay' in self.row:
            self.result['clearance_replay'] = self.row['clearance_replay']
        write_json(self.timeline.with_suffix('.json'), self.result)
        self.finished = True
        return self.result

    def close_stream(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
