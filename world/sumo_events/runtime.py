"""SUMO execution and synchronized factual reports for three event forms.

Execution sources: T-REX obstacle operator (trex_blockage.py), SUMO TraCI
Change_Lane_State (permissions and max speed), and Variable_Speed_Signs
(scheduled speed-limit semantics). No LLM, policy, or training dependency.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import sumolib

from .schema import Report, Schedule
from . import trex_blockage
from utils.scene_context import scene_context_at as build_scene_context


MOTOR_CLASSES = ('passenger', 'taxi', 'bus', 'coach', 'delivery', 'truck',
                 'trailer', 'motorcycle', 'moped', 'emergency', 'evehicle')
TURN_NAMES = {'l': 'left-turn', 's': 'through', 'r': 'right-turn', 't': 'U-turn'}


class SumoEventRuntime:
    """Connection-injected runtime with deterministic reset and no report lag.

    Direct users call bind(engine) on a fresh simulation, then synchronize()
    after EVERY simulationStep (including the episode horizon). World users
    should use install_events instead. reports()/texts() are policy-facing;
    audit() contains privileged schedule/target information for validation only.

    Direct connections must disable timeout teleport (--time-to-teleport -1).
    For dynamic closures, validate_routes first and use --ignore-route-errors
    true to keep otherwise valid planned routes through temporarily closed
    edges. install_events performs these checks/settings for the hz4x4 World.
    """

    def __init__(self, net_file, schedule):
        if not isinstance(schedule, Schedule):
            raise TypeError('Expected a Schedule')
        self.schedule = schedule
        self.net_file = str(Path(net_file).resolve())
        self.net_sha256 = hashlib.sha256(Path(net_file).read_bytes()).hexdigest()
        self.schedule_sha256 = hashlib.sha256(json.dumps(schedule.to_dict(), sort_keys=True).encode()).hexdigest()
        net = sumolib.net.readNet(self.net_file, withInternal=True)
        self.intersection_ids = tuple(sorted(t.getID() for t in net.getTrafficLights()))
        self._lanes = {}
        self._edges = {}
        for edge in net.getEdges(withInternal=True):
            self._edges[edge.getID()] = tuple(lane.getID() for lane in edge.getLanes())
            for lane in edge.getLanes():
                shape = lane.getShape()
                dx, dy = shape[-1][0] - shape[0][0], shape[-1][1] - shape[0][1]
                direction = (('eastbound' if dx > 0 else 'westbound') if abs(dx) > abs(dy)
                             else ('northbound' if dy > 0 else 'southbound'))
                self._lanes[lane.getID()] = {
                    'lane_id': lane.getID(), 'edge_id': edge.getID(), 'lane_index': lane.getIndex(),
                    'length': lane.getLength(), 'from_junction': edge.getFromNode().getID(),
                    'to_junction': edge.getToNode().getID(), 'direction': direction,
                    'movements': sorted({TURN_NAMES[c.getDirection()] for c in lane.getOutgoing()
                                         if c.getDirection() in TURN_NAMES}),
                    'internal': edge.isSpecial(),
                    'motor_vehicle_lane': any(lane.allows(c) for c in MOTOR_CLASSES),
                }
        self._rain_lanes = tuple(sorted(key for key, lane in self._lanes.items() if lane['motor_vehicle_lane']))
        self._validate_targets()
        self._engine = None
        self._failed = None
        self._time = None
        self._states = {}
        self._reports = {}
        self._positions = {}
        self._obstacles = {}
        self._transitions = []
        self._closed = set()
        self._factor = 1.0
        self._base_speed = {}
        self._base_permissions = {}

    def _validate_targets(self):
        if any(e.kind == 'global_rain' for e in self.schedule.events) and not self._rain_lanes:
            raise ValueError('Network has no motor-vehicle lanes for global_rain')
        for event in self.schedule.events:
            if event.kind == 'lane_blockage':
                lane = self._lanes.get(event.lane_id)
                if lane is None or lane['internal'] or not lane['motor_vehicle_lane']:
                    raise ValueError(f'Invalid external motor-vehicle lane: {event.lane_id}')
                if event.position >= lane['length']:
                    raise ValueError(f'Obstacle position exceeds lane length: {event.event_id}')
            elif event.kind == 'road_closure':
                if event.edge_id not in self._edges or event.edge_id.startswith(':'):
                    raise ValueError(f'Invalid directed road: {event.edge_id}')

    def network_catalog(self):
        """Static road/lane facts usable by downstream text grounding; no events."""
        return json.loads(json.dumps(self._lanes))

    def set_schedule(self, schedule):
        """Replace the schedule before the next bind; forbidden while bound.

        Episode plans select one immutable Schedule per reset.  Swapping is
        only legal between episodes (unbound), so a bound simulation never
        sees its schedule change mid-episode.
        """
        if not isinstance(schedule, Schedule):
            raise TypeError('Expected a Schedule')
        if self._engine is not None:
            raise RuntimeError('Cannot change the schedule of a bound runtime')
        self.schedule = schedule
        self.schedule_sha256 = hashlib.sha256(
            json.dumps(schedule.to_dict(), sort_keys=True).encode()).hexdigest()
        self._validate_targets()

    def bind(self, engine):
        """Bind to a new SUMO episode at time zero; discard old connection state."""
        now = float(engine.simulation.getTime())
        if abs(now) > 1e-8:
            raise ValueError('Bind events immediately after starting SUMO, at time 0')
        step = float(engine.simulation.getDeltaT())
        for event in self.schedule.events:
            for boundary in (event.begin, event.end):
                if not math.isclose(boundary / step, round(boundary / step), abs_tol=1e-7, rel_tol=0):
                    raise ValueError(f'{event.event_id} boundary {boundary} is not aligned to SUMO step {step}')
        live_lanes = set(engine.lane.getIDList())
        if live_lanes != set(self._lanes):
            raise ValueError('Running SUMO lane IDs do not match the supplied network')
        for domain in (engine.vehicle, engine.vehicletype, engine.route):
            if any(v.startswith(trex_blockage.PREFIX) for v in domain.getIDList()):
                raise ValueError('Reserved event IDs already exist; bind to a fresh episode')
        self._engine = engine
        self._step = step
        self._failed = None
        self._time = None
        self._states = {e.event_id: 'pending' for e in self.schedule.events}
        self._reports = {}
        self._positions = {}
        self._obstacles = {}
        self._transitions = []
        self._closed = set()
        self._factor = 1.0
        speed_lanes = self._rain_lanes if any(e.kind == 'global_rain' for e in self.schedule.events) else ()
        self._base_speed = {lane: float(engine.lane.getMaxSpeed(lane)) for lane in speed_lanes}
        closure_lanes = {lane for e in self.schedule.events if e.kind == 'road_closure'
                         for lane in self._edges[e.edge_id]}
        self._base_permissions = {lane: tuple(engine.lane.getDisallowed(lane)) for lane in closure_lanes}
        self.synchronize()

    def _check_ready(self):
        if self._engine is None:
            raise RuntimeError('Event runtime is not bound; reset the world first')
        if self._failed is not None:
            raise RuntimeError(f'Event runtime failed; reset required: {self._failed}')

    def synchronize(self):
        """Apply all changes at current time, then publish confirmed reports.

        A failed mutation invalidates the episode and reports; callers must
        close/reset the simulation. No failed event is reported as active.
        """
        self._check_ready()
        now = float(self._engine.simulation.getTime())
        if self._time is not None and (now < self._time or now - self._time > self._step + 1e-7):
            self._failed = 'Simulation steps were skipped or time moved backwards'
            raise RuntimeError(self._failed)
        if self._time == now:
            return
        try:
            active = [e for e in self.schedule.events if e.begin <= now < e.end]
            active_ids = {e.event_id for e in active}
            for event_id in list(self._obstacles):
                if event_id not in active_ids:
                    trex_blockage.remove(self._engine, self._obstacles[event_id])
                    del self._obstacles[event_id]
            for event in active:
                if event.kind == 'lane_blockage' and event.event_id not in self._obstacles:
                    target = self._lanes[event.lane_id]
                    other_positions = [self._positions[e.event_id] for e in active
                                       if e.lane_id == event.lane_id and e.event_id in self._obstacles]
                    position = trex_blockage.choose_position(self._engine, event.lane_id,
                                                            event.position, event.position_tolerance,
                                                            other_positions)
                    self._obstacles[event.event_id] = trex_blockage.insert(
                        self._engine, event, target['edge_id'], target['lane_index'], position, now, self._step)
                    self._positions[event.event_id] = position
            closed = {lane for event in active if event.kind == 'road_closure'
                      for lane in self._edges[event.edge_id]}
            for lane in sorted(closed ^ self._closed):
                self._engine.lane.setDisallowed(lane, ['all'] if lane in closed else self._base_permissions[lane])
                actual = self._engine.lane.getDisallowed(lane)
                if lane in closed:
                    if not set(MOTOR_CLASSES) <= set(actual):
                        raise RuntimeError(f'Road closure not applied: {lane}')
                elif set(actual) != set(self._base_permissions[lane]):
                    raise RuntimeError(f'Road permissions not restored: {lane}')
            self._closed = closed
            factor = min([e.speed_factor for e in active if e.kind == 'global_rain'] or [1.0])
            if factor != self._factor:
                for lane, speed in self._base_speed.items():
                    expected = speed * factor
                    self._engine.lane.setMaxSpeed(lane, expected)
                    if not math.isclose(self._engine.lane.getMaxSpeed(lane), expected, abs_tol=1e-7):
                        raise RuntimeError(f'Rain speed change not applied: {lane}')
                self._factor = factor
            self._verify_obstacles()
            # Publish only after the entire batch is confirmed in SUMO.
            for event in self.schedule.events:
                status = 'active' if event.event_id in active_ids else ('cleared' if now >= event.end else 'pending')
                if status != self._states[event.event_id]:
                    self._states[event.event_id] = status
                    report = Report(event.event_id, status, now, self._render(event, status))
                    self._reports[event.event_id] = report
                    self._transitions.append(asdict(report))
            self._time = now
        except Exception as error:
            self._failed = f'{type(error).__name__}: {error}'
            raise

    def _verify_obstacles(self):
        for event in self.schedule.events:
            if event.event_id in self._obstacles:
                vehicle = self._obstacles[event.event_id]
                if (self._engine.vehicle.getLaneID(vehicle) != event.lane_id
                        or abs(self._engine.vehicle.getLanePosition(vehicle) - self._positions[event.event_id]) > 1e-5
                        or abs(self._engine.vehicle.getSpeed(vehicle)) > 1e-5):
                    raise RuntimeError(f'Obstacle disappeared or moved: {event.event_id}')

    def _render(self, event, status):
        prefix = f'Event {event.event_id}. '
        if event.kind == 'lane_blockage':
            lane = self._lanes[event.lane_id]
            label = '/'.join(lane['movements']) or 'unspecified-movement'
            where = (f"lane {event.lane_id} ({label} lane), on {lane['direction']} road {lane['edge_id']} "
                     f"approaching junction {lane['to_junction']}")
            if status == 'active':
                return prefix + (f'A stationary obstacle locally blocks {where}, '
                                 f'at {self._positions[event.event_id]:.2f} m from the lane start.')
            return prefix + f'The obstacle on {where} has been removed.'
        if event.kind == 'road_closure':
            lane = self._lanes[self._edges[event.edge_id][0]]
            where = (f"directed road {event.edge_id}, from junction {lane['from_junction']} "
                     f"to junction {lane['to_junction']}")
            if status == 'active':
                return prefix + f'All lanes of {where} are closed to vehicle entry.'
            return prefix + f'The entry restriction imposed by this event on {where} has been removed.'
        if status == 'active':
            return prefix + (f'Rain affects the whole network. This event limits motor-vehicle lane speeds '
                             f'to {100 * event.speed_factor:g}% of their normal limits.')
        return prefix + 'This network-wide rain event has ended and its speed restriction has been removed.'

    def reports(self):
        """Latest message per occurred event until reset; pending events stay hidden."""
        self._check_ready()
        if self._time != float(self._engine.simulation.getTime()):
            raise RuntimeError('Synchronize events before reading reports')
        return tuple(self._reports[key] for key in sorted(self._reports))

    def scene_context_at(self, now=None):
        """Return the unique public Normal/Event scene at the synchronized time.

        The runtime still retains cleared reports in ``reports()`` and audit
        transitions.  ``SceneContext.events`` contains active reports only;
        after an event's end the control scene is Normal even though the
        cleared report remains available for logging.
        """
        self._check_ready()
        if now is None:
            now = self._time
        now = float(now)
        if not math.isclose(now, self._time, abs_tol=1e-8):
            raise ValueError('Only the current synchronized scene timestamp is available')
        return build_scene_context(now, self.reports())

    def texts(self, now, intersection_ids):
        """Compatibility with TARL text input; broadcast public reports to all nodes.

        Downstream entity mapping must parse text + network_catalog; this method
        deliberately does not route reports with simulator-truth entity masks.
        """
        reports = self.reports()
        if not math.isclose(float(now), self._time, abs_tol=1e-8):
            raise ValueError('Only the current synchronized report snapshot is available')
        if not set(intersection_ids) <= set(self.intersection_ids):
            raise ValueError('Unknown reporting intersection')
        text = ' '.join(r.text for r in reports)
        return [text for _ in intersection_ids]

    def audit(self):
        """Privileged experiment evidence; do not pass this into the policy."""
        self._check_ready()
        self.reports()
        self._verify_obstacles()
        return {
            'time': self._time, 'schedule': self.schedule.to_dict(),
            'schedule_sha256': self.schedule_sha256, 'network_sha256': self.net_sha256,
            'sumo_version': self._engine.getVersion(), 'states': dict(self._states),
            'actual_positions': dict(self._positions), 'obstacle_ids': dict(self._obstacles),
            'closed_lanes': sorted(self._closed), 'rain_factor': self._factor,
            'rain_lane_count': len(self._rain_lanes), 'transitions': [dict(r) for r in self._transitions],
            'operators': {
                'lane_blockage': 'T-REX Deployment.triggering adaptation; a91a3d8c4ac27982c75553b932eabb27acd63ca7',
                'road_closure': 'SUMO TraCI lane.setDisallowed(all); directed edge; no forced rerouting',
                'global_rain': 'SUMO TraCI lane.setMaxSpeed; network-wide speed-limit proxy; overlap=min',
            },
        }

    def close(self):
        """Restore the live connection before detaching; idempotent after success."""
        if self._engine is None:
            return
        for vehicle in tuple(self._obstacles.values()):
            if vehicle in self._engine.vehicle.getIDList():
                trex_blockage.remove(self._engine, vehicle)
        for lane, disallowed in self._base_permissions.items():
            self._engine.lane.setDisallowed(lane, disallowed)
        for lane, speed in self._base_speed.items():
            self._engine.lane.setMaxSpeed(lane, speed)
        self._engine = None
        self._obstacles.clear()
