"""Opt-in dynamic obstacle placement over the frozen v1 event lifecycle.

The adapter owns only obstacle placement. Timing, closure/rain, reporting,
traffic filtering and cleanup continue to use the existing runtime.
"""
import copy
import math

from . import trex_blockage as obstacle

VERSION = 'stopping-distance-v1'
FORCED_VERSION = 'clear-conflicts-v1'
LEGACY = {'version': 'static-gap-v1'}


class UnsafePlacementError(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('No dynamically safe obstacle position: ' + str(diagnostics))


def validate_policy(policy):
    if policy == LEGACY:
        return dict(policy)
    if isinstance(policy, dict):
        policy = dict(policy)
        policy.setdefault('obstacle_length_m', obstacle.LENGTH)
    if not isinstance(policy, dict) or set(policy) != {'version', 'clearance_m', 'reaction_steps', 'obstacle_length_m'}:
        raise ValueError('placement requires version, clearance_m and reaction_steps')
    if policy['version'] not in (VERSION, FORCED_VERSION):
        raise ValueError('Unknown obstacle placement version')
    for key in ('clearance_m', 'reaction_steps', 'obstacle_length_m'):
        x = policy[key]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 1:
            raise ValueError('placement ' + key + ' must be finite and >= 1')
    return dict(policy)


def choose_position(engine, lane_id, requested, tolerance, policy, extra_positions=(), ignored=()):
    """Nearest feasible front position, checking every potential same-lane follower.

    Required gap = minGap + v * (max(tau, actionStep, simStep) +
    reaction_steps * simStep) + v**2/(2*decel) + clearance_m.
    Comfortable deceleration is used, never emergency braking. A separate
    simStep allowance covers boundary insertion/discrete SUMO integration.
    This function is read-only. Forced mode returns the ordinary vehicles to
    clear if no free position exists; the runtime applies that decision.
    """
    policy = validate_policy(policy)
    if policy == LEGACY:
        return obstacle.choose_position(engine, lane_id, requested, tolerance, extra_positions), {}
    step = float(engine.simulation.getDeltaT())
    length_m = policy['obstacle_length_m']
    lower = max(length_m, requested - tolerance)
    upper = min(engine.lane.getLength(lane_id) - .1, requested + tolerance)
    live = set(engine.vehicle.getIDList()) - set(ignored)
    intervals, vehicles = [], []
    # A vehicle's front can be on an internal/downstream lane while its rear
    # (or SUMO's minimum-gap envelope) still occupies the end of this lane.
    # Lane vehicle lists only locate fronts; include reachable downstream tails.
    offsets = {lane_id: 0.0}
    frontier = [lane_id]
    tail_distance = max((engine.vehicle.getLength(v) for v in live), default=0) + obstacle.GAP
    while frontier:
        lane = frontier.pop()
        end = offsets[lane] + engine.lane.getLength(lane)
        if end > upper + tail_distance:
            continue
        for link in engine.lane.getLinks(lane):
            target = link[4] or link[0]  # via internal lane before outgoing lane
            if target and target not in offsets:
                offsets[target] = end
                frontier.append(target)
    candidates_on_lanes = {v: (lane, offset) for lane, offset in offsets.items()
                           for v in engine.lane.getLastStepVehicleIDs(lane)}
    for vehicle, (actual_lane, offset) in candidates_on_lanes.items():
        if vehicle not in live:
            continue
        actual_pos = float(engine.vehicle.getLanePosition(vehicle))
        pos = offset + actual_pos
        length = float(engine.vehicle.getLength(vehicle))
        if offset and pos - length - obstacle.GAP > upper:
            continue
        speed = float(engine.vehicle.getSpeed(vehicle))
        decel = float(engine.vehicle.getDecel(vehicle))
        tau = float(engine.vehicle.getTau(vehicle))
        action_step = float(engine.vehicle.getActionStepLength(vehicle))
        gap = float(engine.vehicle.getMinGap(vehicle))
        if not all(math.isfinite(x) for x in (pos, length, speed, decel, tau, action_step, gap, step)) or decel <= 0 or speed < 0 or step <= 0:
            raise ValueError('Invalid vehicle dynamics for safe placement')
        required = gap + speed * (max(tau, action_step, step) + policy['reaction_steps'] * step)
        required += speed * speed / (2 * decel) + policy['clearance_m']
        # Ahead vehicles need static body clearance; behind vehicles need a
        # stopping gap to the obstacle's rear bumper.
        intervals.append((pos - length - obstacle.GAP, pos + length_m + required))
        vehicles.append({'vehicle': vehicle, 'position': actual_pos, 'lane': actual_lane,
                         'projected_position': pos, 'speed': speed,
                         'required_gap': required, 'forbidden_interval': list(intervals[-1])})
    intervals.extend((p - length_m - obstacle.GAP, p + length_m + obstacle.GAP)
                     for p in extra_positions)
    candidates = [requested, lower, upper]
    for left, right in intervals:
        candidates.extend((left - .01, right + .01))
    selected = next((p for p in sorted(set(candidates), key=lambda p: (abs(p-requested), p))
                     if lower <= p <= upper and all(not left <= p <= right for left, right in intervals)), None)
    removals = []
    if selected is None and policy['version'] == FORCED_VERSION:
        # Synthetic obstacles are never cleared to make another event fit.
        protected = [v['forbidden_interval'] for v in vehicles if v['vehicle'].startswith(obstacle.PREFIX)]
        protected.extend((p - length_m - obstacle.GAP, p + length_m + obstacle.GAP) for p in extra_positions)
        feasible = [p for p in set(candidates) if lower <= p <= upper
                    and all(not left <= p <= right for left, right in protected)]
        def conflicts(p):
            return [v for v in vehicles if v['forbidden_interval'][0] <= p <= v['forbidden_interval'][1]]
        if feasible:
            selected = min(feasible, key=lambda p: (len(conflicts(p)), abs(p-requested), p))
            removals = conflicts(selected)
    detail = {'version': policy['version'], 'lane': lane_id, 'requested_position': requested,
              'tolerance': tolerance, 'bounds': [lower, upper], 'selected_position': selected,
              'vehicles_checked': len(vehicles), 'obstacle_length_m': length_m}
    if policy['version'] == FORCED_VERSION:
        detail['vehicles_to_remove'] = removals
        detail['forced_clearance'] = bool(removals)
    if selected is None:
        detail['reason'] = 'no_position_with_static_clearance_and_stopping_distance'
        raise UnsafePlacementError(detail)
    removed_ids = {v['vehicle'] for v in removals}
    followers = [dict(v, margin_m=selected - length_m - v['projected_position'] - v['required_gap'])
                 for v in vehicles if v['vehicle'] not in removed_ids and v['projected_position'] <= selected - length_m]
    detail['minimum_margin_m'] = min((v['margin_m'] for v in followers), default=None)
    detail['limiting_follower'] = min(followers, key=lambda v: v['margin_m']) if followers else None
    return selected, detail


def install_safe_placement(runtime, policy):
    """Decorate a closed v1 runtime; never patch global functions or schedules."""
    policy = validate_policy(policy)
    if policy == LEGACY:
        return runtime
    if runtime._engine is not None or hasattr(runtime, 'placement_policy'):
        raise ValueError('Install placement policy exactly once before binding')
    runtime.placement_policy = policy
    runtime.placement_decisions = []
    runtime.removed_vehicles = []
    base_bind, base_sync, base_audit = runtime.bind, runtime.synchronize, runtime.audit

    def bind(engine):
        runtime.placement_decisions = []
        runtime.removed_vehicles = []
        return base_bind(engine)

    def synchronize():
        runtime._check_ready()
        engine = runtime._engine
        now = float(engine.simulation.getTime())
        if runtime._time == now:
            return
        if runtime._time is not None and (now < runtime._time or now-runtime._time > runtime._step + 1e-7):
            return base_sync()  # existing skipped-step failure contract
        try:
            active = [e for e in runtime.schedule.events if e.begin <= now < e.end]
            active_ids = {e.event_id for e in active}
            retiring = [v for key, v in runtime._obstacles.items() if key not in active_ids]
            pending = []
            clearing = set()
            for event in active:
                if event.kind != 'lane_blockage' or event.event_id in runtime._obstacles:
                    continue
                other = [p for e, p, _ in pending if e.lane_id == event.lane_id]
                position, detail = choose_position(engine, event.lane_id, event.position,
                    event.position_tolerance, policy, other, [*retiring, *clearing])
                clearing.update(v['vehicle'] for v in detail.get('vehicles_to_remove', []))
                pending.append((event, position, detail))
            # Validate the whole new-obstacle batch before mutating the scene.
            for key in list(runtime._obstacles):
                if key not in active_ids:
                    obstacle.remove(engine, runtime._obstacles.pop(key))
            if pending:
                type_id = obstacle.PREFIX + 'obstacle'
                if type_id not in engine.vehicletype.getIDList():
                    engine.vehicletype.copy('DEFAULT_VEHTYPE', type_id)
                    engine.vehicletype.setVehicleClass(type_id, 'ignoring')
                    engine.vehicletype.setLength(type_id, policy['obstacle_length_m'])
                    engine.vehicletype.setMinGap(type_id, obstacle.GAP)
                    engine.vehicletype.setColor(type_id, (255, 80, 0, 255))
            for event, position, detail in pending:
                for vehicle in detail.get('vehicles_to_remove', []):
                    engine.vehicle.remove(vehicle['vehicle'])
                    if vehicle['vehicle'] in engine.vehicle.getIDList():
                        raise RuntimeError('Conflicting vehicle was not removed: ' + vehicle['vehicle'])
                    runtime.removed_vehicles.append(dict(vehicle, event_id=event.event_id,
                        time=now, reason='forced_event_insertion'))
                target = runtime._lanes[event.lane_id]
                vehicle = obstacle.insert(engine, event, target['edge_id'], target['lane_index'],
                                          position, now, runtime._step)
                runtime._obstacles[event.event_id] = vehicle
                if abs(engine.vehicle.getLength(vehicle) - policy['obstacle_length_m']) > 1e-6:
                    raise RuntimeError('Obstacle footprint differs from placement contract')
                runtime._positions[event.event_id] = position
                runtime.placement_decisions.append(dict(detail, event_id=event.event_id, time=now))
            return base_sync()  # verifies obstacles, applies other events, publishes reports
        except Exception as error:
            runtime._failed = f'{type(error).__name__}: {error}'
            raise

    def audit():
        value = base_audit()
        value['placement_policy'] = dict(policy)
        value['placement_decisions'] = copy.deepcopy(runtime.placement_decisions)
        value['removed_vehicles'] = copy.deepcopy(runtime.removed_vehicles)
        return value

    runtime.bind, runtime.synchronize, runtime.audit = bind, synchronize, audit
    return runtime


def install_removal_view(world, runtime):
    """Exclude cleared traffic from stale lane lists and SUMO arrival notices.

    Call after reset on the live World. Existing traffic filtering and all
    control commands are reused; clearing is never reported as trip completion.
    """
    if getattr(runtime, 'placement_policy', {}).get('version') != FORCED_VERSION:
        return
    from .integration import TrafficConnectionView, _LaneView, _SimulationView

    def removed():
        return {v['vehicle'] for v in runtime.removed_vehicles}

    class LaneView(_LaneView):
        def getLastStepVehicleIDs(self, lane):
            cleared = removed()
            return tuple(v for v in super().getLastStepVehicleIDs(lane) if v not in cleared)

        def _has_obstacle(self, lane):
            return super()._has_obstacle(lane) or bool(removed().intersection(self._domain.getLastStepVehicleIDs(lane)))

    class SimulationView(_SimulationView):
        def getArrivedIDList(self):
            cleared = removed()
            return tuple(v for v in super().getArrivedIDList() if v not in cleared)

    view = TrafficConnectionView(runtime._engine)
    view.lane = LaneView(runtime._engine)
    view.simulation = SimulationView(runtime._engine.simulation)
    world.eng = view
    for intersection in world.intersections:
        intersection.eng = view


def install_clearance_replay(runtime, removals):
    """Replay a recorded clearance without obstacles, for a paired audit only."""
    if runtime.schedule.events or runtime._engine is not None:
        raise ValueError('Clearance-only replay needs an empty, unbound runtime')
    if getattr(runtime, 'placement_policy', {}).get('version') != FORCED_VERSION:
        raise ValueError('Clearance replay requires the explicit removal accounting policy')
    rows = copy.deepcopy(removals)
    if len({r['vehicle'] for r in rows}) != len(rows):
        raise ValueError('Duplicate clearance vehicle')
    if any(r['time'] < 0 or r['vehicle'].startswith(obstacle.PREFIX) for r in rows):
        raise ValueError('Invalid clearance replay record')
    base = runtime.synchronize

    def synchronize():
        now = float(runtime._engine.simulation.getTime())
        done = {r['vehicle'] for r in runtime.removed_vehicles}
        due = [r for r in rows if r['vehicle'] not in done and r['time'] <= now]
        for row in due:
            v = row['vehicle']
            if row['time'] != now or v not in runtime._engine.vehicle.getIDList():
                raise RuntimeError('Clearance replay time/vehicle mismatch')
            if (runtime._engine.vehicle.getLaneID(v) != row['lane'] or
                    abs(runtime._engine.vehicle.getLanePosition(v)-row['position']) > 1e-6 or
                    abs(runtime._engine.vehicle.getSpeed(v)-row['speed']) > 1e-6):
                raise RuntimeError('Clearance replay pre-intervention state differs')
        for row in due:
            runtime._engine.vehicle.remove(row['vehicle'])
            runtime.removed_vehicles.append(dict(row, reason='clearance_only_replay'))
        return base()

    runtime.synchronize = synchronize
    return runtime
