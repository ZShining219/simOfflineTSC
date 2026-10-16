"""Adapted obstacle operator from T-REX (MIT; see LICENSE.T-REX).

Upstream: MLSM-at-DTU/T-REX, a91a3d8c4ac27982c75553b932eabb27acd63ca7,
T_REX.py::Deployment.triggering (lines 674-728).
Retained: route.add, vehicle.add/moveTo/setSpeed/setLaneChangeMode and remove.
Adaptations: injected connection, unique IDs, explicit stop, safe placement,
and no removal of ordinary traffic. ICM and driver adaptation are not imported.
"""

PREFIX = '__tsc_event__'
LENGTH = 5.0
GAP = 2.5


def choose_position(engine, lane_id, requested, tolerance, extra_positions=()):
    """Find the nearest free position within the configured tolerance, or fail.

    Positions refer to front bumpers. Both vehicles' lengths and minimum gaps
    are respected. No vehicles are moved or deleted to make room.
    """
    lower = max(LENGTH, requested - tolerance)
    upper = min(engine.lane.getLength(lane_id) - 0.1, requested + tolerance)
    occupied = []
    live_vehicles = set(engine.vehicle.getIDList())
    for vehicle in engine.lane.getLastStepVehicleIDs(lane_id):
        # Last-step lane lists can still contain an obstacle removed at this
        # same boundary (end of one event, start of the next).
        if vehicle not in live_vehicles:
            continue
        pos = engine.vehicle.getLanePosition(vehicle)
        occupied.append((pos - engine.vehicle.getLength(vehicle) - GAP,
                         pos + LENGTH + engine.vehicle.getMinGap(vehicle)))
    occupied.extend((p - LENGTH - GAP, p + LENGTH + GAP) for p in extra_positions)
    candidates = [requested, lower, upper]
    for left, right in occupied:
        candidates.extend((left - 0.01, right + 0.01))
    for candidate in sorted(set(candidates), key=lambda p: (abs(p - requested), p)):
        if lower <= candidate <= upper and all(not left <= candidate <= right for left, right in occupied):
            return candidate
    raise RuntimeError(f'No safe obstacle position on {lane_id} within {requested} +/- {tolerance} m')


def insert(engine, event, edge_id, lane_index, position, now, step_length):
    """Insert a stationary obstacle. Only the created obstacle is rolled back."""
    vehicle_id = PREFIX + event.event_id
    route_id = PREFIX + 'route_' + event.event_id
    type_id = PREFIX + 'obstacle'
    if type_id not in engine.vehicletype.getIDList():
        engine.vehicletype.copy('DEFAULT_VEHTYPE', type_id)
        # 'ignoring' allows insertion even when another event closes this road.
        engine.vehicletype.setVehicleClass(type_id, 'ignoring')
        engine.vehicletype.setLength(type_id, LENGTH)
        engine.vehicletype.setMinGap(type_id, GAP)
        engine.vehicletype.setColor(type_id, (255, 80, 0, 255))
    if route_id not in engine.route.getIDList():
        engine.route.add(route_id, [edge_id])
    try:
        engine.vehicle.add(vehicle_id, route_id, typeID=type_id, depart='now',
                           departLane=str(lane_index), departPos=str(position), departSpeed='0')
        engine.vehicle.moveTo(vehicle_id, event.lane_id, position)
        engine.vehicle.setSpeed(vehicle_id, 0)
        engine.vehicle.setLaneChangeMode(vehicle_id, 0)
        # A planned stop prevents waiting-time teleport of the obstacle itself.
        # It outlasts the event so removal, not stop expiry, defines clearance.
        engine.vehicle.setStop(vehicle_id, edge_id, pos=position, laneIndex=lane_index,
                               duration=event.end - now + step_length + 1)
        if engine.vehicle.getLaneID(vehicle_id) != event.lane_id:
            raise RuntimeError('SUMO did not place the obstacle on its requested lane')
        if abs(engine.vehicle.getLanePosition(vehicle_id) - position) > 1e-6:
            raise RuntimeError('SUMO did not apply the obstacle position')
    except Exception:
        try:
            engine.vehicle.remove(vehicle_id)
        except Exception:
            pass
        raise
    return vehicle_id


def remove(engine, vehicle_id):
    engine.vehicle.remove(vehicle_id)
    if vehicle_id in engine.vehicle.getIDList():
        raise RuntimeError(f'Obstacle was not removed: {vehicle_id}')
