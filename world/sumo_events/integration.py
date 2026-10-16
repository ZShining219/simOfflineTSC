"""Instance adapter for the existing World, without changing frozen source.

The real connection executes events. A traffic-only view excludes artificial
obstacle vehicles from the existing observation, reward and trip accounting
paths. All control commands are forwarded to that same connection.
"""
import xml.etree.ElementTree as ET

import sumolib

from .runtime import SumoEventRuntime
from .schema import Schedule, load_schedule
from .trex_blockage import PREFIX


def _traffic(ids):
    return tuple(v for v in ids if not v.startswith(PREFIX))


def validate_routes(net_file, route_file):
    """Validate explicit routes before tolerating TEMPORARY permission errors.

    hz4x4 uses explicit vehicle routes. Unresolved trips/flows and route files
    with no explicit routes are rejected instead of hiding routing failures.
    """
    net = sumolib.net.readNet(net_file)
    route_ids, references, count = set(), [], 0
    for _, element in ET.iterparse(route_file, events=('end',)):
        if element.get('id', '').startswith(PREFIX):
            raise ValueError('Traffic inputs must not use reserved event IDs')
        if element.tag in ('trip', 'flow') and not (element.get('route') or element.find('route') is not None):
            raise ValueError('Closure integration requires explicit vehicle routes')
        if element.tag == 'route':
            edges = element.get('edges', '').split()
            if not edges:
                raise ValueError('Empty or unresolved route')
            try:
                resolved = [net.getEdge(edge) for edge in edges]
            except KeyError as error:
                raise ValueError(f'Unknown route edge: {error}') from error
            for start, end in zip(resolved, resolved[1:]):
                connections = start.getOutgoing().get(end, ())
                if not any(c.getFromLane().allows('passenger') and c.getToLane().allows('passenger')
                           for c in connections):
                    raise ValueError(f'Invalid baseline route connection: {start.getID()} -> {end.getID()}')
            if element.get('id'):
                route_ids.add(element.get('id'))
            count += 1
        elif element.tag in ('vehicle', 'flow'):
            if element.get('route'):
                references.append(element.get('route'))
            elif element.find('route') is None:
                raise ValueError('Vehicle/flow is missing an explicit route')
    if not count or not set(references) <= route_ids:
        raise ValueError('Missing routes or unresolved route references')
    return count


class _DomainView:
    def __init__(self, domain):
        self._domain = domain

    def __getattr__(self, name):
        return getattr(self._domain, name)


class _VehicleView(_DomainView):
    def getIDList(self):
        return _traffic(self._domain.getIDList())

    def getIDCount(self):
        return len(self.getIDList())


class _SimulationView(_DomainView):
    def getDepartedIDList(self):
        return _traffic(self._domain.getDepartedIDList())

    def getArrivedIDList(self):
        return _traffic(self._domain.getArrivedIDList())

    def getDepartedNumber(self):
        return len(self.getDepartedIDList())

    def getArrivedNumber(self):
        return len(self.getArrivedIDList())


class _LaneView(_DomainView):
    def __init__(self, engine):
        super().__init__(engine.lane)
        self._vehicle = engine.vehicle

    def getLastStepVehicleIDs(self, lane):
        return _traffic(self._domain.getLastStepVehicleIDs(lane))

    def getLastStepVehicleNumber(self, lane):
        return len(self.getLastStepVehicleIDs(lane))

    def _has_obstacle(self, lane):
        return any(v.startswith(PREFIX) for v in self._domain.getLastStepVehicleIDs(lane))

    def getLastStepHaltingNumber(self, lane):
        if not self._has_obstacle(lane):
            return self._domain.getLastStepHaltingNumber(lane)
        return sum(self._vehicle.getSpeed(v) < 0.1 for v in self.getLastStepVehicleIDs(lane))

    def getWaitingTime(self, lane):
        if not self._has_obstacle(lane):
            return self._domain.getWaitingTime(lane)
        return sum(self._vehicle.getWaitingTime(v) for v in self.getLastStepVehicleIDs(lane))

    def getLastStepMeanSpeed(self, lane):
        if not self._has_obstacle(lane):
            return self._domain.getLastStepMeanSpeed(lane)
        vehicles = self.getLastStepVehicleIDs(lane)
        return (sum(self._vehicle.getSpeed(v) for v in vehicles) / len(vehicles)
                if vehicles else self._domain.getMaxSpeed(lane))


class TrafficConnectionView:
    """Traffic-only APIs used by LibSignal; other TraCI APIs are forwarded.

    Raw SUMO detector/subscription/edge aggregates are not rewritten. Downstream
    consumers using those APIs must explicitly exclude runtime obstacle IDs.
    """
    def __init__(self, engine):
        self._engine = engine
        self.vehicle = _VehicleView(engine.vehicle)
        self.simulation = _SimulationView(engine.simulation)
        self.lane = _LaneView(engine)

    def __getattr__(self, name):
        return getattr(self._engine, name)


def install_events(world, schedule):
    """Attach once, before world.reset(), and return the reusable runtime.

    Example::

        runtime = install_events(world, 'configs/events/hz4x4.yml')
        world.reset()
        world.step(actions)
        texts = runtime.texts(world.get_current_time(), world.intersection_ids)

    Reset replays the same schedule. The experiment owner sets the SUMO and
    training seeds. Waiting-time teleport is disabled for these experiments;
    no driver rerouting is added. Explicit routes are checked before enabling
    --ignore-route-errors for temporary closures: vehicles keep their planned
    routes and wait for reopening. Keep these settings in no-event controls.
    """
    if not isinstance(schedule, Schedule):
        schedule = load_schedule(schedule)
    existing = getattr(world, '_sumo_event_runtime', None)
    if existing is not None:
        if existing.schedule != schedule:
            raise ValueError('A different event schedule is already installed')
        return existing
    if world.step_ratio != 1:
        raise ValueError('This World adapter requires step_ratio=1; direct runtime supports other step lengths')
    if getattr(world, '_connection_open', False):
        raise ValueError('Install events on a closed World before reset')
    for name in ('_tarl_runtime', '_tarl_v21_runtime'):
        old = getattr(world, name, None)
        if old is not None and old.events:
            raise ValueError('Cannot combine the new module with a nonempty legacy event runtime')
    runtime = SumoEventRuntime(world.net, schedule)
    runtime.validated_routes = validate_routes(world.net, world.route)
    # Physical queues must not disappear through SUMO's timeout teleport.
    command = list(world.sumo_cmd)
    if '--time-to-teleport' in command:
        if float(command[command.index('--time-to-teleport') + 1]) != -1:
            raise ValueError('Events require --time-to-teleport -1')
    else:
        command.extend(['--time-to-teleport', '-1'])
    if '--ignore-route-errors' in command:
        if command[command.index('--ignore-route-errors') + 1].lower() not in ('true', '1'):
            raise ValueError('Temporary road closures require --ignore-route-errors true after route validation')
    else:
        command.extend(['--ignore-route-errors', 'true'])
    world.sumo_cmd = command
    original_reset, original_step, original_close = world.reset, world.step_sim, world.close

    def close(*args, **kwargs):
        try:
            if getattr(world, '_connection_open', False):
                runtime.close()
        finally:
            return_value = original_close(*args, **kwargs)
        return return_value

    def reset(*args, **kwargs):
        try:
            result = original_reset(*args, **kwargs)
            runtime.bind(world.eng)
            world.eng = TrafficConnectionView(world.eng)
            for intersection in world.intersections:
                intersection.eng = world.eng
                intersection.observe(world.step_length, world.max_distance)
            world._update_infos()
            return result
        except Exception:
            close()
            raise

    def step_sim(*args, **kwargs):
        # Detect a conflicting legacy installer even if attached afterwards.
        for name in ('_tarl_runtime', '_tarl_v21_runtime'):
            old = getattr(world, name, None)
            if old is not None and old.events:
                raise RuntimeError('Conflicting legacy event schedule')
        runtime.synchronize()
        result = original_step(*args, **kwargs)
        # Publish at the new time before World.step obtains observations.
        runtime.synchronize()
        return result

    world.reset, world.step_sim, world.close = reset, step_sim, close
    world._sumo_event_runtime = runtime
    return runtime
