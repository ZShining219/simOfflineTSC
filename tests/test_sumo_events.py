"""Contract and real-SUMO behavioural tests on the project's hz4x4 network."""
from dataclasses import replace
from pathlib import Path

import pytest

libsumo = pytest.importorskip('libsumo')
sumolib = pytest.importorskip('sumolib')

from world.sumo_events import Event, Schedule, SumoEventRuntime, load_schedule
from world.sumo_events.integration import TrafficConnectionView, validate_routes
from world.sumo_events.trex_blockage import PREFIX

ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT / 'data/raw_data/hangzhou_4x4_gudang_18041610_1h/hangzhou_4x4_gudang_18041610_1h'
NET = str(STEM) + '.net.xml'
LANE = 'road_1_1_0_2'
EDGE = 'road_1_1_0'


@pytest.fixture
def engine():
    libsumo.start([sumolib.checkBinary('sumo'), '-n', NET, '--seed', '7',
                  '--no-step-log', 'true', '--no-warnings', 'true',
                  '--time-to-teleport', '-1', '--ignore-route-errors', 'true'])
    try:
        yield libsumo
    finally:
        libsumo.close()


def runtime(engine, *events):
    result = SumoEventRuntime(NET, Schedule(events))
    result.bind(engine)
    return result


def advance(engine, rt, seconds):
    while engine.simulation.getTime() < seconds:
        engine.simulationStep()
        rt.synchronize()
        assert engine.simulation.getCollidingVehiclesNumber() == 0
        assert engine.simulation.getStartingTeleportNumber() == 0


def test_forced_blockage_clears_downstream_tail_and_replays_actual_lane(engine):
    from world.sumo_events.safe_placement import install_safe_placement, install_clearance_replay
    lane = 'road_4_3_2_1'
    event = Event('tail', 'lane_blockage', 1, 6, lane_id=lane,
                  position=771.8, position_tolerance=.5)
    rt = SumoEventRuntime(NET, Schedule((event,)))
    install_safe_placement(rt, {'version':'clear-conflicts-v1', 'clearance_m':2,
                               'reaction_steps':1, 'obstacle_length_m':30})
    rt.bind(engine)
    engine.route.add('tail_route', ['road_4_3_2', 'road_3_3_2'])
    engine.vehicle.add('tail_car', 'tail_route')
    engine.vehicle.moveTo('tail_car', ':intersection_3_3_12_0', 4.53)
    engine.vehicle.setSpeed('tail_car', 0)
    advance(engine, rt, 1)
    removed = rt.removed_vehicles
    assert len(removed) == 1 and removed[0]['vehicle'] == 'tail_car'
    assert removed[0]['lane'] == ':intersection_3_3_12_0'
    assert removed[0]['position'] == pytest.approx(4.53)
    assert removed[0]['projected_position'] == pytest.approx(772.8+4.53)
    advance(engine, rt, 5)
    assert PREFIX+'tail' in engine.vehicle.getIDList()
    advance(engine, rt, 6)
    assert PREFIX+'tail' not in engine.vehicle.getIDList()
    # A clearance-only counterfactual must replay the actual internal lane,
    # not the obstacle's external lane or its projected coordinate.
    engine.load(['-n', NET, '--seed', '7', '--no-step-log', 'true',
                 '--no-warnings', 'true', '--time-to-teleport', '-1'])
    control = SumoEventRuntime(NET, Schedule(()))
    install_safe_placement(control, rt.placement_policy)
    install_clearance_replay(control, removed)
    control.bind(engine)
    engine.route.add('tail_route', ['road_4_3_2', 'road_3_3_2'])
    engine.vehicle.add('tail_car', 'tail_route')
    engine.vehicle.moveTo('tail_car', ':intersection_3_3_12_0', 4.53)
    engine.vehicle.setSpeed('tail_car', 0)
    advance(engine, control, 6)
    assert len(control.removed_vehicles) == 1
    assert control.removed_vehicles[0]['vehicle'] == 'tail_car'
    assert control.removed_vehicles[0]['reason'] == 'clearance_only_replay'


@pytest.mark.parametrize('changes', [
    {'begin': -1}, {'end': 0}, {'begin': float('nan')}, {'end': float('inf')},
    {'speed_factor': 0}, {'speed_factor': 1}, {'speed_factor': True},
    {'lane_id': LANE}, {'position_tolerance': 2}, {'event_id': ''},
])
def test_invalid_schedule_rejected(changes):
    args = dict(event_id='rain', kind='global_rain', begin=0, end=10, speed_factor=.5)
    with pytest.raises(ValueError):
        Event(**dict(args, **changes))


def test_schedule_and_static_grounding():
    schedule = load_schedule(ROOT / 'configs/events/hz4x4.yml')
    rt = SumoEventRuntime(NET, schedule)
    catalog = rt.network_catalog()
    assert len(rt.intersection_ids) == 16
    assert catalog[LANE]['movements'] == ['left-turn']
    assert catalog[LANE]['direction'] == 'eastbound'
    assert catalog[LANE]['to_junction'] == 'intersection_2_1'
    assert validate_routes(NET, str(STEM) + '.rou.xml') == 2983
    with pytest.raises(ValueError, match='Duplicate'):
        Schedule((schedule.events[0], schedule.events[0]))
    with pytest.raises(ValueError, match='Invalid external'):
        SumoEventRuntime(NET, Schedule((replace(schedule.events[0], lane_id='missing'),)))
    with pytest.raises(ValueError, match='lane length'):
        SumoEventRuntime(NET, Schedule((replace(schedule.events[0], position=10000),)))


def test_invalid_baseline_route_not_hidden(tmp_path):
    route = tmp_path / 'invalid.rou.xml'
    route.write_text('<routes><vehicle id="x" depart="0"><route edges="road_1_1_0 road_4_4_0"/></vehicle></routes>')
    with pytest.raises(ValueError, match='Invalid baseline route'):
        validate_routes(NET, route)


def test_exact_boundaries_overlap_restore_and_no_future_reports(engine):
    engine.lane.setDisallowed(EDGE + '_0', ['truck'])
    original = {lane: engine.lane.getMaxSpeed(lane) for lane in engine.lane.getIDList()}
    events = (
        Event('rain1', 'global_rain', 2, 8, speed_factor=.7),
        Event('rain2', 'global_rain', 4, 6, speed_factor=.4),
        Event('close1', 'road_closure', 2, 5, edge_id=EDGE),
        Event('close2', 'road_closure', 4, 7, edge_id=EDGE),
    )
    rt = runtime(engine, *events)
    assert rt.reports() == ()
    for now in range(1, 10):
        advance(engine, rt, now)
        factor = .4 if 4 <= now < 6 else .7 if 2 <= now < 8 else 1
        for lane, base in original.items():
            assert engine.lane.getMaxSpeed(lane) == pytest.approx(base * factor)
        for index in range(3):
            denied = set(engine.lane.getDisallowed(f'{EDGE}_{index}'))
            if 2 <= now < 7:
                assert 'passenger' in denied
            else:
                assert denied == ({'truck'} if index == 0 else set())
        assert {r.event_id for r in rt.reports()} == {e.event_id for e in events if e.begin <= now}
    transitions = rt.audit()['transitions']
    assert {(r['event_id'], r['status'], r['updated_at']) for r in transitions} == {
        (e.event_id, state, time) for e in events for state, time in [('active', e.begin), ('cleared', e.end)]}
    assert 'has been removed' in next(r.text for r in rt.reports() if r.event_id == 'close1')
    assert 'reopened' not in next(r.text for r in rt.reports() if r.event_id == 'close1')
    assert len(set(rt.texts(9, rt.intersection_ids))) == 1
    rt.close()
    rt.close()


def add_probe(engine, vehicle_id, edges, lane, position, speed=0):
    engine.route.add('route_' + vehicle_id, edges)
    engine.vehicle.add(vehicle_id, 'route_' + vehicle_id, departLane=lane.rsplit('_', 1)[1],
                       departPos=str(position), departSpeed=str(speed))
    engine.vehicle.moveTo(vehicle_id, lane, position)
    engine.vehicle.setLaneChangeMode(vehicle_id, 0)
    engine.vehicle.setSpeedFactor(vehicle_id, 1)


def test_obstacle_stops_only_target_lane_then_releases_without_vehicle_loss(engine):
    rt = runtime(engine, Event('block', 'lane_blockage', 0, 30, lane_id=LANE, position=650))
    add_probe(engine, 'follower', [EDGE, 'road_2_1_1'], LANE, 590, 5)
    add_probe(engine, 'adjacent', [EDGE, 'road_2_1_0'], EDGE + '_1', 590, 5)
    view = TrafficConnectionView(engine)
    passed_adjacent = False
    for now in range(1, 30):
        advance(engine, rt, now)
        assert engine.vehicle.getLanePosition('follower') < 645
        assert 'follower' in engine.vehicle.getIDList()
        assert not any(v.startswith(PREFIX) for v in view.vehicle.getIDList())
        assert not any(v.startswith(PREFIX) for v in view.lane.getLastStepVehicleIDs(LANE))
        passed_adjacent |= engine.vehicle.getLanePosition('adjacent') > 650
    assert passed_adjacent
    assert engine.vehicle.getSpeed('follower') < .1
    advance(engine, rt, 30)
    assert rt.reports()[0].status == 'cleared'
    assert not any(v.startswith(PREFIX) for v in engine.vehicle.getIDList())
    advance(engine, rt, 40)
    assert engine.vehicle.getLanePosition('follower') > 650
    rt.close()


def test_closure_blocks_actual_entry_and_restores_it(engine):
    rt = runtime(engine, Event('close', 'road_closure', 0, 30, edge_id=EDGE))
    for junction in engine.trafficlight.getIDList():
        engine.trafficlight.setRedYellowGreenState(junction, 'G' * len(engine.trafficlight.getRedYellowGreenState(junction)))
    upstream = 'road_0_1_0'
    add_probe(engine, 'entering', [upstream, EDGE, 'road_2_1_0'], upstream + '_1',
              engine.lane.getLength(upstream + '_1') - 50, 2)
    for now in range(1, 30):
        advance(engine, rt, now)
        assert engine.vehicle.getRoadID('entering') != EDGE
    advance(engine, rt, 50)
    assert engine.vehicle.getRoadID('entering') == EDGE
    rt.close()


def test_rain_limits_new_vehicle_speed_and_restores(engine):
    baseline = engine.lane.getMaxSpeed(EDGE + '_1')
    rt = runtime(engine, Event('rain', 'global_rain', 0, 30, speed_factor=.4))
    advance(engine, rt, 5)
    add_probe(engine, 'new', [EDGE, 'road_2_1_0'], EDGE + '_1', 100)
    advance(engine, rt, 20)
    assert 0 < engine.vehicle.getSpeed('new') <= baseline * .4 + 1e-6
    advance(engine, rt, 40)
    assert engine.vehicle.getSpeed('new') > baseline * .4
    rt.close()


def test_occupied_position_is_rejected_without_deleting_traffic(engine):
    add_probe(engine, 'ordinary', [EDGE, 'road_2_1_1'], LANE, 650)
    engine.vehicle.setSpeed('ordinary', 0)
    rt = runtime(engine, Event('block', 'lane_blockage', 1, 10, lane_id=LANE, position=650))
    engine.simulationStep()
    with pytest.raises(RuntimeError, match='No safe obstacle position'):
        rt.synchronize()
    assert 'ordinary' in engine.vehicle.getIDList()
    with pytest.raises(RuntimeError, match='reset required'):
        rt.reports()
    rt.close()


def test_tolerated_placement_reports_actual_position_and_preserves_traffic(engine):
    add_probe(engine, 'ordinary', [EDGE, 'road_2_1_1'], LANE, 650)
    engine.vehicle.setSpeed('ordinary', 0)
    rt = runtime(engine, Event('block', 'lane_blockage', 1, 10, lane_id=LANE,
                               position=650, position_tolerance=20))
    advance(engine, rt, 1)
    actual = rt.audit()['actual_positions']['block']
    assert actual != 650 and abs(actual - 650) <= 20
    assert f'{actual:.2f} m' in rt.reports()[0].text
    assert 'ordinary' in engine.vehicle.getIDList()
    rt.close()


def test_blockage_can_overlap_closed_road_and_close_mid_event(engine):
    baseline = engine.lane.getMaxSpeed(LANE)
    rt = runtime(engine, Event('closure', 'road_closure', 0, 20, edge_id=EDGE),
                 Event('block', 'lane_blockage', 2, 10, lane_id=LANE, position=650),
                 Event('rain', 'global_rain', 0, 20, speed_factor=.5))
    advance(engine, rt, 5)
    assert len(rt.audit()['obstacle_ids']) == 1
    rt.close()
    assert engine.lane.getDisallowed(LANE) == ()
    assert engine.lane.getMaxSpeed(LANE) == baseline
    assert not any(v.startswith(PREFIX) for v in engine.vehicle.getIDList())


def test_step_alignment_and_stale_report_rejection(engine):
    with pytest.raises(ValueError, match='aligned'):
        runtime(engine, Event('rain', 'global_rain', .5, 3, speed_factor=.5))
    rt = runtime(engine)
    engine.simulationStep()
    with pytest.raises(RuntimeError, match='Synchronize'):
        rt.reports()
    engine.simulationStep()
    with pytest.raises(RuntimeError, match='skipped'):
        rt.synchronize()
    rt.close()


def test_back_to_back_obstacles_on_same_lane(engine):
    rt = runtime(engine, Event('first', 'lane_blockage', 0, 5, lane_id=LANE, position=650),
                 Event('second', 'lane_blockage', 5, 10, lane_id=LANE, position=650))
    advance(engine, rt, 5)
    assert rt.audit()['obstacle_ids'] == {'second': PREFIX + 'second'}
    assert {r.event_id: r.status for r in rt.reports()} == {'first': 'cleared', 'second': 'active'}
    advance(engine, rt, 10)
    assert not rt.audit()['obstacle_ids']
    rt.close()


def test_world_reset_during_active_event_and_empty_adapter_parity(tmp_path, monkeypatch):
    """Exercise original World observations/accounting, not a mock environment."""
    from types import SimpleNamespace
    from common.registry import Registry
    from world.world_sumo import World
    from world.sumo_events import install_events

    monkeypatch.setitem(Registry.mapping['command_mapping'], 'setting', SimpleNamespace(param={'sumo_seed': 7}))
    monkeypatch.setitem(Registry.mapping['logger_mapping'], 'path', SimpleNamespace(path=str(tmp_path)))
    config = str(ROOT / 'configs/sim/hz4x4.cfg')

    def trace(schedule):
        world = World(config, interface='libsumo')
        world.sumo_cmd += ['--no-step-log', 'true']
        rt = install_events(world, schedule) if schedule is not None else None
        if rt is None:
            world.sumo_cmd += ['--time-to-teleport', '-1', '--ignore-route-errors', 'true']
        try:
            sequences = []
            for _ in range(2):
                world.reset()
                rows = []
                for tick in range(6):
                    world.step([0] * len(world.intersections))
                    rows.append((world.get_lane_vehicle_count(), dict(world.inside_vehicles),
                                 tuple((v, world.eng.vehicle.getLaneID(v), world.eng.vehicle.getLanePosition(v))
                                       for v in world.eng.vehicle.getIDList())))
                    assert not any(v.startswith(PREFIX) for v in world.inside_vehicles)
                    assert not any(v.startswith(PREFIX) for v in world.vehicle_trajectory)
                sequences.append(rows)
                if rt and rt.schedule.events:
                    assert len(rt.audit()['obstacle_ids']) == 1
            assert sequences[0] == sequences[1]
            return sequences[0]
        finally:
            world.close()

    assert trace(None) == trace(Schedule(()))
    trace(Schedule((Event('reset', 'lane_blockage', 0, 20, lane_id=LANE, position=650),)))
