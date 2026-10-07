"""P0 gate for SUMO logical green timing and generated yellow transitions.

This is an engineering baseline check, not a traffic-control efficacy test.
It uses the real hz4x4 World and FixedTimeAgent so a phase trajectory cannot
pass by exercising only a mock state machine.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('libsumo')

from common.registry import Registry
from agent.fixedtime import FixedTimeAgent
from world.world_sumo import World
from world.sumo_signal_control import install_signal_control
from world.sumo_events import Schedule, install_events


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def controlled_world(tmp_path, monkeypatch, request):
    monkeypatch.chdir(ROOT)
    monkeypatch.setitem(Registry.mapping['command_mapping'], 'setting',
                        SimpleNamespace(param={'sumo_seed': 7}))
    monkeypatch.setitem(Registry.mapping['model_mapping'], 'setting',
                        SimpleNamespace(param={'t_fixed': 20}))
    monkeypatch.setitem(Registry.mapping['logger_mapping'], 'path',
                        SimpleNamespace(path=str(tmp_path)))

    interface = getattr(request, 'param', 'libsumo')
    world = World('configs/sim/hz4x4.cfg', interface=interface)
    world.sumo_cmd += ['--no-step-log', 'true']
    install_signal_control(world, {'version': 'sumo-green-yellow-v1', 'yellow_seconds': 5})
    install_events(world, Schedule(()))
    try:
        yield world
    finally:
        world.close()


@pytest.mark.parametrize('controlled_world', ['libsumo', 'traci'], indirect=True)
def test_fixedtime_uses_real_yellow_and_does_not_switch_each_second(controlled_world):
    world = controlled_world
    repeats = []
    for repeat in range(2):
        world.reset()
        agents = [FixedTimeAgent(world, rank) for rank in range(len(world.intersections))]
        world._update_infos()
        intersection = world.intersections[0]
        green_count = len(intersection.green_phases)
        trace = []
        for time in range(100):
            actions = [agent.get_action(agent.get_ob(), agent.get_phase()) for agent in agents]
            world.step(actions)
            # Expected intervals are [0,20) G0, [20,25) yellow 0->1,
            # [25,45) G1, [45,50) yellow 1->2, etc. Read actual signal
            # strings after simulation, not the adapter's bookkeeping flags.
            cycle, offset = divmod(time, 25)
            for intersection in world.intersections:
                source = intersection.green_phases[cycle % green_count].state
                target = intersection.green_phases[(cycle + 1) % green_count].state
                expected = source if offset < 20 else ''.join(
                    'y' if a in 'Gg' and b not in 'Gg' else a for a, b in zip(source, target))
                actual = world.eng.trafficlight.getRedYellowGreenState(intersection.id)
                assert actual == expected, (time, intersection.id, actual, expected)
                assert 0 <= intersection.current_phase < green_count
            trace.append((actions, [world.eng.trafficlight.getRedYellowGreenState(i.id)
                                    for i in world.intersections]))
        repeats.append(trace)
    assert repeats[0] == repeats[1]


def test_all_phase_pairs_clear_and_hold_and_reject_invalid_actions(controlled_world):
    world = controlled_world
    world.reset()
    inter = world.intersections[0]
    for bad in [-1, 8, 1.5, True]:
        with pytest.raises(ValueError, match='valid integer'):
            inter.pseudo_step(bad)
    for source in range(8):
        # Stabilize the source, including any transition from the previous case.
        for _ in range(12):
            world.step([source] * 16)
        for target in range(8):
            if source == target:
                continue
            for _ in range(12):
                world.step([source] * 16)
            a = inter.green_phases[source].state
            b = inter.green_phases[target].state
            yellow = ''.join('y' if x in 'Gg' and y not in 'Gg' else x for x, y in zip(a, b))
            for time in range(6):
                # Mid-yellow requests must neither bypass nor extend clearance.
                action = target if time in (0, 5) else source
                world.step([action] * 16)
                expected = yellow if 'y' in yellow and time < 5 else b
                assert world.eng.trafficlight.getRedYellowGreenState(inter.id) == expected
    for _ in range(80):
        world.step([7] * 16)
    assert inter.current_phase_time >= 75
    assert world.eng.trafficlight.getRedYellowGreenState(inter.id) == inter.green_phases[7].state
