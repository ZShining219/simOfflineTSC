"""Real report -> entity -> observation verification, not RL efficacy evidence.

Run in colight from the project root:
    python -m pytest -q tests/test_text_grounding_sumo.py

SUMO/libsumo; existing FixedTimeAgent; hz4x4; seed 7; two 245-second
episodes. The controller is only a traffic driver for this mapping test;
this does not qualify it as a standard fixed-cycle performance baseline.
Pytest's temporary directory receives validation.json with commands, versions,
source hashes, report transitions and replay evidence.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('libsumo')
pytest.importorskip('sumolib')

from common.registry import Registry
from generator.text_entity import TextEntityBinder
from world.sumo_events import ReportGrounder, install_events, load_schedule
from world.world_sumo import World


ROOT = Path(__file__).resolve().parents[1]


def test_hz4x4_report_feature_roundtrip(tmp_path, monkeypatch):
    from agent.fixedtime import FixedTimeAgent

    monkeypatch.chdir(ROOT)
    monkeypatch.setitem(Registry.mapping['command_mapping'], 'setting', SimpleNamespace(param={'sumo_seed': 7}))
    monkeypatch.setitem(Registry.mapping['model_mapping'], 'setting', SimpleNamespace(param={'t_fixed': 20}))
    monkeypatch.setitem(Registry.mapping['logger_mapping'], 'path', SimpleNamespace(path=str(tmp_path)))
    schedule = load_schedule('configs/events/hz4x4.yml')
    world = World('configs/sim/hz4x4.cfg', interface='libsumo')
    world.sumo_cmd += ['--no-step-log', 'true']
    records = []
    try:
        runtime = install_events(world, schedule)
        catalog = runtime.network_catalog()
        mapper = ReportGrounder(catalog)
        # Ground truth is only used here in the verifier, never passed to the mapper.
        expected_targets = {}
        for event in schedule.events:
            if event.kind == 'lane_blockage':
                expected_targets[event.event_id] = {event.lane_id}
            elif event.kind == 'road_closure':
                expected_targets[event.event_id] = {i for i, lane in catalog.items() if lane['edge_id'] == event.edge_id}
            else:
                expected_targets[event.event_id] = {i for i, lane in catalog.items() if lane['motor_vehicle_lane']}
        previous_binder = None
        for repeat in range(2):
            world.reset()
            if previous_binder is not None:
                with pytest.raises(ValueError, match='stale'):
                    previous_binder.bind(())
            agents = [FixedTimeAgent(world, rank) for rank in range(len(world.intersections))]
            generators = [agent.ob_generator for agent in agents]
            world._update_infos()
            binder = TextEntityBinder(mapper.catalog, world.intersection_ids, generators)
            assert binder.bind(mapper.map_reports(runtime.reports())).texts == ()
            trace = hashlib.sha256()
            seen = {}
            transitions = []
            comparisons = 0
            positive_observations = 0
            first_active = None
            for _ in range(245):
                actions = [agent.get_action(agent.get_ob(), agent.get_phase()) for agent in agents]
                world.step(actions)
                now = world.get_current_time()
                observations = [gen.generate() for gen in generators]
                grounded = mapper.map_reports(runtime.reports())
                batch = binder.bind(grounded)
                expected_ids = {event.event_id for event in schedule.events if event.begin <= now}
                assert {r.report_id for r in grounded} == expected_ids
                for bound in batch.reports:
                    result = bound.report
                    event = next(e for e in schedule.events if e.event_id == result.report_id)
                    expected_status = 'active' if now < event.end else 'cleared'
                    assert result.report_status == expected_status
                    assert set(result.target_lane_ids) == expected_targets[result.report_id]
                    if result.report_status != seen.get(result.report_id):
                        seen[result.report_id] = result.report_status
                        transitions.append(result.to_dict())
                    if first_active is None:
                        first_active = batch
                    for binding in bound.feature_bindings:
                        assert observations[binding.intersection_index].shape == (batch.feature_sizes[binding.intersection_index],)
                        # Independently count policy-visible vehicles via public engine
                        # APIs, respecting World's 200 m observation range.
                        vehicles = world.eng.lane.getLastStepVehicleIDs(binding.lane_id)
                        count = 0
                        for vehicle in vehicles:
                            tls = world.eng.vehicle.getNextTLS(vehicle)
                            count += bool(tls and tls[0][2] <= world.max_distance)
                        actual = observations[binding.intersection_index][binding.feature_index]
                        assert actual == count
                        comparisons += 1
                        positive_observations += count > 0
                    observed = {b.lane_id for b in bound.feature_bindings}
                    assert observed | set(bound.unobserved_lane_ids) == set(result.target_lane_ids)
                trace.update(json.dumps([now, actions, [r.to_dict() for r in grounded],
                                         [o.tolist() for o in observations]], sort_keys=True).encode())
                trace.update(batch.feature_report_mask.tobytes())
                assert not np.any(batch.feature_report_mask[~batch.valid_feature_mask])
            assert len(transitions) == 6
            assert comparisons > 0 and positive_observations > 0
            assert first_active.reports[0].report.report_status == 'active'
            assert all(r.report_status == 'cleared' for r in grounded)
            records.append({'repeat': repeat, 'steps': 245, 'comparisons': comparisons,
                            'positive_observations': positive_observations, 'transitions': transitions,
                            'trace_sha256': trace.hexdigest()})
            previous_binder = binder
        assert records[0]['trace_sha256'] == records[1]['trace_sha256']
        source_paths = ['utils/text_grounding.py', 'world/sumo_events/grounding.py',
                        'generator/text_entity.py', 'tests/test_text_grounding_sumo.py',
                        'configs/events/hz4x4.yml', 'configs/sim/hz4x4.cfg',
                        'generator/lane_vehicle.py', 'world/sumo_events/runtime.py',
                        'world/world_sumo.py', 'agent/fixedtime.py']
        source_paths.extend([world.net, world.route])
        evidence = {'status': 'passed', 'scope': 'report_entity_feature_mapping',
                    'simulator': 'SUMO', 'version': world.eng.getVersion(), 'interface': 'libsumo',
                    'agent': 'FixedTimeAgent (mapping validation only)', 'network': 'hz4x4', 'seed': 7,
                    'python': sys.executable, 'invocation': [sys.executable, '-m', 'pytest'] + sys.argv[1:],
                    'sumo_command': list(world.sumo_cmd), 'runs': records,
                    'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths}}
        (tmp_path / 'validation.json').write_text(json.dumps(evidence, indent=2) + '\n')
        print(f'Mapping validation evidence: {tmp_path / "validation.json"}')
    finally:
        world.close()
