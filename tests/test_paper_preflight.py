"""Configuration reuse and development-only event assessment contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from common.paper_experiment import load_config, describe_config, plan, read_json
from world.sumo_events.preflight import placement_probe
from world.sumo_events.safe_placement import choose_position as safe_position, UnsafePlacementError

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'configs/tsc/colight_event150.yml'
SMOKE = ROOT / 'configs/tsc/paper_experiment_smoke.yml'


def test_inheritance_grid_and_budget_are_one_contract(tmp_path):
    result = describe_config(PROFILE)
    assert result['config']['events']['begin'] == list(range(600, 1501, 10))
    assert result['decisions_per_run'] == 54000
    assert sum(result['budget'].values()) == 4240
    variant = tmp_path / 'variant.yml'
    variant.write_text(yaml.safe_dump({'extends': str(SMOKE),
                                      'seed_policy': {'training_event_base': 500000,
                                                      'training_sumo_base': 800000,
                                                      'training_sumo_stride': 2000}}))
    manifest = plan(variant, tmp_path / 'planned')
    episode_plan = read_json(manifest.parent / 'plans/mixed_s7.json')
    assert episode_plan['event_seed'] == 500007
    assert episode_plan['rows'][0]['sumo_seed'] == 814001
    assert not (manifest.parent / 'train').exists()


def test_configuration_cycles_empty_cycles_and_unknown_groups_rejected(tmp_path):
    path = tmp_path / 'cycle.yml'
    path.write_text('extends: cycle.yml\n')
    with pytest.raises(ValueError, match='Cyclic'):
        load_config(path)
    spec = load_config(PROFILE)
    spec['events']['mixed_cycle'] = []
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match='Invalid mixed_cycle'):
        load_config(path)
    spec = load_config(PROFILE)
    spec['target_groups']['validation_junctions'].append('intersection_1_1')
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match='disjoint'):
        load_config(path)
    spec = load_config(PROFILE)
    spec['seed_policy']['training_sumo_stride'] = 10
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match='stride'):
        load_config(path)


def test_free_space_is_not_a_stopping_distance_guarantee():
    engine = SimpleNamespace(
        lane=SimpleNamespace(getLength=lambda lane: 1000,
                             getLastStepVehicleIDs=lambda lane: ['follower']),
        vehicle=SimpleNamespace(getIDList=lambda: ['follower'], getLanePosition=lambda v: 100.,
                                getLength=lambda v: 5., getMinGap=lambda v: 2.5,
                                getSpeed=lambda v: 10., getDecel=lambda v: 5., getTau=lambda v: 1.))
    near = placement_probe(engine, 'lane', 110., 0.)
    assert near['space_available'] and near['braking_margin_m'] < 0
    far = placement_probe(engine, 'lane', 140., 0.)
    assert far['space_available'] and far['braking_margin_m'] > 0
    blocked = placement_probe(engine, 'lane', 100., 0.)
    assert not blocked['space_available']


def test_dynamic_placement_rejects_or_moves_within_tolerance_without_touching_traffic():
    engine = SimpleNamespace(
        simulation=SimpleNamespace(getDeltaT=lambda: 1.),
        lane=SimpleNamespace(getLength=lambda lane: 1000., getLastStepVehicleIDs=lambda lane: ['follower']),
        vehicle=SimpleNamespace(getIDList=lambda: ['follower'], getLanePosition=lambda v: 100.,
            getLength=lambda v: 5., getMinGap=lambda v: 2.5, getSpeed=lambda v: 10.,
            getDecel=lambda v: 5., getTau=lambda v: 1., getActionStepLength=lambda v: 1.))
    policy = {'version': 'stopping-distance-v1', 'clearance_m': 2., 'reaction_steps': 1}
    with pytest.raises(UnsafePlacementError) as failure:
        safe_position(engine, 'lane', 110., 0., policy)
    assert failure.value.diagnostics['selected_position'] is None
    position, detail = safe_position(engine, 'lane', 110., 40., policy)
    # Moving behind the follower is allowed if nearest; it cannot trap it by overlap.
    assert 70 <= position <= 150
    assert detail['minimum_margin_m'] is None or detail['minimum_margin_m'] >= 0
    position, detail = safe_position(engine, 'lane', 140., 2., policy)
    assert position >= 139.5 and detail['minimum_margin_m'] >= 0
    probe = placement_probe(engine, 'lane', 140., 0., policy)
    assert probe['braking_margin_m'] == pytest.approx(.5)
    # Simultaneous events must also respect the obstacle reserved by an earlier event.
    with pytest.raises(UnsafePlacementError):
        safe_position(engine, 'lane', position, 0., policy, extra_positions=[position])
    # A roadworks footprint cannot reuse a passenger-car stopping envelope.
    with pytest.raises(UnsafePlacementError):
        safe_position(engine, 'lane', 140., 0., dict(policy, obstacle_length_m=30))


def test_execution_policy_and_demand_floor_are_frozen_and_validated(tmp_path):
    spec = load_config(PROFILE)
    spec['events']['blockage_min_vehicles'] = 100
    variant = tmp_path / 'demand.yml'
    variant.write_text(yaml.safe_dump(spec))
    manifest = read_json(plan(variant, tmp_path / 'plan'))
    pools = read_json(tmp_path / 'plan/target_pools.json')
    for group in pools.values():
        assert group['blockage_lanes']
        assert all(group['planned_lane_demand'][lane] >= 100 for lane in group['blockage_lanes'])
    row = read_json(tmp_path / 'plan/plans/mixed_s7.json')
    assert row['event_execution'] == manifest['config']['event_execution']
    # A disabled filter still records actual demand, not misleading zeros.
    spec['events']['blockage_min_vehicles'] = 0
    variant.write_text(yaml.safe_dump(spec))
    plan(variant, tmp_path / 'unfiltered')
    unfiltered = read_json(tmp_path / 'unfiltered/target_pools.json')
    assert any(v > 0 for group in unfiltered.values() for v in group['planned_lane_demand'].values())
    spec['event_execution']['placement']['clearance_m'] = float('nan')
    variant.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match='clearance_m'):
        load_config(variant)


def test_forced_placement_plans_minimal_clearance_without_mutating_probe():
    data = {'occupied': (110., 0.), 'fast_follower': (90., 10.), 'far_ahead': (180., 0.)}
    engine = SimpleNamespace(
        simulation=SimpleNamespace(getDeltaT=lambda: 1.),
        lane=SimpleNamespace(getLength=lambda lane: 1000., getLastStepVehicleIDs=lambda lane: list(data)),
        vehicle=SimpleNamespace(getIDList=lambda: list(data), getLanePosition=lambda v: data[v][0],
            getLength=lambda v: 5., getMinGap=lambda v: 2.5, getSpeed=lambda v: data[v][1],
            getDecel=lambda v: 5., getTau=lambda v: 1., getActionStepLength=lambda v: 1.))
    policy = dict(version='clear-conflicts-v1', clearance_m=2, reaction_steps=1, obstacle_length_m=5)
    position, decision = safe_position(engine, 'lane', 110., 0., policy)
    assert position == 110.
    assert {v['vehicle'] for v in decision['vehicles_to_remove']} == {'occupied', 'fast_follower'}
    assert len(data) == 3  # availability scans must never delete traffic
    with pytest.raises(UnsafePlacementError):
        safe_position(engine, 'lane', 110., 0., policy, extra_positions=[110.])
    position, decision = safe_position(engine, 'lane', 300., 0., policy)
    assert position == 300 and not decision['vehicles_to_remove']
    assert decision['minimum_margin_m'] >= 0
    config = load_config(PROFILE)
    assert config['event_execution']['placement']['version'] == 'clear-conflicts-v1'
    assert config['preflight']['admission_basis'] == 'injection'


def test_preflight_cli_finishes_assessment_without_training(tmp_path):
    output = tmp_path / 'preflight'
    cmd = [sys.executable, '-m', 'tools.run_paper_baseline', 'preflight', '--config', str(SMOKE),
           '--output', str(output), '--workers', '1']
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120,
                            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1'))
    # A completed assessment with weak effects returns 2, not a false pass.
    assert result.returncode == 2, result.stdout + result.stderr
    summary = read_json(output / 'preflight_summary.json')
    assert summary['status'] == 'assessment_completed' and summary['admission'] == 'not_ready'
    assert summary['physical_cases_passed'] and summary['reset_replay_passed']
    assert summary['coverage']['completed_simulations'] == 10
    assert summary['placement_probes']['checks'] > 0
    assert not (output / 'train').exists()
    assert read_json(output / 'manifest.json')['budget']['training_episodes'] == 0
    assert read_json(output / 'runs/fixedtime_s101/results.json')['model_unchanged']
    manifest = read_json(output / 'manifest.json')
    assert all(name in manifest['source_sha256'] for name in
               ['agent/fixedtime.py', 'agent/maxpressure.py', 'agent/colight.py'])


def test_forced_injection_clears_occupied_position_at_exact_onset():
    from test_sumo_events import NET, LANE, EDGE, add_probe, advance
    from world.sumo_events import Event, Schedule, SumoEventRuntime
    import libsumo as engine
    import sumolib
    engine.start([sumolib.checkBinary("sumo"), "-n", NET, "--seed", "7",
                  "--no-step-log", "true", "--no-warnings", "true", "--time-to-teleport", "-1"])
    try:
        from world.sumo_events.safe_placement import install_safe_placement
        events = (Event('forced', 'lane_blockage', 1, 5, lane_id=LANE, position=650),
                  Event('next', 'lane_blockage', 5, 8, lane_id=LANE, position=650))
        rt = SumoEventRuntime(NET, Schedule(events))
        install_safe_placement(rt, dict(version='clear-conflicts-v1', clearance_m=2,
                                       reaction_steps=1, obstacle_length_m=5))
        rt.bind(engine)
        add_probe(engine, 'occupied', [EDGE, 'road_2_1_1'], LANE, 650)
        engine.vehicle.setSpeed('occupied', 0)
        add_probe(engine, 'unaffected', [EDGE, 'road_2_1_0'], EDGE + '_1', 590, 5)
        advance(engine, rt, 1)
        assert 'occupied' not in engine.vehicle.getIDList()
        assert 'unaffected' in engine.vehicle.getIDList()
        assert rt.audit()['actual_positions']['forced'] == 650
        assert rt.audit()['removed_vehicles'][0]['vehicle'] == 'occupied'
        assert rt.reports()[0].status == 'active'
        # Same-boundary retirement/new insertion and eventual restoration work.
        advance(engine, rt, 9)
        audit = rt.audit()
        assert len(audit['removed_vehicles']) == 1
        assert {(r['event_id'], r['status'], r['updated_at']) for r in audit['transitions']} == {
            (e.event_id, state, time) for e in events for state, time in [('active', e.begin), ('cleared', e.end)]}
        assert not rt._obstacles
        rt.close()

    finally:
        engine.close()
