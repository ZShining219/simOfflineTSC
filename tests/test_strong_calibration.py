"""Strong profiles, frozen-policy identity, exposure selection and sham replay."""
import copy
import csv
import json
from pathlib import Path

import pytest

from common.paper_experiment import load_config, plan, read_json, write_json, file_digest
from world.sumo_events.calibration import describe, prepare, freeze_cases

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'configs/tsc/colight_strong_calibration.yml'


def test_budget_and_profile_validation(tmp_path):
    import yaml
    spec = load_config(PROFILE)
    assert describe(spec)['budget'] == {'training_episodes': 0, 'base_simulations': 152,
                                        'maximum_clearance_controls': 48, 'maximum_simulations': 200}
    with pytest.raises(ValueError, match='preflight'):
        plan(PROFILE, tmp_path / 'forbidden_training')
    spec['calibration']['profiles']['S2']['blocked_lanes'] = 3
    path = tmp_path / 'bad.yml';path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match='lane count'):
        load_config(path)


def test_pinned_policies_and_exposure_only_case_bank(tmp_path):
    if not all((ROOT / v['publication']).exists() for v in load_config(PROFILE)['calibration']['frozen_policies'].values()):
        pytest.skip('Historical calibration policy artifacts are not installed')
    manifest = read_json(prepare(PROFILE, tmp_path / 'prepared'))
    root = Path(manifest['root'])
    assert len(read_json(root / 'frozen_policies.json')) == 2
    assert manifest['budget']['maximum_simulations'] == 200
    assert len({t['output_path'] for t in manifest['tasks']}) == len(manifest['tasks'])
    assert all(t['depends_on'] == ['reference_physical_gate'] for t in manifest['tasks']
               if (t.get('controller') or '').startswith('colight_'))
    lanes = read_json(root / 'target_pools.json')['validation']['lanes']
    for name in manifest['config']['preflight']['controllers']:
        for seed in manifest['config']['preflight']['seeds']:
            path = root / 'runs' / f'{name}_s{seed}' / 'exposure.csv';path.parent.mkdir(parents=True)
            fields = ['simulation_time','queue_vehicles','running','sum_speed_mps','pending_due',
                      'arrived','lane_queues_json','lane_exits_json','lane_entries_json']
            with path.open('w') as f:
                w = csv.DictWriter(f, fieldnames=fields);w.writeheader()
                w.writerow(dict(zip(fields,[901,0,1,10,0,0,'{}',json.dumps({l:10 for l in lanes}),'{}'])))
            write_json(path.parent / 'cases/normal.json', {'status':'passed','result':{'timeline':str(path)}})
    frozen = freeze_cases(manifest)
    bank = read_json(root / 'case_bank.json')
    assert file_digest(root / 'case_bank.json') == frozen['bank_sha256']
    assert len(bank['cases']) == 18
    for row in bank['cases']:
        events = row['schedule']['events']
        if row['kind'] == 'lane_blockage' and row['severity'] == 'S2':
            assert len(events) == 2
            assert events[0]['lane_id'] != events[1]['lane_id']
            assert events[0]['lane_id'].rsplit('_',1)[0] == events[1]['lane_id'].rsplit('_',1)[0]
        duration = 600 if row['severity']=='S1' else 900
        assert all(e['end']-e['begin']==duration for e in events)
    assert freeze_cases(manifest) == frozen


def test_clearance_only_replays_two_lanes_without_obstacles():
    import libsumo as engine
    import sumolib
    from test_sumo_events import NET, LANE, EDGE, add_probe, advance
    from world.sumo_events import SumoEventRuntime, Schedule, Event
    from world.sumo_events.safe_placement import install_safe_placement, install_clearance_replay
    policy = dict(version='clear-conflicts-v1', clearance_m=2, reaction_steps=1, obstacle_length_m=30)
    removed = None
    for sham in (False, True):
        engine.start([sumolib.checkBinary('sumo'), '-n', NET, '--seed','7',
                      '--no-step-log','true','--no-warnings','true','--time-to-teleport','-1'])
        try:
            schedule = Schedule(()) if sham else Schedule(tuple(
                Event('block'+str(i),'lane_blockage',1,5,lane_id=lane,position=650)
                for i,lane in enumerate((LANE,EDGE+'_1'))))
            rt = SumoEventRuntime(NET,schedule);install_safe_placement(rt,policy)
            if sham:install_clearance_replay(rt,removed)
            rt.bind(engine)
            for ident,lane,next_edge in [('a',LANE,'road_2_1_1'),('b',EDGE+'_1','road_2_1_0')]:
                add_probe(engine,ident,[EDGE,next_edge],lane,650)
                engine.vehicle.setSpeed(ident,0)
            advance(engine,rt,1)
            assert {v['vehicle'] for v in rt.removed_vehicles} == {'a','b'}
            if sham:
                assert not rt._obstacles and not rt.reports()
                assert all(v['reason']=='clearance_only_replay' for v in rt.removed_vehicles)
            else:
                removed=copy.deepcopy(rt.removed_vehicles)
                assert len(rt._obstacles)==2
            advance(engine,rt,6)
            rt.close()
        finally:engine.close()


def test_exposure_reuse_preserves_global_budget_and_rejects_changed_artifact(tmp_path):
    import yaml
    from common.paper_experiment import digest
    from world.sumo_events.calibration import evaluate
    if not all((ROOT / v['publication']).exists() for v in load_config(PROFILE)['calibration']['frozen_policies'].values()):
        pytest.skip('Historical calibration policy artifacts are not installed')
    old = read_json(prepare(PROFILE, tmp_path / 'old'))
    root = Path(old['root']);budget = read_json(root / 'execution_budget.json')
    for name in old['config']['preflight']['controllers']:
        for seed in old['config']['preflight']['seeds']:
            folder = root / 'runs' / f'{name}_s{seed}'
            path = folder / 'timeline.csv';path.parent.mkdir(parents=True)
            path.write_text('immutable normal exposure fixture\n')
            case = dict(case_id='normal',kind='normal',sumo_seed=seed,target_split='none',
                        schedule={'schema_version':'sumo-events-v1','events':[]})
            write_json(folder / 'cases/normal.json', dict(status='passed',case_hash=digest(case),
                       protocol_hash=old['protocol_hash'],result={'timeline':str(path)}))
            budget['attempts'].append({'group':folder.name,'case_id':'normal'})
    write_json(root / 'execution_budget.json',budget)
    config = tmp_path / 'reuse.yml'
    config.write_text(yaml.safe_dump({'extends':str(PROFILE),'calibration':{'exposure_source':str(root)}}))
    new = read_json(prepare(config,tmp_path / 'new'))
    reused = read_json(Path(new['root']) / 'execution_budget.json')
    assert reused['started_at_unix'] == budget['started_at_unix']
    assert reused['attempts'] == budget['attempts']
    task = next(t for t in new['tasks'] if t['run_id']=='exposure_fixedtime_s101')
    case = dict(case_id='normal',kind='normal',sumo_seed=101,target_split='none',
                schedule={'schema_version':'sumo-events-v1','events':[]})
    assert evaluate(new,task,None,None,None,case)['reused_from']['protocol_hash']==old['protocol_hash']
    (root / 'runs/fixedtime_s101/timeline.csv').write_text('changed\n')
    with pytest.raises(ValueError,match='artifact changed'):
        evaluate(new,task,None,None,None,case)
