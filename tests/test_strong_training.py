"""Strong training plans, joint severity, and the auditable evaluation path."""
from collections import Counter
from pathlib import Path
import json
import copy
import pytest

from common.paper_experiment import plan, describe_config, read_json, digest
from tools.training_dashboard import ExperimentProjection

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'configs/tsc/colight_strong150.yml'


def test_formal_plan_balances_joint_profiles_and_keeps_holdouts(tmp_path):
    path = plan(PROFILE,tmp_path/'formal');m = read_json(path)
    assert m['budget'] == describe_config(PROFILE)['budget'] == {
        'training_episodes':1500,'validation_episodes':490,'test_episodes':750,'maximum_clearance_episodes':440}
    cases=read_json(path.parent/'cases.json')
    assert len(cases['validation'])==7 and len(cases['test'])==75
    assert Counter((c['kind'],c.get('severity')) for c in cases['validation']) == Counter({
        ('normal',None):1,('lane_blockage','S1'):1,('lane_blockage','S2'):1,
        ('road_closure','S1'):1,('road_closure','S2'):1,('global_rain','S1'):1,('global_rain','S2'):1})
    held={digest(c['schedule']) for group in cases.values() for c in group if c['kind']!='normal'}
    for seed in m['config']['seeds']:
        mixed=read_json(path.parent/f'plans/mixed_s{seed}.json')['rows']
        normal=read_json(path.parent/f'plans/normal_s{seed}.json')['rows']
        assert Counter(c['kind'] for c in mixed)=={'normal':75,'road_closure':45,'lane_blockage':30}
        assert Counter(c['severity'] for c in mixed if c['kind']=='road_closure')=={'S1':22,'S2':23}
        assert Counter(c['severity'] for c in mixed if c['kind']=='lane_blockage')=={'S1':15,'S2':15}
        assert [c['sumo_seed'] for c in normal]==[c['sumo_seed'] for c in mixed]
        assert all(c['severity'] is None and not c['schedule']['events'] for c in normal)
        for c in mixed:
            if c['kind']=='normal':continue
            assert digest(c['schedule']) not in held
            p=m['config']['strong_events']['profiles'][c['severity']]
            assert all(e['end']-e['begin']==p['duration'] for e in c['schedule']['events'])
            if c['kind']=='lane_blockage':
                events=c['schedule']['events'];assert len(events)==p['blocked_lanes']
                assert len({e['lane_id'].rsplit('_',1)[0] for e in events})==1
                assert len({e['lane_id'] for e in events})==len(events)
    for c in cases['test']:
        if c['kind']=='global_rain':
            assert c['schedule']['events'][0]['speed_factor']==m['config']['strong_events']['profiles'][c['severity']]['rain_factor']


def test_dashboard_does_not_average_severity_levels_together(tmp_path):
    run=tmp_path/'train/normal_s1';run.mkdir(parents=True)
    out=tmp_path/'validation';out.mkdir()
    cases=[dict(kind='road_closure',severity=level,system_time_per_vehicle=j,
                physical_system_time_per_vehicle=j,net_event_J_percent=j-300,
                mean_queue=1,arrived=20,forced_removed=0) for level,j in [('S1',310),('S2',350)]]
    (out/'results.json').write_text(json.dumps(dict(protocol_hash='p',checkpoint_episode=25,cases=cases)))
    m=dict(root=str(tmp_path),protocol_hash='p',config={'episodes':150},runs=[dict(run_id='normal_s1',regime='normal',seed=1,output_path=str(run))],
           tasks=[dict(kind='evaluate',role='validation',episode=25,run='normal_s1',output_path=str(out))])
    p=tmp_path/'manifest.json';p.write_text(json.dumps(m));points=ExperimentProjection(p).points()
    assert points['N150_seed1','Validation/road_closure/S1/J_seconds',25]==310
    assert points['N150_seed1','Validation/road_closure/S2/net_event_J_percent',25]==50
    assert ('N150_seed1','Validation/road_closure/J_seconds',25) not in points
    m['protocol_hash'] = 'recovery-view'
    m['tasks'][0]['source_protocol_hash'] = 'p'
    p.write_text(json.dumps(m))
    assert ExperimentProjection(p).points() == points
    m['tasks'][0]['source_protocol_hash'] = 'unexpected'
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='another protocol'):
        ExperimentProjection(p).points()


def test_frozen_evaluation_repair_rejects_policy_and_semantic_changes():
    from common.paper_experiment import verify_evaluation_compatibility, file_digest
    name = 'world/sumo_events/safe_placement.py'
    current_hash = file_digest(ROOT/name)
    original = {'hash':'old', 'implementation_sha256':{name:'previous'},
                'model':{'phase':True}, 'seeds':{'training':147}}
    current = copy.deepcopy(original)
    current.update(hash='new', implementation_sha256={name:current_hash})
    repair = {name:{'before':'previous', 'after':current_hash}}
    verify_evaluation_compatibility(original,current,repair)
    bad = copy.deepcopy(current);bad['model']['phase'] = False
    with pytest.raises(ValueError,match='semantics'):
        verify_evaluation_compatibility(original,bad,repair)
    with pytest.raises(ValueError,match='policy'):
        verify_evaluation_compatibility(original,current,{'agent/colight.py':{}})
    with pytest.raises(ValueError,match='identity'):
        verify_evaluation_compatibility(original,current,{name:{'before':'wrong','after':current_hash}})


def test_extension_rejects_changed_prefix_settings_and_parent_artifacts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from common import paper_experiment as experiment
    parent = {'trainer':{'episodes':150, 'learning_start':1000},
              'implementation_sha256':{}, 'hash':'parent', 'simulator_config_sha256':'old'}
    original = {'rows':[{'episode':i} for i in range(1,151)],'seconds':3600}
    current = copy.deepcopy(parent)
    current['trainer']['episodes']=500
    target = dict(original, rows=original['rows']+[{'episode':151}])
    sim={'combined_file':'s','roadnetFile':'n','flowFile':'f'}
    documents={'parent':parent,'prefix':original,'old_sim':sim,'new_sim':sim}
    monkeypatch.setattr(experiment,'read_json',lambda p:copy.deepcopy(documents[str(p)]))
    monkeypatch.setattr(experiment,'file_digest',lambda p:'expected')
    ext={'inputs':{},'contract':'parent','simulator_config':'old_sim','parent_plan':'prefix',
         'parent_episode':150,'source_changes':{}}
    trainer=SimpleNamespace(experiment_contract=current,path='new_sim',
                            episode_events=SimpleNamespace(plan=target))
    target['rows'][0]={'episode':99}
    with pytest.raises(ValueError,match='prefix'):
        experiment.extension_checkpoint(trainer,ext)
    target['rows'][0]={'episode':1}
    current['trainer']['learning_start']=1
    with pytest.raises(ValueError,match='trainer semantics'):
        experiment.extension_checkpoint(trainer,ext)
    ext['inputs']={'checkpoint':'tampered'}
    with pytest.raises(ValueError,match='artifact changed'):
        experiment.extension_checkpoint(trainer,ext)
