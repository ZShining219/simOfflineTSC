"""Staged, exposure-selected event calibration through the shared preflight CLI.

This module owns a development assessment, not another trainer or launcher.
Normal exposure -> immutable case bank -> reference policies -> frozen policies.
"""
import collections
import copy
import csv
import fcntl
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import yaml

from common.paper_experiment import (ROOT, SCHEMA, load_config, _assets, _case_factory,
    source_identity, environment_identity, digest, file_digest, write_json, read_json, checked_json)
from common.registry import Registry
from .schema import Event, Schedule
from .runtime import SumoEventRuntime
from .episodes import EpisodeEvents
from .preflight import _controllers


def validate_config(spec):
    c = spec['calibration']
    c.setdefault('exposure_source', None)
    fields = {'version', 'profiles', 'local_begin', 'rain_begins', 'target_count',
              'exposure_fraction', 'selection_seed', 'frozen_policies', 'max_simulations',
              'wall_seconds', 'thresholds', 'recovery_seconds', 'recovery_hold_seconds',
              'recovery_absolute_queue', 'recovery_relative_queue', 'exposure_source'}
    if set(c) != fields or c['version'] != 'strong-events-stage-a-v1':
        raise ValueError('Invalid calibration configuration fields/version')
    if spec['purpose'] != 'engineering' or spec['agent'] != 'paper_colight':
        raise ValueError('Calibration is an engineering-only CoLight assessment')
    if spec['preflight']['controllers'] != ['fixedtime', 'maxpressure']:
        raise ValueError('Exposure must use FixedTime and MaxPressure only')
    if spec['event_execution']['placement']['version'] != 'clear-conflicts-v1':
        raise ValueError('Calibration requires audited forced placement')
    for key in ('target_count', 'max_simulations', 'wall_seconds', 'recovery_seconds', 'recovery_hold_seconds'):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError('Invalid calibration ' + key)
    if not 0 < c['exposure_fraction'] <= 1 or not 0 <= c['selection_seed'] < 2**31:
        raise ValueError('Invalid exposure selection')
    if not c['profiles'] or set(c['profiles']) != {'S1', 'S2'}:
        raise ValueError('Calibration requires explicit S1 and S2 profiles')
    onsets = [c['local_begin'], *c['rain_begins']]
    if len(c['rain_begins']) != c['target_count'] or len(set(c['rain_begins'])) != len(c['rain_begins']):
        raise ValueError('One distinct rain onset per development instance required')
    if any(type(x) is not int or x <= 0 or x % spec['action_interval'] for x in onsets):
        raise ValueError('Calibration onsets must be positive and action-aligned')
    for level, p in c['profiles'].items():
        if set(p) != {'duration', 'blocked_lanes', 'rain_factor'}:
            raise ValueError('Profile must jointly define duration, lanes, rain factor')
        if type(p['duration']) is not int or p['duration'] <= 0 or p['duration'] % spec['action_interval']:
            raise ValueError('Invalid profile duration')
        if type(p['blocked_lanes']) is not int or not 1 <= p['blocked_lanes'] <= 2 or not 0 < p['rain_factor'] < 1:
            raise ValueError('Invalid profile lane count or rain factor')
        if max(onsets) + p['duration'] + c['recovery_seconds'] >= spec['seconds']:
            raise ValueError('Calibration leaves insufficient recovery horizon')
    if (c['profiles']['S2']['duration'] < c['profiles']['S1']['duration'] or
            c['profiles']['S2']['blocked_lanes'] < c['profiles']['S1']['blocked_lanes'] or
            c['profiles']['S2']['rain_factor'] > c['profiles']['S1']['rain_factor']):
        raise ValueError('S2 must not be weaker than S1')
    if len(c['frozen_policies']) != 2:
        raise ValueError('Stage A requires two frozen CoLight policies')
    for name, p in c['frozen_policies'].items():
        if not name.startswith('colight_') or set(p) != {'publication', 'sha256'}:
            raise ValueError('Frozen policies require named publication and pinned SHA256')
    required = {'local_queue_delta', 'flow_drop_fraction', 'speed_drop_fraction', 'net_J_fraction', 'minimum_success_fraction'}
    if set(c['thresholds']) != required or any(not math.isfinite(v) or v <= 0 for v in c['thresholds'].values()):
        raise ValueError('Invalid physical/response thresholds')
    if any(c[k] < 0 or not math.isfinite(c[k]) for k in ('recovery_absolute_queue', 'recovery_relative_queue')):
        raise ValueError('Invalid recovery band')
    if describe(spec)['budget']['maximum_simulations'] > c['max_simulations']:
        raise ValueError('Configured simulation cap below worst-case stage budget')


def describe(spec):
    c = spec['calibration']
    groups = (len(spec['preflight']['controllers']) + len(c['frozen_policies'])) * len(spec['preflight']['seeds'])
    events = 3 * len(c['profiles']) * c['target_count']
    base = groups * (1 + events)
    return {'config': spec, 'config_hash': digest(spec), 'stage': 'strong_calibration',
            'budget': {'training_episodes': 0, 'base_simulations': base,
                       'maximum_clearance_controls': groups * len(c['profiles']) * c['target_count'],
                       'maximum_simulations': base + groups * len(c['profiles']) * c['target_count']}}


def checkpoint_identity(name, entry, spec, world):
    """Pin external weights and check the unchanged model/observation contract."""
    import torch
    publication = ROOT / entry['publication']
    pub = read_json(publication)
    if pub['evaluation_sha256'] != entry['sha256'] or file_digest(pub['evaluation']) != entry['sha256']:
        raise ValueError('Frozen policy checkpoint hash mismatch')
    run = publication.parent.parent
    binding, reward = read_json(run / 'experiment_contract.json'), read_json(run / 'reward_contract.json')
    if (binding['network_sha256'] != file_digest(world['roadnetFile']) or
            binding['route_sha256'] != file_digest(world['flowFile']) or
            binding['signal_control'] != spec['signal_control'] or
            binding['trainer']['action_interval'] != spec['action_interval'] or
            binding['model']['name'] != 'paper_colight'):
        raise ValueError('Frozen policy network/traffic/action/signal mismatch')
    from common.utils import load_config as legacy_config
    model = legacy_config(str(ROOT / 'configs/tsc/colight.yml'))[0]['model']
    model.update(spec['model'])
    for k in ('phase', 'one_hot', 'vehicle_max', 'NEIGHBOR_NUM', 'N_LAYERS',
              'INPUT_DIM', 'OUTPUT_DIM', 'NODE_EMB_DIM', 'NUM_HEADS', 'NODE_LAYER_DIMS_EACH_HEAD', 'OUTPUT_LAYERS'):
        if binding['model'][k] != model[k]:
            raise ValueError('Frozen policy model setting changed: ' + k)
    verified = {}
    for filename, expected in binding['implementation_sha256'].items():
        if filename.startswith(('agent/', 'generator/')) or filename in (
                'world/world_sumo.py', 'world/sumo_signal_control.py', 'common/paper_rewards.py', 'configs/rewards/colight.yml'):
            if file_digest(ROOT / filename) != expected:
                raise ValueError('Frozen policy observation/model source changed: ' + filename)
            verified[filename] = expected
    payload = torch.load(pub['evaluation'], map_location='cpu')
    if (payload['checkpoint_type'] != 'evaluation' or payload['episode'] != 150 or
            payload['experiment_contract_hash'] != binding['hash'] or
            payload['reward_contract_hash'] != reward['hash'] or len(payload['agents']) != 1):
        raise ValueError('Frozen policy payload/training/reward binding mismatch')
    return {'name': name, 'checkpoint': pub['evaluation'], 'sha256': entry['sha256'],
            'publication': str(publication.resolve()), 'publication_sha256': file_digest(publication),
            'model_seed': binding['seeds']['training'], 'training_contract_hash': binding['hash'],
            'reward_contract_hash': reward['hash'], 'verified_policy_sources': verified,
            'purpose': 'Frozen greedy development probe in a new event protocol; no replay/optimizer transfer.'}


def prepare(config_path, output):
    spec = load_config(config_path)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'experiment.yml').write_text(yaml.safe_dump(spec, sort_keys=False))
    world, provenance = _assets(spec, root)
    _, pools = _case_factory(spec, world['roadnetFile'], world['flowFile'])
    policies = {name: checkpoint_identity(name, p, spec, world) for name, p in spec['calibration']['frozen_policies'].items()}
    write_json(root / 'frozen_policies.json', policies)
    source = source_identity(spec['agent'])
    for name in ('agent/fixedtime.py', 'agent/maxpressure.py'):
        source[name] = file_digest(ROOT / name)
    for name in source:
        target = root / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    protocol = digest({'stage': 'strong_calibration', 'config': spec, 'sources': source,
                       'policies': policies, 'scenario': provenance, 'environment': environment_identity(spec['agent'])})
    sim = read_json(ROOT / 'configs/sim' / (spec['network'] + '.cfg'))
    sim.update(world, name='strong_calibration')
    write_json(root / 'simulator.cfg', sim)
    write_json(root / 'target_pools.json', pools)
    episode_plan = {'schema_version': SCHEMA, 'protocol_hash': protocol, 'seconds': spec['seconds'],
                    'action_interval': spec['action_interval'], 'event_execution': spec['event_execution'],
                    'record_lane_flow': True, 'rows': []}
    write_json(root / 'episode_plan.json', episode_plan)
    tasks = []
    def task(task_id, phase, controller=None, seed=None, dependencies=()):
        out = root / 'runs' / f'{controller}_s{seed}' if controller else root
        marker = out / (phase + '_completed.json')
        cmd = [sys.executable, '-m', 'tools.run_paper_baseline', 'worker', '--manifest', str(root/'manifest.json'), '--task-id', task_id]
        tasks.append({'run_id': task_id, 'kind': 'preflight', 'phase': phase, 'controller': controller,
            'seed': seed, 'output_path': str(root / 'task_runs' / task_id), 'case_output_path': str(out),
            'pool': 'eval', 'depends_on': list(dependencies),
            'command': cmd, 'resume_command': cmd + ['--resume'], 'cwd': str(ROOT),
            'env': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '-1'},
            'completion': {'path': str(marker), 'json_field': 'protocol_hash', 'equals': protocol}})
    refs = spec['preflight']['controllers'];seeds = spec['preflight']['seeds']
    for name in refs:
        for seed in seeds:task(f'exposure_{name}_s{seed}', 'exposure', name, seed)
    exposure = [t['run_id'] for t in tasks]
    task('freeze_case_bank', 'freeze', dependencies=exposure)
    for name in refs:
        for seed in seeds:task(f'assess_{name}_s{seed}', 'assessment', name, seed, ['freeze_case_bank'])
    references = [t['run_id'] for t in tasks if t['phase']=='assessment']
    task('reference_physical_gate', 'reference_gate', dependencies=references)
    for name in policies:
        for seed in seeds:task(f'assess_{name}_s{seed}', 'assessment', name, seed, ['reference_physical_gate'])
    manifest = {'schema_version': SCHEMA, 'stage': 'strong_calibration', 'root': str(root),
        'config': spec, 'protocol_hash': protocol, 'source_sha256': source, 'scenario': provenance,
        'environment': environment_identity(spec['agent']), 'runs': [], 'tasks': tasks,
        'budget': describe(spec)['budget'],
        'input_sha256': {str(p.relative_to(root)): file_digest(p) for p in root.rglob('*') if p.is_file()}}
    manifest['manifest_hash'] = digest(manifest)
    write_json(root/'manifest.json', manifest)
    # Runtime metadata is separate from the immutable manifest input list.
    budget = {'started_at_unix': time.time(), 'attempts': []}
    if spec['calibration']['exposure_source']:
        budget = reuse_exposure(manifest, spec['calibration']['exposure_source'])
    write_json(root/'execution_budget.json', budget)
    return root/'manifest.json'


def reuse_exposure(manifest, source):
    """Reuse only completed normal probes under identical simulator/policy code.

    The old immutable artifacts remain authoritative; new wrappers explicitly
    record their origin. Queue/planning fixes do not require rerunning SUMO.
    """
    import ast
    from common.paper_experiment import load_manifest
    old = load_manifest(ROOT / source / 'manifest.json', verify=False)
    root = Path(manifest['root']); previous = Path(old['root'])
    for key in ('seconds', 'action_interval', 'model', 'signal_control', 'event_execution'):
        if old['config'][key] != manifest['config'][key]:
            raise ValueError('Exposure reuse configuration changed: ' + key)
    for key in ('controllers', 'seeds', 'fixedtime_green', 'maxpressure_min_green'):
        if old['config']['preflight'][key] != manifest['config']['preflight'][key]:
            raise ValueError('Exposure reuse controller setting changed')
    if old['environment'] != manifest['environment'] or old['scenario'] != manifest['scenario']:
        raise ValueError('Exposure reuse environment/assets changed')
    for name, expected in old['source_sha256'].items():
        if name.startswith(('agent/', 'generator/', 'world/')) and name != 'world/sumo_events/calibration.py':
            if file_digest(ROOT / name) != expected:
                raise ValueError('Exposure reuse simulation source changed: ' + name)
    # Compare the complete simulation/cleanup block, excluding cache provenance
    # checks and output-directory wiring outside that block.
    def evaluation_ast(path):
        node = next(n for n in ast.parse(Path(path).read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
        simulation = next(n for n in node.body if isinstance(n, ast.Try))
        return ast.dump(simulation, include_attributes=False)
    if evaluation_ast(previous/'source_snapshot/world/sumo_events/calibration.py') != evaluation_ast(ROOT/'world/sumo_events/calibration.py'):
        raise ValueError('Exposure evaluation behavior changed')
    artifacts = []
    for name in manifest['config']['preflight']['controllers']:
        for seed in manifest['config']['preflight']['seeds']:
            source_path = previous/'runs'/f'{name}_s{seed}'/'cases/normal.json'
            row = read_json(source_path)
            if row['status'] != 'passed' or row['protocol_hash'] != old['protocol_hash']:
                raise ValueError('Exposure reuse requires a completed normal case')
            row = dict(row, protocol_hash=manifest['protocol_hash'], reused_from={
                'path': str(source_path), 'sha256': file_digest(source_path),
                'timeline_sha256': file_digest(row['result']['timeline']), 'protocol_hash': old['protocol_hash']})
            write_json(root/'runs'/f'{name}_s{seed}'/'cases/normal.json', row)
            artifacts.append(row['reused_from'])
    old_budget = read_json(previous/'execution_budget.json')
    if len(old_budget['attempts']) != len(artifacts) or any(x['case_id'] != 'normal' for x in old_budget['attempts']):
        raise ValueError('Reuse is limited to an exposure-only interrupted assessment')
    write_json(root/'exposure_reuse.json', {'source': str(previous), 'artifacts': artifacts,
                                          'reason': 'Task output isolation fix; normal dynamics unchanged.'})
    return dict(old_budget, exposure_reused_from=str(previous))


def attempt(manifest, group, case):
    root = Path(manifest['root']);c = manifest['config']['calibration']
    with (root/'budget.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        state = read_json(root/'execution_budget.json')
        if time.time()-state['started_at_unix'] > c['wall_seconds']:
            raise RuntimeError('Calibration wall-clock budget exhausted')
        if len(state['attempts']) >= c['max_simulations']:
            raise RuntimeError('Calibration simulation budget exhausted')
        state['attempts'].append({'group': group, 'case_id': case, 'started_at_unix': time.time()})
        write_json(root/'execution_budget.json', state)


def timeline(path):
    data = []
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            data.append({'time': int(float(r['simulation_time'])), 'queue': float(r['queue_vehicles']),
                'running': int(r['running']), 'sum_speed': float(r['sum_speed_mps']),
                'pending': int(r['pending_due']), 'arrived': int(r['arrived']),
                'lanes': json.loads(r['lane_queues_json']), 'exits': json.loads(r['lane_exits_json']),
                'entries': json.loads(r['lane_entries_json'])})
    return data


def window(rows, begin, end, lanes):
    a = [r for r in rows if begin < r['time'] <= end]
    count = sum(r['running'] for r in a)
    return {'seconds': len(a), 'mean_queue': statistics.mean(r['queue'] for r in a),
            'local_queue': statistics.mean(sum(r['lanes'].get(l, 0) for l in lanes) for r in a),
            'queue_integral': sum(r['queue'] for r in a),
            'local_queue_integral': sum(sum(r['lanes'].get(l, 0) for l in lanes) for r in a),
            'flow': sum(sum(r['exits'].get(l, 0) for l in lanes) for r in a),
            'mean_speed': sum(r['sum_speed'] for r in a)/count if count else 0,
            'last_pending': a[-1]['pending'], 'arrivals': a[-1]['arrived']-a[0]['arrived']}


def freeze_cases(manifest):
    root = Path(manifest['root']);spec = manifest['config'];c = spec['calibration']
    cfg = read_json(root/'simulator.cfg');net = Path(cfg['dir'])/cfg['roadnetFile']
    catalog = SumoEventRuntime(net, Schedule(())).network_catalog()
    pool = read_json(root/'target_pools.json')['validation']
    scores = collections.Counter();source_hashes = {}
    for name in spec['preflight']['controllers']:
        for seed in spec['preflight']['seeds']:
            p = root/'runs'/f'{name}_s{seed}'/'cases/normal.json';row = read_json(p)
            if row['status'] != 'passed':raise RuntimeError('Normal exposure failed')
            path = Path(row['result']['timeline']);source_hashes[str(p)] = file_digest(p)
            source_hashes[str(path)] = file_digest(path)
            for t in timeline(path):
                if c['local_begin'] < t['time'] <= c['local_begin']+c['profiles']['S1']['duration']:
                    scores.update(t['exits'])
    edge_lanes = collections.defaultdict(list)
    for lane,info in catalog.items():
        if not info['internal'] and info['motor_vehicle_lane']:edge_lanes[info['edge_id']].append(lane)
    candidates = []
    max_block = max(p['blocked_lanes'] for p in c['profiles'].values())
    for edge in pool['edges']:
        available = sorted([l for l in pool['blockage_lanes'] if catalog[l]['edge_id']==edge],key=lambda l:(-scores[l],l))
        if len(available)>=max_block and len(edge_lanes[edge])>max_block:
            candidates.append({'edge':edge,'lanes':available[:max_block],
                'blockage_exposure':sum(scores[l] for l in available[:max_block]),
                'closure_exposure':sum(scores[l] for l in edge_lanes[edge])})
    rng = random.Random(c['selection_seed']);selected = {}
    for kind,key in [('lane_blockage','blockage_exposure'),('road_closure','closure_exposure')]:
        ranked = sorted(candidates,key=lambda x:(-x[key],x['edge']))
        upper = ranked[:math.ceil(len(ranked)*c['exposure_fraction'])]
        upper = [x for x in upper if x[key]>0]
        if len(upper)<c['target_count']:raise ValueError('Insufficient exposed targets; do not retarget from policy failures')
        selected[kind] = rng.sample(upper,c['target_count'])
    bank=[]
    for level,p in c['profiles'].items():
        for kind in ('lane_blockage','road_closure','global_rain'):
            for i in range(c['target_count']):
                begin=c['rain_begins'][i] if kind=='global_rain' else c['local_begin'];events=[]
                if kind=='lane_blockage':
                    for j,lane in enumerate(selected[kind][i]['lanes'][:p['blocked_lanes']]):
                        events.append(Event(f'event_{j}',kind,begin,begin+p['duration'],lane_id=lane,
                            position=catalog[lane]['length']-spec['events']['block_distance'],
                            position_tolerance=spec['events']['position_tolerance']))
                elif kind=='road_closure':
                    events.append(Event('event_0',kind,begin,begin+p['duration'],edge_id=selected[kind][i]['edge']))
                else:events.append(Event('event_0',kind,begin,begin+p['duration'],speed_factor=p['rain_factor']))
                schedule=Schedule(tuple(events));SumoEventRuntime(net,schedule)
                bank.append({'case_id':f'{kind}_{level}_{i}','kind':kind,'severity':level,
                             'target_split':'global' if kind=='global_rain' else 'validation','schedule':schedule.to_dict()})
    value={'cases':bank,'selection':selected,'all_candidates':candidates,'exposure_lane_counts':dict(scores),
           'exposure_source_sha256':source_hashes,'protocol_hash':manifest['protocol_hash']}
    write_json(root/'case_bank.json',value)
    return {'bank_sha256':file_digest(root/'case_bank.json')}


def evaluate(manifest, task, world, owner, agents, case):
    import torch
    import traceback
    root=Path(manifest['root']);out=Path(task['case_output_path']);path=out/'cases'/(case['case_id']+'.json')
    if path.exists():
        prior=read_json(path)
        if prior['case_hash']!=digest(case) or prior['protocol_hash']!=manifest['protocol_hash']:
            raise ValueError('Existing calibration case identity mismatch')
        if 'reused_from' in prior:
            origin=prior['reused_from']
            if (file_digest(origin['path'])!=origin['sha256'] or
                    file_digest(prior['result']['timeline'])!=origin['timeline_sha256'] or
                    read_json(origin['path'])['result']!=prior['result']):
                raise ValueError('Reused exposure artifact changed')
        return prior
    attempt(manifest,out.name,case['case_id'])
    try:
        owner.select(case,'calibration');world.reset()
        for agent in agents:agent.reset()
        interval=1 if task['controller']=='fixedtime' else manifest['config']['action_interval']
        deadline=read_json(root/'execution_budget.json')['started_at_unix']+manifest['config']['calibration']['wall_seconds']
        with torch.no_grad():
            for t in range(manifest['config']['seconds']):
                if t%interval==0:
                    actions=np.concatenate([np.asarray(a.get_action(a.get_ob(),a.get_phase(),test=True)).reshape(-1) for a in agents])
                world.step(actions)
                if t%100==0 and time.time()>deadline:raise RuntimeError('Calibration wall-clock budget exhausted')
        result=owner.finish();rt=owner.current;raw=rt._engine
        assert not rt._obstacles and not rt._closed and rt._factor==1
        assert all(math.isclose(raw.lane.getMaxSpeed(l),v,abs_tol=1e-7) for l,v in rt._base_speed.items())
        assert all(set(raw.lane.getDisallowed(l))==set(v) for l,v in rt._base_permissions.items())
        expected={(e.event_id,s,at) for e in rt.schedule.events for s,at in [('active',e.begin),('cleared',e.end)]}
        assert {(x['event_id'],x['status'],x['updated_at']) for x in result['events']['transitions']}==expected
        if case.get('clearance_replay') is not None:
            assert {x['vehicle'] for x in result['events']['removed_vehicles']}=={x['vehicle'] for x in case['clearance_replay']}
        row={'status':'passed','result':result}
    except Exception as error:
        row={'status':'failed','error':str(error),'error_type':type(error).__name__,'traceback':traceback.format_exc()}
        try:row['simulation_time']=world.get_current_time();row['removals']=getattr(owner.current,'removed_vehicles',[])
        except Exception:pass
    finally:owner.close_stream();world.close()
    row.update(case_id=case['case_id'],kind=case['kind'],severity=case.get('severity'),
               case_hash=digest(case),protocol_hash=manifest['protocol_hash'],controller=task['controller'],seed=task['seed'])
    write_json(path,row)
    return row


def compare(spec, normal, event, control):
    c=spec['calibration'];base=normal['result'];result=event['result'];ref=control['result']
    begin=min(e['begin'] for e in result['schedule']['events']);end=max(e['end'] for e in result['schedule']['events'])
    # Flow is measured only across the target lanes/edge; closure queue also
    # includes upstream approaches, whose growth is part of the road's impact.
    queue_lanes=set(result['local_regions']['blockage_lanes']+result['local_regions']['closure_area_lanes'])
    flow_lanes=set(result['local_regions']['blockage_lanes'])
    if event['kind']=='road_closure':
        flow_lanes={l for l in queue_lanes if any(l.rsplit('_',1)[0]==e['edge_id'] for e in result['schedule']['events'])}
    b,e,s=timeline(base['timeline']),timeline(result['timeline']),timeline(ref['timeline'])
    bw,ew,sw=window(b,begin,end,queue_lanes),window(e,begin,end,queue_lanes),window(s,begin,end,queue_lanes)
    ef,sf=window(e,begin,end,flow_lanes)['flow'],window(s,begin,end,flow_lanes)['flow']
    recovery=0;recovered=None
    for br,er in zip(s,e):
        if not end<er['time']<=end+c['recovery_seconds']:continue
        bq=sum(br['lanes'].get(l,0) for l in queue_lanes) if queue_lanes else br['queue']
        eq=sum(er['lanes'].get(l,0) for l in queue_lanes) if queue_lanes else er['queue']
        bound=max(bq+c['recovery_absolute_queue'],bq*(1+c['recovery_relative_queue']))
        recovery=recovery+1 if eq<=bound else 0
        if recovery>=c['recovery_hold_seconds']:
            recovered=er['time']-end-c['recovery_hold_seconds']+1;break
    j=base['physical_system_time_per_vehicle'];net=result['physical_system_time_per_vehicle']-ref['physical_system_time_per_vehicle']
    return {'case_id':event['case_id'],'kind':event['kind'],'severity':event['severity'],
        'normal_J_physical':j,'event_J_physical':result['physical_system_time_per_vehicle'],
        'event_J_accounted':result['system_time_per_vehicle'],'control_J_physical':ref['physical_system_time_per_vehicle'],
        'total_J_delta':result['physical_system_time_per_vehicle']-j,'net_J_delta':net,'net_J_fraction':net/j,
        'removed':result['forced_removed'],'control_case_id':control['case_id'],'event_window':ew,
        'normal_window':bw,'control_window':sw,'local_queue_delta':ew['local_queue']-sw['local_queue'],
        'flow_drop_fraction':1-ef/sf if sf else None,
        'speed_drop_fraction':1-ew['mean_speed']/sw['mean_speed'] if sw['mean_speed'] else None,
        'recovery_seconds':recovered,'recovery_observation_seconds':c['recovery_seconds'],
        'recovery_window':window(e,end,end+c['recovery_seconds'],queue_lanes),
        'control_recovery_window':window(s,end,end+c['recovery_seconds'],queue_lanes),
        'pre_event_matches':result['pre_event_trace_sha256']==base['trace_prefixes_before'][str(int(begin))],
        'control_pre_event_matches':ref['trace_prefixes_before'][str(int(begin))]==base['trace_prefixes_before'][str(int(begin))],
        'ended_running':result['running'],'ended_pending':result['pending_due'],'arrived':result['arrived']}


def worker(manifest, task):
    import torch
    from utils.logger import hash_torch_state_dict
    from world.world_sumo import World
    from world.sumo_signal_control import install_signal_control
    root=Path(manifest['root']);out=Path(task['case_output_path']);out.mkdir(parents=True,exist_ok=True)
    phase=task['phase'];metadata={}
    if phase=='freeze':metadata=freeze_cases(manifest)
    elif phase=='reference_gate':
        for name in manifest['config']['preflight']['controllers']:
            for seed in manifest['config']['preflight']['seeds']:
                d=read_json(root/'runs'/f'{name}_s{seed}'/'assessment.json')
                if d['failures'] or any(not x['pre_event_matches'] or not x['control_pre_event_matches'] for x in d['comparisons']):
                    raise RuntimeError('Reference physical/pairing gate failed; frozen policies not launched')
    else:
        spec=manifest['config'];name=task['controller'];seed=task['seed']
        Registry.mapping['command_mapping']['setting']=SimpleNamespace(param={'sumo_seed':seed})
        Registry.mapping['logger_mapping']['path']=SimpleNamespace(path=str(out))
        world=World(str(root/'simulator.cfg'),interface='libsumo');world.sumo_cmd+=['--no-step-log','true']
        install_signal_control(world,spec['signal_control'])
        plan=read_json(root/'episode_plan.json');owner=EpisodeEvents(world,root/'episode_plan.json',digest(plan),out)
        policies=read_json(root/'frozen_policies.json')
        if name in policies:
            p=policies[name]
            if file_digest(p['checkpoint'])!=p['sha256']:raise ValueError('Frozen checkpoint changed')
            agents=_controllers(world,spec,'colight_untrained',p['model_seed'])
            payload=torch.load(p['checkpoint'],map_location='cpu');agents[0].model.load_state_dict(payload['agents'][0]['online_model_state_dict'],strict=True)
            agents[0].model.eval()
        else:agents=_controllers(world,spec,name,seed)
        models=[a.model for a in agents if a.model is not None]
        before=[hash_torch_state_dict(x.state_dict()) for x in models]
        normal_case={'case_id':'normal','kind':'normal','sumo_seed':seed,'target_split':'none','schedule':Schedule(()).to_dict()}
        results=[];comparisons=[]
        try:
            normal=evaluate(manifest,task,world,owner,agents,normal_case);results.append(normal)
            if normal['status']!='passed':raise RuntimeError('Normal baseline failed')
            if phase=='assessment':
                frozen=read_json(root/'freeze_completed.json');bank=read_json(root/'case_bank.json')
                if file_digest(root/'case_bank.json')!=frozen['bank_sha256']:raise ValueError('Frozen case bank changed')
                for template in bank['cases']:
                    case=dict(template,sumo_seed=seed);event=evaluate(manifest,task,world,owner,agents,case);results.append(event)
                    if event['status']=='passed':
                        removed=event['result']['events']['removed_vehicles'];control=normal
                        if removed:
                            sham=dict(case,case_id=case['case_id']+'_clearance',kind='clearance_only',
                                      schedule=Schedule(()).to_dict(),clearance_replay=removed)
                            control=evaluate(manifest,task,world,owner,agents,sham);results.append(control)
                        if control['status']=='passed':comparisons.append(compare(spec,normal,event,control))
                    write_json(out/'progress.json',{'cases_finished':len(results),'base_cases':1+len(bank['cases']),
                        'failures':sum(r['status']!='passed' for r in results),'comparisons':len(comparisons)})
                write_json(out/'assessment.json',{'protocol_hash':manifest['protocol_hash'],'controller':name,'seed':seed,
                    'comparisons':comparisons,'failures':[r for r in results if r['status']!='passed'],
                    'simulations':len(results),'model_unchanged':before==[hash_torch_state_dict(x.state_dict()) for x in models]})
            if before!=[hash_torch_state_dict(x.state_dict()) for x in models]:raise RuntimeError('Calibration changed frozen model')
        finally:owner.close_stream();world.close()
    marker=Path(task['completion']['path']);write_json(marker,dict(metadata,protocol_hash=manifest['protocol_hash'],status='assessment_completed'))
    return 0


def summarize(manifest):
    root=Path(manifest['root']);spec=manifest['config'];c=spec['calibration'];thresholds=c['thresholds']
    missing=[];failures=[];groups=[];all_rows=[]
    for task in manifest['tasks']:
        if not Path(task['completion']['path']).exists():missing.append(task['run_id'])
        if task['phase']=='assessment':
            p=Path(task['case_output_path'])/'assessment.json'
            if p.exists():
                d=read_json(p);groups.append(d);failures.extend(d['failures'])
                all_rows.extend(dict(x,controller=d['controller'],seed=d['seed']) for x in d['comparisons'])
    per_kind={}
    for kind in ('lane_blockage','road_closure','global_rain'):
        levels={}
        for level in c['profiles']:
            refs=[r for r in all_rows if r['kind']==kind and r['severity']==level and r['controller'] in spec['preflight']['controllers']]
            probes=[r for r in all_rows if r['kind']==kind and r['severity']==level and r['controller'] in c['frozen_policies']]
            def physical(r):
                if kind=='global_rain':return r['speed_drop_fraction'] is not None and r['speed_drop_fraction']>=thresholds['speed_drop_fraction']
                return r['local_queue_delta']>=thresholds['local_queue_delta'] or (r['flow_drop_fraction'] is not None and r['flow_drop_fraction']>=thresholds['flow_drop_fraction'])
            levels[level]={'reference_pairs':len(refs),'reference_physical_success_fraction':sum(physical(r) for r in refs)/len(refs) if refs else None,
                'frozen_pairs':len(probes),'frozen_net_J_median_fraction':statistics.median(r['net_J_fraction'] for r in probes) if probes else None,
                'frozen_net_J_range':[min(r['net_J_fraction'] for r in probes),max(r['net_J_fraction'] for r in probes)] if probes else None,
                'frozen_recovered_fraction':sum(r['recovery_seconds'] is not None for r in probes)/len(probes) if probes else None,
                'removed_vehicles':sum(r['removed'] for r in refs+probes)}
        strong=levels['S2'];expected=2*len(spec['preflight']['seeds'])*c['target_count']
        eligible=(strong['reference_pairs']==expected and strong['frozen_pairs']==expected and
            strong['reference_physical_success_fraction']>=thresholds['minimum_success_fraction'] and
            strong['frozen_net_J_median_fraction']>=thresholds['net_J_fraction'])
        per_kind[kind]={'levels':levels,'impact_gate_passed':eligible,
            'control_space_review':'Required: compare recovery, pending demand and policies; impact alone does not prove learnable headroom.'}
    pairing=all(r['pre_event_matches'] and r['control_pre_event_matches'] for r in all_rows)
    executed=read_json(root/'execution_budget.json')
    complete=not missing and not failures and pairing
    result={'schema_version':'strong-calibration-summary-v1','status':'assessment_completed' if not missing else 'incomplete',
        'admission':'development_checks_passed' if complete else 'not_ready','protocol_hash':manifest['protocol_hash'],
        'missing_tasks':missing,'failures':failures,'pairing_passed':pairing,'per_kind':per_kind,'comparisons':all_rows,
        'groups':[{k:d[k] for k in ('controller','seed','simulations','model_unchanged')} for d in groups],
        'attempted_simulations':len(executed['attempts']),
        'budget':manifest['budget'],'elapsed_wall_seconds':time.time()-executed['started_at_unix'],
        'training_started':False,'decision':'Review per-type impact and control space before any pilot; no automatic training launch.'}
    write_json(root/'preflight_summary.json',result)
    return result
