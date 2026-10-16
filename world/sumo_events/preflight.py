"""Development-only event intensity/placement assessment, through the shared CLI.

No optimizer updates, test-performance selection or event retargeting occurs.
The frozen event operators and the normal episode recorder are reused.
"""
import copy
import csv
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import yaml

from common.paper_experiment import (ROOT, SCHEMA, load_config, _assets, _case_factory,
    source_identity, environment_identity, digest, file_digest, write_json, read_json)
from common.registry import Registry
from world.world_sumo import World
from world.sumo_signal_control import install_signal_control
from .episodes import EpisodeEvents
from .trex_blockage import choose_position, LENGTH


def prepare_preflight(config_path, output):
    spec = load_config(config_path)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'experiment.yml').write_text(yaml.safe_dump(spec, sort_keys=False))
    world, provenance = _assets(spec, root)
    generate, pools = _case_factory(spec, world['roadnetFile'], world['flowFile'])
    gate = spec['preflight']
    rng = random.Random(gate['case_seed'])
    begin, duration = gate['event_begin'], gate['event_duration']
    cases = [{'case_id': 'normal', 'kind': 'normal', 'target_split': 'none',
              'schedule': generate('normal', rng, 'validation', begin, duration)}]
    for kind, field in [('lane_blockage', 'lane_id'), ('road_closure', 'edge_id')]:
        used = set()
        for index in range(gate['target_count']):
            for attempt in range(1000):
                schedule = generate(kind, rng, 'validation', begin, duration)
                target = schedule['events'][0][field]
                if target not in used:
                    used.add(target)
                    break
            else:
                raise ValueError('Insufficient distinct development targets')
            cases.append({'case_id': f'{kind}_{index}', 'kind': kind,
                          'target_split': 'validation', 'schedule': schedule})
    for factor in spec['events']['rain_factors']:
        schedule = generate('global_rain', rng, 'validation', begin, duration)
        schedule['events'][0]['speed_factor'] = factor
        cases.append({'case_id': f'rain_{factor:g}', 'kind': 'global_rain',
                      'target_split': 'global', 'schedule': schedule})
    for index, pair in enumerate(('lane_blockage+road_closure', 'lane_blockage+global_rain', 'road_closure+global_rain')):
        cases.append({'case_id': f'combined_{index}', 'kind': 'combined', 'components': pair,
                      'target_split': 'validation', 'schedule': generate(pair, rng, 'validation', begin, duration)})
    for repeat in range(1, gate['normal_repeats']):
        cases.append(dict(cases[0], case_id=f'normal_repeat_{repeat}'))
    if gate['case_ids'] is not None:
        if set(gate['case_ids']) - {c['case_id'] for c in cases}:
            raise ValueError('Unknown development case ID')
        cases = [c for c in cases if c['kind'] == 'normal' or c['case_id'] in gate['case_ids']]
    source = source_identity(spec['agent'])
    for name in ('agent/fixedtime.py', 'agent/maxpressure.py', 'agent/colight.py'):
        source[name] = file_digest(ROOT / name)
    for name in source:
        target = root / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    protocol = digest({'stage': 'preflight', 'config': spec, 'source': source,
                       'environment': environment_identity(spec['agent']), 'scenario': provenance, 'cases': cases})
    config = read_json(ROOT / 'configs/sim' / (spec['network'] + '.cfg'))
    config.update(world, name='paper_preflight')
    write_json(root / 'simulator.cfg', config)
    write_json(root / 'cases.json', cases)
    write_json(root / 'target_pools.json', pools)
    # The episode owner runs only explicitly selected development cases.
    episode_plan = {'schema_version': SCHEMA, 'protocol_hash': protocol,
                    'seconds': spec['seconds'], 'action_interval': spec['action_interval'],
                    'event_execution': spec['event_execution'], 'rows': []}
    write_json(root / 'episode_plan.json', episode_plan)
    tasks = []
    for controller in gate['controllers']:
        for seed in gate['seeds']:
            run_id = f'{controller}_s{seed}'
            run_dir = root / 'runs' / run_id
            cmd = [sys.executable, '-m', 'tools.run_paper_baseline', 'worker',
                   '--manifest', str(root / 'manifest.json'), '--task-id', run_id]
            tasks.append({'run_id': run_id, 'kind': 'preflight', 'controller': controller, 'seed': seed,
                          'output_path': str(run_dir), 'command': cmd, 'resume_command': cmd + ['--resume'],
                          'cwd': str(ROOT), 'pool': 'eval',
                          'env': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '-1'},
                          'completion': {'path': str(run_dir / 'completed.json'), 'json_field': 'protocol_hash', 'equals': protocol}})
    manifest = {'schema_version': SCHEMA, 'stage': 'preflight', 'root': str(root), 'config': spec,
                'protocol_hash': protocol, 'source_sha256': source, 'scenario': provenance,
                'environment': environment_identity(spec['agent']), 'runs': [], 'tasks': tasks,
                'input_sha256': {str(p.relative_to(root)): file_digest(p) for p in root.rglob('*') if p.is_file()},
                'budget': {'training_episodes': 0, 'preflight_episodes': len(cases) * len(tasks)}}
    manifest['manifest_hash'] = digest(manifest)
    write_json(root / 'manifest.json', manifest)
    return root / 'manifest.json'


def placement_probe(engine, lane, requested, tolerance, policy=None):
    """Read-only availability and conservative stopping-distance diagnostic."""
    row = {'lane': lane, 'requested': requested, 'tolerance': tolerance}
    try:
        if policy is None or policy['version'] == 'static-gap-v1':
            actual = choose_position(engine, lane, requested, tolerance)
        else:
            from .safe_placement import choose_position as safe_position
            actual, decision = safe_position(engine, lane, requested, tolerance, policy)
            row['dynamic_decision'] = decision
        row.update(space_available=True, selected_position=actual)
        if 'dynamic_decision' in row:
            # Report the same conservative envelope used to admit insertion,
            # including footprint, discrete reaction allowance and clearance.
            decision = row['dynamic_decision']
            row['braking_margin_m'] = decision['minimum_margin_m']
            follower = decision['limiting_follower']
            if follower:
                row.update(follower=follower['vehicle'], follower_speed=follower['speed'])
            return row
    except RuntimeError as error:
        row.update(space_available=False, error=str(error), braking_margin_m=None)
        return row
    followers = [(engine.vehicle.getLanePosition(v), v) for v in engine.lane.getLastStepVehicleIDs(lane)
                 if engine.vehicle.getLanePosition(v) <= actual - LENGTH]
    if not followers:
        row['braking_margin_m'] = None
    else:
        position, vehicle = max(followers)
        speed = engine.vehicle.getSpeed(vehicle)
        decel = engine.vehicle.getDecel(vehicle)
        required = engine.vehicle.getMinGap(vehicle) + speed * engine.vehicle.getTau(vehicle) + speed**2 / (2 * decel)
        row.update(braking_margin_m=actual - LENGTH - position - required, follower=vehicle, follower_speed=speed)
    return row


def _controllers(world, spec, name, seed):
    import torch
    from common.utils import load_config as legacy_config, build_index_intersection_map_sumo
    from agent.fixedtime import FixedTimeAgent
    from agent.maxpressure import MaxPressureAgent
    from agent.colight import CoLightAgent
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = legacy_config(str(ROOT / 'configs/tsc/colight.yml'))[0]['model']
    model.update(spec['model'])
    model.update(t_fixed=spec['preflight']['fixedtime_green'], t_min=spec['preflight']['maxpressure_min_green'])
    Registry.mapping['model_mapping']['setting'] = SimpleNamespace(param=model)
    Registry.mapping['trainer_mapping']['setting'] = SimpleNamespace(param={'buffer_size': spec['trainer'].get('buffer_size', 5000)})
    Registry.mapping['logger_mapping']['setting'] = SimpleNamespace(param={'attention': False})
    Registry.mapping['world_mapping']['setting'] = SimpleNamespace(param={'network': spec['network']})
    Registry.mapping['world_mapping']['graph_setting'] = SimpleNamespace(graph=build_index_intersection_map_sumo(world.net))
    if name == 'colight_untrained':
        return [CoLightAgent(world, 0)]
    cls = FixedTimeAgent if name == 'fixedtime' else MaxPressureAgent
    return [cls(world, i) for i in range(len(world.intersections))]


def run_preflight(manifest, task):
    import traceback
    from utils.logger import hash_torch_state_dict
    root, out = Path(manifest['root']), Path(task['output_path'])
    out.mkdir(parents=True, exist_ok=True)
    spec = manifest['config']
    seed, controller = task['seed'], task['controller']
    Registry.mapping['command_mapping']['setting'] = SimpleNamespace(param={'sumo_seed': seed})
    Registry.mapping['logger_mapping']['path'] = SimpleNamespace(path=str(out))
    world = World(str(root / 'simulator.cfg'), interface='libsumo')
    world.sumo_cmd += ['--no-step-log', 'true']
    install_signal_control(world, spec['signal_control'])
    episode_plan = read_json(root / 'episode_plan.json')
    owner = EpisodeEvents(world, root / 'episode_plan.json', digest(episode_plan), out)
    agents = _controllers(world, spec, controller, seed)
    models = [a.model for a in agents if a.model is not None]
    before = [hash_torch_state_dict(m.state_dict()) for m in models]
    cases = read_json(root / 'cases.json')
    pools = read_json(root / 'target_pools.json')
    probe_lanes = sorted(set(pools['train']['blockage_lanes'] + pools['validation']['blockage_lanes']))
    probe_times = set(spec['events']['begin'])
    results = []
    probe_counts = {'checks': 0, 'no_space': 0, 'negative_braking_margin': 0,
                    'requires_clearance': 0, 'vehicles_to_clear': 0}
    try:
        for template in cases:
            case = dict(template, sumo_seed=seed)
            result_path = out / 'cases' / (case['case_id'] + '.json')
            if result_path.exists():
                prior = read_json(result_path)
                if prior['case_hash'] != digest(case) or prior['protocol_hash'] != manifest['protocol_hash']:
                    raise ValueError('Preflight case identity mismatch')
                results.append(prior)
                if case['case_id'] == 'normal':
                    probe_counts = prior.get('placement_probes', probe_counts)
                continue
            probe_stream = None
            try:
                owner.select(case, 'preflight')
                world.reset()
                for agent in agents:
                    agent.reset()
                if case['case_id'] == 'normal':
                    probe_stream = (out / 'placement_probes.jsonl').open('w')
                catalog = owner.current.network_catalog()
                interval = 1 if controller == 'fixedtime' else spec['action_interval']
                for time in range(spec['seconds']):
                    if time % interval == 0:
                        actions = np.concatenate([np.asarray(a.get_action(a.get_ob(), a.get_phase(), test=True)).reshape(-1)
                                                  for a in agents])
                    world.step(actions)
                    now = int(world.get_current_time())
                    if probe_stream and now in probe_times:
                        for lane in probe_lanes:
                            probe = placement_probe(owner.current._engine, lane,
                                catalog[lane]['length'] - spec['events']['block_distance'], spec['events']['position_tolerance'], spec['event_execution']['placement'])
                            probe_counts['checks'] += 1
                            probe_counts['no_space'] += not probe['space_available']
                            probe_counts['negative_braking_margin'] += probe.get('braking_margin_m') is not None and probe['braking_margin_m'] < 0
                            clearing = probe.get('dynamic_decision', {}).get('vehicles_to_remove', [])
                            probe_counts['requires_clearance'] += bool(clearing)
                            probe_counts['vehicles_to_clear'] += len(clearing)
                            probe_stream.write(json.dumps(dict(time=now, **probe)) + '\n')
                        probe_stream.flush()
                result = owner.finish()
                runtime, raw = owner.current, owner.current._engine
                assert not runtime._obstacles and not runtime._closed and runtime._factor == 1
                assert all(math.isclose(raw.lane.getMaxSpeed(l), v, abs_tol=1e-7) for l, v in runtime._base_speed.items())
                assert all(set(raw.lane.getDisallowed(l)) == set(v) for l, v in runtime._base_permissions.items())
                expected = {(e.event_id, state, when) for e in runtime.schedule.events
                            for state, when in [('active', e.begin), ('cleared', e.end)]}
                assert {(r['event_id'], r['status'], r['updated_at']) for r in result['events']['transitions']} == expected
                row = {'status': 'passed', 'result': result}
                if case['case_id'] == 'normal':
                    row['placement_probes'] = dict(probe_counts)
            except Exception as error:
                row = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                       'simulation_time': world.get_current_time(), 'case': case,
                       'traceback': traceback.format_exc()}
                if hasattr(error, 'diagnostics'):
                    row['placement_rejection'] = error.diagnostics
                try:
                    raw = owner.current._engine
                    row['physical_diagnostics'] = {
                        'colliding_vehicles': list(raw.simulation.getCollidingVehiclesIDList()),
                        'starting_teleports': list(raw.simulation.getStartingTeleportIDList()),
                        'collisions': [{key: getattr(collision, key, None) for key in
                                        ('collider', 'victim', 'colliderType', 'victimType', 'colliderSpeed',
                                         'victimSpeed', 'type', 'lane', 'pos')}
                                       for collision in raw.simulation.getCollisions()],
                        'actual_obstacle_positions': dict(owner.current._positions),
                        'placement_decisions': getattr(owner.current, 'placement_decisions', [])}
                except Exception as detail_error:
                    row['diagnostic_error'] = str(detail_error)
            finally:
                if probe_stream:
                    probe_stream.close()
                owner.close_stream()
                world.close()
            row.update(case_id=case['case_id'], kind=case['kind'], controller=controller, seed=seed,
                       case_hash=digest(case), protocol_hash=manifest['protocol_hash'])
            write_json(result_path, row)
            results.append(row)
            write_json(out / 'progress.json', {'cases_finished': len(results), 'cases_total': len(cases),
                                             'failures': sum(r['status'] != 'passed' for r in results)})
        assert before == [hash_torch_state_dict(m.state_dict()) for m in models]
        normal = next((r.get('result') for r in results if r['case_id'] == 'normal'), None)
        for row in results:
            if row['status'] == 'passed' and normal is not None:
                if row['kind'] == 'normal':
                    row['reset_replay_matches'] = row['result']['trace_sha256'] == normal['trace_sha256']
                else:
                    begin = str(int(min(e['begin'] for e in row['result']['schedule']['events'])))
                    row['pre_event_matches'] = row['result']['pre_event_trace_sha256'] == normal['trace_prefixes_before'][begin]
        write_json(out / 'results.json', {'controller': controller, 'seed': seed, 'cases': results,
                                        'placement_probes': probe_counts, 'model_unchanged': True,
                                        'protocol_hash': manifest['protocol_hash']})
        # completed means all assessments ran; individual failures remain data.
        write_json(out / 'completed.json', {'protocol_hash': manifest['protocol_hash'], 'status': 'assessment_completed'})
        return 0
    finally:
        owner.close_stream()
        world.close()


def _window(path, begin, end, lanes=()):
    rows = []
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            if begin < float(row['simulation_time']) <= end:
                rows.append(row)
    if not rows:
        raise ValueError('Empty preflight event window')
    vehicles = sum(float(r['running']) for r in rows)
    return {'mean_queue': sum(float(r['queue_vehicles']) for r in rows) / len(rows),
            'local_queue': sum(sum(json.loads(r['lane_queues_json']).get(l, 0) for l in lanes) for r in rows) / len(rows),
            'speed': sum(float(r['sum_speed_mps']) for r in rows) / vehicles if vehicles else 0.}


def summarize_preflight(manifest):
    root = Path(manifest['root'])
    groups, missing = [], []
    comparisons, failures = [], []
    probe_totals = {'checks': 0, 'no_space': 0, 'negative_braking_margin': 0,
                    'requires_clearance': 0, 'vehicles_to_clear': 0}
    replay_ok, physical_ok = True, True
    rejected, collisions = [], []
    for task in manifest['tasks']:
        path = Path(task['output_path']) / 'results.json'
        if not path.exists():
            missing.append(task['run_id'])
            continue
        group = read_json(path)
        groups.append(group)
        for key in probe_totals:
            probe_totals[key] += group['placement_probes'].get(key, 0)
        normal = next((r.get('result') for r in group['cases'] if r['case_id'] == 'normal'), None)
        for row in group['cases']:
            if row['status'] != 'passed':
                failures.append({k: row[k] for k in ('case_id', 'controller', 'seed', 'error_type', 'error')})
                physical_ok = False
                if 'placement_rejection' in row:
                    rejected.append({k: row[k] for k in ('case_id', 'controller', 'seed', 'placement_rejection')})
                diagnostic = row.get('physical_diagnostics', {})
                if diagnostic.get('colliding_vehicles') or diagnostic.get('starting_teleports'):
                    collisions.append({k: row[k] for k in ('case_id', 'controller', 'seed', 'physical_diagnostics')})
                continue
            result = row['result']
            if row['kind'] == 'normal':
                replay_ok &= row.get('reset_replay_matches', False)
                continue
            physical_ok &= row.get('pre_event_matches', False)
            if normal is None:
                continue
            begin = min(e['begin'] for e in result['schedule']['events'])
            end = max(e['end'] for e in result['schedule']['events'])
            lanes = sorted(set(result['local_regions']['blockage_lanes'] + result['local_regions']['closure_area_lanes']))
            baseline, event = _window(normal['timeline'], begin, end, lanes), _window(result['timeline'], begin, end, lanes)
            comparisons.append({'controller': row['controller'], 'seed': row['seed'], 'case_id': row['case_id'],
                'kind': row['kind'], 'schedule': result['schedule'], 'normal': baseline, 'event': event,
                'local_queue_delta': event['local_queue'] - baseline['local_queue'],
                'global_queue_delta': event['mean_queue'] - baseline['mean_queue'],
                'speed_drop_fraction': 1 - event['speed'] / baseline['speed'] if baseline['speed'] else None,
                'system_time_delta': result['system_time_per_vehicle'] - normal['system_time_per_vehicle'],
                'arrived_delta': result['arrived'] - normal['arrived'], 'pending_delta': result['pending_due'] - normal['pending_due'],
                'forced_removed': result.get('forced_removed', 0)})
    gate = manifest['config']['preflight']
    templates = read_json(root / 'cases.json')
    intensity = {}
    for kind in ('lane_blockage', 'road_closure', 'global_rain'):
        rows = [r for r in comparisons if r['kind'] == kind]
        if kind == 'global_rain':
            positive = [r for r in rows if r['speed_drop_fraction'] is not None
                        and r['speed_drop_fraction'] >= gate['minimum_speed_drop']]
        else:
            positive = [r for r in rows if r['local_queue_delta'] >= gate['minimum_local_queue_delta']]
        expected = sum(c['kind'] == kind for c in templates) * len(manifest['tasks'])
        intensity[kind] = {'expected_cases': expected, 'completed_cases': len(rows),
                           'above_development_threshold': len(positive),
                           'all_cases_above_threshold': bool(expected) and expected == len(positive)}
    injection_ready = (not missing and physical_ok and replay_ok and not probe_totals['no_space']
                       and not probe_totals['negative_braking_margin'])
    intensity_ready = all(v['all_cases_above_threshold'] for v in intensity.values())
    ready = injection_ready and (gate['admission_basis'] == 'injection' or intensity_ready)
    summary = {'schema_version': 'paper-event-preflight-v2', 'status': 'assessment_completed' if not missing else 'incomplete',
               'admission': 'development_checks_passed' if ready else 'not_ready',
               'admission_basis': gate['admission_basis'],
               'injection_checks_passed': injection_ready, 'intensity_checks_passed': intensity_ready,
               'forced_removed_total': sum(r.get('result', {}).get('forced_removed', 0) for g in groups for r in g['cases']),
               'protocol_hash': manifest['protocol_hash'], 'config_hash': digest(manifest['config']),
               'physical_cases_passed': physical_ok and not missing, 'reset_replay_passed': replay_ok and not missing,
               'missing_workers': missing, 'failures': failures, 'placement_probes': probe_totals,
               'placement_rejections': rejected, 'collision_or_teleport_cases': collisions,
               'placement_policy': manifest['config']['event_execution']['placement'],
               'intensity': intensity, 'comparisons': comparisons,
               'coverage': {'development_seeds': gate['seeds'], 'controllers': gate['controllers'],
                            'attempted_simulations': sum(len(g['cases']) for g in groups),
                            'completed_simulations': sum(c['status'] == 'passed' for g in groups for c in g['cases']),
                            'seconds_per_simulation': manifest['config']['seconds'],
                            'placement_scope': 'normal trajectories, configured training onset grid, train+validation lane pools'},
               'limits': ['Development sample, not a guarantee for all learned policies or event combinations.',
                          'Negative braking margin is a conservative kinematic diagnostic, not an observed collision.',
                          'Untrained CoLight is a stress controller, not a trained policy result.',
                          'No 150-episode training or formal test-policy evaluation was run.']}
    write_json(root / 'preflight_summary.json', summary)
    return summary
