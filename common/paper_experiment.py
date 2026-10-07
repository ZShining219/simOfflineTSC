"""Compile immutable paper-baseline experiments; no simulator is started here.

The user-facing CLI remains tools.run_paper_baseline. This module owns the
configuration/plan contract, not a second training loop or launcher script.
"""
import copy
import math
import hashlib
import json
from pathlib import Path
import random
import sys
import importlib.metadata
import xml.etree.ElementTree as ET

import yaml

from common.experiment_queue import atomic_text


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'paper-experiment-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_json(path, expected):
    value = read_json(path)
    if digest(value) != expected:
        raise ValueError(f'Plan digest mismatch: {path}')
    return value


def source_identity(agent):
    files = ['tools/run_paper_baseline.py', 'common/paper_experiment.py',
             'common/experiment_queue.py', 'trainer/paper_trainer.py',
             'trainer/tsc_trainer.py', 'trainer/base_trainer.py', 'run.py',
             'environment.py', 'utils/training_monitor.py', 'utils/logger.py',
             'agent/paper_baselines.py', f'agent/{agent.removeprefix("paper_")}.py',
             'world/world_sumo.py', 'world/sumo_signal_control.py',
             'world/paper_rewards.py', 'common/paper_rewards.py',
             'agent/base.py', 'agent/rl_agent.py', 'agent/utils.py', 'agent/__init__.py',
             'common/interface.py', 'common/registry.py', 'common/utils.py', 'common/metrics.py',
             'task/task.py', 'sequential/launcher.py',
             'configs/tsc/base.yml', f'configs/tsc/{agent}.yml',
             'configs/tsc/paper_experiment_base.yml',
             f'configs/tsc/{agent.removeprefix("paper_")}.yml',
             f'configs/rewards/{agent.removeprefix("paper_")}.yml']
    files += [str(p.relative_to(ROOT)) for folder in ('world/sumo_events', 'generator')
              for p in (ROOT / folder).glob('*.py')]
    return {name: file_digest(ROOT / name) for name in sorted(set(files))}


def load_manifest(path, verify=True):
    path = Path(path).resolve()
    value = read_json(path)
    payload = {k: v for k, v in value.items() if k != 'manifest_hash'}
    if value.get('schema_version') != SCHEMA or digest(payload) != value.get('manifest_hash'):
        raise ValueError('Experiment manifest identity mismatch')
    if Path(value['root']).resolve() != path.parent:
        raise ValueError('Manifest moved: regenerate the operational plan at its new location')
    if verify:
        if value['environment'] != environment_identity(value['config']['agent']):
            raise ValueError('Python/dependency environment changed since planning')
        for name, expected in value['source_sha256'].items():
            if file_digest(ROOT / name) != expected:
                raise ValueError(f'Source changed since planning: {name}')
        for name, expected in value['input_sha256'].items():
            if file_digest(path.parent / name) != expected:
                raise ValueError(f'Frozen experiment input changed: {name}')
    return value


def environment_identity(agent):
    packages = ['torch', 'numpy', 'libsumo', 'sumolib', 'gym', 'PyYAML']
    if agent == 'paper_colight':
        packages += ['torch-scatter', 'torch-geometric']
    return {'python': sys.version, 'packages': {p: importlib.metadata.version(p) for p in packages}}


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _read_profile(path, chain=()):
    path = Path(path).resolve()
    if path in chain:
        raise ValueError('Cyclic experiment configuration inheritance')
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('Experiment configuration must be a mapping')
    parent = value.pop('extends', None)
    if parent is None:
        return value
    if not isinstance(parent, str):
        raise ValueError('extends must name one YAML configuration')
    return _merge_profile(_read_profile(path.parent / parent, chain + (path,)), value)


def _merge_profile(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = (_merge_profile(result[key], value)
                       if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value))
    return result


def _time_grid(value, name):
    if isinstance(value, dict):
        if set(value) != {'start', 'stop', 'step'} or any(type(v) is not int for v in value.values()):
            raise ValueError(f'{name} grid requires integer start/stop/step')
        if value['step'] <= 0 or value['start'] < 0 or value['stop'] < value['start']:
            raise ValueError(f'Invalid {name} time grid')
        if (value['stop'] - value['start']) % value['step']:
            raise ValueError(f'{name} stop must lie on its grid')
        return list(range(value['start'], value['stop'] + 1, value['step']))
    return value


def load_config(path):
    """Resolve inheritance, normalize grids and validate the complete profile."""
    spec = _read_profile(path)
    defaults = yaml.safe_load((ROOT / 'configs/tsc/paper_experiment_base.yml').read_text())
    for key in ('seed_policy', 'target_groups', 'preflight'):
        spec[key] = _merge_profile(defaults[key], spec.get(key, {}))
    from world.sumo_events.safe_placement import validate_policy, LEGACY
    spec['event_execution'] = spec.get('event_execution', {'placement': LEGACY})
    if set(spec['event_execution']) != {'placement'}:
        raise ValueError('event_execution requires placement only')
    spec['event_execution']['placement'] = validate_policy(spec['event_execution']['placement'])
    fields = {'schema_version', 'purpose', 'agent', 'network', 'seeds', 'regimes',
              'episodes', 'seconds', 'action_interval', 'validation_every', 'test_episodes',
              'signal_control', 'model', 'trainer', 'monitor', 'events', 'seed_policy', 'target_groups', 'preflight', 'event_execution'}
    if 'calibration' in spec:
        fields.add('calibration')
    if 'strong_events' in spec:
        fields.add('strong_events')
        if 'calibration' in spec:
            raise ValueError('Calibration and training profiles are separate purposes')
    if not isinstance(spec, dict) or set(spec) != fields or spec['schema_version'] != SCHEMA:
        raise ValueError('Experiment configuration must exactly match paper-experiment-v1 fields')
    if spec['purpose'] not in {'engineering', 'research'}:
        raise ValueError('purpose must be engineering or research')
    if spec['agent'] not in {'paper_colight', 'paper_frap', 'paper_presslight'} or spec['network'] != 'hz4x4':
        raise ValueError('This event design supports paper_* on hz4x4')
    for key in ('episodes', 'seconds', 'action_interval', 'validation_every'):
        _positive(spec[key], key)
    if spec['seconds'] % spec['action_interval']:
        raise ValueError('seconds must be divisible by action_interval')
    signal = spec['signal_control']
    from world.sumo_signal_control import VERSION
    if (not isinstance(signal, dict) or set(signal) != {'version', 'yellow_seconds'}
            or signal['version'] != VERSION or type(signal['yellow_seconds']) is not int
            or not 0 < signal['yellow_seconds'] < spec['action_interval']):
        raise ValueError('signal_control requires a supported version and yellow_seconds below action_interval')
    seeds = spec['seeds']
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Training seeds must be explicit, distinct nonnegative integers')
    policy = spec['seed_policy']
    if set(policy) != {'training_event_base', 'training_sumo_base', 'training_sumo_stride', 'validation_plan_seed', 'test_plan_seed'}:
        raise ValueError('seed_policy fields are incomplete')
    if any(type(v) is not int or v < 0 for v in policy.values()):
        raise ValueError('seed_policy values must be nonnegative integers')
    if policy['training_sumo_stride'] < spec['episodes']:
        raise ValueError('training_sumo_stride must cover the full episode budget')
    groups = spec['target_groups']
    if set(groups) != {'train_junctions', 'validation_junctions', 'test_junctions'}:
        raise ValueError('target_groups must explicitly define train/validation/test junctions')
    if any(not isinstance(v, list) or not v for v in groups.values()):
        raise ValueError('target_groups must be nonempty lists')
    if any(len(set(v)) != len(v) or any(not isinstance(x, str) for x in v) for v in groups.values()):
        raise ValueError('target_groups require distinct junction IDs')
    if set(groups['train_junctions']) & set(groups['validation_junctions']) or set(groups['train_junctions']) & set(groups['test_junctions']) or set(groups['validation_junctions']) & set(groups['test_junctions']):
        raise ValueError('target_groups must be disjoint')
    if (not spec['regimes'] or set(spec['regimes']) - {'normal', 'mixed'}
            or len(set(spec['regimes'])) != len(spec['regimes'])):
        raise ValueError('regimes must be a unique subset of normal/mixed')
    if any(type(e) is not int or not 0 < e <= spec['episodes'] for e in spec['test_episodes']):
        raise ValueError('test_episodes outside budget')
    if spec['test_episodes'] != sorted(set(spec['test_episodes'])):
        raise ValueError('test_episodes must be sorted and distinct')
    allowed_trainer = {'learning_start', 'buffer_size', 'update_model_rate', 'update_target_rate'}
    if set(spec['trainer']) - allowed_trainer or set(spec['monitor']) - {'render_every'}:
        raise ValueError('Unsupported trainer/monitor override')
    base_model = yaml.safe_load((ROOT / 'configs/tsc/base.yml').read_text())['model']
    method_model = yaml.safe_load((ROOT / 'configs/tsc' / (spec['agent'].removeprefix('paper_') + '.yml')).read_text())['model']
    allowed_model = (set(base_model) | set(method_model)) - {'name', 'train_model', 'test_model', 'load_model', 'graphic'}
    if set(spec['model']) - allowed_model:
        raise ValueError('Unknown or identity-changing model override')
    _positive(spec['monitor']['render_every'], 'render_every')
    batch = spec['model'].get('batch_size', base_model['batch_size'])
    capacity = spec['trainer'].get('buffer_size', 5000)
    start = spec['trainer'].get('learning_start', 1000)
    if type(batch) is not int or not 0 < batch <= min(capacity, start + 1):
        raise ValueError('batch_size exceeds replay capacity or first update readiness')
    spec['events'].setdefault('blockage_min_vehicles', 0)
    if type(spec['events']['blockage_min_vehicles']) is not int or spec['events']['blockage_min_vehicles'] < 0:
        raise ValueError('blockage_min_vehicles must be a nonnegative integer')
    event_fields = {'blockage_min_vehicles', 'begin', 'durations', 'rain_factors', 'block_distance', 'position_tolerance',
                    'validation_begin', 'test_begin', 'evaluation_duration', 'recovery_seconds',
                    'cases_per_kind', 'validation_sumo_seeds', 'test_sumo_seeds', 'mixed_cycle'}
    settings = spec['events']
    if set(settings) != event_fields:
        raise ValueError('events fields must match the explicit event-design schema')
    if (not settings['mixed_cycle'] or set(settings['mixed_cycle']) -
            {'normal', 'lane_blockage', 'road_closure', 'global_rain'}):
        raise ValueError('Invalid mixed_cycle')
    if spec['purpose'] == 'research' and spec['episodes'] % len(settings['mixed_cycle']):
        raise ValueError('Research budget must contain complete balanced event cycles')
    for key in ('begin', 'durations', 'validation_begin', 'test_begin'):
        settings[key] = _time_grid(settings[key], key)
        if not settings[key] or any(type(t) is not int or t < 0 or t % spec['action_interval'] for t in settings[key]):
            raise ValueError(f'{key} must use the explicit action-interval time grid')
    if min(settings['durations']) <= 0 or settings['evaluation_duration'] <= 0:
        raise ValueError('Event durations must be positive')
    if max(settings['begin'] + settings['validation_begin'] + settings['test_begin']) + max(
            settings['durations'] + [settings['evaluation_duration']]) + settings['recovery_seconds'] >= spec['seconds']:
        raise ValueError('Events must clear with the requested recovery window before horizon')
    if any(not 0 < f < 1 for f in settings['rain_factors']):
        raise ValueError('Rain factors must be in (0,1)')
    _positive(settings['cases_per_kind'], 'cases_per_kind')
    for key in ('validation_sumo_seeds', 'test_sumo_seeds'):
        values = settings[key]
        if not values or len(set(values)) != len(values) or any(type(s) is not int or s < 0 for s in values):
            raise ValueError('Explicit distinct evaluation SUMO seeds required')
    if set(settings['validation_sumo_seeds']) & set(settings['test_sumo_seeds']):
        raise ValueError('Validation and final test seeds overlap')
    gate = spec['preflight']
    if set(gate) != set(defaults['preflight']):
        raise ValueError('Unknown preflight configuration field')
    if gate['admission_basis'] not in ('injection', 'injection_and_intensity'):
        raise ValueError('preflight.admission_basis must be injection or injection_and_intensity')
    if (not gate['seeds'] or len(set(gate['seeds'])) != len(gate['seeds'])
            or any(type(s) is not int or s < 0 for s in gate['seeds'])
            or set(gate['seeds']) & set(seeds + settings['validation_sumo_seeds'] + settings['test_sumo_seeds'])):
        raise ValueError('Preflight seeds must be distinct from research seeds')
    if (not gate['controllers'] or len(set(gate['controllers'])) != len(gate['controllers'])
            or set(gate['controllers']) - {'fixedtime', 'maxpressure', 'colight_untrained'}):
        raise ValueError('Unsupported preflight controller')
    if gate['case_ids'] is not None and (not isinstance(gate['case_ids'], list)
            or not gate['case_ids'] or any(not isinstance(c, str) for c in gate['case_ids'])
            or len(set(gate['case_ids'])) != len(gate['case_ids'])):
        raise ValueError('preflight.case_ids must be null or distinct case IDs')
    for key in ('fixedtime_green', 'maxpressure_min_green', 'event_duration', 'target_count', 'normal_repeats'):
        _positive(gate[key], 'preflight.' + key)
    if gate['normal_repeats'] < 2:
        raise ValueError('Preflight needs a normal reset replay')
    if any(type(gate[k]) is not int or gate[k] < 0 for k in ('event_begin', 'case_seed')):
        raise ValueError('Invalid preflight onset or case seed')
    if (gate['event_begin'] % spec['action_interval'] or gate['event_duration'] % spec['action_interval']
            or gate['event_begin'] + gate['event_duration'] + settings['recovery_seconds'] >= spec['seconds']):
        raise ValueError('Preflight events must align and leave the recovery window')
    for key in ('minimum_local_queue_delta', 'minimum_speed_drop'):
        if not isinstance(gate[key], (int, float)) or not math.isfinite(gate[key]) or gate[key] <= 0:
            raise ValueError('Preflight response thresholds must be positive and finite')
    if 'calibration' in spec:
        from world.sumo_events.calibration import validate_config
        validate_config(spec)
    if 'strong_events' in spec:
        from world.sumo_events.profiles import validate
        validate(spec)
    return spec


_config = load_config  # compatibility for earlier callers; new code uses the public API


def describe_config(path):
    spec = load_config(path)
    if 'calibration' in spec:
        from world.sumo_events.calibration import describe
        return describe(spec)
    checkpoints = checkpoint_episodes(spec)
    runs = len(spec['seeds']) * len(spec['regimes'])
    levels = len(spec.get('strong_events', {}).get('profiles', {})) or 1
    result = {'config': spec, 'config_hash': digest(spec), 'checkpoints': checkpoints,
            'decisions_per_run': spec['episodes'] * spec['seconds'] // spec['action_interval'],
            'budget': {'training_episodes': runs * spec['episodes'],
                       'validation_episodes': runs * len(checkpoints) * (1 + 3*levels) * len(spec['events']['validation_sumo_seeds']),
                       'test_episodes': runs * len(spec['test_episodes']) * (1 + 4 * spec['events']['cases_per_kind']) * len(spec['events']['test_sumo_seeds'])}}

    if 'strong_events' in spec:
        settings = spec['events']
        block_test = settings['cases_per_kind'] + sum(i % 3 in (0, 1) for i in range(settings['cases_per_kind']))
        result['budget']['maximum_clearance_episodes'] = runs * (
            len(checkpoints) * levels * len(settings['validation_sumo_seeds']) +
            len(spec['test_episodes']) * block_test * len(settings['test_sumo_seeds']))
    return result


def checkpoint_episodes(spec):
    """Single source of truth for budget inspection and materialized plans."""
    return sorted(set(range(0, spec['episodes'] + 1, spec['validation_every'])) |
                  set(spec['test_episodes']) | {spec['episodes']})


def _assets(spec, root):
    """Freeze a corrected copy, never mutate shared historical demand files."""
    source = json.loads((ROOT / 'configs/sim' / (spec['network'] + '.cfg')).read_text())
    assets = root / 'assets'
    assets.mkdir()
    original_net = ROOT / source['dir'] / source['roadnetFile']
    net = assets / original_net.name
    net.write_bytes(original_net.read_bytes())
    original_route = ROOT / source['dir'] / source['flowFile']
    tree = ET.parse(original_route)
    if not any(t.get('id') == 'pkw' for t in tree.getroot().findall('vType')):
        raise ValueError('Expected the audited hz4x4 pkw vehicle type')
    for vehicle in tree.getroot().findall('vehicle'):
        if vehicle.get('type') not in (None, 'pkw'):
            raise ValueError('Unexpected vehicle type; do not silently replace heterogeneous demand')
        vehicle.set('type', 'pkw')
    route = assets / original_route.name
    tree.write(route, encoding='utf-8', xml_declaration=True)
    config = ET.Element('configuration')
    inputs = ET.SubElement(config, 'input')
    ET.SubElement(inputs, 'net-file', value=net.name)
    ET.SubElement(inputs, 'route-files', value=route.name)
    simtime = ET.SubElement(config, 'time')
    ET.SubElement(simtime, 'begin', value='0')
    ET.SubElement(simtime, 'end', value=str(spec['seconds']))
    combined = assets / 'experiment.sumocfg'
    ET.ElementTree(config).write(combined, encoding='utf-8', xml_declaration=True)
    return {'dir': '', 'roadnetFile': str(net), 'flowFile': str(route),
            'combined_file': str(combined), 'saveReplay': False}, {
                'source_network_sha256': file_digest(original_net),
                'source_route_sha256': file_digest(original_route),
                'vehicle_type_binding': 'all vehicles explicitly reference pkw',
                'vehicles': len(tree.getroot().findall('vehicle'))}


def _case_factory(spec, net, route_file=None):
    from world.sumo_events import Schedule, Event, SumoEventRuntime
    catalog = SumoEventRuntime(net, Schedule(())).network_catalog()
    from collections import Counter
    import sumolib
    demand = Counter()
    if spec['events']['blockage_min_vehicles'] and route_file is None:
        raise ValueError('Route snapshot required for blockage demand screening')
    if route_file is not None:
        roadnet = sumolib.net.readNet(str(net))
        for vehicle in ET.parse(route_file).getroot().findall('vehicle'):
            if float(vehicle.get('depart')) >= spec['seconds']:
                continue
            route = vehicle.find('route')
            if route is None:
                raise ValueError('Demand screening requires inline explicit vehicle routes')
            edges = route.get('edges').split()
            for a, b in zip(edges, edges[1:]):
                lanes = {c.getFromLane().getID() for c in roadnet.getEdge(a).getOutgoing()[roadnet.getEdge(b)]
                         if c.getFromLane().allows('passenger') and c.getToLane().allows('passenger')}
                for lane in lanes:
                    demand[lane] += 1 / len(lanes)
    val = set(spec['target_groups']['validation_junctions'])
    test = set(spec['target_groups']['test_junctions'])
    train = set(spec['target_groups']['train_junctions'])
    controlled = train | val | test
    known = set(SumoEventRuntime(net, Schedule(())).intersection_ids)
    if controlled - known:
        raise ValueError(f'Unknown target junctions: {sorted(controlled - known)}')
    pools = {}
    settings = spec['events']
    for split, nodes in [('train', train), ('validation', val), ('test', test)]:
        lanes = sorted(k for k, d in catalog.items() if not d['internal'] and d['motor_vehicle_lane']
                       and d['from_junction'] in controlled and d['to_junction'] in nodes
                       and set(d['movements']) & {'left-turn', 'through'}
                       and d['length'] > settings['block_distance'] + settings['position_tolerance'] + 5)
        edges = sorted({catalog[k]['edge_id'] for k in lanes})
        if not lanes or not edges:
            raise ValueError('Event target pool is empty')
        blockage_lanes = [lane for lane in lanes if demand[lane] >= settings['blockage_min_vehicles']]
        if not blockage_lanes:
            raise ValueError('No blockage lanes meet the configured demand floor in ' + split)
        pools[split] = {'junctions': sorted(nodes), 'lanes': lanes, 'edges': edges,
                        'blockage_lanes': blockage_lanes,
                        'planned_lane_demand': {lane: demand[lane] for lane in lanes}}

    if 'strong_events' in spec:
        from world.sumo_events.profiles import select_pools
        select_pools(spec, pools, catalog)

    def generate(kind, rng, split, begin, duration, severity=None):
        if 'strong_events' in spec:
            from world.sumo_events.profiles import generate as profiled
            return profiled(spec, pools, catalog, kind, rng, split, begin, severity)
        if kind == 'normal':
            return Schedule(()).to_dict()
        kinds = kind.split('+')
        events = []
        blocked_edge = None
        for index, name in enumerate(kinds):
            args = dict(event_id=f'event_{index}', kind=name, begin=begin, end=begin + duration)
            if name == 'lane_blockage':
                lane = rng.choice(pools[split]['blockage_lanes'])
                blocked_edge = catalog[lane]['edge_id']
                args.update(lane_id=lane, position=catalog[lane]['length'] - settings['block_distance'],
                            position_tolerance=settings['position_tolerance'])
            elif name == 'road_closure':
                args['edge_id'] = rng.choice([e for e in pools[split]['edges'] if e != blocked_edge])
            else:
                args['speed_factor'] = rng.choice(settings['rain_factors'])
            events.append(Event(**args))
        schedule = Schedule(tuple(events))
        SumoEventRuntime(net, schedule)  # validate targets without starting SUMO
        return schedule.to_dict()
    return generate, pools


def plan(config_path, output):
    spec = _config(config_path)
    if 'calibration' in spec:
        raise ValueError('Calibration profiles use preflight, not training plan')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    atomic_text(root / 'experiment.yml', yaml.safe_dump(spec, sort_keys=False))
    world, provenance = _assets(spec, root)
    generate, pools = _case_factory(spec, world['roadnetFile'], world['flowFile'])
    events = spec['events']
    cases = {'validation': [], 'test': []}
    levels = list(spec.get('strong_events', {}).get('profiles', {}))
    excluded = set()
    for role, seed in [('validation', spec['seed_policy']['validation_plan_seed']),
                       ('test', spec['seed_policy']['test_plan_seed'])]:
        if role == 'test' and not spec['test_episodes']:
            continue
        rng = random.Random(seed)
        kinds = ['normal', 'lane_blockage', 'road_closure', 'global_rain']
        for kind in kinds + (['combined'] if role == 'test' else []):
            count = 1 if role == 'validation' or kind == 'normal' else events['cases_per_kind']
            if levels and role == 'validation' and kind != 'normal':
                count = len(levels)
            for index in range(count):
                severity = levels[index // (count // len(levels))] if levels and kind != 'normal' else None
                split = 'validation' if role == 'validation' else (
                    'test' if kind in ('lane_blockage', 'road_closure') and index >= count // 2 else 'train')
                if levels and role == 'test' and kind in ('lane_blockage','road_closure'):
                    split = 'train' if index % (count // len(levels)) == 0 else 'test'
                actual_kind = kind
                if kind == 'combined':
                    actual_kind = ['lane_blockage+road_closure', 'lane_blockage+global_rain',
                                   'road_closure+global_rain'][index % 3]
                for attempt in range(1000):
                    schedule = generate(actual_kind, rng, split, rng.choice(events[role + '_begin']), events['evaluation_duration'], severity)
                    signature = digest(schedule)
                    if kind == 'normal' or signature not in excluded:
                        break
                else:
                    raise ValueError('Insufficient distinct evaluation schedules')
                if kind != 'normal':
                    excluded.add(signature)
                for sumo_seed in events[role + '_sumo_seeds']:
                    case = {'case_id': f'{kind}_{index:02d}_s{sumo_seed}', 'kind': kind,
                                        'target_split': ('global' if kind == 'global_rain' else 'none' if kind == 'normal' else split),
                                        'sumo_seed': sumo_seed, 'schedule': schedule}
                    if levels:
                        case['severity'] = severity
                    cases[role].append(case)
    source = source_identity(spec['agent'])
    # Audit copies are immutable outputs, never alternative launch scripts.
    for name in source:
        target = root / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    if levels:
        origin = ROOT / spec['strong_events']['exposure_file']
        (root / 'exposure_source.json').write_bytes(origin.read_bytes())
    protocol = digest({'config': spec, 'source': source, 'scenario': provenance,
                       'environment': environment_identity(spec['agent']),
                       'network': file_digest(world['roadnetFile']), 'route': file_digest(world['flowFile'])})
    write_json(root / 'target_pools.json', pools)
    write_json(root / 'cases.json', cases)
    runs, tasks = [], []
    checkpoints = checkpoint_episodes(spec)
    for seed in spec['seeds']:
        # Generate the full mixed plan once; a normal control simply clears its
        # schedules, without consuming policy RNG or changing demand seeds.
        rng = random.Random(spec['seed_policy']['training_event_base'] + seed)
        order = []
        while len(order) < spec['episodes']:
            cycle = list(events['mixed_cycle'])
            rng.shuffle(cycle)
            order.extend(cycle)
        mixed = []
        severity_order = {}
        if levels:
            from collections import Counter
            for kind, n in Counter(order[:spec['episodes']]).items():
                if kind != 'normal':
                    # Odd counts allocate the extra instance to S2.
                    severity_order[kind] = [levels[0]]*(n//2) + [levels[1]]*(n-n//2)
                    rng.shuffle(severity_order[kind])
        for ep, kind in enumerate(order[:spec['episodes']], 1):
            severity = severity_order[kind].pop() if levels and kind != 'normal' else None
            for attempt in range(1000):
                schedule = generate(kind, rng, 'train', rng.choice(events['begin']), rng.choice(events['durations']), severity)
                if kind == 'normal' or digest(schedule) not in excluded:
                    break
            else:
                raise ValueError('Training events overlap all available held-out schedules')
            sumo_seed = spec['seed_policy']['training_sumo_base'] + spec['seed_policy']['training_sumo_stride'] * seed + ep
            if sumo_seed >= 2**31 or sumo_seed in set(events['validation_sumo_seeds'] + events['test_sumo_seeds']):
                raise ValueError('Derived SUMO seed invalid or overlaps evaluation')
            mixed.append({'episode': ep, 'case_id': f'episode_{ep:04d}', 'kind': kind,
                          'sumo_seed': sumo_seed, 'schedule': schedule})
            if levels:
                mixed[-1]['severity'] = severity
        for regime in spec['regimes']:
            run_id = f'{regime}_s{seed}'
            rows = copy.deepcopy(mixed)
            if regime == 'normal':
                for row in rows:
                    row.update(kind='normal', schedule={'schema_version': 'sumo-events-v1', 'events': []})
                    if levels:
                        row['severity'] = None
            episode_plan = {'schema_version': SCHEMA, 'protocol_hash': protocol,
                            'training_seed': seed, 'event_seed': spec['seed_policy']['training_event_base'] + seed,
                            'seconds': spec['seconds'], 'action_interval': spec['action_interval'],
                            'event_execution': spec['event_execution'], 'rows': rows}
            if levels:
                episode_plan['record_lane_flow'] = True
            plan_path = root / 'plans' / (run_id + '.json')
            write_json(plan_path, episode_plan)
            run_dir = root / 'train' / run_id
            overlay = {'command': {'sumo_seed': rows[0]['sumo_seed'], 'output_path': str(run_dir)},
                       'world': world, 'model': spec['model'], 'logger': {'save_rate': spec['episodes']},
                       'trainer': dict(spec['trainer'], episodes=spec['episodes'], steps=spec['seconds'],
                                       test_steps=spec['seconds'], action_interval=spec['action_interval'],
                                       evaluation_episodes=checkpoints, resumable_checkpoint_episodes=checkpoints,
                                       signal_control=spec['signal_control'], event_schedule=None,
                                       episode_plan=str(plan_path), episode_plan_hash=digest(episode_plan),
                                       monitor_render_every=spec['monitor']['render_every'])}
            overlay_path = root / 'overlays' / (run_id + '.yml')
            overlay_path.parent.mkdir(exist_ok=True)
            atomic_text(overlay_path, yaml.safe_dump(overlay, sort_keys=False))
            runs.append({'run_id': run_id, 'seed': seed, 'regime': regime,
                         'output_path': str(run_dir), 'overlay': str(overlay_path)})
            tasks.append({'run_id': run_id, 'kind': 'train', 'pool': 'train', 'output_path': str(run_dir),
                          'run': run_id, 'completion': {'path': str(run_dir / 'completed.json'),
                                                      'json_field': 'protocol_hash', 'equals': protocol}})
    for run in runs:
        for role, episodes in [('validation', checkpoints), ('test', spec['test_episodes'])]:
            for episode in episodes:
                folder = root / role / run['run_id'] / f'episode_{episode:04d}'
                ready = Path(run['output_path']) / 'published' / f'episode_{episode:04d}.json'
                task = {'run_id': f'{role}_{run["run_id"]}_e{episode:04d}', 'kind': 'evaluate',
                        'pool': 'eval' if role == 'validation' else 'test', 'role': role, 'episode': episode, 'run': run['run_id'],
                        'priority': 1 if role == 'validation' else 2, 'output_path': str(folder),
                        'publication': str(ready), 'producer': run['run_id'],
                        'requires': [{'path': str(ready), 'json_field': 'protocol_hash', 'equals': protocol}],
                        'completion': {'path': str(folder / 'completed.json'), 'json_field': 'protocol_hash', 'equals': protocol}}
                if role == 'test':
                    task['depends_on'] = [r['run_id'] for r in runs]
                tasks.append(task)
    manifest_path = root / 'manifest.json'
    for task in tasks:
        command = [sys.executable, '-m', 'tools.run_paper_baseline', 'worker',
                   '--manifest', str(manifest_path), '--task-id', task['run_id']]
        task.update(command=command, resume_command=command + ['--resume'], cwd=str(ROOT),
                    env={'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '-1'})
    inputs = {str(p.relative_to(root)): file_digest(p) for p in root.rglob('*') if p.is_file()}
    manifest = {'schema_version': SCHEMA, 'root': str(root), 'protocol_hash': protocol,
                'environment': environment_identity(spec['agent']),
                'config': spec, 'source_sha256': source, 'input_sha256': inputs,
                'scenario': provenance, 'runs': runs, 'tasks': tasks,
                'budget': {'training_episodes': len(runs) * spec['episodes'],
                           'validation_episodes': len(runs) * len(checkpoints) * len(cases['validation']),
                           'test_episodes': len(runs) * len(spec['test_episodes']) * len(cases['test'])}}
    if levels:
        maximum = sum(len(runs) * len(checkpoints if role == 'validation' else spec['test_episodes']) *
                      sum(any(e['kind']=='lane_blockage' for e in c['schedule']['events']) for c in cases[role])
                      for role in ('validation','test'))
        manifest['budget']['maximum_clearance_episodes'] = maximum
    manifest['manifest_hash'] = digest(manifest)
    write_json(manifest_path, manifest)
    return manifest_path


def preflight(config_path, output):
    if 'calibration' in load_config(config_path):
        from world.sumo_events.calibration import prepare
        return prepare(config_path, output)
    from world.sumo_events.preflight import prepare_preflight
    return prepare_preflight(config_path, output)


EXTENSION_SOURCE_CHANGES = frozenset({
    'common/paper_experiment.py', 'tools/run_paper_baseline.py',
    'trainer/paper_trainer.py', 'world/sumo_events/safe_placement.py',
})


def extension_checkpoint(trainer, extension):
    """Verify the parent and the complete new binding before importing state."""
    import copy
    import torch
    for path, expected in extension['inputs'].items():
        if file_digest(path) != expected:
            raise ValueError('Extension parent artifact changed: ' + path)
    parent = read_json(extension['contract'])
    current = trainer.experiment_contract
    expected = copy.deepcopy(parent)
    old_sim = read_json(extension['simulator_config'])
    new_sim = read_json(trainer.path)
    for field in ('combined_file', 'roadnetFile', 'flowFile'):
        if file_digest(old_sim[field]) != file_digest(new_sim[field]):
            raise ValueError('Extension simulator asset changed: ' + field)
        old_sim[field] = new_sim[field]
    if old_sim != new_sim:
        raise ValueError('Extension simulator options changed')
    expected['simulator_config_sha256'] = current['simulator_config_sha256']
    allowed_settings = {'episodes', 'evaluation_episodes', 'resumable_checkpoint_episodes',
                        'episode_plan', 'episode_plan_hash', 'training_extension'}
    old_settings, new_settings = parent['trainer'], current['trainer']
    for k in set(old_settings) | set(new_settings):
        if k not in allowed_settings and old_settings.get(k) != new_settings.get(k):
            raise ValueError('Extension changes trainer semantics: ' + k)
    if new_settings['episodes'] <= old_settings['episodes']:
        raise ValueError('Extension must increase the training horizon')
    prefix = read_json(extension['parent_plan'])
    target = trainer.episode_events.plan
    n = extension['parent_episode']
    if n != old_settings['episodes'] or target['rows'][:n] != prefix['rows']:
        raise ValueError('Extension changes the completed training prefix')
    for key in set(prefix) | set(target):
        if key not in {'protocol_hash', 'rows'} and prefix.get(key) != target.get(key):
            raise ValueError('Extension changes episode semantics: ' + key)
    before, after = parent['implementation_sha256'], current['implementation_sha256']
    if set(before) != set(after):
        raise ValueError('Extension source set changed')
    changes = {p for p in before if before[p] != after[p]}
    if changes - EXTENSION_SOURCE_CHANGES:
        raise ValueError('Extension changes unaudited implementation: ' + str(changes))
    if {p: {'before': before[p], 'after': after[p]} for p in changes} != extension['source_changes']:
        raise ValueError('Extension repair identity differs from plan')
    for key in ('trainer', 'implementation_sha256', 'episode_plan_hash', 'protocol_hash'):
        expected[key] = current[key]
    expected.pop('hash')
    if expected != {k: v for k, v in current.items() if k != 'hash'}:
        changed = [k for k in expected if expected[k] != current.get(k)]
        raise ValueError('Extension changes model/environment/reward contract: ' + str(changed))
    payload = torch.load(extension['checkpoint'], map_location='cpu')
    trainer.validate_checkpoint_payload(payload, expected_type='resumable')
    if (payload['episode'] != n or payload['experiment_contract_hash'] != parent['hash'] or
            payload['reward_contract_hash'] != trainer.reward_contract['hash'] or
            payload['config_hash'] != read_json(extension['run_manifest'])['config_hash']):
        raise ValueError('Extension checkpoint identity mismatch')
    # Only identifiers migrate in memory. All learned state and RNG tensors
    # remain exactly the parent state. The original file is immutable.
    payload['config_hash'] = trainer.config_hash
    payload['experiment_contract_hash'] = current['hash']
    return payload


def plan_extension(config_path, output, source_manifests):
    """Append an event-only curriculum to immutable completed training plans."""
    import copy
    spec = load_config(config_path)
    if spec['regimes'] != ['mixed'] or spec['events']['mixed_cycle'] != ['road_closure', 'lane_blockage']:
        raise ValueError('This extension requires mixed runs and balanced closure/blockage suffix')
    candidates = {}
    parent_inputs = {}
    for path in source_manifests:
        source = load_manifest(path, verify=False)
        if source['environment'] != environment_identity(spec['agent']):
            raise ValueError('Parent environment changed')
        for name, expected in source['input_sha256'].items():
            full = Path(source['root'])/name
            if file_digest(full) != expected:
                raise ValueError('Parent frozen input changed: ' + str(full))
        parent_inputs[str(Path(path).resolve())] = file_digest(path)
        for run in source['runs']:
            marker = Path(run['output_path'])/'completed.json'
            if run['regime'] != 'mixed' or run['seed'] not in spec['seeds'] or not marker.exists():
                continue
            if run['run_id'] in candidates:
                raise ValueError('Ambiguous completed parent: ' + run['run_id'])
            if read_json(marker)['protocol_hash'] != source['protocol_hash']:
                raise ValueError('Parent completion protocol mismatch')
            for key in ('agent','network','seconds','action_interval','signal_control','model','trainer',
                        'event_execution','seed_policy','target_groups','strong_events'):
                if source['config'][key] != spec[key]:
                    raise ValueError('Extension changes parent setting: ' + key)
            candidates[run['run_id']] = (source, run)
    if set(candidates) != {f'mixed_s{s}' for s in spec['seeds']}:
        raise ValueError('One completed parent required for every requested seed')
    # Use the common planner for assets, task wiring and source snapshots;
    # finalize its new output before launch with the immutable parent prefix.
    manifest_path = plan(config_path, output)
    root = manifest_path.parent
    manifest = read_json(manifest_path)
    cases = read_json(Path(next(iter(candidates.values()))[0]['root'])/'cases.json')
    if read_json(root/'cases.json') != cases:
        raise ValueError('Extension evaluation library differs from E150')
    world = yaml.safe_load(Path(manifest['runs'][0]['overlay']).read_text())['world']
    generate, pools = _case_factory(spec, world['roadnetFile'], world['flowFile'])
    excluded = {digest(c['schedule']) for group in cases.values() for c in group if c['kind'] != 'normal'}
    extensions = {}; plans = {}
    for run in manifest['runs']:
        old, original = candidates[run['run_id']]
        folder = Path(original['output_path'])
        old_overlay = yaml.safe_load(Path(original['overlay']).read_text())
        prefix_path = Path(old_overlay['trainer']['episode_plan'])
        prefix = checked_json(prefix_path, old_overlay['trainer']['episode_plan_hash'])
        n = old['config']['episodes']; extra = spec['episodes']-n
        if extra <= 0 or extra % 2:
            raise ValueError('Extension suffix must have a positive even number of episodes')
        pub_path = folder/'published'/f'episode_{n:04d}.json'; pub = read_json(pub_path)
        checkpoint = Path(pub['resumable'])
        if pub['episode'] != n or pub['protocol_hash'] != old['protocol_hash'] or file_digest(checkpoint) != pub['resumable_sha256']:
            raise ValueError('Parent resumable checkpoint mismatch')
        binding = read_json(folder/'experiment_contract.json')
        changes = {p: {'before': h, 'after': file_digest(ROOT/p)} for p,h in binding['implementation_sha256'].items() if file_digest(ROOT/p)!=h}
        if set(changes)-EXTENSION_SOURCE_CHANGES:
            raise ValueError('Parent source incompatible: ' + str(set(changes)-EXTENSION_SOURCE_CHANGES))
        inputs = {str(p.resolve()):file_digest(p) for p in [prefix_path, pub_path, checkpoint,
            folder/'experiment_contract.json',folder/'reward_contract.json',folder/'run_manifest.json',folder/'completed.json',folder/'config/simulator_resolved.cfg']}
        extension = {'parent_episode':n, 'checkpoint':str(checkpoint), 'checkpoint_sha256':pub['resumable_sha256'],
            'contract':str(folder/'experiment_contract.json'),'parent_plan':str(prefix_path),
            'run_manifest':str(folder/'run_manifest.json'),'simulator_config':str(folder/'config/simulator_resolved.cfg'),
            'inputs':inputs,'source_changes':changes}
        extensions[run['run_id']] = extension;parent_inputs.update(inputs)
        rng = random.Random(spec['seed_policy']['training_event_base']+run['seed']+1000000)
        # Each pair has one of each event, random order; no long type streaks.
        order=[]
        for _ in range(extra//2):
            pair=['road_closure','lane_blockage'];rng.shuffle(pair);order.extend(pair)
        levels={}
        for kind in set(order):
            count=order.count(kind);levels[kind]=['S1']*(count//2)+['S2']*(count-count//2);rng.shuffle(levels[kind])
        rows=copy.deepcopy(prefix['rows'])
        for ep,kind in enumerate(order,n+1):
            severity=levels[kind].pop()
            for _ in range(1000):
                schedule=generate(kind,rng,'train',rng.choice(spec['events']['begin']),spec['events']['evaluation_duration'],severity)
                if digest(schedule) not in excluded:break
            else:raise ValueError('No train-only extension scenario')
            rows.append({'episode':ep,'case_id':f'episode_{ep:04d}','kind':kind,
                'sumo_seed':spec['seed_policy']['training_sumo_base']+spec['seed_policy']['training_sumo_stride']*run['seed']+ep,
                'schedule':schedule,'severity':severity})
        plans[run['run_id']] = dict(prefix,rows=rows)
    protocol=digest({'base_plan':manifest['protocol_hash'],'extensions':extensions,'rows':{k:v['rows'] for k,v in plans.items()}})
    manifest['protocol_hash']=protocol
    manifest['stage']='event_training_extension'
    for run in manifest['runs']:
        ext=extensions[run['run_id']];ep_plan=plans[run['run_id']];ep_plan['protocol_hash']=protocol
        path=root/'plans'/f'{run["run_id"]}.json';write_json(path,ep_plan)
        overlay=yaml.safe_load(Path(run['overlay']).read_text())
        checkpoints=sorted({ext['parent_episode']} | {t['episode'] for t in manifest['tasks'] if t.get('role')=='validation' and t['episode']>ext['parent_episode']})
        overlay['trainer'].update(training_extension=ext,episode_plan_hash=digest(ep_plan),
            evaluation_episodes=[0]+checkpoints,resumable_checkpoint_episodes=[0]+checkpoints)
        overlay['command']['sumo_seed']=ep_plan['rows'][0]['sumo_seed']
        Path(run['overlay']).write_text(yaml.safe_dump(overlay,sort_keys=False))
    tasks=[]
    for task in manifest['tasks']:
        n=extensions[task['run']]['parent_episode']
        if task.get('role')=='validation' and task['episode']<=n:
            if task['episode']!=0:continue
            old_id=task['run_id'];task['episode']=n
            task['run_id']=f'validation_{task["run"]}_e{n:04d}'
            task['output_path']=str(root/'validation'/task['run']/f'episode_{n:04d}')
            task['publication']=str(root/'train'/task['run']/'published'/f'episode_{n:04d}.json')
            task['requires'][0]['path']=task['publication']
            for field in ('command','resume_command'):
                task[field]=[task['run_id'] if arg==old_id else arg for arg in task[field]]
        task['completion']={'path':str(Path(task['output_path'])/'completed.json'),'json_field':'protocol_hash','equals':protocol}
        for req in task.get('requires',[]):req['equals']=protocol
        tasks.append(task)
    manifest['tasks']=tasks
    manifest['budget']['training_episodes']=sum(spec['episodes']-e['parent_episode'] for e in extensions.values())
    manifest['budget']['validation_episodes']=sum(t.get('role')=='validation' for t in tasks)*len(cases['validation'])
    manifest['budget']['maximum_clearance_episodes']=sum(sum(any(e['kind']=='lane_blockage' for e in c['schedule']['events']) for c in cases[t['role']]) for t in tasks if t['kind']=='evaluate')
    write_json(root/'extension_provenance.json',{'parents':extensions,'normal_episodes_added':0,'event_suffix':'equal closure/blockage, randomized pair blocks; balanced severity per kind','initialization':'full resumable state; no epsilon/optimizer/replay reset'})
    manifest['input_sha256']={str(p.relative_to(root)):file_digest(p) for p in root.rglob('*') if p.is_file() and p!=manifest_path}
    manifest['input_sha256'].update(parent_inputs)
    manifest.pop('manifest_hash',None);manifest['manifest_hash']=digest(manifest);write_json(manifest_path,manifest)
    return manifest_path


# A narrow migration for frozen-policy inference, never optimizer/replay resume.
# Observation, action, reward, signal control and model implementations stay exact.
EVALUATION_REPAIR_SOURCES = frozenset({
    'tools/run_paper_baseline.py', 'common/paper_experiment.py',
    'world/sumo_events/safe_placement.py',
})


def verify_evaluation_compatibility(original, current, repair):
    import copy
    expected = copy.deepcopy(original)
    if set(repair) - EVALUATION_REPAIR_SOURCES:
        raise ValueError('Repair changes policy or unsupported simulator code')
    for name, hashes in repair.items():
        if (expected['implementation_sha256'].get(name) != hashes['before'] or
                current['implementation_sha256'].get(name) != hashes['after'] or
                file_digest(ROOT / name) != hashes['after']):
            raise ValueError('Evaluation repair source identity mismatch: ' + name)
        expected['implementation_sha256'][name] = hashes['after']
    expected.pop('hash', None)
    actual = {k: v for k, v in current.items() if k != 'hash'}
    if expected != actual:
        changed = sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
        raise ValueError('Evaluation repair changed training semantics: ' + str(changed))


def plan_frozen_evaluation(source_manifest, output, role='test', run_ids=None, checkpoint=None):
    """Evaluate completed checkpoints on their original case library after repair.

    Incomplete runs are explicitly excluded and listed; no original files change.
    External publications/contracts/checkpoints become frozen inputs in this plan.
    """
    import copy
    old = load_manifest(source_manifest, verify=False)
    if role not in ('validation', 'test'):
        raise ValueError('Invalid evaluation role')
    if run_ids and set(run_ids) - {r['run_id'] for r in old['runs']}:
        raise ValueError('Unknown requested run')
    if old['environment'] != environment_identity(old['config']['agent']):
        raise ValueError('Frozen evaluation dependency environment changed')
    previous = Path(old['root'])
    for name, expected in old['input_sha256'].items():
        if file_digest(previous / name) != expected:
            raise ValueError('Original frozen input changed: ' + name)
    changes = {name: {'before': before, 'after': file_digest(ROOT / name)}
               for name, before in old['source_sha256'].items()
               if before != file_digest(ROOT / name)}
    if set(changes) - EVALUATION_REPAIR_SOURCES:
        raise ValueError('Unsupported source changes for frozen evaluation: ' + str(set(changes) - EVALUATION_REPAIR_SOURCES))
    root = Path(output).resolve()
    if root.exists():
        raise ValueError('Evaluation output already exists')
    runs, tasks, external, excluded = [], [], {}, []
    for run in old['runs']:
        if run_ids and run['run_id'] not in run_ids:
            continue
        source = Path(run['output_path'])
        marker = source / 'completed.json'
        if not marker.exists():
            excluded.append(run['run_id'])
            continue
        if read_json(marker)['protocol_hash'] != old['protocol_hash']:
            raise ValueError('Completed training protocol mismatch')
        binding = read_json(source / 'experiment_contract.json')
        migration = {k: v for k, v in changes.items() if k in binding['implementation_sha256']}
        for k, v in migration.items():
            if binding['implementation_sha256'][k] != v['before']:
                raise ValueError('Original training source mismatch: ' + k)
        imported = dict(run, source_protocol_hash=old['protocol_hash'], evaluation_compatibility=migration)
        runs.append(imported)
        for path in (marker, source/'experiment_contract.json', source/'reward_contract.json', Path(run['overlay'])):
            external[str(path)] = file_digest(path)
        for original_task in old['tasks']:
            if (original_task.get('run') != run['run_id'] or original_task.get('role') != role or
                    (checkpoint is not None and original_task.get('episode') != checkpoint)):
                continue
            task = copy.deepcopy(original_task)
            publication = Path(task['publication']); pub = read_json(publication)
            if pub['protocol_hash'] != old['protocol_hash'] or file_digest(pub['evaluation']) != pub['evaluation_sha256']:
                raise ValueError('Frozen checkpoint/publication mismatch')
            for path in (publication, Path(pub['evaluation'])):
                external[str(path)] = file_digest(path)
            task.pop('depends_on', None)
            task.pop('producer', None)
            task['output_path'] = str(root/role/run['run_id']/f'episode_{task["episode"]:04d}')
            tasks.append(task)
    if not tasks:
        raise ValueError('No completed final checkpoints to evaluate')
    source = source_identity(old['config']['agent'])
    protocol = digest({'source_protocol': old['protocol_hash'], 'source': source,
                       'external': external, 'purpose': 'repaired_frozen_evaluation'})
    root.mkdir(parents=True)
    for name in source:
        path = root/'source_snapshot'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    (root/'cases.json').write_bytes((previous/'cases.json').read_bytes())
    write_json(root/'repair_provenance.json', {'source_manifest': str(Path(source_manifest).resolve()),
        'source_manifest_sha256': file_digest(source_manifest), 'source_changes': changes,
        'excluded_incomplete_runs': excluded, 'mode': 'greedy_evaluation_only_no_training_state_transfer'})
    for task in tasks:
        task['completion'] = {'path': str(Path(task['output_path'])/'completed.json'),
                              'json_field': 'protocol_hash', 'equals': protocol}
        cmd = [sys.executable, '-m', 'tools.run_paper_baseline', 'worker', '--manifest', str(root/'manifest.json'),
               '--task-id', task['run_id']]
        task.update(command=cmd, resume_command=cmd+['--resume'], cwd=str(ROOT))
    cases = read_json(root/'cases.json')[role]
    inputs = {str(p.relative_to(root)): file_digest(p) for p in root.rglob('*') if p.is_file()}
    inputs.update(external)
    inputs[str(Path(source_manifest).resolve())] = file_digest(source_manifest)
    manifest = {'schema_version': SCHEMA, 'root': str(root), 'protocol_hash': protocol,
        'environment': old['environment'], 'config': old['config'], 'source_sha256': source,
        'input_sha256': inputs, 'scenario': old['scenario'], 'runs': runs, 'tasks': tasks,
        'budget': {'training_episodes': 0, 'validation_episodes': len(tasks)*len(cases) if role=='validation' else 0,
                   'test_episodes': len(tasks)*len(cases) if role=='test' else 0,
                   'maximum_clearance_episodes': len(tasks)*sum(any(e['kind']=='lane_blockage' for e in c['schedule']['events']) for c in cases)}}
    manifest['manifest_hash'] = digest(manifest)
    write_json(root/'manifest.json', manifest)
    return root/'manifest.json'
