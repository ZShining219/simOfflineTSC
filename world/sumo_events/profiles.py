"""Joint severity profiles and exposure-selected targets for reusable experiments."""
import math
from collections import defaultdict

from .schema import Event, Schedule


def validate(spec):
    from common.paper_experiment import ROOT, file_digest
    p = spec['strong_events']
    if set(p) != {'profiles', 'exposure_file', 'exposure_sha256', 'exposure_fraction', 'clearance_controls'}:
        raise ValueError('Invalid strong_events fields')
    if spec['event_execution']['placement']['version'] != 'clear-conflicts-v1' or p['clearance_controls'] is not True:
        raise ValueError('Strong profiles require audited forced placement and clearance controls')
    if not 0 < p['exposure_fraction'] <= 1 or set(p['profiles']) != {'S1', 'S2'}:
        raise ValueError('Explicit S1/S2 exposure profiles required')
    if file_digest(ROOT / p['exposure_file']) != p['exposure_sha256']:
        raise ValueError('Exposure selection input changed')
    for name, row in p['profiles'].items():
        if set(row) != {'duration','blocked_lanes','rain_factor'}:
            raise ValueError('Profiles jointly define duration, blocked lanes and rain factor')
        if (type(row['duration']) is not int or row['duration'] <= 0 or row['duration'] % spec['action_interval']
                or type(row['blocked_lanes']) is not int or not 1 <= row['blocked_lanes'] <= 2
                or not 0 < row['rain_factor'] < 1):
            raise ValueError('Invalid joint profile')
        if max(spec['events']['begin'] + spec['events']['validation_begin'] + spec['events']['test_begin']) + row['duration'] + spec['events']['recovery_seconds'] >= spec['seconds']:
            raise ValueError('Profile violates recovery horizon')
    if spec['events']['cases_per_kind'] % len(p['profiles']):
        raise ValueError('Test cases must balance severity profiles')


def select_pools(spec, pools, catalog):
    from common.paper_experiment import ROOT, read_json
    settings = spec['strong_events']
    scores = read_json(ROOT / settings['exposure_file'])['exposure_lane_counts']
    all_lanes = defaultdict(list)
    for lane, info in catalog.items():
        if not info['internal'] and info['motor_vehicle_lane']:
            all_lanes[info['edge_id']].append(lane)
    max_lanes = max(p['blocked_lanes'] for p in settings['profiles'].values())
    length = spec['event_execution']['placement']['obstacle_length_m']
    for pool in pools.values():
        bundles = []
        for edge in pool['edges']:
            lanes = sorted([l for l in pool['blockage_lanes'] if catalog[l]['edge_id'] == edge and
                            catalog[l]['length'] - spec['events']['block_distance'] >= length],
                           key=lambda l: (-scores.get(l, 0), l))
            if len(lanes) >= max_lanes and len(all_lanes[edge]) > max_lanes:
                bundles.append({'edge': edge, 'lanes': lanes[:max_lanes],
                    'blockage_exposure': sum(scores.get(l,0) for l in lanes[:max_lanes]),
                    'closure_exposure': sum(scores.get(l,0) for l in all_lanes[edge])})
        for kind, score in [('lane_blockage','blockage_exposure'), ('road_closure','closure_exposure')]:
            ranked = sorted(bundles,key=lambda b:(-b[score],b['edge']))
            selected = [b for b in ranked[:math.ceil(len(ranked)*settings['exposure_fraction'])] if b[score]>0]
            if not selected:
                raise ValueError('No exposed target bundle for ' + kind)
            pool[kind + '_targets'] = selected
    return pools


def generate(spec, pools, catalog, kind, rng, split, begin, severity):
    if kind == 'normal':
        return Schedule(()).to_dict()
    p = spec['strong_events']['profiles'][severity]
    events, used = [], set()
    for component in kind.split('+'):
        base = dict(kind=component,begin=begin,end=begin+p['duration'])
        if component == 'global_rain':
            events.append(Event(event_id=f'event_{len(events)}',speed_factor=p['rain_factor'],**base))
            continue
        candidates = [x for x in pools[split][component+'_targets'] if x['edge'] not in used]
        if not candidates:
            raise ValueError('Combined event needs distinct exposed roads')
        target = rng.choice(candidates);used.add(target['edge'])
        if component == 'road_closure':
            events.append(Event(event_id=f'event_{len(events)}',edge_id=target['edge'],**base))
        else:
            for lane in target['lanes'][:p['blocked_lanes']]:
                events.append(Event(event_id=f'event_{len(events)}',lane_id=lane,
                    position=catalog[lane]['length']-spec['events']['block_distance'],
                    position_tolerance=spec['events']['position_tolerance'],**base))
    return Schedule(tuple(events)).to_dict()
