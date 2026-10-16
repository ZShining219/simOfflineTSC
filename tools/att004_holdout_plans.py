#!/usr/bin/env python3
"""Generate ATT-ENTITY-004 T3/T4 held-out event plans (protocol doc §6.2).

T4 (combo holdout): train episodes keep normal + all single events + only the
multi combos {blockage+rain, closure+rain}; the eval schedule is the held-out
combination {blockage+closure}. Single blockage and closure both appear in
training; only the combination is held out.

T3 (duration extrapolation): every train event duration is rescaled into
[180, 360]s; the eval schedule uses durations in [420, 600]s with the same
targets/kinds as the shared eval compound.

Outputs four plans + a manifest with sha256 under analysis/att_entity_004.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
POOL = 'configs/events/plans/hz4x4_random_v1.yml'
OUT_DIR = ROOT / 'configs/events/plans'
MANIFEST = ROOT / 'data/output_data/analysis/att_entity_004/holdout_plans_manifest.json'

T4_ALLOWED_PAIRS = {frozenset({'lane_blockage', 'global_rain'}),
                    frozenset({'road_closure', 'global_rain'})}
T4_HELDOUT = frozenset({'lane_blockage', 'road_closure'})
T3_TRAIN_BAND = (180, 360)
T3_EVAL_DURS = {'lane_blockage': 420, 'road_closure': 540, 'global_rain': 600}
HORIZON = 3600


def kinds_of(events):
    return frozenset(e['kind'] for e in events)


def renumber(events, ep_idx):
    out = []
    for j, e in enumerate(events):
        e = copy.deepcopy(e)
        e['event_id'] = f'ep{ep_idx:04d}_e{j}'
        out.append(e)
    return out


def t4_train_episodes(pool, n, seed):
    keep = []
    for row in pool:
        ks = kinds_of(row['events'])
        if len(ks) <= 1 or ks in T4_ALLOWED_PAIRS:
            keep.append(row)
    assert len(keep) >= n, f'T4 pool too small: {len(keep)} < {n}'
    rng = random.Random(seed)
    order = list(range(len(keep)))
    rng.shuffle(order)
    eps = []
    for i, k in enumerate(order[:n]):
        row = keep[k]
        eps.append({'episode': i, 'label': f"t4_{row['label']}",
                    'events': renumber(row['events'], i)})
    return eps


def t3_train_episodes(pool, n, seed):
    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    eps = []
    for i, k in enumerate(order[:n]):
        row = pool[k]
        evs = []
        for e in renumber(row['events'], i):
            dur = e['end'] - e['begin']
            lo, hi = T3_TRAIN_BAND
            new_dur = lo + (dur % (hi - lo + 1))
            e['end'] = e['begin'] + new_dur
            assert e['end'] <= HORIZON - 900, (e['event_id'], e['end'])
            evs.append(e)
        eps.append({'episode': i, 'label': f"t3_{row['label']}",
                    'events': evs})
    return eps


def std_eval():
    d = yaml.safe_load(open(ROOT / POOL))
    return d['eval']


def t4_eval_schedule():
    ev = copy.deepcopy(std_eval()['events'])
    held = [e for e in ev if e['kind'] in T4_HELDOUT]
    assert kinds_of(held) == T4_HELDOUT
    return {'schema_version': 'sumo-events-v1', 'events': held}


def t3_eval_schedule():
    ev = copy.deepcopy(std_eval()['events'])
    for e in ev:
        e['end'] = e['begin'] + T3_EVAL_DURS[e['kind']]
    return {'schema_version': 'sumo-events-v1', 'events': ev}


def write_plan(name, episodes, eval_schedule):
    plan = {'schema_version': 'sumo-episode-plan-v1', 'plan_id': name,
            'episodes': episodes, 'eval': eval_schedule}
    path = OUT_DIR / f'{name}.yml'
    with open(path, 'w') as fh:
        yaml.safe_dump(plan, fh, sort_keys=False, allow_unicode=True)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(path.relative_to(ROOT)), sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20261003)
    args = ap.parse_args()

    pool = yaml.safe_load(open(ROOT / POOL))['episodes']
    eval_only_row = {'episode': 0, 'label': 'eval_only_normal', 'events': []}
    results = {}
    results['t4_train'] = write_plan(
        'att004_t4_train_v1', t4_train_episodes(pool, args.episodes, args.seed),
        std_eval())
    results['t4_eval'] = write_plan(
        'att004_t4_eval_v1', [eval_only_row], t4_eval_schedule())
    results['t3_train'] = write_plan(
        'att004_t3_train_v1', t3_train_episodes(pool, args.episodes, args.seed),
        std_eval())
    results['t3_eval'] = write_plan(
        'att004_t3_eval_v1', [eval_only_row], t3_eval_schedule())

    # validation
    errs = []
    t4t = yaml.safe_load(open(ROOT / results['t4_train'][0]))
    for ep in t4t['episodes']:
        ks = kinds_of(ep['events'])
        if len(ks) > 1 and ks not in T4_ALLOWED_PAIRS:
            errs.append(f"T4 train leak: {ep['label']}")
    t4e = yaml.safe_load(open(ROOT / results['t4_eval'][0]))
    if kinds_of(t4e['eval']['events']) != T4_HELDOUT:
        errs.append('T4 eval missing held-out combo')
    single_kinds = set()
    for ep in t4t['episodes']:
        ks = kinds_of(ep['events'])
        if len(ks) == 1:
            single_kinds |= ks
    for req in ('lane_blockage', 'road_closure', 'global_rain'):
        if req not in single_kinds:
            errs.append(f'T4 train missing single kind {req}')
    t3t = yaml.safe_load(open(ROOT / results['t3_train'][0]))
    for ep in t3t['episodes']:
        for e in ep['events']:
            d = e['end'] - e['begin']
            if not (T3_TRAIN_BAND[0] <= d <= T3_TRAIN_BAND[1]):
                errs.append(f"T3 train duration {d} out of band in {ep['label']}")
    t3e = yaml.safe_load(open(ROOT / results['t3_eval'][0]))
    for e in t3e['eval']['events']:
        d = e['end'] - e['begin']
        if not (420 <= d <= 600):
            errs.append(f'T3 eval duration {d} out of band')

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'generated': 'att004_holdout_plans', 'seed': args.seed,
                'episodes': args.episodes, 'pool': POOL,
                'plans': {k: {'path': v[0], 'sha256': v[1]}
                          for k, v in results.items()},
                'validation_errors': errs}
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    if errs:
        raise SystemExit(f'validation failed: {errs}')


if __name__ == '__main__':
    main()
