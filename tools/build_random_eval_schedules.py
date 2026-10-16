"""Generate held-out random evaluation schedules for random-E runs.

Each output file is one sumo-events-v1 schedule drawn from the SAME
distribution as the training plan but a DIFFERENT seed, so evaluation
measures generalization to unseen event instances rather than memorized
episodes.  Files are emitted as eval "conditions" under
configs/events/eval_random/<set_id>/ and consumed per controller via the
evaluation-manifest `event_schedule` field.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_random_event_plan import (catalog, conflicts, sample_event,  # noqa: E402
                                           KINDS)
from world.sumo_events.runtime import SumoEventRuntime  # noqa: E402
from world.sumo_events.schema import load_schedule  # noqa: E402


def _pick_kinds(rng, stratum):
    if stratum == 'normal':
        return []
    if stratum == 'single':
        return [rng.choice(KINDS)]
    return rng.sample(KINDS, 2)


def build_schedules(args):
    rng = random.Random(args.seed)
    _, lanes, edges = catalog(args.net)
    strata = (['normal'] * args.n_normal +
              ['single'] * args.n_single +
              ['multi'] * args.n_multi)
    rng.shuffle(strata)
    out = []
    for index, stratum in enumerate(strata):
        events = []
        for slot, kind in enumerate(_pick_kinds(rng, stratum)):
            for _ in range(64):
                candidate = sample_event(rng, kind, slot, index, args, lanes, edges)
                candidate['event_id'] = f'{args.set_id}_{index:02d}_e{slot}'
                if all(not conflicts(candidate, existing, lanes) for existing in events):
                    events.append(candidate)
                    break
            else:
                raise RuntimeError(f'No conflict-free target for eval row {index}')
        out.append({'condition': f'{args.set_id}_{index:02d}_{stratum}',
                    'schedule': {'schema_version': 'sumo-events-v1',
                                 'events': events}})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--net', default=str(ROOT / 'data/raw_data/hangzhou_4x4_gudang_18041610_1h/hangzhou_4x4_gudang_18041610_1h.net.xml'))
    parser.add_argument('--seed', type=int, required=True,
                        help='MUST differ from the training-plan seed')
    parser.add_argument('--train-seed', type=int, default=20260928)
    parser.add_argument('--set-id', default='randeval')
    parser.add_argument('--n-normal', type=int, default=3)
    parser.add_argument('--n-single', type=int, default=5)
    parser.add_argument('--n-multi', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=3600)
    parser.add_argument('--recovery', type=int, default=600)
    parser.add_argument('--begin-min', type=int, default=600)
    parser.add_argument('--begin-max', type=int, default=1800)
    parser.add_argument('--dur-min', type=int, default=180)
    parser.add_argument('--dur-max', type=int, default=600)
    parser.add_argument('--rain-min', type=float, default=0.7)
    parser.add_argument('--rain-max', type=float, default=0.9)
    parser.add_argument('--position-min-ratio', type=float, default=0.25)
    parser.add_argument('--position-tolerance', type=float, default=20.0)
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()
    if args.seed == args.train_seed:
        raise ValueError('eval seed must differ from the training-plan seed')
    out_dir = Path(args.out_dir or
                   ROOT / 'configs/events/eval_random' / args.set_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_schedules(args)
    conditions = {}
    for row in rows:
        path = out_dir / f"{row['condition']}.yml"
        path.write_text(yaml.safe_dump(row['schedule'], sort_keys=False))
        conditions[row['condition']] = str(path.relative_to(ROOT))
    # Round-trip validation: every emitted file must reload through the schema
    # and pass the runtime's target validator before entering the eval library.
    runtime = SumoEventRuntime(args.net, load_schedule(
        str(out_dir / f"{rows[0]['condition']}.yml")))
    for row in rows:
        runtime.set_schedule(load_schedule(
            str(out_dir / f"{row['condition']}.yml")))
    manifest = {
        'set_id': args.set_id, 'seed': args.seed, 'count': len(conditions),
        'train_plan_seed': args.train_seed,
        'conditions': conditions,
        'set_sha256': hashlib.sha256(json.dumps(
            sorted(conditions.items()), sort_keys=True).encode()).hexdigest(),
    }
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
