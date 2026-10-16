#!/usr/bin/env python3
"""Paired-curriculum episode-plan builder (att004 pairaux arm).

Emits a plan whose episodes are PAIRED: each source episode becomes two
consecutive plan rows carrying the identical event schedule and demand seed,
differing only in the row-level ``text_condition`` (canonical vs empty) that
``PlanController``/the tarl adapter consume per episode.  The contrast within
a pair is therefore text-only: identical physics, identical demand.

Order inside each pair alternates by pair index (even pairs canonical-first,
odd pairs empty-first) so neither condition systematically trains earlier.

Usage:
    python3 dev/pairaux/build_pair_plan.py \
        --src configs/events/plans/hz4x4_random_v1.yml \
        --out configs/events/plans/hz4x4_pair_v1.yml \
        --pairs 100 [--per-episode-seeds 9000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairs', type=int, default=100)
    ap.add_argument('--plan-id', default=None)
    ap.add_argument('--per-episode-seeds', type=int, default=None,
                    help='optional absolute seed base; pair p gets this+p')
    args = ap.parse_args()

    src = yaml.safe_load(Path(args.src).read_text())
    episodes = src['episodes'][:args.pairs]
    if len(episodes) < args.pairs:
        raise SystemExit(f'source plan has only {len(episodes)} episodes')

    rows = []
    for p, ep in enumerate(episodes):
        order = ('canonical', 'empty') if p % 2 == 0 else ('empty', 'canonical')
        for cond in order:
            row = {'episode': len(rows),
                   'label': f"{ep.get('label', 'episode')}|pair{p}|{cond}",
                   'events': ep['events'],
                   'text_condition': cond}
            if args.per_episode_seeds is not None:
                row['sumo_seed'] = args.per_episode_seeds + p
            rows.append(row)

    plan = {'schema_version': 'sumo-episode-plan-v1',
            'plan_id': args.plan_id
                     or f"{src.get('plan_id', 'plan')}_pair{args.pairs}",
            'episodes': rows,
            'eval': src['eval']}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(plan, allow_unicode=True, sort_keys=False))

    # Manifest sha via the real loader so it matches the runtime contract.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from world.sumo_events.plans import load_plan
    sha = load_plan(out).plan_sha256
    manifest = {'plan_id': plan['plan_id'], 'plan_sha256': sha,
                'source_plan': str(args.src), 'pairs': args.pairs,
                'per_episode_seeds': args.per_episode_seeds,
                'episodes': len(rows)}
    man = out.with_suffix('.manifest.json')
    man.write_text(json.dumps(manifest, indent=1))
    print(f'wrote {out} ({len(rows)} rows, {args.pairs} pairs) + {man}')
    print(f'plan_sha256={sha}')


if __name__ == '__main__':
    main()
