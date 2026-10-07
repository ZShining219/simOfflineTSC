#!/usr/bin/env python3
"""Aggregate frozen-evaluation packages into paired cross-arm comparisons.

Reads every ``summary.csv`` under one or more eval package roots, groups rows
by (agent/run-family, training_seed, evaluation condition) and reports
per-condition travel_time means plus paired deltas against a chosen baseline
family.  Run families are inferred from the package/run name; pass
``--family`` regexes to override.

Usage:
    python3 tools/stage_eval_aggregate.py \
        --roots artifacts/att_entity_003/eval_stage4/packages \
                artifacts/att_entity_003/eval_v1/packages \
        --baseline colight --metric travel_time --out report.csv
"""
import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path


def infer_family(run_id):
    """Map run names to comparison arms; extend as new families land."""
    table = [
        (r'align_trans_meta', 'flx_Bprime_meta'),
        (r'align_trans_l1\.0', 'flx_B_l1.0'),
        (r'align_trans_l0\.1|align_trans(?!_meta)', 'flx_B_fused'),
        (r'mplight', 'mplight'),
        (r'maxpressure|tarl_mp', 'maxpressure'),
        (r'tarl.*att|tarl.*attention|tarlp_att', 'tarl_att'),
        (r'tarl.*gatg|tarl.*gating|tarlp_gatg', 'tarl_gating'),
        (r'tarl.*gat|tarlp_gat', 'tarl_gat'),
        (r'tarl.*sen|tarl.*sensor|tarlp_sen', 'tarl_sensor'),
        (r'sga_flx', 'sga_flx'),
        (r'sga', 'sga'),
        (r'colight', 'colight'),
    ]
    for pat, fam in table:
        if re.search(pat, run_id):
            return fam
    return run_id


def condition_key(row):
    scene = (row.get('evaluation_scene') or '').strip()
    cid = row.get('controller_id', '')
    m = re.search(r'(randeval_v1_\d+_\w+|none|all)$', cid)
    return scene or (m.group(1) if m else cid)


def collect(roots):
    rows = []
    for root in roots:
        for f in sorted(Path(root).glob('*/summary.csv')):
            run_id = f.parent.name
            family = infer_family(run_id)
            with open(f, newline='') as h:
                for r in csv.DictReader(h):
                    r['run_id'] = run_id
                    r['family'] = family
                    r['condition'] = condition_key(r)
                    rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--baseline', default='colight')
    ap.add_argument('--metric', default='travel_time')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rows = collect(args.roots)
    metric = args.metric
    # (family, condition, training_seed) -> metric value
    cell = defaultdict(dict)
    for r in rows:
        if not r.get(metric):
            continue
        cell[(r['family'], r['condition'])][r['training_seed']] = float(r[metric])

    families = sorted({k[0] for k in cell})
    conditions = sorted({k[1] for k in cell})
    out_rows = []
    print(f"{'condition':34s} " + ' '.join(f'{f:>16s}' for f in families))
    for cond in conditions:
        line = f'{cond:34s} '
        base_vals = {}
        for fam in families:
            vals = cell.get((fam, cond), {})
            if vals:
                mean = sum(vals.values()) / len(vals)
                line += f'{mean:8.2f}(n{len(vals)}) '
                if fam == args.baseline:
                    base_vals = vals
            else:
                line += f'{"-":>16s} '
        print(line)
        for fam in families:
            vals = cell.get((fam, cond), {})
            if not vals or not base_vals:
                continue
            paired = [(s, vals[s] - base_vals[s]) for s in vals if s in base_vals]
            if not paired:
                continue
            dmean = sum(d for _, d in paired) / len(paired)
            out_rows.append({
                'condition': cond, 'family': fam,
                'baseline': args.baseline, 'n_paired': len(paired),
                'mean_' + metric: sum(vals.values()) / len(vals),
                'baseline_mean': sum(base_vals.values()) / len(base_vals),
                'paired_delta_mean': dmean,
                'paired_delta_per_seed': json.dumps(dict(paired)),
            })
    if args.out and out_rows:
        w = csv.DictWriter(open(args.out, 'w'), fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
        print(f'-> {args.out}')


if __name__ == '__main__':
    main()
