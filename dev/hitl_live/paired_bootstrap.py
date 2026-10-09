#!/usr/bin/env python
"""paired_bootstrap.py — 同 seed 配对 bootstrap 统计（spec §11.4 口径）。

对每个 condition × seed 取冻结自主 eval 序列，算：
  * AULC_delta:  该臂 AULC − 同 seed C0 臂 AULC（梯形积分，横轴=决策数）
  * final_delta: 末集指标差（travel_time/unfinished/throughput/rrt）
然后在 seed 维度做 10000 次配对重采样 → 均值差 + 95% CI。

用法：
    python dev/hitl_live/paired_bootstrap.py \
        --root output_data/hitl_probe/pilot5 \
        --plan output_data/hitl_probe/pilot5/plan.json \
        --out output_data/hitl_probe/pilot5/stats/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

METRICS = ('travel_time', 'unfinished_vehicles', 'throughput', 'rrt')
BASELINE = 'HITL_C0'


def _seed_of(run_id):
    for part in str(run_id).split('-'):
        if part.startswith('SD'):
            return int(part[2:])
    return None


def collect(root, plan_path):
    """(condition, seed, episode) -> summary metrics."""
    plan = json.load(open(plan_path))
    cond_of = {c['logical_run_id']: c.get('condition', 'HITL_C0')
               for c in plan.get('children', [])}
    rows = []
    pattern = os.path.join(
        root, '_shared_evaluation', 'physical', '*', 'attempt_*')
    seen = set()
    for attempt_dir in glob.glob(pattern):
        req_p = os.path.join(attempt_dir, 'request.json')
        sum_p = os.path.join(attempt_dir, 'summary.json')
        if not (os.path.isfile(req_p) and os.path.isfile(sum_p)):
            continue
        request = json.load(open(req_p))
        snapshot = request.get('snapshot_path', '')
        run_id = next((p for p in Path(snapshot).parts
                       if p in cond_of), None)
        if run_id is None:
            continue
        ep = int(request.get('checkpoint_episode', -1))
        key = (run_id, ep)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            'run_id': run_id,
            'condition': cond_of[run_id],
            'seed': _seed_of(run_id),
            'episode': ep,
            'metrics': json.load(open(sum_p)),
        })
    return rows


def aulc(points, dpe):
    """trapezoid AULC over actual decision counts."""
    pts = sorted(points.items())
    if len(pts) < 2:
        return None
    x = np.asarray([p[0] * dpe for p in pts], dtype=np.float64)
    y = np.asarray([p[1] for p in pts], dtype=np.float64)
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    return float(trapz(y, x) / (x[-1] - x[0]))


def paired_deltas(rows, metric, dpe):
    """condition -> {seed: (aulc_delta, final_delta)} vs baseline."""
    series = {}
    for r in rows:
        v = r['metrics'].get(metric)
        if v is None:
            continue
        series.setdefault((r['condition'], r['seed']), {})[r['episode']] = v
    # Snapshot-dedup aliasing: while weights are still identical to the ep0
    # checkpoint (no gradient updates yet), the evaluator runs the shared
    # snapshot once and attributes it to the first requesting run — other
    # arms legitimately "miss" episodes 1..k even though their checkpoints
    # are bit-identical.  Fill those leading episodes with the earliest
    # evaluated value available for the same seed (same frozen snapshot).
    seed_early = {}
    for (cond, seed), pts in series.items():
        early = min(pts)
        if seed not in seed_early or early < seed_early[seed][0]:
            seed_early[seed] = (early, pts[early])
    for (cond, seed), pts in series.items():
        if seed not in seed_early:
            continue
        early_val = seed_early[seed][1]
        for ep in range(1, min(pts)):
            pts.setdefault(ep, early_val)
    # Same-snapshot interior aliasing: e.g. ep2 checkpoints are still
    # bit-identical to ep1 (learning_start gates updates until ep3), so
    # the evaluator aliases the request to the ep1 attempt and no ep2
    # request exists.  Forward-fill interior gaps with the run's own
    # most recent evaluated value.
    for (cond, seed), pts in series.items():
        for ep in range(min(pts) + 1, max(pts) + 1):
            if ep not in pts:
                prev = max(e for e in pts if e < ep)
                pts[ep] = pts[prev]
    per_seed = {}
    for (cond, seed), pts in series.items():
        per_seed.setdefault(cond, {})[seed] = {
            'aulc': aulc(pts, dpe),
            'final': pts[max(pts)] if pts else None,
        }
    out = {}
    base = per_seed.get(BASELINE, {})
    for cond, seeds in per_seed.items():
        if cond == BASELINE:
            continue
        deltas = {}
        for seed, val in seeds.items():
            if seed not in base or base[seed]['aulc'] is None \
                    or val['aulc'] is None:
                continue
            deltas[seed] = {
                'aulc_delta': val['aulc'] - base[seed]['aulc'],
                'final_delta': val['final'] - base[seed]['final'],
            }
        out[cond] = deltas
    return out


def bootstrap(deltas, resamples=10000, seed=11):
    """Paired bootstrap over seeds -> mean delta + percentile CI."""
    keys = sorted(deltas)
    if not keys:
        return None
    vals = np.asarray([deltas[k] for k in keys], dtype=np.float64)
    rng = np.random.RandomState(seed)
    draws = rng.choice(len(vals), size=(resamples, len(vals)))
    means = vals[draws].mean(axis=1)
    return {
        'n_seeds': len(keys),
        'mean_delta': float(vals.mean()),
        'ci95': [float(np.percentile(means, 2.5)),
                 float(np.percentile(means, 97.5))],
        'p_one_sided_improvement': float(np.mean(means < 0)),
        'resamples': resamples,
    }


def main(argv=None):
    global BASELINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--baseline', default=BASELINE)
    parser.add_argument('--decisions-per-episode', type=int, default=120)
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    BASELINE = args.baseline
    rows = collect(args.root, args.plan)
    os.makedirs(args.out, exist_ok=True)
    result = {'baseline': BASELINE, 'metrics': {}}
    for metric in METRICS:
        deltas = paired_deltas(rows, metric, args.decisions_per_episode)
        metric_out = {}
        for cond, dd in sorted(deltas.items()):
            metric_out[cond] = {
                'seeds': dd,
                'aulc_bootstrap': bootstrap(
                    {s: d['aulc_delta'] for s, d in dd.items()},
                    args.resamples, args.seed),
                'final_bootstrap': bootstrap(
                    {s: d['final_delta'] for s, d in dd.items()},
                    args.resamples, args.seed),
            }
        result['metrics'][metric] = metric_out
    out_path = os.path.join(args.out, 'paired_bootstrap.json')
    with open(out_path, 'w') as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    # compact console table
    for metric in METRICS:
        print(f'== {metric} (delta vs {BASELINE}; negative = better)==')
        for cond, mo in sorted(result['metrics'][metric].items()):
            b = mo['aulc_bootstrap']
            if b:
                print(f"  {cond:14s} AULC d={b['mean_delta']:+.2f} "
                      f"CI95=[{b['ci95'][0]:+.2f},{b['ci95'][1]:+.2f}] "
                      f"P(<0)={b['p_one_sided_improvement']:.3f} "
                      f"n={b['n_seeds']}")
    return result


if __name__ == '__main__':
    main()
