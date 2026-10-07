#!/usr/bin/env python3
"""Null-model comparison for focus-set scoring (ATT-ENTITY-004).

Null hypothesis: the text-token attention pattern carries no alignment with
the physically grounded focus set. Operationalized by permuting the key/node
dimension (j) of fusion_w per decision step and recomputing focus_ratio.
Reported per cell: true mean vs null distribution (mean±sd, empirical p,
z-score) over event-active steps.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID = 4
LOCAL_TYPES = {1.0, 2.0}


def neighbors(idx):
    x, y = idx // GRID + 1, idx % GRID + 1
    return [(x + dx - 1) * GRID + (y + dy - 1)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 1 <= x + dx <= GRID and 1 <= y + dy <= GRID]


def episode_ratios(npz_path, rng, n_perm):
    """Return (per-step true ratios, per-permutation step-MEAN ratios).

    The null statistic is the mean over the episode's event steps — same
    aggregation as the true score — not the per-step null distribution.
    """
    z = np.load(npz_path)
    zt, fw = z['z_task'], z['fusion_w']
    bound_mask = (zt[..., 0] > 0.5) & np.isin(zt[..., 1], list(LOCAL_TYPES))
    true, perm_sums = [], np.zeros(n_perm)
    n_steps = 0
    for k in range(zt.shape[0]):
        bound = np.where(bound_mask[k])[0]
        if not len(bound):
            continue
        n_steps += 1
        F = sorted(set(bound.tolist()) |
                   {n for b in bound for n in neighbors(int(b))})
        denom = len(F) / 16.0
        w = fw[k, 0]
        true.append(w[bound][:, F].sum(-1).mean() / denom)
        for p in range(n_perm):
            wp = w[:, rng.permutation(16)]
            perm_sums[p] += wp[bound][:, F].sum(-1).mean() / denom
    return np.asarray(true), perm_sums, n_steps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state_lib',
                    default='data/output_data/analysis/att_entity_004/state_lib')
    ap.add_argument('--output',
                    default='data/output_data/analysis/att_entity_004/focus_score/null_model.json')
    ap.add_argument('--n_perm', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    lib = (ROOT / args.state_lib).resolve()
    rng = np.random.default_rng(args.seed)
    rows = []
    for cell_dir in sorted(p for p in lib.iterdir()
                           if p.is_dir() and p.name.startswith('a3s')):
        tt, perm_sums, steps = [], np.zeros(args.n_perm), 0
        for npz in sorted(cell_dir.glob('ep*.npz')):
            t_, ps_, ns_ = episode_ratios(npz, rng, args.n_perm)
            tt.append(t_)
            perm_sums += ps_
            steps += ns_
        if not tt or steps == 0:
            continue
        t_all = np.concatenate(tt)
        null_means = perm_sums / steps      # one null mean per permutation
        rows.append({
            'cell': cell_dir.name,
            'event_steps': int(len(t_all)),
            'true_mean': float(t_all.mean()),
            'true_sd': float(t_all.std()),
            'null_mean': float(null_means.mean()),
            'null_sd': float(null_means.std()),
            'z_vs_null': float((t_all.mean() - null_means.mean())
                               / (null_means.std() + 1e-12)),
            'p_empirical': float(
                (np.sum(null_means >= t_all.mean()) + 1)
                / (args.n_perm + 1)),
        })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {'n_perm': args.n_perm, 'perm': 'fusion_w key/node dim shuffled per step',
         'rows': rows}, indent=1))
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
