"""Convergence-time analysis for the o3 st3 (c3->s1) entry-shock cell.

Combines three eval sources into one frozen-eval trajectory per
(backbone, condition, seed):

  ep0/50    in-train EVALUATION records of the stage-3 run (sumo_seed=0,
            marked hollow — different eval seed than the snap cells)
  ep5/10/15/20  snap3_* model_snapshot eval cells (eval seeds 101/102/103;
            dqn-family snapshots are target-net weights -> conservative,
            frap snapshots are online-net weights)
  ep100     ev3 frozen stage-end eval cells (eval seeds 101/102/103)

Reference = the same backbone+seed single-scene baseline ATT
(matrix_report.json); 'settled' = first measured episode from which all
subsequent points stay <= ref*(1+TOL). Runs that never settle are kept
as 'never', not dropped from summaries.

Run: python dev/cl_hz1x1/plot_soa_snap.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'output_data' / 'tsc'
ART = ROOT / 'artifacts' / 'cl_hz1x1'
TOL = 0.05
SNAP_EPS = (5, 10, 15, 20)
SEEDS = (0, 1, 2)
EVAL_SEEDS = (101, 102, 103)

BB = {  # backbone -> (agent_dir, train-run prefix per condition)
    'ddqn': ('sumo_dqn', {'clear': 'cl1_ddqn_clear_o3',
                         'semi': 'so1_soddqn_o3',
                         'p': 'soa3_soddqnp_o3'}),
    'duel': ('sumo_dqn', {'clear': 'cl1_duel_clear_o3',
                         'semi': 'so1_soduel_o3',
                         'p': 'soa3_soduelp_o3'}),
    'frap': ('sumo_frap', {'clear': 'cl1_frap_clear_o3',
                          'semi': 'so1_sofrap_o3',
                          'p': 'soa3_sofrapp_o3'}),
}
CONDS = [('clear', '#d62728', 'pure online'),
         ('semi', '#1f77b4', 'semi-offline r=0.25'),
         ('p', '#2ca02c', 'semi-offline + P anchor')]

BASELINES = json.loads((ART / 'matrix_report.json').read_text())['baselines']


def final_eval(run_dir):
    """(ATT, throughput, unfinished) from the FINAL_EVALUATION record."""
    p = Path(run_dir) / 'metrics' / 'records.jsonl'
    if not p.exists():
        return None
    for line in p.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r.get('record_type') == 'FINAL_EVALUATION':
            return (r.get('travel_time'), r.get('throughput'),
                    r.get('unfinished_vehicles'))
    return None


def in_train_eval(run_dir, ep):
    p = Path(run_dir) / 'metrics' / 'records.jsonl'
    if not p.exists():
        return None
    for line in p.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if (r.get('record_type') == 'EVALUATION'
                and r.get('episode') == ep):
            return (r.get('travel_time'), r.get('throughput'),
                    r.get('unfinished_vehicles'))
    return None


def trajectory(bb, cond, sd):
    """{episode: [att over eval seeds]} + {episode: seed0 in-train eval}."""
    adir, prefixes = BB[bb]
    prefix = prefixes[cond]
    snap_pts, train_pts = defaultdict(list), {}
    for ep in SNAP_EPS:
        for es in EVAL_SEEDS:
            v = final_eval(OUT / adir / 'sumohz1x1'
                           / f'snap3_{bb}_{cond}_o3_sd{sd}_e{ep}_es{es}')
            if v and v[0] is not None:
                snap_pts[ep].append(v)
    for es in EVAL_SEEDS:
        v = final_eval(OUT / adir / 'sumohz1x1'
                       / f'{prefix}_sd{sd}_ev3_s1_es{es}')
        if v and v[0] is not None:
            snap_pts[100].append(v)
    src = OUT / adir / 'sumohz1x1' / f'{prefix}_sd{sd}_st3'
    for ep in (0, 50):
        v = in_train_eval(src, ep)
        if v and v[0] is not None:
            train_pts[ep] = v
    return snap_pts, train_pts


def settled_episode(snap_pts, ref):
    """First episode from which every later point stays <= ref*(1+TOL)."""
    if ref is None:
        return None
    eps = sorted(snap_pts)
    for i, ep in enumerate(eps):
        later = [a for e2 in eps[i:] for a, *_ in snap_pts[e2]]
        if later and all(a <= ref * (1 + TOL) for a in later):
            return ep
    return 'never'


def main():
    metrics = {}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for c, bb in enumerate(BB):
        ax = axes[c]
        refs = [BASELINES[f'{bb}_s1_sd{sd}']['travel_time'] for sd in SEEDS]
        ref_mean = float(np.mean(refs))
        ax.axhspan(ref_mean, ref_mean * (1 + TOL), color='gray', alpha=0.15)
        ax.axhline(ref_mean, color='gray', ls='--', lw=1,
                   label='s1 baseline ±5%' if c == 0 else None)
        for cond, col, lab in CONDS:
            xs, means, lo, hi = [], [], [], []
            settle = []
            for sd in SEEDS:
                snap_pts, train_pts = trajectory(bb, cond, sd)
                ref = BASELINES[f'{bb}_s1_sd{sd}']['travel_time']
                metrics[f'{bb}|{cond}|sd{sd}'] = {
                    'ref_att': ref,
                    'snapshot_eval': {
                        str(ep): [round(v[0], 3) for v in snap_pts[ep]]
                        for ep in sorted(snap_pts)},
                    'intrain_eval': {
                        str(ep): round(v[0], 3)
                        for ep, v in sorted(train_pts.items())},
                    'settled_episode': settled_episode(snap_pts, ref),
                }
                settle.append(settled_episode(snap_pts, ref))
                per_seed = sorted(snap_pts)
                ax.plot([e for e in per_seed],
                        [np.mean([v[0] for v in snap_pts[e]])
                         for e in per_seed],
                        'o-', color=col, lw=0.6, ms=2.5, alpha=0.35)
                for ep, v in sorted(train_pts.items()):
                    ax.plot(ep, v[0], 'o', mfc='none', mec=col, ms=5,
                            alpha=0.5)
            # mean curve over seeds (only episodes with all 3 seeds)
            all_eps = sorted({e for sd in SEEDS
                              for e in trajectory(bb, cond, sd)[0]})
            for ep in all_eps:
                vals = []
                for sd in SEEDS:
                    sp, _ = trajectory(bb, cond, sd)
                    if ep in sp:
                        vals.append(np.mean([v[0] for v in sp[ep]]))
                if len(vals) == 3:
                    xs.append(ep)
                    means.append(np.mean(vals))
                    lo.append(min(vals))
                    hi.append(max(vals))
            ax.plot(xs, means, 'o-', color=col, lw=2.2, ms=4, label=lab)
            ax.fill_between(xs, lo, hi, color=col, alpha=0.12)
            s_str = ', '.join(str(s) for s in settle)
            print(f'{bb:5s} {cond:5s} settled@: {s_str}')
        ax.set_xscale('symlog', linthresh=5)
        ax.set_xticks([0, 5, 10, 15, 20, 100])
        ax.set_xticklabels(['0', '5', '10', '15', '20', '100'])
        ax.set_title(f'{bb} — o3 st3: c3→s1 frozen-eval convergence',
                     fontsize=10)
        ax.set_xlabel('stage-3 episode of the evaluated snapshot')
        ax.grid(alpha=0.3)
        if c == 0:
            ax.set_ylabel('eval ATT on s1 (s)')
            ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('cl_hz1x1 snap3 — frozen-eval convergence on new scene s1 '
                 '(filled/thin = snapshot eval seeds 101-103; hollow = '
                 'in-train eval seed 0; dqn-family snapshots = target net)',
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ART / 'fig_soa_snap_convergence.png', dpi=150)
    plt.close(fig)
    (ART / 'snap3_metrics.json').write_text(
        json.dumps(metrics, indent=1, ensure_ascii=False), encoding='utf-8')
    print('wrote fig_soa_snap_convergence.png + snap3_metrics.json')


if __name__ == '__main__':
    main()
