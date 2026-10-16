"""Dose-response figure: replay ratio rho x {replay, +policy anchor} (o2).

Reads matrix_report_soa_soar.json (rho arms r12/r50) plus the r=0.25 cells
from matrix_report_so_recomputed.json / matrix_report_soa_soa1.json.

Output: fig_soa_rho.png — per backbone, two panels:
  top: worst old-scene ΔATT vs rho (seed mean line + per-seed markers)
  bottom: mean old-scene ΔATT vs rho (stability) with diagonal ΔATT
          (plasticity) as dashed overlay on a twin axis? -> keep simple:
          second row shows diagonal mean ΔATT vs rho.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts' / 'cl_hz1x1'

rho = json.loads((ART / 'matrix_report_soa_soar.json').read_text())
so = json.loads((ART / 'matrix_report_so_recomputed.json').read_text())
soa1 = json.loads((ART / 'matrix_report_soa_soa1.json').read_text())

RHOS = [0.125, 0.25, 0.5]
BACKBONES = {'dqn': 'soddqn', 'duel': 'soduel', 'frap': 'sofrap'}
SEEDS = [0, 1, 2]


def chain(rep, arm, seed):
    return rep.get(f'{arm}_o2_sd{seed}', {})


def arm_for(bb, cond, r):
    stem = BACKBONES[bb]
    tag = {0.125: '_r12', 0.5: '_r50'}.get(r, '')
    return f'{stem}{"p" if cond == "p" else ""}{tag}'


def rep_for(r):
    return rho if r != 0.25 else None


def stat(bb, cond, r, field):
    rep = rho if r != 0.25 else (soa1 if cond == 'p' else so)
    arm = arm_for(bb, cond, r)
    vals = []
    for sd in SEEDS:
        c = chain(rep, arm, sd)
        if field == 'worst':
            v = c.get('worst_old_degradation')
        elif field == 'mean_old':
            v = c.get('mean_old_delta')
        else:
            v = np.mean(c.get('diagonal_deltas', [])) \
                if c.get('diagonal_deltas') else None
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    return vals


fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
for c, bb in enumerate(['dqn', 'duel', 'frap']):
    for cond, color, lab in (('replay', '#1f77b4', 'replay'),
                             ('p', '#2ca02c', 'replay + policy anchor')):
        for row, field in ((0, 'worst'), (1, 'diag')):
            ax = axes[row][c]
            means, lows, highs = [], [], []
            for r in RHOS:
                v = stat(bb, cond, r, field)
                means.append(np.mean(v) if v else np.nan)
                ax.scatter([r] * len(v), v, c=color, s=14, alpha=0.5)
            ax.plot(RHOS, means, c=color, marker='o', lw=2, label=lab)
            ax.axhline(0, color='gray', lw=0.8)
            ax.set_xscale('log')
            ax.set_xticks(RHOS)
            ax.set_xticklabels(['0.125', '0.25', '0.5'])
    axes[0][c].set_title(f'{bb} — worst old-scene ΔATT vs ρ')
    axes[1][c].set_title(f'{bb} — diagonal ΔATT vs ρ (plasticity)')
    axes[0][c].set_ylabel('ΔATT (s)') if c == 0 else None
    axes[1][c].set_ylabel('ΔATT (s)') if c == 0 else None
    axes[1][c].set_xlabel('offline ratio ρ')
axes[0][0].legend(fontsize=8)
fig.suptitle('cl_hz1x1 o2 — replay-ratio dose-response: retention & plasticity')
fig.tight_layout()
fig.savefig(ART / 'fig_soa_rho.png', dpi=150)
print('fig ->', ART / 'fig_soa_rho.png')
