"""Plot clear/fifo/semi-offline retention trajectories for the cl_hz1x1 runs.

Figure A (retention trajectory): for each backbone x order, the ATT on the
*first* scene of the order across stages 1-4.  Thin lines = individual
training seeds, thick line = seed mean, dashed = single-task baseline.

Figure B (dose-response): max off-diagonal degradation per chain
(st4 end value minus that cell's value right after learning) aggregated
over scenes, shown as per-seed bars grouped by condition.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts' / 'cl_hz1x1'

cl1 = json.loads((ART / 'matrix_report.json').read_text())
so = json.loads((ART / 'matrix_report_so.json').read_text())

ORDERS = {'o2': ['s1', 'c4', 'c2', 'c3'], 'o3': ['c2', 'c3', 's1', 'c4']}
BACKBONES = {
    'dqn':  {'clear': 'ddqn_clear', 'fifo': 'ddqn_fifo', 'so': 'soddqn'},
    'duel': {'clear': 'duel_clear', 'fifo': 'duel_fifo', 'so': 'soduel'},
    'frap': {'clear': 'frap_clear', 'fifo': 'frap_fifo', 'so': 'sofrap'},
}
COND_STYLE = {'clear': {'c': '#d62728', 'm': 'o', 'label': 'clear (online only)'},
              'fifo':  {'c': '#ff7f0e', 'm': 's', 'label': 'fifo (5000-window)'},
              'so':    {'c': '#1f77b4', 'm': '^', 'label': 'semi-offline 25% causal'}}
SEEDS = [0, 1, 2]

baseline_att = {}
for key, vals in cl1['baselines'].items():
    arm, net, seed = key.rsplit('_', 2)
    baseline_att.setdefault((arm, net), []).append(vals['travel_time'])
baseline_att = {k: sum(v) / len(v) for k, v in baseline_att.items()}
ARM_BASE = {'dqn': 'ddqn', 'duel': 'duel', 'frap': 'frap'}


def cell_att(cond, order, seed, stage, col):
    """ATT matrix cell (stage row, scene col) or None."""
    if cond in ('clear', 'fifo'):
        arm = BACKBONES[BB][cond]
        blk = cl1['conditions'].get(arm, {}).get(f'{order}_sd{seed}')
        if not blk or not blk.get('matrix'):
            return None
        return blk['matrix']['travel_time'][stage - 1][col]
    arm = BACKBONES[BB]['so']
    net = ORDERS[order][col]
    cell = so[f'{arm}_{order}_sd{seed}'].get(str(stage), {}).get(net)
    return None if cell is None else cell['travel_time']


# ---------- Figure A: first-scene retention trajectories ----------
fig, axes = plt.subplots(3, 2, figsize=(13, 13), sharex=True)
for r, BB in enumerate(['dqn', 'duel', 'frap']):
    for c, order in enumerate(['o2', 'o3']):
        ax = axes[r][c]
        first_net = ORDERS[order][0]
        base = baseline_att[(ARM_BASE[BB], first_net)]
        for cond in ['clear', 'fifo', 'so']:
            per_seed = []
            for sd in SEEDS:
                ys = [cell_att(cond, order, sd, st, 0) for st in range(1, 5)]
                per_seed.append(ys)
                ax.plot(range(1, 5), ys, color=COND_STYLE[cond]['c'],
                        alpha=0.28, lw=1.2, marker=COND_STYLE[cond]['m'],
                        ms=4)
            mean = np.nanmean(np.array(per_seed, dtype=float), axis=0)
            ax.plot(range(1, 5), mean, color=COND_STYLE[cond]['c'], lw=2.6,
                    marker=COND_STYLE[cond]['m'], ms=7,
                    label=COND_STYLE[cond]['label'])
        ax.axhline(base, ls='--', c='gray', lw=1.5, alpha=0.8,
                   label=f'{first_net} baseline ({base:.0f})')
        ax.set_title(f'{BACKBONES[BB]["so"]} · order {order} · scene {first_net} column',
                     fontsize=11)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels([f'st{i}' for i in range(1, 5)])
        ax.set_ylabel('eval ATT (s)' if c == 0 else '')
        ax.grid(alpha=0.3)
        if r == 0 and c == 0:
            ax.legend(fontsize=8.5, loc='upper left')
fig.suptitle('cl_hz1x1 forgetting trajectory: first-scene eval ATT across stages (thin=per-seed)', fontsize=14)
fig.tight_layout()
fig.savefig(ART / 'fig_so_retention_trajectory.png', dpi=150)
plt.close(fig)

# ---------- Figure B: worst off-diagonal degradation ----------
# per chain: max over stages/cells of (cell ATT - that cell's own learned level)
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
for i, BB in enumerate(['dqn', 'duel', 'frap']):
    ax = axes[i]
    width = 0.8 / 3
    for j, cond in enumerate(['clear', 'fifo', 'so']):
        per_seed_worst = []
        for sd in SEEDS:
            worst = 0.0
            for order in ['o2', 'o3']:
                cols = ORDERS[order]
                for col in range(4):
                    learned = cell_att(cond, order, sd, col + 1, col)
                    if learned is None:
                        continue
                    for st in range(col + 1, 5):
                        v = cell_att(cond, order, sd, st, col)
                        if v is not None:
                            worst = max(worst, v - learned)
            per_seed_worst.append(worst)
        xs = np.arange(3) + j * width
        ax.bar(xs, per_seed_worst, width=width * 0.9,
               color=COND_STYLE[cond]['c'], alpha=0.85,
               label=COND_STYLE[cond]['label'] if i == 0 else None)
        for x, y in zip(xs, per_seed_worst):
            ax.text(x, y + max(per_seed_worst) * 0.01, f'{y:.0f}',
                    ha='center', fontsize=8)
    ax.set_title(BACKBONES[BB]['so'])
    ax.set_xticks(np.arange(3) + width)
    ax.set_xticklabels(['sd0', 'sd1', 'sd2'])
    ax.set_ylabel('max ΔATT over learned level (s)' if i == 0 else '')
    ax.grid(alpha=0.3, axis='y')
axes[0].legend(fontsize=9, loc='upper left')
fig.suptitle('cl_hz1x1 deepest off-diagonal degradation per chain (max over orders, per-seed)', fontsize=13)
fig.tight_layout()
fig.savefig(ART / 'fig_so_worst_degradation.png', dpi=150)
plt.close(fig)
print('wrote fig_so_retention_trajectory.png + fig_so_worst_degradation.png')
