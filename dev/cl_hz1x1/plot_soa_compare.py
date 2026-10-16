"""Plot replay vs Q-anchor (policy / policy+value) retention for cl_hz1x1 o2.

Companion to plot_so_compare.py — consumes the same-schema harvest reports:
  matrix_report_so_recomputed.json  (plain replay arms, recomputed w/ soa harvester)
  matrix_report_soa_soa1.json       (anchor arms: policy-only and policy+value)

Figure A (retention trajectory): ATT on the order's first scene (o2: s1)
across stages 1-4; thin = per-seed, thick = seed mean, dashed = baseline.

Figure B (worst old-scene degradation): max off-diagonal ΔATT per chain
(per-seed bars grouped by condition within each backbone).
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
so = json.loads((ART / 'matrix_report_so_recomputed.json').read_text())
soa = json.loads((ART / 'matrix_report_soa_soa1.json').read_text())
soav_path = ART / 'matrix_report_soa_soav.json'
soav = (json.loads(soav_path.read_text())
        if soav_path.exists() else {})

ORDER = 'o2'
NETS = ['s1', 'c4', 'c2', 'c3']
BACKBONES = {
    'ddqn': {'replay': 'soddqn',  'p': 'soddqnp',  'pv': 'soddqnpv'},
    'duel': {'replay': 'soduel',  'p': 'soduelp',  'pv': 'soduelpv'},
    'frap': {'replay': 'sofrap',  'p': 'sofrapp',  'pv': 'sofrappv'},
    'vdqn': {'replay': 'sodqn',   'p': 'sodqnp'},
}
COND_STYLE = {
    'replay': {'c': '#1f77b4', 'm': 'o', 'label': 'replay only (r=0.25)'},
    'p':      {'c': '#2ca02c', 'm': '^', 'label': 'replay + policy anchor (λπ=1)'},
    'pv':     {'c': '#9467bd', 'm': 's', 'label': 'replay + policy + value anchor'},
}
SEEDS = [0, 1, 2]
ARM_BASE = {'ddqn': 'ddqn', 'duel': 'duel', 'frap': 'frap', 'vdqn': 'dqn'}

baseline_att = {}
for key, vals in cl1['baselines'].items():
    arm, net, seed = key.rsplit('_', 2)
    baseline_att.setdefault((arm, net), []).append(vals['travel_time'])
baseline_att = {k: sum(v) / len(v) for k, v in baseline_att.items()}


SOA_ARMS = {'soddqnp', 'soddqnpv', 'soduelp', 'soduelpv', 'sofrapp', 'sofrappv'}
SOAV_ARMS = {'sodqn', 'sodqnp'}


def report(arm):
    if arm in SOA_ARMS:
        return soa
    if arm in SOAV_ARMS:
        return soav
    return so


def cell_att(arm, seed, stage, col):
    blk = report(arm).get(f'{arm}_{ORDER}_sd{seed}', {})
    cell = blk.get(str(stage), {}).get(NETS[col])
    return None if cell is None else cell['travel_time']


def chain_worst(arm, seed):
    blk = report(arm).get(f'{arm}_{ORDER}_sd{seed}', {})
    vals = []
    for st in range(2, 5):
        row = blk.get(str(st), {})
        for col in range(st - 1):
            net = NETS[col]
            if net in row and row[net].get('travel_time') is not None:
                vals.append(row[net]['travel_time'] - baseline_att[(ARM_BASE[BB], net)])
    return max(vals) if vals else None


# ---------- Figure A: first-scene (s1) retention trajectories ----------
fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), sharex=True)
for c, BB in enumerate(BACKBONES):
    ax = axes[c]
    base = baseline_att[(ARM_BASE[BB], 's1')]
    ax.axhline(base, ls='--', c='gray', lw=1, label='single-task baseline')
    for cond, arm in BACKBONES[BB].items():
        st_ = COND_STYLE[cond]
        xs = range(1, 5)
        mat = np.array([[cell_att(arm, sd, st, 0) or np.nan for st in xs] for sd in SEEDS])
        for sd_i, sd in enumerate(SEEDS):
            ax.plot(xs, mat[sd_i], c=st_['c'], marker=st_['m'], ms=4, lw=0.8, alpha=0.35)
        ax.plot(xs, np.nanmean(mat, axis=0), c=st_['c'], marker=st_['m'], lw=2.2,
                label=st_['label'])
    ax.set_title(f'{BB} — scene s1 retention (o2)')
    ax.set_xlabel('stage'); ax.set_xticks([1, 2, 3, 4])
    if c == 0:
        ax.set_ylabel('ATT on s1 (s)')
    ax.legend(fontsize=7, loc='upper left')
fig.suptitle('cl_hz1x1 Q-anchor: first-scene retention across stages')
fig.tight_layout()
fig.savefig(ART / 'fig_soa_retention_trajectory.png', dpi=150)
plt.close(fig)

# ---------- Figure B: worst old-scene degradation ----------
fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), sharey=True)
for c, BB in enumerate(BACKBONES):
    ax = axes[c]
    conds = list(BACKBONES[BB])
    x = np.arange(len(conds))
    for i, cond in enumerate(conds):
        arm = BACKBONES[BB][cond]
        vals = [chain_worst(arm, sd) for sd in SEEDS]
        vals = [v for v in vals if v is not None]
        ax.bar(i, np.mean(vals), width=0.55, color=COND_STYLE[cond]['c'], alpha=0.75,
               label=COND_STYLE[cond]['label'])
        ax.scatter([i] * len(vals), vals, c='k', s=18, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, fontsize=9)
    ax.set_title(f'{BB} — worst old-scene ΔATT')
    ax.set_xlabel('condition')
    if c == 0:
        ax.set_ylabel('max degradation vs baseline (s)')
axes[0].legend(fontsize=7)
fig.suptitle('cl_hz1x1 Q-anchor (o2, n=3 seeds): stability gain at ~zero plasticity cost')
fig.tight_layout()
fig.savefig(ART / 'fig_soa_worst_degradation.png', dpi=150)
plt.close(fig)
print('figs ->', ART / 'fig_soa_retention_trajectory.png', ART / 'fig_soa_worst_degradation.png')
