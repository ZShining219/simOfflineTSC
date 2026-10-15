"""Composite Q-anchor comparison panels for cl_hz1x1 (o2 + o3 + diagnostics).

Outputs (gitignored artifacts):
  fig_soa_cellmap.png   per-cell ΔATT heat-triangles (stage x scene),
                        rows = backbone, cols = {replay, +P, +P+V}, o2 seed-mean
  fig_soa_bars.png      worst old-scene degradation bars, o2 and o3 side by side
  fig_soa_diag.png      training diagnostics: anchor KL loss + td_loss per
                        episode-quartile, per backbone, stages 2-4 pooled

Data: matrix_report_so_recomputed.json (replay), matrix_report_soa_soa1.json
(o2 anchors), matrix_report_soa_soa3.json (o3 +P), records.jsonl per run dir.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts' / 'cl_hz1x1'
OUT = ROOT / 'data' / 'output_data' / 'tsc'

cl1 = json.loads((ART / 'matrix_report.json').read_text())
so = json.loads((ART / 'matrix_report_so_recomputed.json').read_text())
soa1 = json.loads((ART / 'matrix_report_soa_soa1.json').read_text())
soa3 = json.loads((ART / 'matrix_report_soa_soa3.json').read_text())
soav_path = ART / 'matrix_report_soa_soav.json'
soav = (json.loads(soav_path.read_text())
        if soav_path.exists() else {})

ORDERS = {'o2': ['s1', 'c4', 'c2', 'c3'], 'o3': ['c2', 'c3', 's1', 'c4']}
NETS = {'s1': 'sumohz1x1', 'c4': 'sumohz1x1_config4', 'c2': 'sumohz1x1_config2',
        'c3': 'sumohz1x1_config3'}
BACKBONES = {'dqn': 'ddqn', 'duel': 'duel', 'frap': 'frap', 'vdqn': 'dqn'}
ARMS = {
    'dqn':  {'replay': 'soddqn', 'p': 'soddqnp', 'pv': 'soddqnpv'},
    'duel': {'replay': 'soduel', 'p': 'soduelp', 'pv': 'soduelpv'},
    'frap': {'replay': 'sofrap', 'p': 'sofrapp', 'pv': 'sofrappv'},
    'vdqn': {'replay': 'sodqn', 'p': 'sodqnp'},
}
SEEDS = [0, 1, 2]
REP = {'replay': so, 'p': soa1, 'pv': soa1}
SOAV_ARMS = {'sodqn', 'sodqnp'}

baseline_att = {}
for key, vals in cl1['baselines'].items():
    arm, net, sd = key.rsplit('_', 2)
    baseline_att.setdefault((arm, net), []).append(vals['travel_time'])
baseline_att = {k: np.mean(v) for k, v in baseline_att.items()}


def cell_delta(cond, bb, order, seed, stage, col):
    """ΔATT of eval cell (stage row, scene col) vs baseline, or None."""
    arm = ARMS[bb].get(cond)
    if arm is None:
        return None
    rep = (soav if arm in SOAV_ARMS else
           soa3 if (order == 'o3' and cond in ('p', 'replay') and
                    f'{arm}_o3_sd{seed}' in soa3) else REP[cond])
    net = ORDERS[order][col]
    blk = rep.get(f'{arm}_{order}_sd{seed}', {})
    cell = blk.get(str(stage), {}).get(net)
    if cell is None:
        return None
    return cell['travel_time'] - baseline_att[(BACKBONES[bb], net)]


def seed_mean_cell(cond, bb, order, stage, col):
    vals = [cell_delta(cond, bb, order, sd, stage, col) for sd in SEEDS]
    vals = [v for v in vals if v is not None]
    return np.mean(vals) if vals else np.nan


# ================= Figure 1: per-cell ΔATT heat triangles =================
CONDS = [('replay', 'replay r=0.25'), ('p', '+policy anchor'),
         ('pv', '+policy+value')]
BB_ROWS = ['dqn', 'duel', 'frap', 'vdqn']
fig, axes = plt.subplots(len(BB_ROWS), 3, figsize=(13, 15))
vmax = 6.0
for r, bb in enumerate(BB_ROWS):
    for c, (cond, label) in enumerate(CONDS):
        ax = axes[r][c]
        M = np.full((4, 4), np.nan)
        for st in range(4):
            for col in range(4):
                M[st, col] = seed_mean_cell(cond, bb, 'o2', st + 1, col)
        mask = np.triu(np.ones_like(M, dtype=bool), k=0)  # upper incl diag -> mask
        # keep diagonal? show everything below diag as old-scene; mark diag grey
        show = np.where(mask, np.nan, M)
        im = ax.imshow(show, cmap='RdYlGn_r', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(4), ORDERS['o2'])
        ax.set_yticks(range(4), [f'st{i+1}' for i in range(4)])
        ax.set_xlabel('scene (o2 order)'); ax.set_ylabel('stage')
        if r == 0:
            ax.set_title(label, fontsize=11)
        if c == 0:
            ax.annotate(bb, xy=(-0.45, 0.5), xycoords='axes fraction',
                        fontsize=12, rotation=90, va='center')
        for st in range(4):
            for col in range(4):
                if not np.isnan(show[st, col]):
                    ax.text(col, st, f'{show[st, col]:+.1f}', ha='center',
                            va='center', fontsize=8,
                            color='k')
                elif st == col:
                    ax.text(col, st, '·', ha='center', va='center', color='gray')
fig.suptitle('cl_hz1x1 o2 — ΔATT per eval cell (seed mean, s; red=degraded, green=below baseline)')
fig.tight_layout(rect=[0, 0, 0.95, 0.96])
fig.colorbar(im, ax=axes, shrink=0.6, label='ΔATT (s)')
fig.savefig(ART / 'fig_soa_cellmap.png', dpi=150)
plt.close(fig)
print('cellmap done')


# ================= Figure 2: worst-degradation bars, o2 vs o3 ==============
def worst(rep, arm, order, seed):
    blk = rep.get(f'{arm}_{order}_sd{seed}', {})
    vals = []
    nets = ORDERS[order]
    for st in range(2, 5):
        row = blk.get(str(st), {})
        for col in range(st - 1):
            cell = row.get(nets[col])
            if cell and cell.get('travel_time') is not None:
                bb = [b for b, arms in ARMS.items() if arm in arms.values()][0]
                vals.append(cell['travel_time'] - baseline_att[(BACKBONES[bb], nets[col])])
    return max(vals) if vals else None


fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), sharey=True)
STYLE = {'replay': ('#1f77b4', 'replay'), 'p': ('#2ca02c', '+P'),
         'pv': ('#9467bd', '+P+V')}
for c, bb in enumerate(BB_ROWS):
    ax = axes[c]
    groups = [('o2', 'replay', so), ('o2', 'p', soa1), ('o2', 'pv', soa1),
              ('o3', 'replay', so), ('o3', 'p', soa3)]
    xpos = np.arange(len(groups))
    for i, (order, cond, rep) in enumerate(groups):
        arm = ARMS[bb].get(cond)
        if arm is None:
            continue
        rep = soav if arm in SOAV_ARMS else rep
        vals = [worst(rep, arm, order, sd) for sd in SEEDS]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        col, lab = STYLE[cond]
        ax.bar(i, np.mean(vals), width=0.6, color=col, alpha=0.8)
        ax.scatter([i] * len(vals), vals, c='k', s=16, zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f'{o}\n{c2}' for o, c2, _ in groups], fontsize=8)
    ax.set_title(bb)
    ax.axhline(0, color='gray', lw=0.8)
    if c == 0:
        ax.set_ylabel('worst old-scene ΔATT (s)')
handles = [plt.Rectangle((0, 0), 1, 1, color=STYLE[k][0]) for k in STYLE]
axes[0].legend(handles, [STYLE[k][1] for k in STYLE], fontsize=8)
fig.suptitle('cl_hz1x1 — worst old-scene degradation, orders o2 & o3')
fig.tight_layout()
fig.savefig(ART / 'fig_soa_bars.png', dpi=150)
plt.close(fig)
print('bars done')


# ================= Figure 3: training diagnostics ==========================
def run_dir(rid, order):
    adir = 'sumo_frap' if 'frap' in rid else 'sumo_dqn'
    m = re.search(r'_st(\d+)$', rid)
    net = NETS[ORDERS[order][int(m.group(1)) - 1]]
    return OUT / adir / net / rid


def diag_series(prefix, arm, order, stage, field):
    """Per-seed series of train_diagnostics[field] for a stage run."""
    out = []
    for sd in SEEDS:
        rid = f'{prefix}_{arm}_{order}_sd{sd}_st{stage}'
        f = run_dir(rid, order) / 'metrics' / 'records.jsonl'
        if not f.exists():
            continue
        vals = []
        for l in open(f):
            d = json.loads(l).get('train_diagnostics') or {}
            if d.get(field) is not None:
                vals.append(d[field])
        out.append(vals)
    return out


GRID = np.linspace(0, 100, 200)


def seed_mean(series_list):
    """Resample each seed series to a common 0-100 grid and average."""
    rs = []
    for s in series_list:
        if len(s) < 4:
            continue
        rs.append(np.interp(GRID, np.linspace(0, 100, len(s)), s))
    return np.mean(rs, axis=0) if rs else np.full_like(GRID, np.nan)


fig, axes = plt.subplots(3, 2, figsize=(13, 10))
for r, bb in enumerate(['dqn', 'duel', 'frap']):
    for st in (2, 3, 4):
        m = seed_mean(diag_series('soa1', ARMS[bb]['p'], 'o2', st,
                                  'anchor_policy_loss'))
        axes[r][0].plot(GRID, m, lw=1.6, label=f'st{st}')
    axes[r][0].set_title(f'{bb} — anchor policy KL (seed mean, st2-4)')
    axes[r][0].set_ylabel('KL')
    axes[r][0].legend(fontsize=8)
    for st in (2, 3, 4):
        m = seed_mean(diag_series('soa1', ARMS[bb]['p'], 'o2', st, 'td_loss'))
        axes[r][1].plot(GRID, m, lw=1.6, label=f'st{st}')
    axes[r][1].set_title(f'{bb} — td loss (+P arm, seed mean)')
    axes[r][1].legend(fontsize=8)
for ax in axes[2]:
    ax.set_xlabel('episode progress %')
fig.suptitle('cl_hz1x1 +P arms — anchor & TD diagnostics (o2)')
fig.tight_layout()
fig.savefig(ART / 'fig_soa_diag.png', dpi=150)
plt.close(fig)
print('diag done')
