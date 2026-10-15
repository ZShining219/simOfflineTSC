"""Within-episode microscopic anatomy of the o3 st3 (c3->s1) entry shock.

Reads the snap3o_eval package (run.py --evaluation-manifest output):
DECISION_METRICS rows every 10s of simulation -> queue_network_sum(t),
throughput_cumulative(t), per-lane queue heatmaps.  All controllers share
the same evaluation seed -> identical vehicle realizations, so differences
are pure policy differences between checkpoint episodes.

All checkpoints here hold ONLINE weights (resumable payload), unlike the
model/{e}_0.pt target-net snapshots used by the earlier snap3 probe.

Outputs (artifacts/cl_hz1x1/):
  fig_snap3o_queue.png     queue_network_sum(t) per ckpt episode, one panel
                           per (backbone x condition) on the shock seed
  fig_snap3o_throughput.png  same layout, cumulative throughput(t)
  fig_snap3o_laneheat.png  lane x time queue heatmap, worst vs best ckpt
  fig_snap3o_conv.png      online-fidelity ATT-vs-checkpoint-episode curve
                           (summary.csv), replotted next to snap3 target-net
Run: python dev/cl_hz1x1/plot_snap3o_micro.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / 'artifacts' / 'cl_hz1x1' / 'snap3o_eval' / 'package'
ART = ROOT / 'artifacts' / 'cl_hz1x1'
EPS = (0, 5, 10, 15, 20, 50, 100)
CONDS = [('clear', 'pure online'), ('semi', 'semi-offline r=0.25'),
         ('p', 'semi-offline + P anchor')]
BBS = ('ddqn', 'duel')
# per-backbone shock seed: the seed whose ep0 frozen eval was catastrophic
SHOCK_SEED = {'ddqn': 2, 'duel': 0}
EVAL_SEED = 101


def load_records():
    """{(controller_id): sorted list of DECISION_METRICS rows}."""
    path = PKG / 'records.jsonl'
    series = defaultdict(list)
    for line in path.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r.get('record_type') != 'DECISION_METRICS':
            continue
        series[r['controller_id']].append(r)
    for rows in series.values():
        rows.sort(key=lambda r: r['simulation_time_seconds'])
    return series


def load_summary():
    """{(controller_id, evaluation_seed): summary row dict}."""
    path = PKG / 'summary.csv'
    out = {}
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            out[row['controller_id']] = row
    return out


def ep_of(controller_id):
    return int(controller_id.rsplit('_e', 1)[1])


def cmap_for(cond):
    return {'clear': 'Reds', 'semi': 'Blues', 'p': 'Greens'}[cond]


def series_xy(rows, key):
    return ([r['simulation_time_seconds'] for r in rows],
            [r[key] for r in rows])


def fig_curves(series, key, ylabel, title, out_name):
    fig, axes = plt.subplots(len(CONDS), len(BBS), figsize=(13, 9),
                             sharex=True)
    for r, (cond, clab) in enumerate(CONDS):
        for c, bb in enumerate(BBS):
            ax = axes[r][c]
            sd = SHOCK_SEED[bb]
            cmap = plt.get_cmap(cmap_for(cond))
            for i, ep in enumerate(EPS):
                cid = f'{bb}_{cond}_sd{sd}_e{ep}'
                rows = series.get(cid)
                if not rows:
                    continue
                x, y = series_xy(rows, key)
                col = cmap(0.35 + 0.6 * i / (len(EPS) - 1))
                ax.plot(x, y, color=col, lw=1.6 if ep in (0, 100) else 0.9,
                        alpha=0.95 if ep in (0, 100) else 0.75,
                        label=f'ep{ep}')
            ax.set_title(f'{bb} {cond} sd{sd} — {clab}', fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6, ncol=2, loc='upper right')
            if r == len(CONDS) - 1:
                ax.set_xlabel('simulation time (s)')
            if c == 0:
                ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ART / out_name, dpi=150)
    plt.close(fig)
    print('wrote', out_name)


def fig_laneheat(series):
    """Lane x time queue heatmap: worst entry (ep0) vs recovered (ep100)."""
    cells = [('ddqn', 'clear', SHOCK_SEED['ddqn'], 0),
             ('ddqn', 'clear', SHOCK_SEED['ddqn'], 100)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, (bb, cond, sd, ep) in zip(axes, cells):
        rows = series.get(f'{bb}_{cond}_sd{sd}_e{ep}')
        lanes = sorted(rows[0]['queue_lanes'])
        mat = np.array([[r['queue_lanes'].get(l, 0.0) for l in lanes]
                        for r in rows])
        t = [r['simulation_time_seconds'] for r in rows]
        im = ax.imshow(mat.T, aspect='auto', origin='lower', cmap='magma',
                       extent=[t[0], t[-1], -0.5, len(lanes) - 0.5])
        ax.set_yticks(range(len(lanes)))
        ax.set_yticklabels([l.replace('north_in_', 'N_').replace('_', ' ')
                            for l in lanes], fontsize=6)
        ax.set_title(f'{bb} {cond} sd{sd} ep{ep} — queue per lane', fontsize=9)
        ax.set_xlabel('simulation time (s)')
        fig.colorbar(im, ax=ax, label='vehicles queued')
    fig.suptitle('snap3o — per-lane queue on identical traffic (s1, es101)')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ART / 'fig_snap3o_laneheat.png', dpi=150)
    plt.close(fig)
    print('wrote fig_snap3o_laneheat.png')


def fig_convergence(summary):
    """Online-fidelity ATT vs ckpt episode vs the snap3 target-net curve."""
    snap3 = json.loads((ART / 'snap3_metrics.json').read_text())
    fig, axes = plt.subplots(1, len(BBS), figsize=(12, 4.5), sharey=True)
    colmap = {'clear': '#d62728', 'semi': '#1f77b4', 'p': '#2ca02c'}
    for ax, bb in zip(axes, BBS):
        for cond, col, _ in [(c, m, l) for (c, m, l) in CONDS]:
            xs, ys = [], []
            for sd in (0, 1, 2):
                for ep in EPS:
                    row = summary.get(f'{bb}_{cond}_sd{sd}_e{ep}')
                    if row:
                        xs.append(ep)
                        ys.append(float(row['travel_time']))
                        ax.plot(ep, float(row['travel_time']), '.',
                                color=col, ms=4, alpha=0.3)
            # mean over seeds per ep
            mx = [np.mean([float(summary[f'{bb}_{cond}_sd{sd}_e{ep}']
                                   ['travel_time'])
                           for sd in (0, 1, 2)
                           if f'{bb}_{cond}_sd{sd}_e{ep}' in summary])
                  for ep in EPS]
            ax.plot(EPS, mx, 'o-', color=col, lw=2, ms=4,
                    label=f'{cond} (online)')
            # snap3 target-net comparison for the same bb/cond
            tgt_eps, tgt_vals = [], []
            for sd in (0, 1, 2):
                key = f'{bb}|{cond}|sd{sd}'
                if key in snap3:
                    for e, vals in snap3[key]['snapshot_eval'].items():
                        tgt_eps.append(int(e))
                        tgt_vals.append(np.mean(vals))
            tg = {e: [] for e in sorted(set(tgt_eps))}
            for e, v in zip(tgt_eps, tgt_vals):
                tg[e].append(v)
            ax.plot(sorted(tg), [np.mean(tg[e]) for e in sorted(tg)],
                    'x--', color=col, lw=1, ms=5, alpha=0.7,
                    label=f'{cond} (target snap)')
        ax.set_title(f'{bb} — o3 st3→s1 frozen eval by ckpt episode')
        ax.set_xlabel('checkpoint episode')
        ax.set_ylabel('eval ATT (s)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle('snap3o online-net vs snap3 target-net eval conventions '
                 '(dots=per-seed online, line=seed mean)')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ART / 'fig_snap3o_conv.png', dpi=150)
    plt.close(fig)
    print('wrote fig_snap3o_conv.png')


def main():
    series = load_records()
    summary = load_summary()
    print(f'{len(series)} controller series, {len(summary)} summaries')
    fig_curves(series, 'queue_network_sum', 'network queue (veh)',
               'snap3o — network queue(t) per checkpoint on identical '
               'traffic (o3 st3→s1, es101, ONLINE weights)',
               'fig_snap3o_queue.png')
    fig_curves(series, 'throughput_cumulative', 'cumulative throughput (veh)',
               'snap3o — cumulative throughput(t) per checkpoint on '
               'identical traffic (o3 st3→s1, es101, ONLINE weights)',
               'fig_snap3o_throughput.png')
    fig_laneheat(series)
    fig_convergence(summary)


if __name__ == '__main__':
    main()
