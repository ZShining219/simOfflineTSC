"""Forgetting-side microscopic anatomy: old-scene s1 revisited.

Reads the forget_eval packages (run.py --evaluation-manifest output):
DECISION_METRICS every 10s -> queue_network_sum(t), throughput_cumulative(t),
per-lane queue heatmaps.  All controllers share eval seed 101 on sumohz1x1,
so differences are pure policy drift along each continual-learning chain.

Cells: worst-forgetting o2 cells -- frap sd0/sd1 replay vs +P anchor
(+9.79s peak), duel sd2 (+5.23s transient at st3).  Stage-end ckpts for
every stage plus intra-stage ckpts at the erosion stage (frap st4,
duel st3) show WHERE inside a stage the old-scene skill erodes.

Outputs (artifacts/cl_hz1x1/):
  fig_forget_queue.png       queue(t) per stage-end ckpt, per cell x arm
  fig_forget_throughput.png  same layout, cumulative throughput(t)
  fig_forget_laneheat.png    lane x time heat across the st4 erosion
                             (frap sd1 replay: st1 teacher, st4 e0/e50/e100)
  fig_forget_att.png         ATT-on-s1 vs chain position (stage ends +
                             intra-stage points) vs baseline
Run: python dev/cl_hz1x1/plot_forget_micro.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = ROOT / 'artifacts' / 'cl_hz1x1' / 'forget_eval'
PKGS = [PKG_ROOT / 'pkg_a', PKG_ROOT / 'pkg_b']
ART = ROOT / 'artifacts' / 'cl_hz1x1'
CELLS = [('frap', 0), ('frap', 1), ('duel', 2)]
CONDS = [('r', 'replay-only'), ('p', 'replay + P anchor')]
MID_E = {('frap', 4): (0, 50, 100), ('duel', 3): (0, 50, 100)}
CID_RE = re.compile(r'(frap|duel)(r|p)_sd(\d)_st(\d)_e(\d+)')
BASE_RE = re.compile(r'(frap|duel)_base_sd(\d)')
STAGE_COL = {1: '#1f77b4', 2: '#9467bd', 3: '#ff7f0e', 4: '#d62728'}


def load_records():
    series = defaultdict(list)
    for pkg in PKGS:
        path = pkg / 'records.jsonl'
        if not path.exists():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r.get('record_type') != 'DECISION_METRICS':
                continue
            series[r['controller_id']].append(r)
    for rows in series.values():
        rows.sort(key=lambda r: r['simulation_time_seconds'])
    return series


def load_summary():
    out = {}
    for pkg in PKGS:
        path = pkg / 'summary.csv'
        if not path.exists():
            continue
        with open(path, newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                out.setdefault(row['controller_id'], row)
    return out


def cid(bb, cond, sd, st, ep):
    return f'{bb}{cond}_sd{sd}_st{st}_e{ep}'


def base_cid(bb, sd):
    return f'{bb}_base_sd{sd}'


def fig_curves(series, key, ylabel, title, out_name):
    fig, axes = plt.subplots(len(CONDS), len(CELLS), figsize=(14, 7),
                             sharex=True, sharey=True)
    for r, (cond, clab) in enumerate(CONDS):
        for c, (bb, sd) in enumerate(CELLS):
            ax = axes[r][c]
            for st in (1, 2, 3, 4):
                for ep in MID_E.get((bb, st), (100,)):
                    rows = series.get(cid(bb, cond, sd, st, ep))
                    if not rows:
                        continue
                    x = [q['simulation_time_seconds'] for q in rows]
                    y = [q[key] for q in rows]
                    ls = {100: '-', 50: '--', 0: ':'}[ep]
                    lab = f'st{st}' + ('' if ep == 100 else f' e{ep}')
                    ax.plot(x, y, color=STAGE_COL[st], ls=ls,
                            lw=1.6 if ep == 100 else 1.0,
                            alpha=0.9 if ep == 100 else 0.6, label=lab)
            rows = series.get(base_cid(bb, sd))
            if rows:
                x = [q['simulation_time_seconds'] for q in rows]
                y = [q[key] for q in rows]
                ax.plot(x, y, 'k-', lw=1.4, alpha=0.5, label='base')
            ax.set_title(f'{bb} {clab} sd{sd} on s1', fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6, ncol=2, loc='upper left')
            if r == len(CONDS) - 1:
                ax.set_xlabel('simulation time (s)')
            if c == 0:
                ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ART / out_name, dpi=150)
    plt.close(fig)
    print('wrote', out_name)


def fig_laneheat(series):
    """frap sd1 replay: teacher (st1) vs st4 ep0/50/100 -- where does the
    old-scene skill erode inside stage 4?"""
    picks = [('st1 e100 (teacher)', 'frapr_sd1_st1_e100'),
             ('st4 e0 (enter c3)', 'frapr_sd1_st4_e0'),
             ('st4 e50', 'frapr_sd1_st4_e50'),
             ('st4 e100 (+9.79s)', 'frapr_sd1_st4_e100')]
    mats, lanes, t = [], None, None
    for _, key in picks:
        rows = series[key]
        if lanes is None:
            lanes = sorted(rows[0]['queue_lanes'])
            t = [r['simulation_time_seconds'] for r in rows]
        mats.append(np.array([[r['queue_lanes'].get(l, 0.0) for l in lanes]
                              for r in rows]))
    vmax = max(m.max() for m in mats)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (lab, _), mat in zip(axes, picks, mats):
        im = ax.imshow(mat.T, aspect='auto', origin='lower', cmap='magma',
                       vmin=0, vmax=vmax,
                       extent=[t[0], t[-1], -0.5, len(lanes) - 0.5])
        ax.set_yticks(range(len(lanes)))
        ax.set_yticklabels([l.replace('_', ' ') for l in lanes], fontsize=6)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel('simulation time (s)')
    fig.colorbar(im, ax=axes, label='vehicles queued', shrink=0.8)
    fig.suptitle('forget_eval — per-lane queue on identical s1 traffic '
                 '(es101): frap replay sd1 across stage-4 erosion')
    fig.tight_layout(rect=(0, 0, 0.95, 0.94))
    fig.savefig(ART / 'fig_forget_laneheat.png', dpi=150)
    plt.close(fig)
    print('wrote fig_forget_laneheat.png')


def fig_att(summary):
    """ATT on s1 vs chain position.  Stage-end points at integer x with
    intra-stage points at fractional offsets; baseline = dashed hline."""
    fig, axes = plt.subplots(1, len(CELLS), figsize=(14, 4.5), sharey=True)
    colmap = {'r': '#1f77b4', 'p': '#2ca02c'}
    for ax, (bb, sd) in zip(axes, CELLS):
        b = summary.get(base_cid(bb, sd))
        bval = float(b['travel_time']) if b else None
        for cond, clab in CONDS:
            xs, ys = [], []
            for st in (1, 2, 3, 4):
                eps = MID_E.get((bb, st), (100,))
                for ep in sorted(eps):
                    row = summary.get(cid(bb, cond, sd, st, ep))
                    if not row:
                        continue
                    x = (st - 1) * 100 + ep
                    xs.append(x)
                    ys.append(float(row['travel_time']))
            ends = [(x, y) for x, y in zip(xs, ys) if x % 100 == 0]
            mid = [(x, y) for x, y in zip(xs, ys) if x % 100 != 0]
            ax.plot([x for x, _ in ends], [y for _, y in ends], 'o-',
                    color=colmap[cond], lw=1.8, ms=5, label=clab)
            ax.plot([x for x, _ in mid], [y for _, y in mid], 'x',
                    color=colmap[cond], ms=6, alpha=0.8)
        if bval is not None:
            ax.axhline(bval, color='k', ls='--', lw=1, alpha=0.6,
                       label=f'base {bval:.1f}s')
        ax.set_title(f'{bb} sd{sd} — frozen eval on old scene s1')
        ax.set_xlabel('episodes since chain start')
        ax.set_xticks([100, 200, 300, 400])
        ax.set_xticklabels([f'end st{s}' for s in (1, 2, 3, 4)])
        ax.set_ylabel('eval ATT (s)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle('forget_eval — old-scene ATT along the o2 chain '
                 '(same es101 traffic; x marks = intra-stage ckpts)')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ART / 'fig_forget_att.png', dpi=150)
    plt.close(fig)
    print('wrote fig_forget_att.png')


def audit(summary):
    """Cross-check manifest ATT vs matrix_report stage values."""
    print(f'{"controller":24s} {"ATT":>7s} {"thr":>6s} {"unf":>4s}')
    for bb, sd in CELLS:
        b = summary.get(base_cid(bb, sd))
        if b:
            print(f'{base_cid(bb, sd):24s} {float(b["travel_time"]):7.2f} '
                  f'{b["throughput"]:>6s} {b["unfinished_vehicles"]:>4s}')
        for cond, _ in CONDS:
            for st in (1, 2, 3, 4):
                for ep in sorted(MID_E.get((bb, st), (100,))):
                    c = cid(bb, cond, sd, st, ep)
                    row = summary.get(c)
                    if row:
                        print(f'{c:24s} {float(row["travel_time"]):7.2f} '
                              f'{row["throughput"]:>6s} '
                              f'{row["unfinished_vehicles"]:>4s}')


def main():
    series = load_records()
    summary = load_summary()
    print(f'{len(series)} controller series, {len(summary)} summaries')
    audit(summary)
    fig_curves(series, 'queue_network_sum', 'network queue (veh)',
               'forget_eval — old-scene queue(t) along the o2 chain '
               '(s1, es101, ONLINE weights; dotted = intra-stage ckpts)',
               'fig_forget_queue.png')
    fig_curves(series, 'throughput_cumulative',
               'cumulative throughput (veh)',
               'forget_eval — old-scene throughput(t) along the o2 chain '
               '(s1, es101, ONLINE weights; dotted = intra-stage ckpts)',
               'fig_forget_throughput.png')
    fig_laneheat(series)
    fig_att(summary)


if __name__ == '__main__':
    main()
