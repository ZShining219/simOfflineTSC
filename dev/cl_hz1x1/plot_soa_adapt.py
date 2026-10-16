"""Adaptation-cost analysis for cl_hz1x1 scene transitions.

The stage-level retention story (worst old-scene dATT) is one axis; this
script quantifies the OTHER axis the records contain: what a transition costs
on the NEW scene during the first episodes after switching.

Conditions compared at every transition (per-episode TRAIN records +
episode-0 frozen EVALUATION):
  clear = cl1_*_clear_*     pure online, buffer cleared at switch
  semi  = so1_*             causal semi-offline replay r=0.25
  +P    = soa1/soa3/soav    semi-offline + policy-output anchor

Outputs:
  fig_soa_adapt_main.png   headline: o3 c3->s1 transition, 3 backbone panels,
                           thin per-seed curves + mean + ep0 frozen-eval marks
  fig_soa_adapt_all.png    every transition (o2 st2-4, o3 st2-4) x backbone,
                           first-30-episode curves, all seeds overlaid
  fig_soa_adapt_tradeoff.png  stability-plasticity scatter: x = cumulative
                           excess cost on the new scene (first 20 train
                           episodes vs common single-scene baseline), y =
                           worst old-scene dATT at that stage end
  adapt_metrics.json       per (backbone, cond, order, stage, seed): entry
                           gap, excess10/20, worst-3ep window, detrended sd,
                           rebound count/amplitude vs COMMON reference

Common reference rule: a scene's reference = that scene's single-scene
baseline final eval ATT for the SAME backbone and training seed
(matrix_report.json baselines) — never the arm's own final value.

Run: python dev/cl_hz1x1/plot_soa_adapt.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'output_data' / 'tsc'
ART = ROOT / 'artifacts' / 'cl_hz1x1'

NETS = {'s1': 'sumohz1x1', 'c4': 'sumohz1x1_config4',
        'c2': 'sumohz1x1_config2', 'c3': 'sumohz1x1_config3'}
NET_TAG = {v: k for k, v in NETS.items()}
ORDERS = {
    'o2': ['s1', 'c4', 'c2', 'c3'],
    'o3': ['c2', 'c3', 's1', 'c4'],
}
SEEDS = [0, 1, 2]
EVAL_SEEDS = [101, 102, 103]
MAX_EP = 100
TOL = 0.05  # rebound tolerance: 5% above the common reference

# backbone -> (agent_dir, clear prefix fmt, semi prefix fmt, +P prefix fmt)
BB = {
    'ddqn': ('sumo_dqn', 'cl1_ddqn_clear_{o}', 'so1_soddqn_{o}',
             '{o2}soa1_soddqnp_o2|{o3}soa3_soddqnp_o3'),
    'duel': ('sumo_dqn', 'cl1_duel_clear_{o}', 'so1_soduel_{o}',
             '{o2}soa1_soduelp_o2|{o3}soa3_soduelp_o3'),
    'frap': ('sumo_frap', 'cl1_frap_clear_{o}', 'so1_sofrap_{o}',
             '{o2}soa1_sofrapp_o2|{o3}soa3_sofrapp_o3'),
    'vdqn': ('sumo_dqn', None, 'soav_sodqn_{o}', '{o2}soav_sodqnp_o2'),
}
CONDS = [('clear', '#d62728', 'pure online'),
         ('semi', '#1f77b4', 'semi-offline r=0.25'),
         ('p', '#2ca02c', 'semi-offline + P anchor')]


def arm_prefix(bb, cond, order):
    """Run-id prefix for an arm, or None if the cell was never run."""
    _, cclr, csemi, cp = BB[bb]
    if cond == 'clear':
        return None if cclr is None else cclr.format(o=order)
    if cond == 'semi':
        return csemi.format(o=order)
    # cp format: '{o2}<o2 prefix>|{o3}<o3 prefix>'
    parts = dict(seg.split('}', 1) for seg in cp.strip('{}').split('|{'))
    return parts.get(order)


def stage_run(adir, net, prefix, sd, st):
    return OUT / adir / net / f'{prefix}_sd{sd}_st{st}'


def load_records(run_dir):
    """(episode->train ATT list per metric, ep0 eval dict)."""
    p = Path(run_dir) / 'metrics' / 'records.jsonl'
    if not p.exists():
        return None, None
    tr, ev0 = [], None
    for line in p.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r.get('record_type') == 'TRAIN':
            tr.append(r)
        elif r.get('record_type') == 'EVALUATION' and r.get('episode') == 0:
            ev0 = r
    tr.sort(key=lambda r: r['episode'])
    return tr, ev0


BASELINES = json.loads((ART / 'matrix_report.json').read_text())['baselines']
BB_BASE = {'ddqn': 'ddqn', 'duel': 'duel', 'frap': 'frap', 'vdqn': 'dqn'}


def ref_att(bb, tag, sd):
    b = BASELINES.get(f'{BB_BASE[bb]}_{tag}_sd{sd}') or {}
    return b.get('travel_time')


# ---------------- per-transition curve + metric extraction ----------------
def transition_series(bb, cond, order, st):
    """Per-seed train curves + ep0 evals for one transition.

    order stage index st (1-based) trains scene nets[st-1].
    Returns {seed: {'curve': np.array(1..MAX_EP), 'ep0': dict|None}}.
    """
    adir = BB[bb][0]
    prefix = arm_prefix(bb, cond, order)
    net = NETS[ORDERS[order][st - 1]]
    out = {}
    if prefix is None:
        return out
    for sd in SEEDS:
        tr, ev0 = load_records(stage_run(adir, net, prefix, sd, st))
        if tr is None:
            continue
        arr = np.full(MAX_EP + 1, np.nan)
        for r in tr:
            ep = int(r['episode'])
            if 0 <= ep <= MAX_EP and r.get('travel_time') is not None:
                arr[ep] = float(r['travel_time'])
        out[sd] = {'curve': arr, 'ep0': ev0}
    return out


def transition_metrics(bb, cond, order, st):
    """Metrics for one transition, computed per seed then averaged."""
    tag = ORDERS[order][st - 1]
    rows = {}
    for sd, pack in transition_series(bb, cond, order, st).items():
        curve, ref = pack['curve'], ref_att(bb, tag, sd)
        if ref is None:
            continue
        eps = np.arange(MAX_EP + 1)
        first20 = curve[1:21]
        finite = np.isfinite(first20)
        entry = None
        if pack['ep0'] is not None and pack['ep0'].get('travel_time'):
            entry = float(pack['ep0']['travel_time']) - ref
        excess = float(np.nansum(np.maximum(first20 - ref, 0.0)))
        win3 = np.full(18, np.nan)
        for i in range(18):
            if np.isfinite(first20[i:i + 3]).all():
                win3[i] = first20[i:i + 3].mean()
        trend = np.full_like(curve, np.nan, dtype=float)
        for i in range(len(curve)):
            lo = max(0, i - 9)
            chunk = curve[lo:i + 1]
            chunk = chunk[np.isfinite(chunk)]
            if len(chunk):
                trend[i] = chunk.mean()
        resid = curve[1:21] - trend[1:21]
        resid = resid[np.isfinite(resid)]
        crossed = np.isfinite(curve[1:]) & (curve[1:] <= ref * (1 + TOL))
        rebound_n, rebound_amp, never = 0, 0.0, False
        if crossed.any():
            first_ok = int(np.argmax(crossed)) + 1
            tail = curve[first_ok:]
            over = np.isfinite(tail) & (tail > ref * (1 + TOL))
            rebound_n = int(over.sum())
            rebound_amp = float(np.nanmax(tail[over]) - ref) if over.any() else 0.0
        else:
            never = True
        rows[sd] = {
            'scene': tag, 'ref_att': ref,
            'entry_gap': entry,
            'excess_cost_10': float(np.nansum(
                np.maximum(curve[1:11] - ref, 0.0))),
            'excess_cost_20': excess,
            'worst3_window': float(np.nanmax(win3) - ref)
                if np.isfinite(win3).any() else None,
            'detrended_sd_20': float(resid.std()) if resid.size else None,
            'rebound_count': rebound_n, 'rebound_amp': rebound_amp,
            'never_reached_ref': never,
        }
    return rows


def worst_old_datt(bb, cond, order, sd, st):
    """Worst old-scene dATT at end of stage st (matrix reports)."""
    if cond == 'clear':
        blk = (json.loads((ART / 'matrix_report.json').read_text())
               ['conditions'].get(f'{bb}_clear', {})
               .get(f'{order}_sd{sd}', {}).get('matrix', {}))
        mat = (blk or {}).get('travel_time')
        if not mat:
            return None
        vals = [mat[st - 1][j] - ref_att(bb, ORDERS[order][j], sd)
                for j in range(st - 1)
                if mat[st - 1][j] is not None]
        return max(vals) if vals else None
    if bb == 'vdqn':
        rep_name = 'matrix_report_soa_soav.json'
    elif cond == 'semi':
        rep_name = 'matrix_report_so_recomputed.json'
    elif cond == 'p':
        rep_name = {'o2': 'matrix_report_soa_soa1.json',
                    'o3': 'matrix_report_soa_soa3.json'}.get(order)
    else:
        rep_name = None
    rep_path = ART / rep_name if rep_name else None
    if rep_path is None or not rep_path.exists():
        return None
    rep = json.loads(rep_path.read_text())
    arm = arm_prefix(bb, cond, order)
    if arm is None:
        return None
    # 'soa1_soddqnp_o2' -> 'soddqnp' ; 'soav_sodqn_o2' -> 'sodqn'
    arm_id = arm.split('_', 1)[1].rsplit('_', 1)[0]
    blk = rep.get(f'{arm_id}_{order}_sd{sd}', {})
    row = blk.get(str(st), {})
    nets = ORDERS[order]
    vals = [row.get(nets[j], {}).get('travel_time') - ref_att(bb, nets[j], sd)
            for j in range(st - 1) if row.get(nets[j])]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


# ============================ figure 1: headline ============================
def fig_main():
    order, st = 'o3', 3  # c3 -> s1: the largest measured entry shock
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for c, bb in enumerate(['ddqn', 'duel', 'frap']):
        ax, ax2 = axes[c], axes[c].inset_axes([0.52, 0.42, 0.44, 0.44])
        for target, xmax in ((ax, MAX_EP), (ax2, 30)):
            for cond, col, lab in CONDS:
                ser = transition_series(bb, cond, order, st)
                if not ser:
                    continue
                curves = np.stack([p['curve'] for p in ser.values()])
                x = np.arange(MAX_EP + 1)
                target.plot(x, curves.T, color=col, lw=0.6, alpha=0.35)
                target.plot(x, np.nanmean(curves, axis=0), color=col,
                            lw=2.0, label=lab if target is ax else None)
                for sd, p in ser.items():
                    if p['ep0'] and p['ep0'].get('travel_time'):
                        target.plot(-4, p['ep0']['travel_time'], 'x',
                                    color=col, ms=6, alpha=0.9)
            target.set_xlim(-8, xmax)
            target.grid(alpha=0.3)
        ax2.set_title('zoom ep0-30', fontsize=7)
        ax2.tick_params(labelsize=7)
        ax.set_title(f'{bb} — o3 st3: c3→s1 adaptation', fontsize=10)
        ax.set_xlabel('episode in stage (x<0 = ep0 frozen eval)')
        if c == 0:
            ax.set_ylabel('train ATT on s1 (s)')
            ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('cl_hz1x1 — new-scene adaptation cost after switch '
                 '(thin = seeds, thick = mean; x-marks = ep0 frozen eval)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ART / 'fig_soa_adapt_main.png', dpi=150)
    plt.close(fig)
    print('wrote fig_soa_adapt_main.png')


# ==================== figure 2: all transitions supplementary =================
def fig_all():
    trans = [(o, st) for o in ('o2', 'o3') for st in (2, 3, 4)]
    fig, axes = plt.subplots(4, len(trans), figsize=(24, 12),
                             sharex=True)
    for r, bb in enumerate(BB):
        for c, (order, st) in enumerate(trans):
            ax = axes[r][c]
            tag = ORDERS[order][st - 1]
            for cond, col, lab in CONDS:
                ser = transition_series(bb, cond, order, st)
                if not ser:
                    continue
                curves = np.stack([p['curve'] for p in ser.values()])
                x = np.arange(31)
                ax.plot(x, curves[:, :31].T, color=col, lw=0.5, alpha=0.3)
                ax.plot(x, np.nanmean(curves[:, :31], axis=0), color=col,
                        lw=1.8)
            ax.set_title(f'{bb} {order} st{st} → {tag}', fontsize=8)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel('train ATT (s)', fontsize=8)
    handles = [plt.Line2D([0], [0], color=c, lw=2) for _, c, _ in CONDS]
    fig.legend(handles, [l for _, _, l in CONDS], loc='upper right',
               fontsize=9)
    fig.suptitle('cl_hz1x1 — first-30-episode adaptation at every transition '
                 '(thin = seeds, thick = mean)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.97, 0.96))
    fig.savefig(ART / 'fig_soa_adapt_all.png', dpi=150)
    plt.close(fig)
    print('wrote fig_soa_adapt_all.png')


# ====================== figure 3: stability-plasticity ======================
def fig_tradeoff(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    markers = {'ddqn': 'o', 'duel': 's', 'frap': '^', 'vdqn': 'D'}
    for c, order in enumerate(('o2', 'o3')):
        ax = axes[c]
        for bb in BB:
            for cond, col, lab in CONDS:
                pts = []
                for st in (2, 3, 4):
                    for sd in SEEDS:
                        m = metrics.get((bb, cond, order, st, sd))
                        y = worst_old_datt(bb, cond, order, sd, st)
                        if m and y is not None and m['excess_cost_20'] is not None:
                            pts.append((m['excess_cost_20'], y))
                if pts:
                    xs, ys = zip(*pts)
                    ax.scatter(xs, ys, marker=markers[bb], c=col, s=28,
                               alpha=0.7, edgecolors='none')
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_xlabel('new-scene excess cost, first 20 ep '
                      '(sum of max(0, ATT−baseline), s·ep)')
        ax.set_title(f'order {order}')
        if c == 0:
            ax.set_ylabel('worst old-scene ΔATT at stage end (s)')
    h1 = [plt.Line2D([0], [0], marker=m, ls='', c='gray', ms=8) for m in
          markers.values()]
    h2 = [plt.Line2D([0], [0], marker='o', ls='', c=c, ms=8)
          for _, c, _ in CONDS]
    axes[1].legend(h1 + h2, list(markers) + [l for _, _, l in CONDS],
                   fontsize=8, loc='upper right')
    fig.suptitle('cl_hz1x1 — stability–plasticity tradeoff per chain×stage '
                 '(each point = one transition in one seed)')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ART / 'fig_soa_adapt_tradeoff.png', dpi=150)
    plt.close(fig)
    print('wrote fig_soa_adapt_tradeoff.png')


def main():
    metrics = {}
    for bb in BB:
        for cond, _, _ in CONDS:
            for order in ORDERS:
                for st in (2, 3, 4):
                    for sd, m in transition_metrics(bb, cond, order,
                                                    st).items():
                        metrics[(bb, cond, order, st, sd)] = m
    out = ART / 'adapt_metrics.json'
    out.write_text(json.dumps(
        {f'{bb}|{cond}|{o}|st{st}|sd{sd}': m
         for (bb, cond, o, st, sd), m in metrics.items()},
        indent=1, ensure_ascii=False), encoding='utf-8')
    print('wrote', out)

    # console: headline transition o3 st3, all-seed view
    print('\n== o3 st3 (c3→s1) first-10ep train ATT, excess20, entry gap ==')
    print(f'{"bb":6s}{"cond":6s} {"TT10":>7s} {"excess20":>9s} '
          f'{"entry":>7s} {"per-seed TT10"}')
    for bb in ('ddqn', 'duel', 'frap'):
        for cond, _, _ in CONDS:
            rows = [transition_metrics(bb, cond, 'o3', 3)]
            ser = transition_series(bb, cond, 'o3', 3)
            if not ser:
                continue
            tt10 = [np.nanmean(p['curve'][1:11]) for p in ser.values()]
            xs = [metrics[(bb, cond, 'o3', 3, sd)]['excess_cost_20']
                  for sd in ser]
            en = [metrics[(bb, cond, 'o3', 3, sd)]['entry_gap']
                  for sd in ser if metrics[(bb, cond, 'o3', 3, sd)]
                  ['entry_gap'] is not None]
            print(f'{bb:6s}{cond:6s} {np.mean(tt10):7.2f} {np.mean(xs):9.1f} '
                  f'{np.mean(en) if en else float("nan"):7.1f}   '
                  f'{[round(v,1) for v in tt10]}')
    fig_main()
    fig_all()
    fig_tradeoff(metrics)


if __name__ == '__main__':
    sys.exit(main())
