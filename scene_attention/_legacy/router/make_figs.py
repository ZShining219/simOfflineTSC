"""Render presentation figures from dev_takeover_probe eval artifacts.

Reads the obj3 evaluation directories (frozen checkpoints, held-out seeds)
and emits PNGs into ``output_data/dev_takeover_probe/figs/``:

    fig1_oracle_regret.png   paired-flip oracle regret, 3 variants
    fig2_p2_holdout.png      P2 holdout queue integral, all arms
    fig3_ablation.png        semantic ablation ladder (attn arm)
    fig4_binding.png         attention-to-annotation consistency
    fig5_training.png        training reward curves, 3 variants
    fig6_timeseries.png      queue-vs-time on the severe event cell
    fig7_attn_heatmap.png    attention heatmap, 3 report conditions
    fig8_queue_curves.png    total queue vs time, all arms, 4 cells
    fig9_throughput_curves.png  cumulative throughput vs time, 4 cells
    fig10_eventlane_curves.png  event-edge max lane queue vs time, surge cells
    fig11_controller_timeline.png  per-window controller choice, 3 router arms
    fig12_repr_probes.png      representation probes per held-out episode:
                               congestion-class acc + event-impact MAE,
                               3 encoders + raw-input baseline

Curves are seed-mean +- sample SD over the 3 held-out seeds; fig11 shows
seed s0 rows with mean switches annotated.

Usage:
    python -m scene_attention._legacy.router.make_figs --root output_data/dev_takeover_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS3 = ['attn', 'selfattn', 'concat']
ARM_LABEL = {'attn': 'Scene-guided', 'selfattn': 'Self-attn',
             'concat': 'Concat', 'fixed_dqn_s1': 'Fixed DQN',
             'fixed_mp': 'Fixed MP', 'attn_no_text': 'No text',
             'attn_wrong_loc': 'Wrong loc',
             'attn_mms': 'SGA-mms',
             'router_attn': 'Scene-guided',
             'router_selfattn': 'Self-attn', 'router_concat': 'Concat',
             'router_attn_mms': 'SGA-mms'}
COLORS = {'attn': '#d62728', 'selfattn': '#1f77b4', 'concat': '#7f7f7f',
          'fixed_dqn_s1': '#2ca02c', 'fixed_mp': '#9467bd',
          'attn_no_text': '#8c564b', 'attn_wrong_loc': '#e377c2',
          'attn_mms': '#ff7f0e',
          'router_attn': '#d62728', 'router_selfattn': '#1f77b4',
          'router_concat': '#7f7f7f', 'router_attn_mms': '#ff7f0e'}


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def load_obj(root):
    out = {}
    for v in ARMS3:
        p = root / f'eval_obj_{v}3' / 'obj_summary.json'
        if p.exists():
            out[v] = json.loads(p.read_text())
    return out


def oracle_regret(summary):
    """Mean (router - best fixed) over members of winner-flipped pairs."""
    members = summary['per_member']
    by_pair = {}
    for m in members:
        by_pair.setdefault((m['cell'], m['seed']), []).append(m)
    rows = []
    for ms in by_pair.values():
        if len(ms) != 2:
            continue
        if ms[0]['winner'] == ms[1]['winner']:
            continue
        multi = '+' in ms[0]['cell']
        for m in ms:
            base = min(m['scores']['fixed_dqn_s1'],
                       m['scores']['fixed_mp'])
            rows.append({'multi': multi,
                         'router': m['scores']['router'] - base,
                         'wrong': m['scores']['router_wrong_obj'] - base,
                         'notext': m['scores']['router_no_text'] - base})
    def agg(key):
        vals = [r[key] for r in rows]
        sv = [r[key] for r in rows if not r['multi']]
        mv = [r[key] for r in rows if r['multi']]
        return (float(np.mean(vals)) if vals else np.nan,
                float(np.mean(sv)) if sv else np.nan,
                float(np.mean(mv)) if mv else np.nan)
    return {'router': agg('router'), 'wrong': agg('wrong'),
            'notext': agg('notext'), 'n_rows': len(rows)}


def fig1(obj, out, plt):
    groups = ['router', 'wrong', 'notext']
    glabels = ['correct objective', 'wrong objective', 'no text']
    x = np.arange(3)
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for gi, g in enumerate(groups):
        vals = [oracle_regret(obj[v])[g] for v in ARMS3]
        for ci, v in enumerate(vals):
            ax.bar(x[ci] + (gi - 1) * w, v[0], w * 0.9,
                   color=plt.cm.RdYlGn_r(0.15 + 0.3 * gi), alpha=0.9)
            ax.text(x[ci] + (gi - 1) * w, v[0], f'{v[0]:.0f}',
                    ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(ARMS3):
        ax.text(x[i] - w, -18, '', fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1,
                             color=plt.cm.RdYlGn_r(0.15 + 0.3 * gi))
               for gi in range(3)]
    ax.legend(handles, glabels, fontsize=8, loc='upper left')
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[v] for v in ARMS3])
    ax.set_ylabel('oracle regret (queue, lower better)')
    ax.set_title('Paired objective-flip evaluation (held-out seeds 9100-9102)')
    fig.tight_layout()
    fig.savefig(out / 'fig1_oracle_regret.png', dpi=160)
    plt.close(fig)


def fig2(summary, out, plt):
    arms = ['fixed_dqn_s1', 'fixed_mp', 'router_attn',
            'router_selfattn', 'router_concat']
    events = []
    for row in summary:
        if row['event'] not in events:
            events.append(row['event'])
    table = {(r['event'], r['arm']): r['queue_integral'] for r in summary}
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(events))
    w = 0.16
    for ai, arm in enumerate(arms):
        vals = [table.get((e, arm), np.nan) for e in events]
        ax.bar(x + (ai - 2) * w, vals, w * 0.9,
               label=ARM_LABEL[arm], color=COLORS[arm])
    labels = []
    for e in events:
        if e == 'none':
            labels.append('no event')
        else:
            labels.append(e.split(':')[1].replace('road_', 'r'))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('queue integral (lower better)')
    ax.set_title('P2 held-out events (3 seeds each)')
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out / 'fig2_p2_holdout.png', dpi=160)
    plt.close(fig)


def fig3(summary, out, plt):
    arms = ['router_attn', 'attn_no_text', 'attn_wrong_loc']
    events = [e for e in
              dict.fromkeys(r['event'] for r in summary) if e != 'none']
    table = {(r['event'], r['arm']): r['queue_integral'] for r in summary}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(events))
    w = 0.26
    shades = {'router_attn': '#d62728', 'attn_no_text': '#8c564b',
              'attn_wrong_loc': '#e377c2'}
    for ai, arm in enumerate(arms):
        vals = [table.get((e, arm), np.nan) for e in events]
        ax.bar(x + (ai - 1) * w, vals, w * 0.9,
               label=ARM_LABEL[arm], color=shades[arm])
    ax.set_xticks(x)
    ax.set_xticklabels([e.split(':')[1].replace('road_', 'r')
                        for e in events])
    ax.set_ylabel('queue integral (lower better)')
    ax.set_title('Semantic ablation on scene-guided router')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'fig3_ablation.png', dpi=160)
    plt.close(fig)


TEST_EVENT_EDGES = ['road_0_1_0', 'road_1_0_1', 'road_2_1_2', None]

# P2 eval cell order in records/windows filenames: e0..e2 surge, e3 none
CELL_META = {
    'e0': ('surge r0_1_0\nonset 600s, dur 450s', 600, 450),
    'e1': ('surge r1_0_1\nonset 300s, dur 450s', 300, 450),
    'e2': ('surge r2_1_2\nonset 900s, dur 750s', 900, 750),
    'e3': ('no event', None, None),
}
CELL_ORDER = ['e3', 'e0', 'e1', 'e2']
SURGE_CELLS = ['e0', 'e1', 'e2']
P2_ARMS = ['fixed_dqn_s1', 'fixed_mp',
           'router_concat', 'router_selfattn', 'router_attn']
CTRL_COLORS = {'dqn_s1': '#2ca02c', 'maxpressure': '#9467bd',
               'fixedtime': '#7f7f7f'}


def _records_series(path, extract):
    """Load records_*.jsonl -> (t, y) on a regular 10 s grid."""
    if not path.exists():
        return None
    t, y = [], []
    for line in open(path):
        r = json.loads(line)
        t.append(float(r['t']))
        y.append(extract(r))
    return np.asarray(t), np.asarray(y, dtype=float)


def _seed_mean_sd(root, arm, cell, extract, grid):
    """Per-seed series interpolated onto grid -> mean, sd arrays."""
    series = []
    for s in ('s0', 's1', 's2'):
        p = root / 'eval_p2_obj3' / f'records_{arm}_{cell}_{s}.jsonl'
        ts = _records_series(p, extract)
        if ts is not None:
            series.append(np.interp(grid, ts[0], ts[1]))
    if not series:
        return None, None
    v = np.vstack(series)
    return v.mean(0), v.std(0, ddof=1) if len(series) > 1 else np.zeros(len(grid))


def _event_lane_max(records_path_edge):
    edge = records_path_edge
    return lambda r: max(
        (r['lane_queue'].get(l, 0) for l in r['lane_queue']
         if _lane_edge(l) == edge), default=0)


def _curve_panels(root, out, plt, extract_by_cell, ylabel, title, fname,
                  cells=CELL_ORDER, ncols=2):
    """Grid of cells; per arm seed-mean curve +- SD band."""
    grid = np.arange(0.0, 3600.1, 10.0)
    nrows = (len(cells) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, sharex=True,
                             figsize=(5.6 * ncols, 3.3 * nrows))
    flat = np.atleast_1d(axes).ravel()
    for ax in flat[len(cells):]:
        ax.set_visible(False)
    for ax, cell in zip(flat, cells):
        extract = extract_by_cell(cell)
        for arm in P2_ARMS:
            mean, sd = _seed_mean_sd(root, arm, cell, extract, grid)
            if mean is None:
                continue
            ax.plot(grid, mean, label=ARM_LABEL[arm], color=COLORS[arm],
                    lw=1.7)
            ax.fill_between(grid, mean - sd, mean + sd,
                            color=COLORS[arm], alpha=0.13, lw=0)
        label, onset, dur = CELL_META[cell]
        if onset is not None:
            for tt, c in [(onset, 'k'), (onset + 30, 'r'),
                          (onset + dur, 'k')]:
                ax.axvline(tt, ls='--', color=c, alpha=0.45, lw=1)
        ax.set_title(label, fontsize=9)
        ax.grid(alpha=0.2)
        ax.set_xlabel('sim time (s)')
    for i in range(0, len(cells), ncols):
        flat[i].set_ylabel(ylabel)
    flat[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig8(root, out, plt):
    """Total-queue trajectory: does the scene-guided router hold queues
    lower than blind variants and fixed arms on each held-out cell."""
    _curve_panels(
        root, out, plt,
        lambda cell: (lambda r: r['total_queue']),
        'total queue (veh)',
        'Total queue vs time — seed mean ± SD (3 held-out seeds); '
        'dashed: surge onset / report / end',
        'fig8_queue_curves.png')


def fig9(root, out, plt):
    """Cumulative throughput trajectory: same frozen runs, discharge-rate
    comparison across arms."""
    _curve_panels(
        root, out, plt,
        lambda cell: (lambda r: r['throughput']),
        'cumulative throughput (veh)',
        'Cumulative throughput vs time — seed mean ± SD '
        '(higher slope = faster discharge)',
        'fig9_throughput_curves.png')


def fig10(root, out, plt):
    """Event-edge worst-lane queue: localised congestion at the reported
    incident edge, surge cells only."""
    _curve_panels(
        root, out, plt,
        lambda cell: _event_lane_max(TEST_EVENT_EDGES[int(cell[1])]),
        'max queue on event edge (veh)',
        'Event-edge max lane queue vs time — surge cells only '
        '(dashed: onset / report / end)',
        'fig10_eventlane_curves.png', cells=SURGE_CELLS, ncols=3)


def fig11(root, out, plt):
    """Controller timeline: which controller each router arm picks per
    120 s window (seed s0), surge window shaded; mean switches/ep over
    3 seeds annotated at the right edge."""
    variants = ['router_concat', 'router_selfattn', 'router_attn']
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.6), sharey=True)
    for ax, cell in zip(axes, CELL_ORDER):
        label, onset, dur = CELL_META[cell]
        if onset is not None:
            ax.axvspan(onset, onset + dur, color='#d62728', alpha=0.08)
            ax.axvline(onset + 30, ls='--', color='r', alpha=0.5, lw=1)
        for yi, arm in enumerate(variants):
            p = (root / 'eval_p2_obj3' /
                 f'windows_{arm}_{cell}_s0.jsonl')
            if not p.exists():
                continue
            rows = [json.loads(l) for l in open(p)]
            ts = [float(r['t']) for r in rows] + [3600.0]
            y = len(variants) - 1 - yi
            for i, r in enumerate(rows):
                ax.barh(y, ts[i + 1] - ts[i], left=ts[i], height=0.62,
                        color=CTRL_COLORS.get(r['controller'], '#333333'),
                        edgecolor='none')
            nsw = []
            for s in ('s0', 's1', 's2'):
                ps = (root / 'eval_p2_obj3' /
                      f'windows_{arm}_{cell}_{s}.jsonl')
                if ps.exists():
                    nsw.append(sum(json.loads(l)['switched']
                                   for l in open(ps)))
            if nsw:
                ax.text(3620, y, f'{np.mean(nsw):.1f} sw',
                        va='center', fontsize=8, color='#333333')
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(
            [ARM_LABEL[v] for v in reversed(variants)], fontsize=8)
        ax.set_title(label.replace('\n', ' '), fontsize=9)
        ax.set_xlim(0, 3720)
        ax.set_xlabel('sim time (s)')
        ax.grid(axis='x', alpha=0.2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CTRL_COLORS[c], label=c)
               for c in ('dqn_s1', 'maxpressure')]
    axes[0].legend(handles=handles, fontsize=8, loc='lower left')
    fig.suptitle('Per-window controller choice (seed s0; shaded = surge '
                 'active, red dash = report time; right = mean switches '
                 'per episode over 3 seeds)', y=1.0)
    fig.tight_layout()
    fig.savefig(out / 'fig11_controller_timeline.png', dpi=160,
                bbox_inches='tight')
    plt.close(fig)


def fig12(root, out, plt):
    """Representation probes (probe_repr/metrics.json): per-held-out-
    episode paired lines — congestion-class accuracy and event-impact
    MAE of linear probes fit on each encoder's frozen z."""
    p = root / 'probe_repr' / 'metrics.json'
    if not p.exists():
        return
    m = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    arm_names = list(m['arms'])
    series = [(a, ARM_LABEL.get(a, a), COLORS.get(a, '#17becf'))
              for a in arm_names]
    raw = m.get('raw_baseline') or {}
    for ax, key, lab, yl in (
            (axes[0], 'congestion_clf', 'acc',
             'congestion-level probe acc (higher better)'),
            (axes[1], 'impact_reg', 'mae',
             'event-impact probe MAE (lower better)')):
        all_groups = []
        for a in arm_names:
            folds = m['arms'][a][key].get('folds') or []
            for f in folds:
                if f['group'] not in all_groups:
                    all_groups.append(f['group'])
        x = np.arange(len(all_groups))
        for a, lbl, c in series:
            vals = {f['group']: f[lab] if lab in f else f.get('acc')
                    for f in m['arms'][a][key].get('folds', [])}
            ys = [vals.get(g, np.nan) for g in all_groups]
            ax.plot(x, ys, 'o-', color=c, lw=1.6, ms=4,
                    label=lbl, alpha=0.9)
            ax.axhline(m['arms'][a][key][lab if lab in
                       m['arms'][a][key] else 'acc'],
                       color=c, ls=':', lw=1, alpha=0.5)
        rv = raw.get(key, {})
        rfolds = {f['group']: f.get(lab, f.get('acc'))
                  for f in rv.get('folds', [])}
        if rfolds:
            ax.plot(x, [rfolds.get(g, np.nan) for g in all_groups],
                    's--', color='#222222', lw=1.2, ms=3.5,
                    label='raw input', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(all_groups, rotation=45, ha='right',
                           fontsize=7)
        ax.set_ylabel(yl.split('(')[0].strip(), fontsize=8)
        ax.set_title(yl, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Frozen-representation probes, leave-one-episode-out '
                 '(each point = that episode held out; dotted = pooled '
                 'score)', y=1.02)
    fig.tight_layout()
    fig.savefig(out / 'fig12_repr_probes.png', dpi=160,
                bbox_inches='tight')
    plt.close(fig)


def _lane_edge(lane):
    return lane.rsplit('_', 1)[0]


def _reported_edge(arm, true_edge, wrong_edge):
    if arm == 'attn_no_text':
        return None
    if arm == 'attn_wrong_loc':
        return wrong_edge
    return true_edge


def binding_stats(root):
    """% of event-report windows where attn_top_lane lands on the true
    event edge (and, separately, on the reported edge)."""
    import glob as g
    stats = {}
    for arm in ['router_attn', 'attn_no_text', 'attn_wrong_loc',
                'router_selfattn', 'router_concat']:
        t_hit = r_hit = n = 0
        for fp in sorted(g.glob(str(root / 'eval_p2_obj3' /
                                    f'windows_{arm}_e*_s*.jsonl'))):
            ev_idx = int(fp.rsplit('_e', 1)[1].split('_')[0])
            true_edge = TEST_EVENT_EDGES[ev_idx]
            if true_edge is None:
                continue
            for line in open(fp):
                w = json.loads(line)
                top = w.get('attn_top_lane')
                if not top or '正常交通' in (w.get('text') or ''):
                    continue
                n += 1
                e = _lane_edge(top)
                if e == true_edge:
                    t_hit += 1
        stats[arm] = (t_hit / n * 100 if n else np.nan, n)
    # wrong-loc reported-edge follow rate
    fh = n = 0
    for fp in sorted(g.glob(str(root / 'eval_p2_obj3' /
                            'windows_attn_wrong_loc_e*_s*.jsonl'))):
        ev_idx = int(fp.rsplit('_e', 1)[1].split('_')[0])
        true_edge = TEST_EVENT_EDGES[ev_idx]
        if true_edge is None:
            continue
        recp = fp.replace('windows_attn_wrong_loc', 'records_fixed_mp')
        with open(recp) as f:
            first = json.loads(f.readline())
        edges = sorted({_lane_edge(l) for l in first['lane_queue']})
        wrong = next(e for e in edges if e != true_edge)
        for line in open(fp):
            w = json.loads(line)
            top = w.get('attn_top_lane')
            if not top or '正常交通' in (w.get('text') or ''):
                continue
            n += 1
            if _lane_edge(top) == wrong:
                fh += 1
    stats['wrong_follow'] = (fh / n * 100 if n else np.nan, n)
    return stats


def fig4(root, out, plt):
    stats = binding_stats(root)
    arms = ['router_attn', 'attn_wrong_loc', 'attn_no_text']
    vals = [stats[a][0] for a in arms]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(['correct report', 'wrong-loc report', 'no report'],
                  vals, color=['#d62728', '#e377c2', '#8c564b'])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1,
                f'{v:.0f}%' if v == v else 'n/a', ha='center', fontsize=10)
    ax.set_ylabel('% report windows: top-attn on true event lane')
    ax.set_ylim(0, 105)
    ax.set_title('Attention-to-annotation consistency (frozen, held-out)')
    fig.tight_layout()
    fig.savefig(out / 'fig4_binding.png', dpi=160)
    plt.close(fig)
    return stats


def fig5(root, out, plt):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for v in ARMS3:
        p = root / f'router_{v}_obj3' / 'train_log.jsonl'
        if not p.exists():
            continue
        eps, rew = [], []
        for line in open(p):
            r = json.loads(line)
            eps.append(r['episode'])
            rew.append(r['mean_window_reward'])
        k = 7
        sm = np.convolve(rew, np.ones(k) / k, 'same')
        ax.plot(eps, sm, label=ARM_LABEL[v], color=COLORS[v], lw=1.8)
    ax.set_xlabel('episode')
    ax.set_ylabel('mean window reward (7-ep smooth)')
    ax.set_title('SMDP training curves (identical seed schedule)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'fig5_training.png', dpi=160)
    plt.close(fig)


def fig6(root, out, plt):
    """Router tracks the better fixed arm per cell: e2 favours DQN
    (MP collapses), e1 favours MP (DQN loses).  Same router, no prior
    knowledge of which arm wins."""
    cells = [('e2', 's0'), ('e2', 's2'), ('e1', 's0'), ('e1', 's2')]
    onset = {'e1': 300, 'e2': 900}
    winner = {'e1': 'MP wins this cell', 'e2': 'DQN wins this cell'}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharey=True)
    for ax, (cell, seed) in zip(axes.ravel(), cells):
        for arm in ['fixed_dqn_s1', 'fixed_mp', 'router_attn']:
            p = root / 'eval_p2_obj3' / f'records_{arm}_{cell}_{seed}.jsonl'
            if not p.exists():
                continue
            t, q = [], []
            for line in open(p):
                r = json.loads(line)
                t.append(r['t'])
                q.append(r['total_queue'])
            ax.plot(t, q, label=ARM_LABEL[arm], color=COLORS[arm], lw=1.4)
        o = onset[cell]
        for tt, c in [(o, 'k'), (o + 30, 'r')]:
            ax.axvline(tt, ls='--', color=c, alpha=0.5, lw=1)
        ax.set_title(f'{cell} seed{seed[-1]} — {winner[cell]}',
                     fontsize=9)
        ax.set_xlabel('sim time (s)')
    for ax in axes[:, 0]:
        ax.set_ylabel('total queue')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Queue trajectory: router locks onto the cell winner',
                 y=1.0)
    fig.tight_layout()
    fig.savefig(out / 'fig6_timeseries.png', dpi=160,
                bbox_inches='tight')
    plt.close(fig)


def fig7(root, out, plt):
    """Attention heatmap: lanes x time under three report conditions."""
    trace_dir = root / 'figs' / 'attn_trace'
    panels = [('correct', 'correct report'),
              ('wrong_loc', 'wrong-location report'),
              ('no_text', 'no report')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (name, title) in zip(axes, panels):
        p = trace_dir / f'attn_trace_{name}.jsonl'
        if not p.exists():
            ax.set_visible(False)
            continue
        rows = [json.loads(l) for l in open(p)]
        rows = [r for r in rows if r.get('attn_weights')]
        lanes = rows[0]['lane_names']
        W = np.array([r['attn_weights'] for r in rows]).T
        t = [r['t'] for r in rows]
        ax.imshow(W, aspect='auto', origin='lower', cmap='hot',
                  extent=[t[0], t[-1], -0.5, len(lanes) - 0.5])
        true_edge = rows[0]['true_edge']
        rep_edge = rows[0]['reported_edge']
        edges = [_lane_edge(l) for l in lanes]
        for edge, color, lbl in [(true_edge, '#00ff88', 'true'),
                                 (rep_edge, '#00ccff', 'reported')]:
            if not edge:
                continue
            idxs = [i for i, e in enumerate(edges) if e == edge]
            for i in idxs:
                ax.axhline(i + 0.5, color=color, lw=0.8, alpha=0.7)
                ax.axhline(i - 0.5, color=color, lw=0.8, alpha=0.7)
        onset = 900.0
        ax.axvline(onset + 30, color='white', ls='--', lw=1,
                   alpha=0.7)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('sim time (s)')
    axes[0].set_yticks(range(len(rows[0]['lane_names'])))
    axes[0].set_yticklabels(
        [l.replace('road_', 'r') for l in rows[0]['lane_names']],
        fontsize=6)
    axes[0].set_ylabel('lane')
    fig.suptitle(
        'Scene-guided attention weights (green=true edge, '
        'blue=reported edge)', y=1.0)
    fig.tight_layout()
    fig.savefig(out / 'fig7_attn_heatmap.png', dpi=160,
                bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='output_data/dev_takeover_probe')
    args = ap.parse_args()
    root = Path(args.root)
    out = root / 'figs'
    out.mkdir(parents=True, exist_ok=True)
    plt = _mpl()
    obj = load_obj(root)
    fig1(obj, out, plt)
    summary = json.loads(
        (root / 'eval_p2_obj3' / 'eval_summary.json').read_text())
    fig2(summary, out, plt)
    fig3(summary, out, plt)
    stats = fig4(root, out, plt)
    fig5(root, out, plt)
    fig6(root, out, plt)
    fig7(root, out, plt)
    fig8(root, out, plt)
    fig9(root, out, plt)
    fig10(root, out, plt)
    fig11(root, out, plt)
    fig12(root, out, plt)
    print('wrote', out)
    print('binding stats:', json.dumps(stats))


if __name__ == '__main__':
    main()
