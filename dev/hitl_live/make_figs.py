"""Quantitative-evidence figures for the paired HITL probe.

探索件（dev/hitl_live/，登记 dev/INDEX.md）。读取
``paired_probe.py`` 的产物目录（runs.json + probe_manifest.json +
sessions/*/interventions.jsonl + transitions.jsonl），产出
"人在回路有效性"的四张佐证对比图 + 机读汇总：

    figs/fig1_queue_trajectory.png
        每臂排队轨迹（逐决策点，seed 均值±min/max 带）；hitl 臂上叠
        gate 查询事件（▲ 接管 / ○ 同判）；inject 事件窗阴影。
    figs/fig2_terminal_metrics.png
        终值三指标（travel_time / throughput / unfinished）：
        臂均值柱 + 逐 seed 配对散点（student↔hitl 连线）。
    figs/fig3_event_window.png
        事件窗局部放大：排队曲线 + 峰值标注 + 回落时间标注
        （主张：HITL 价值集中在扰动段）。
    figs/fig4_cost_effect.png
        成本-效果账本：hitl 臂逐 seed 查询数/接管数/同判数 vs
        相对 student 臂的 Δtravel_time/Δunfinished。

    figs/evidence_summary.json  全部图的机读数据（臂均值、配对差、
        账本计数）——复核与报告共用同一份数字。

用法：

    python dev/hitl_live/make_figs.py --run-dir output_data/hitl_probe/run1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ARM_LABEL = {
    'student': 'student alone (DQN-S2)',
    'hitl': 'student + gated MP expert (HITL)',
    'expert': 'expert alone (MaxPressure)',
}
ARM_COLOR = {'student': '#888888', 'hitl': '#c23b22', 'expert': '#1f6fb4'}


def _load_runs(run_dir):
    run_dir = Path(run_dir)
    runs = json.loads((run_dir / 'runs.json').read_text(encoding='utf-8'))
    manifest = json.loads(
        (run_dir / 'probe_manifest.json').read_text(encoding='utf-8'))
    return run_dir, runs, manifest


def _series(run):
    """(t, queue, throughput) arrays for one run's recorded series."""
    t, queue, thr = [], [], []
    for row in sorted(run.get('series') or [],
                      key=lambda r: (r.get('decision_step') or 0)):
        if row.get('t') is None or row.get('queue') is None:
            continue
        t.append(float(row['t']))
        queue.append(float(row['queue']))
        thr.append(float(row.get('throughput') or 0.0))
    return np.asarray(t), np.asarray(queue), np.asarray(thr)


def _event_window(manifest):
    for spec in manifest.get('injects') or []:
        kind, *rest = spec.split(':')
        if kind == 'surge' and len(rest) >= 4:
            start = float(rest[2])
            return start, start + float(rest[3]), spec
        if kind == 'throttle' and len(rest) >= 3:
            start = float(rest[1])
            return start, start + float(rest[2]), spec
        if kind == 'lane_close' and len(rest) >= 3:
            start = float(rest[1])
            return start, start + float(rest[2]), spec
    return None


def _queries(run_dir, run):
    """Gate query times + intervened flags for one (hitl) run."""
    path = run_dir / 'sessions' / run['session_id'] / 'interventions.jsonl'
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('kind') != 'gate_query':
            continue
        detail = row.get('detail') or {}
        out.append({
            't': float(row.get('t') or 0.0),
            'intervened': bool(detail.get('intervened')),
            'reason': detail.get('reason'),
        })
    return out


def fig1_trajectory(run_dir, runs, manifest, figs):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    event = _event_window(manifest)
    if event:
        ax.axvspan(event[0], event[1], color='#f4c542', alpha=0.18,
                   label='injected event window')
    for arm in sorted({r['arm'] for r in runs}):
        series = [_series(r) for r in runs if r['arm'] == arm]
        series = [(t, q) for t, q, _ in series if len(t)]
        if not series:
            continue
        # align on union grid via nearest step (decision cadence is fixed)
        grid = np.unique(np.concatenate([t for t, _ in series]))
        qs = []
        for t, q in series:
            qs.append(np.interp(grid, t, q))
        mean = np.mean(qs, axis=0)
        lo = np.min(qs, axis=0)
        hi = np.max(qs, axis=0)
        ax.plot(grid, mean, color=ARM_COLOR[arm], lw=1.8,
                label=ARM_LABEL.get(arm, arm))
        if len(series) > 1:
            ax.fill_between(grid, lo, hi, color=ARM_COLOR[arm], alpha=0.15)
        if arm == 'hitl':
            for r in [x for x in runs if x['arm'] == 'hitl']:
                for q in _queries(run_dir, r):
                    ax.axvline(q['t'], color=ARM_COLOR['hitl'],
                               alpha=0.35 if q['intervened'] else 0.15,
                               ls='-' if q['intervened'] else ':',
                               lw=1.0)
    ax.set_xlabel('simulation time (s)')
    ax.set_ylabel('incoming-lane queued vehicles')
    ax.set_title('Queue trajectory: HITL arm vs student vs expert '
                 '(vlines = gate queries: solid takeover / dotted agree)')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / 'fig1_queue_trajectory.png', dpi=150)
    plt.close(fig)


def fig2_terminal(runs, figs):
    metrics = [('travel_time', 'mean travel time (s)'),
               ('throughput', 'arrived vehicles (cumulative)'),
               ('unfinished', 'unfinished vehicles at horizon')]
    arms = sorted({r['arm'] for r in runs})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (key, label) in zip(axes, metrics):
        means = []
        for i, arm in enumerate(arms):
            vals = [r.get(key) for r in runs
                    if r['arm'] == arm and r.get(key) is not None]
            means.append(float(np.mean(vals)) if vals else 0.0)
            ax.scatter([i] * len(vals), vals, color=ARM_COLOR[arm],
                       zorder=3, s=28)
        ax.bar(range(len(arms)), means,
               color=[ARM_COLOR[a] for a in arms], alpha=0.55)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([ARM_LABEL.get(a, a) for a in arms],
                         fontsize=7, rotation=12)
        ax.set_title(label)
        ax.grid(axis='y', alpha=0.3)
        # paired deltas annotation student vs hitl
        if 'student' in arms and 'hitl' in arms:
            pairs = []
            for seed in {r['seed'] for r in runs}:
                s = next((r for r in runs
                          if r['arm'] == 'student' and r['seed'] == seed),
                         None)
                h = next((r for r in runs
                          if r['arm'] == 'hitl' and r['seed'] == seed),
                         None)
                if s and h and s.get(key) is not None \
                        and h.get(key) is not None:
                    pairs.append(float(s[key]) - float(h[key]))
            if pairs:
                ax.text(0.02, 0.95,
                        f'Δ(student−hitl)={np.mean(pairs):+.1f}',
                        transform=ax.transAxes, fontsize=8,
                        va='top')
    fig.suptitle('Terminal metrics (paired by seed)', fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / 'fig2_terminal_metrics.png', dpi=150)
    plt.close(fig)


def fig3_event_window(run_dir, runs, manifest, figs):
    event = _event_window(manifest)
    if not event:
        return False
    start, end, spec = event
    span = end - start
    lo_t = max(0.0, start - 0.35 * span)
    hi_t = end + 0.65 * span
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axvspan(start, end, color='#f4c542', alpha=0.2,
               label='event window')
    stats = {}
    for arm in sorted({r['arm'] for r in runs}):
        for r in runs:
            if r['arm'] != arm:
                continue
            t, q, _ = _series(r)
            if not len(t):
                continue
            mask = (t >= lo_t) & (t <= hi_t)
            ax.plot(t[mask], q[mask], color=ARM_COLOR[arm],
                    alpha=0.45, lw=1.0)
            peak = float(q[mask].max()) if mask.any() else np.nan
            pre = q[(t >= lo_t) & (t < start)]
            base = float(np.mean(pre)) if len(pre) else 0.0
            post = t[(t >= end) & (q <= base + 0.2 * max(peak - base, 1))]
            recovery = float(post[0] - end) if len(post) else np.nan
            stats.setdefault(arm, []).append(
                {'peak': peak, 'recovery_s': recovery})
    for arm in stats:
        mean_peak = float(np.mean([s['peak'] for s in stats[arm]]))
        ax.plot([], [], color=ARM_COLOR[arm],
                label=f'{ARM_LABEL.get(arm, arm)} (peak≈{mean_peak:.0f})')
    ax.set_xlim(lo_t, hi_t)
    ax.set_xlabel('simulation time (s)')
    ax.set_ylabel('incoming-lane queued vehicles')
    ax.set_title(f'Event-window zoom — {spec} '
                 '(one thin line per seed)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / 'fig3_event_window.png', dpi=150)
    plt.close(fig)
    return stats


def fig4_cost_effect(run_dir, runs, figs):
    hitl_runs = [r for r in runs if r['arm'] == 'hitl']
    if not hitl_runs:
        return False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    seeds = sorted(r['seed'] for r in hitl_runs)
    queries = [r.get('queries', 0) for r in hitl_runs]
    takeovers = [r.get('takeovers', 0) for r in hitl_runs]
    agrees = [r.get('agrees', 0) for r in hitl_runs]
    x = np.arange(len(seeds))
    ax1.bar(x - 0.2, takeovers, width=0.4, color=ARM_COLOR['hitl'],
            label='takeover')
    ax1.bar(x + 0.2, agrees, width=0.4, color='#8fbf8f', label='agree')
    for i, q in enumerate(queries):
        ax1.text(i, max(takeovers[i], agrees[i]) + 0.3,
                 f'{q} queries', ha='center', fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'seed {s}' for s in seeds])
    ax1.set_ylabel('count')
    ax1.set_title('Gate ledger: budget spent = takeovers + agrees')
    ax1.legend(fontsize=8)

    # cost-effect: Δmetrics vs student per seed
    d_tt, d_unf, labels = [], [], []
    for r in hitl_runs:
        s = next((x for x in runs
                  if x['arm'] == 'student' and x['seed'] == r['seed']),
                 None)
        if not s:
            continue
        labels.append(f"s{r['seed']}\n({r.get('queries', 0)}q)")
        d_tt.append(float(s.get('travel_time') or 0)
                    - float(r.get('travel_time') or 0))
        d_unf.append(float(s.get('unfinished') or 0)
                     - float(r.get('unfinished') or 0))
    x = np.arange(len(d_tt))
    ax2.bar(x - 0.2, d_tt, width=0.4,
            label='Δ travel_time (student−hitl)')
    ax2.bar(x + 0.2, d_unf, width=0.4,
            label='Δ unfinished (student−hitl)')
    ax2.axhline(0, color='k', lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_title('Per-seed benefit (positive = HITL better)')
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / 'fig4_cost_effect.png', dpi=150)
    plt.close(fig)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run-dir', required=True,
                        help='paired_probe 输出目录')
    parser.add_argument('--out', default=None,
                        help='图输出目录（默认 <run-dir>/figs）')
    args = parser.parse_args(argv)
    run_dir, runs, manifest = _load_runs(args.run_dir)
    figs = Path(args.out or run_dir / 'figs')
    figs.mkdir(parents=True, exist_ok=True)

    fig1_trajectory(run_dir, runs, manifest, figs)
    fig2_terminal(runs, figs)
    event_stats = fig3_event_window(run_dir, runs, manifest, figs)
    ledger = fig4_cost_effect(run_dir, runs, figs)

    # machine-readable evidence summary
    summary = {'arms': {}, 'paired_delta': {}, 'event_window': {},
               'hitl_ledger': {}}
    for arm in sorted({r['arm'] for r in runs}):
        vals = {}
        for key in ('travel_time', 'throughput', 'unfinished',
                    'real_delay', 'waiting_time'):
            xs = [r[key] for r in runs
                  if r['arm'] == arm and r.get(key) is not None]
            vals[key] = {
                'mean': float(np.mean(xs)) if xs else None,
                'per_seed': {
                    str(r['seed']): r[key] for r in runs
                    if r['arm'] == arm},
            }
        summary['arms'][arm] = vals
    for seed in manifest.get('seeds', []):
        s = next((r for r in runs
                  if r['arm'] == 'student' and r['seed'] == seed), None)
        h = next((r for r in runs
                  if r['arm'] == 'hitl' and r['seed'] == seed), None)
        if s and h:
            summary['paired_delta'][str(seed)] = {
                k: (None if s.get(k) is None or h.get(k) is None
                    else float(s[k]) - float(h[k]))
                for k in ('travel_time', 'throughput', 'unfinished')}
    if event_stats:
        summary['event_window'] = {
            arm: {
                'peak_mean': float(np.mean(
                    [x['peak'] for x in stats])),
                'recovery_s_mean': float(np.nanmean(
                    [x['recovery_s'] for x in stats])),
            } for arm, stats in event_stats.items()}
    hitl_runs = [r for r in runs if r['arm'] == 'hitl']
    if hitl_runs:
        summary['hitl_ledger'] = {
            'queries_total': sum(r.get('queries', 0) for r in hitl_runs),
            'takeovers_total': sum(
                r.get('takeovers', 0) for r in hitl_runs),
            'agrees_total': sum(r.get('agrees', 0) for r in hitl_runs),
            'budget_per_episode': (manifest.get('gate') or {}).get(
                'budget'),
        }
    (figs / 'evidence_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(f'figs -> {figs}')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
