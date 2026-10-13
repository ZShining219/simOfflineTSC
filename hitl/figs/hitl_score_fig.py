"""HITL 三通道合并对比图：全部臂 × 4 指标 AULC Δvs C0 森林图。

数据源：pilot5 各 stats_*/paired_bootstrap.json（均 baseline=C0）。
臂按通道着色分组：A 演示模仿 / B 奖励塑形 / B' 裁定面板 / C 专家池。

用法::

    python -m hitl.figs.hitl_score_fig \
        --pilot output_data/hitl_probe/pilot5 \
        --out output_data/hitl_probe/pilot5/figs_ALL/hitl_score.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CHANNEL = {
    'HITL_A0': ('A demo-imitation', '#4c72b0'), 'HITL_API': ('A demo-imitation', '#4c72b0'),
    'HITL_AQ': ('A demo-imitation', '#4c72b0'), 'HITL_APIQ': ('A demo-imitation', '#4c72b0'),
    'HITL_APIQB': ('A demo-imitation', '#4c72b0'),
    'HITL_B1': ('B reward-shaping', '#55a868'), 'HITL_B2': ('B reward-shaping', '#55a868'),
    'HITL_BP': ("B' judge-panel", '#8172b2'), 'HITL_BPC': ("B' judge-panel", '#8172b2'),
    'HITL_BPT': ("B' judge-panel", '#8172b2'), 'HITL_BPM': ("B' judge-panel", '#8172b2'),
    'HITL_BPQ': ("B' judge-panel", '#8172b2'),
    'HITL_CTRL': ('C expert-pool', '#c44e52'), 'HITL_DEMO': ('C expert-pool', '#c44e52'),
    'HITL_EXPERT': ('C expert-pool', '#c44e52'), 'HITL_EXPC': ('C expert-pool', '#c44e52'),
    'HITL_EXPD': ('C expert-pool', '#c44e52'), 'HITL_EXPF': ('C expert-pool', '#c44e52'),
    'HITL_EXPB': ('C expert-pool', '#c44e52'), 'HITL_EXPL': ('C expert-pool', '#c44e52'),
}
SHORT = {
    'HITL_A0': 'A0 no-pretrain', 'HITL_API': 'A-pi actor-only',
    'HITL_AQ': 'A-Q critic-only', 'HITL_APIQ': 'A-pi-Q dual-pretrain',
    'HITL_APIQB': 'A-pi-Q-BIG demo-6x',
    'HITL_B1': 'B1 rule-score', 'HITL_B2': 'B2 reward-net',
    'HITL_BP': "BP 3-judge panel", 'HITL_BPC': "BPC panel+pool",
    'HITL_BPT': "BPT terms-only", 'HITL_BPM': "BPM reward-net-only",
    'HITL_BPQ': "BPQ qgap-only",
    'HITL_CTRL': 'CTRL equal machine data', 'HITL_DEMO': 'DEMO pure demo pool',
    'HITL_EXPERT': 'EXP correct-pool r=.25', 'HITL_EXPC': 'EXPC r=.125',
    'HITL_EXPD': 'EXPD r .25->0 decay', 'HITL_EXPF': 'EXPF gateq pool',
    'HITL_EXPB': 'EXPB big-pool r=.25', 'HITL_EXPL': 'EXPL big-pool r=.05',
}
METRICS = [('travel_time', 'travel_time (↓ better)'),
           ('throughput', 'throughput (↑ better)'),
           ('unfinished_vehicles', 'unfinished (↓ better)'),
           ('rrt', 'rrt (↓ better)')]


def load_stats(pilot: Path):
    """Merge all stat sets sharing baseline C0; return arm→metric→entry."""
    merged = {}
    for d in sorted(pilot.glob('stats*')):
        f = d / 'paired_bootstrap.json'
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        if s.get('baseline') != 'HITL_C0':
            continue
        for met, arms in s['metrics'].items():
            for arm, ent in arms.items():
                if arm == 'HITL_C0':
                    continue
                merged.setdefault(arm, {})[met] = ent
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot', default='output_data/hitl_probe/pilot5')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    merged = load_stats(Path(args.pilot))

    order = [a for a in SHORT if a in merged]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2), sharey=True)
    for ax, (met, title) in zip(axes, METRICS):
        ys, means, los, his, cols = [], [], [], [], []
        for i, arm in enumerate(order):
            ent = merged[arm].get(met, {}).get('aulc_bootstrap')
            if not ent:
                continue
            ys.append(i)
            means.append(ent['mean_delta'])
            lo, hi = ent['ci95']
            los.append(ent['mean_delta'] - lo)
            his.append(hi - ent['mean_delta'])
            cols.append(CHANNEL[arm][1])
        ax.axvline(0, color='k', lw=0.9)
        ax.errorbar(means, ys, xerr=[los, his], fmt='none', ecolor='gray',
                    elinewidth=1.4, capsize=3, zorder=2)
        ax.scatter(means, ys, c=cols, s=55, zorder=3)
        for x, y, arm in zip(means, ys, order):
            p = merged[arm][met]['aulc_bootstrap'].get(
                'p_one_sided_improvement')
            if p is not None and (p >= 0.975 or p <= 0.025):
                ax.annotate('*', (x, y + 0.28), ha='center', fontsize=13,
                            fontweight='bold',
                            color='#2ca02c' if (
                                (met == 'throughput') == (x > 0)) else '#c44e52')
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.25)
    axes[0].set_yticks(range(len(order)), [SHORT[a] for a in order],
                       fontsize=8.5)
    # channel legend
    seen = {}
    for a in order:
        ch, c = CHANNEL[a]
        seen.setdefault(ch, c)
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=c, markersize=8, label=ch)
               for ch, c in seen.items()]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               frameon=False)
    fig.suptitle('HITL pilot5: all arms vs C0 baseline — AULC delta, '
                 '95% paired bootstrap CI, n=5 seeds\n'
                 '(* = one-sided P≥0.975 or ≤0.025; green*=improvement, '
                 'red*=harm)', fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
