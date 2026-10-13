"""一句话柱状图：同一模型只换文本报告，"报告所指走廊"排队下降比例。

走廊排队 = 事件边 + 协调腿(feeder/discharge/截流来路)排队积分。
以"同模型无文本"臂为 0 基准，各臂给出相对下降比例。

用法::

    python -m scene_attention.corridor_delta_fig \
        --results output_data/dev_takeover_probe/sig15_eval/results.json \
        --out output_data/dev_takeover_probe/evidence_fig/corridor_delta.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    summary = json.loads(Path(args.results).read_text())['summary']

    def corr_total(a):
        return summary[a]['ev_queue_int'] + summary[a]['corr_queue_int']

    ref = corr_total('joint_notext')
    arms = [
        ('joint_sga15', 'true report', '#2ca02c'),
        ('joint_lie', 'lie report', '#ff7f0e'),
        ('joint_base', 'base path\n(no scene)', '#7f7f7f'),
        ('maxpressure', 'max-\npressure', '#1f77b4'),
        ('fixedtime', 'fixed\ntime', '#1f77b4'),
    ]
    vals = [100 * (corr_total(a) - ref) / ref for a, _, _ in arms]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(arms))
    colors = [c for _, _, c in arms]
    CAP = 4.0      # visual cap for positive (worse) bars
    FLOOR = -9.0
    disp = [max(FLOOR, min(CAP, v)) for v in vals]
    bars = ax.bar(x, disp, 0.58, color=colors)
    for i, b in enumerate(bars):
        if i >= 2:
            b.set_alpha(0.55)
    for b, v, d in zip(bars, vals, disp):
        clipped = v != d
        ytxt = d - 0.4 if v > 0 else d + 0.4
        ax.annotate(f'{v:+.1f}%' + ('  ≫' if clipped else ''),
                    (b.get_x() + b.get_width() / 2, ytxt),
                    ha='center', va='top' if v > 0 else 'bottom',
                    fontsize=11, fontweight='bold')
        if clipped:
            b.set_hatch('//')
            b.set_edgecolor('white')
    ax.axhline(0, color='k', lw=1.0)
    ax.annotate('same model, report channel OFF (0%)',
                (0.98, 0.06), xycoords='axes fraction', ha='right',
                fontsize=8, color='gray')
    ax.set_xticks(list(x), [l for _, l, _ in arms], fontsize=10)
    ax.set_ylabel('queue on the corridor the report names\n(% change vs no-text arm)',
                  fontsize=10)
    ax.set_title('One model, one scenario — only the report text changes\n'
                 'corridor-load scenario, 12 cases, event edge + coordination legs',
                 fontsize=10.5)
    ax.set_ylim(FLOOR - 1, CAP + 1.5)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f'wrote {out}')
    for (a, l, _), v in zip(arms, vals):
        print(f'  {a:14s} {l.split(chr(10))[0]:20s} {v:+.2f}%')


if __name__ == '__main__':
    main()
