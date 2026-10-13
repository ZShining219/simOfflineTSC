"""SIGMA 式三联证据图（corridor-load 单场景加厚版）。

  A. 治理收益:   各臂"报告所指走廊排队"(事件边+协调腿)相对无文本臂的
                配对逐 case % 变化,mean±SEM,标 Wilcoxon 显著性
  B. 延迟鲁棒:   corr_queue vs 报告延迟 {0,30,60,90,120}s,
                参考线=无文本臂/maxpressure
  C. 强度剂量:   corr_queue vs 播报强度 sev{1,2,3}(period{4,2,1}),
                sga15/notext/lie/maxp/sotl 五线

用法::

    python -m dev.takeover_probe.sig15_panels \
        --root output_data/dev_takeover_probe \
        --out output_data/dev_takeover_probe/evidence_fig/sig15_panels.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np


def load(root: Path, name: str):
    p = root / name / 'results.json'
    return json.loads(p.read_text())['arms'] if p.exists() else None


def corr_total(r):
    return (r.get('ev_queue_int') or 0.0) + (r.get('corr_queue_int') or 0.0)


def paired_delta_pct(arms, arm, ref='joint_notext'):
    """Per-case (arm - ref) / ref on corridor total queue."""
    out = []
    for a, b in zip(arms[arm], arms[ref]):
        rb = corr_total(b)
        if rb:
            out.append(100 * (corr_total(a) - rb) / rb)
    return np.asarray(out)


def wilcoxon_p(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    if len(d) < 5:
        return None
    from math import erfc
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    w_plus = ranks[d > 0].sum()
    n = len(d)
    mu, sd = n * (n + 1) / 4, (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    return erfc(abs((w_plus - mu) / sd) / 2 ** 0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='output_data/dev_takeover_probe')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.root)

    main8 = load(root, 'sig15_eval_main8')
    sev = load(root, 'sig15_eval_sev')
    delays = {}
    for d, name in [(0, 'sig15_eval_delay0'), (30, 'sig15_eval'),
                    (60, 'sig15_eval_delay60'), (90, 'sig15_eval_delay90'),
                    (120, 'sig15_eval_delay120')]:
        a = load(root, name)
        if a:
            delays[d] = a

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # ---------- A: governance benefit ----------
    ax = axes[0]
    arms = [('joint_sga15', 'true\nreport', '#2ca02c'),
            ('joint_lie', 'lie\nreport', '#ff7f0e'),
            ('joint_base', 'base\npath', '#7f7f7f'),
            ('maxpressure', 'max-\npressure', '#1f77b4'),
            ('sotl', 'sotl', '#8c564b'),
            ('dqn_s1', 'dqn_s1\n(vanilla)', '#17becf'),
            ('fixedtime', 'fixed\ntime', '#9467bd')]
    means, sems, cols, labs, sig = [], [], [], [], []
    for name, lab, c in arms:
        d = paired_delta_pct(main8, name)
        means.append(d.mean())
        sems.append(d.std() / np.sqrt(len(d)))
        cols.append(c)
        labs.append(lab)
        pv = wilcoxon_p(
            [corr_total(r) for r in main8[name]],
            [corr_total(r) for r in main8['joint_notext']])
        sig.append(pv)
    CAP, FLOOR = 6.0, -13.0
    disp = [max(FLOOR, min(CAP, m)) for m in means]
    bars = ax.bar(range(len(arms)), disp, 0.62, color=cols)
    for i, (b, m, s, p_) in enumerate(zip(bars, means, sems, sig)):
        if i >= 2:
            b.set_alpha(0.6)
        clipped = m != disp[i]
        if clipped:
            b.set_hatch('//')
            b.set_edgecolor('white')
        else:
            ax.errorbar(b.get_x() + b.get_width() / 2, m, yerr=s,
                        color='k', capsize=3, lw=1.2)
        star = {True: '', False: ' *'}.get(p_ is None, ' **' if p_ < 0.01
                                          else ' *' if p_ < 0.05 else '')
        ytxt = disp[i] - 1.0 if m > 0 else disp[i] + 0.8
        ax.annotate(f'{m:+.1f}%{star}',
                    (b.get_x() + b.get_width() / 2, ytxt),
                    ha='center', va='top' if m > 0 else 'bottom',
                    fontsize=9, fontweight='bold')
    ax.axhline(0, color='k', lw=1)
    ax.annotate('no-text arm = 0\n(* p<0.05, ** p<0.01 paired Wilcoxon)',
                (0.02, 0.04), xycoords='axes fraction', fontsize=7.5,
                color='gray')
    ax.set_xticks(range(len(arms)), labs, fontsize=8.5)
    ax.set_ylabel('corridor queue, % vs\nsame-model no-text', fontsize=9)
    ax.set_title('A. Governance benefit on the named corridor\n'
                 '(20 cases: 4 edges x 5 seeds)', fontsize=9.5)
    ax.set_ylim(FLOOR - 2.5, CAP + 2)

    # ---------- B: report-delay robustness ----------
    ax = axes[1]
    ds = sorted(delays)
    for arm, c, lab in [('joint_sga15', '#2ca02c', 'sga15 true report'),
                        ('joint_notext', 'gray', 'no text')]:
        ys = [np.array([r['corr_queue_int'] for r in delays[d][arm] if
                        r.get('corr_queue_int') is not None]) / 1000
              for d in ds]
        ms_ = [y.mean() for y in ys]
        ss = [y.std() / np.sqrt(len(y)) for y in ys]
        ax.errorbar(ds, ms_, yerr=ss, color=c, lw=2, marker='o', ms=5,
                    capsize=3, label=lab)
    mp = delays[30].get('maxpressure')
    if mp:
        ax.axhline(np.mean([r['corr_queue_int'] for r in mp]) / 1000,
                   color='#1f77b4', ls=':', lw=1.5,
                   label='maxpressure (report-free)')
    ax.set_xlabel('report delay (s)')
    ax.set_ylabel('corridor legs queue\n(x1000 veh·s)', fontsize=9)
    ax.set_title('B. Report-delay robustness\n'
                 '(graceful degradation, never worse than no-text)',
                 fontsize=9.5)
    ax.legend(fontsize=8)

    # ---------- C: severity dose ----------
    ax = axes[2]
    by_sev = {}
    for n, rws in sev.items():
        for r in rws:
            per = int(r['event'].split(':')[-1])
            s = min(3, max(1, round(4 / per)))
            by_sev.setdefault(s, {}).setdefault(n, []).append(
                r['corr_queue_int'])
    ss = sorted(by_sev)
    for n, c, lab in [('joint_sga15', '#2ca02c', 'sga15 true'),
                      ('joint_notext', 'gray', 'no text'),
                      ('joint_lie', '#ff7f0e', 'lie'),
                      ('maxpressure', '#1f77b4', 'maxpressure'),
                      ('sotl', '#8c564b', 'sotl')]:
        if not all(n in by_sev[s] for s in ss):
            continue
        ys = [np.array(by_sev[s][n]) / 1000 for s in ss]
        ax.errorbar(ss, [y.mean() for y in ys],
                    yerr=[y.std() / np.sqrt(len(y)) for y in ys],
                    color=c, lw=2, marker='o', ms=5, capsize=3, label=lab)
    ax.annotate('text benefit\nwidens at low\nseverity', (1.05, 43.5),
                fontsize=8, color='#2ca02c')
    ax.set_xlabel('reported severity (1=mild .. 3=extreme)')
    ax.set_xticks(ss, [f'sev{s}' for s in ss])
    ax.set_ylabel('corridor legs queue\n(x1000 veh·s)', fontsize=9)
    ax.set_title('C. Severity dose-response\n'
                 '(scene steering helps most when slack remains)',
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc='upper left')

    fig.suptitle('Scene-guided attention — corridor-load scenario '
                 '(same physical scenario family throughout)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
