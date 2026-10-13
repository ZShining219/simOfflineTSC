"""sig15 系列评测汇总表：跨 eval 目录拼一张 SIGMA 式主表。

- 主表: arm x {ev_queue, corr_queue, queue_int, throughput} mean±std
- paired Wilcoxon(true vs notext / true vs lie, corr_queue)
- 延迟曲线 / 强度档 / 跨场景各一张小表

用法::

    python -m scene_attention.sig15_table \
        --root output_data/dev_takeover_probe
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
    if not p.exists():
        return None
    return json.loads(p.read_text())['arms']


def ms(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    if not v:
        return None
    return float(np.mean(v)), float(np.std(v))


def fmt(stat, nd=0):
    if stat is None:
        return '—'
    m, s = stat
    return f'{m:,.{nd}f}±{s:,.{nd}f}'


def wilcoxon(a, b):
    """Paired Wilcoxon signed-rank on per-case values (scipy-free)."""
    d = np.array(a) - np.array(b)
    d = d[d != 0]
    if len(d) < 5:
        return None
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    w_plus = ranks[d > 0].sum()
    n = len(d)
    mu, sd = n * (n + 1) / 4, (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    z = (w_plus - mu) / sd
    from math import erfc
    p = erfc(abs(z) / 2 ** 0.5)
    return w_plus, z, p, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='output_data/dev_takeover_probe')
    ap.add_argument('--main', default='sig15_eval_main8')
    args = ap.parse_args()
    root = Path(args.root)

    # ---------- main table ----------
    arms = load(root, args.main)
    if arms:
        print(f'== MAIN TABLE ({args.main}, n={len(next(iter(arms.values())))} cases) ==')
        keys = ['ev_queue_int', 'corr_queue_int', 'queue_int', 'throughput']
        hdr = f'{"arm":<16}' + ''.join(f'{k[:12]:>22}' for k in keys)
        print(hdr)
        for name, rows in arms.items():
            line = f'{name:<16}'
            for k in keys:
                line += f'{fmt(ms(rows, k)):>22}'
            print(line)
        for other in ('joint_notext', 'joint_lie'):
            a = [r['corr_queue_int'] for r in arms['joint_sga15']]
            b = [r['corr_queue_int'] for r in arms[other]]
            w = wilcoxon(a, b)
            if w:
                print(f'  Wilcoxon sga15 vs {other} (corr_queue): '
                      f'W+={w[0]:.0f} z={w[1]:.2f} p={w[2]:.4f} n={w[3]}')

    # ---------- delay sweep ----------
    print('\n== DELAY SWEEP (corr_queue_int) ==')
    rows = {}
    for d in (0, 30, 60, 90, 120):
        dd = 'sig15_eval' if d == 30 else \
            ('sig15_eval_delay90' if d == 90 else f'sig15_eval_delay{d}')
        a = load(root, dd)
        if a is None:
            continue
        rows[d] = a
    hdr = f'{"delay":>6}'
    names = sorted({n for a in rows.values() for n in a})
    for n in names:
        hdr += f'{n[:18]:>20}'
    print(hdr)
    for d, a in rows.items():
        line = f'{d:>5}s'
        for n in names:
            line += f'{fmt(ms(a.get(n, []), "corr_queue_int")):>20}'
        print(line)

    # ---------- severity sweep ----------
    print('\n== SEVERITY SWEEP (period -> report severity) ==')
    a = load(root, 'sig15_eval_sev')
    if a:
        by_per = {}
        for n, rws in a.items():
            for r in rws:
                per = r['event'].split(':')[-1]
                by_per.setdefault(per, {}).setdefault(n, []).append(r)
        hdr = f'{"period":>7}{"sev":>5}'
        for n in a:
            hdr += f'{n[:16]:>18}'
        print(hdr + '   (corr_queue_int)')
        for per in sorted(by_per, key=int):
            sev = min(3, max(1, round(4 / int(per))))
            line = f'{per:>7}{sev:>5}'
            for n in a:
                line += f'{fmt(ms(by_per[per].get(n, []), "corr_queue_int")):>18}'
            print(line)

    # ---------- cross-scene ----------
    print('\n== CROSS-SCENE (corr_queue_int) ==')
    names = {'S2': 'sig15_eval', 'S1': 'sig15_S1_eval',
             'S3': 'sig15_S3_eval', 'S4': 'sig15_S4_eval'}
    zs = {'S1': 'sig15_eval_S1', 'S3': 'sig15_eval_S3',
          'S4': 'sig15_eval_S4'}
    print(f'{"scene":>6}{"arm":>16}{"corr_q":>16}{"ev_q":>16}')
    for sc, d in names.items():
        a = load(root, d)
        if a is None:
            continue
        for n, rws in a.items():
            print(f'{sc:>6}{n:>16}{fmt(ms(rws,"corr_queue_int")):>16}'
                  f'{fmt(ms(rws,"ev_queue_int")):>16}')
    for sc, d in zs.items():
        a = load(root, d)
        if a is None:
            continue
        for n, rws in a.items():
            print(f'{sc+"(zs)":>6}{n:>16}{fmt(ms(rws,"corr_queue_int")):>16}'
                  f'{fmt(ms(rws,"ev_queue_int")):>16}')


if __name__ == '__main__':
    main()
