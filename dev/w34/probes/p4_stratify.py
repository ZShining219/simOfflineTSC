#!/usr/bin/env python3
"""w34_2 P4 — stratified mining of eval_v2 paired diffs.

Answers, without any training:
  1. does the canonical-minus-empty penalty concentrate in particular
     arms/seeds (incl. s17), event kinds, or episodes?
  2. does the per-episode diff correlate with the event window's share of
     the episode (sparse-window artifact check: a long episode with a short
     window should show a smaller *episode-mean* penalty even if the window
     penalty is real)?
  3. distribution diagnostics: is the negative tail a few outliers or
     systematic (quantiles, sign counts, winsorized means)?

Inputs: eval_v2_paired/paired_episodes.csv + p2_binding_audit/audit.json
Output: analysis/att_entity_004/p4_stratify/{stratify.json,stratify.md}
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[3]
PAIRED = ROOT / 'data/output_data/analysis/att_entity_004/eval_v2_paired/paired_episodes.csv'
AUDIT = ROOT / 'data/output_data/analysis/att_entity_004/p2_binding_audit/audit.json'
OUT = ROOT / 'data/output_data/analysis/att_entity_004/p4_stratify'

METRICS = ['travel_time', 'waiting_time', 'queue', 'delay',
           'win_travel_time_in', 'win_travel_time_out',
           'win_queue_in', 'win_queue_out']


def summarize(xs):
    xs = sorted(xs)
    n = len(xs)
    pos = sum(1 for x in xs if x > 0)
    neg = sum(1 for x in xs if x < 0)
    def q(p):
        k = (n - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
    w = xs[1:-1] if n > 2 else xs      # winsorized: drop extremes
    return {'n': n, 'mean': st.mean(xs), 'median': q(.5),
            'q10': q(.1), 'q90': q(.9), 'min': xs[0], 'max': xs[-1],
            'winsor_mean': st.mean(w), 'pos': pos, 'neg': neg,
            'sign_p': binom_sign(pos, n)}


def binom_sign(pos, n):
    """two-sided sign test, exact."""
    from math import comb
    k = min(pos, n - pos)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    audit = json.load(AUDIT.open())
    ep_events = defaultdict(list)
    for r in audit['episodes']:
        ep_events[r['episode']].append(r)

    # paired_util's CSV mixes keyings: metric columns sit on the trainer's
    # 1-based record index (record N ran plan N-1) while win_queue_* /
    # has_event sit on the 0-based npz==plan index.  Re-key explicitly.
    rows_by_ep = {(r['pair'], int(r['episode'])): r
                  for r in csv.DictReader(PAIRED.open())}
    pairs = sorted({p for p, _ in rows_by_ep})
    groups = defaultdict(lambda: defaultdict(list))
    merged_rows = []
    for pe in range(40):
        events = ep_events.get(pe, [])
        kinds = '+'.join(sorted({e['kind'] for e in events})) or 'no_event'
        classes = {e['class_'] for e in events}
        co = any(e.get('co_active') for e in events)
        cls = ('has_gap' if 'gap' in classes else
               'co_active' if co else
               ('all_bound' if classes else 'no_event'))
        wins = [(e['begin'], e['end']) for e in events]
        win_s = sum(b - a for a, b in wins)
        for pair in pairs:
            met = rows_by_ep.get((pair, pe + 1))
            win = rows_by_ep.get((pair, pe))
            if met is None:
                continue
            merged = {'pair': pair, 'plan_ep': pe, 'kinds': kinds, 'cls': cls,
                      'win_s': win_s,
                      'win_queue_in': win.get('win_queue_in') if win else None,
                      'win_queue_out': win.get('win_queue_out') if win else None}
            for m in ('travel_time', 'waiting_time', 'queue', 'delay'):
                merged[m] = met.get(m)
            merged_rows.append(merged)
            arm = pair.rsplit('_', 1)[0]
            for m in METRICS:
                v = merged.get(m)
                if not v:
                    continue
                x = float(v)
                for key in (('arm', arm), ('arm_seed', pair),
                            ('kind', kinds), ('class', cls),
                            ('win_bucket', f'{int(win_s)//300*300}-{int(win_s)//300*300+300}s')):
                    groups[key][m].append(x)

    result = {}
    for (dim, val), mets in sorted(groups.items()):
        result.setdefault(dim, {})[val] = {m: summarize(xs)
                                           for m, xs in mets.items() if xs}

    # sparse-window artifact test: corr(win_share, episode diff) &
    # window-only vs episode-mean diff comparison
    sparse = {}
    for m in ('queue',):
        triples = [(r['win_s'] / 3600.0, float(r[m]), float(r['win_queue_in']))
                   for r in merged_rows
                   if r.get(m) and r.get('win_queue_in')]
        if len(triples) > 5:
            n = len(triples)
            xs = [t[0] for t in triples]
            for idx, name in ((1, 'episode_diff'), (2, 'window_diff')):
                ys = [t[idx] for t in triples]
                mx, my = st.mean(xs), st.mean(ys)
                cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
                sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
                sy = math.sqrt(sum((b - my) ** 2 for b in ys))
                sparse.setdefault(m, {})[name] = {
                    'n': n, 'corr_win_share': cov / (sx * sy) if sx * sy else None}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'stratify.json').write_text(json.dumps(
        {'strata': result, 'sparse_artifact': sparse}, indent=1))

    md = ['# P4 stratified mining — canonical minus empty (eval_v2)', '']
    for dim, vals in result.items():
        md.append(f'## by {dim}')
        md.append('| value | metric | n | mean | median | q10..q90 | pos/neg | sign_p |')
        md.append('|---|---|---|---|---|---|---|---|')
        for val, mets in vals.items():
            for m in ('travel_time', 'win_queue_in', 'win_queue_out'):
                s = mets.get(m)
                if s:
                    md.append(f"| {val} | {m} | {s['n']} | {s['mean']:+.3f} | "
                              f"{s['median']:+.3f} | {s['q10']:+.2f}..{s['q90']:+.2f} | "
                              f"{s['pos']}/{s['neg']} | {s['sign_p']:.3g} |")
        md.append('')
    md.append('## sparse-window artifact check')
    md.append('| metric | target | n | corr(window_share, diff) |')
    md.append('|---|---|---|---|')
    for m, tgt in sparse.items():
        for name, s in tgt.items():
            c = s['corr_win_share']
            md.append(f"| {m} | {name} | {s['n']} | {c:+.3f} |" if c is not None
                      else f"| {m} | {name} | {s['n']} | n/a |")
    (OUT / 'stratify.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {OUT}/stratify.{{json,md}}')


if __name__ == '__main__':
    main()
