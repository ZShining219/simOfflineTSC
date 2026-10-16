#!/usr/bin/env python3
"""Paired-statistics layer for ATT-ENTITY-002 judgment requirements.

Reads summary_long.csv produced by tools/sga_eval_report.py and emits, per
(checkpoint, condition, eval_seed):

  - method contrasts at matched training seed and training dist:
        sga-colight, concat-colight, sga-concat
  - distribution contrasts at matched training seed and method:
        E-N (event-trained minus normal-trained)
  - per-contrast rollups: sign consistency across seeds, mean delta, and
    delta magnitude relative to the colight across-seed std

Outputs: paired_cells.csv (one row per matched cell), paired_stats.csv
(per contrast x dist/method x condition), and a printed ep200 matrix.

Usage:
    python3 tools/sga_paired_stats.py \
        --summary data/output_data/analysis/sga_concat_colight_200_v1/summary_long.csv \
        --out data/output_data/analysis/sga_concat_colight_200_v1
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ['travel_time', 'queue', 'throughput', 'unfinished_vehicles', 'delay']
WINDOW_FIELDS = ['before_queue_network_mean', 'during_queue_network_mean',
                 'after_queue_network_mean', 'during_throughput_interval',
                 'during_delay_network_weighted_mean']


def load_rows(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in list(r):
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                pass
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    rows = load_rows(args.summary)
    # cell = (method, dist, train_seed) -> per (ckpt, cond, eval_seed) record
    by_cell = defaultdict(dict)
    for r in rows:
        by_cell[(r['method'], r['dist'], int(r['seed']))][
            (int(r['checkpoint']), r['condition'], int(r['eval_seed']))] = r

    metrics = METRICS + WINDOW_FIELDS
    cells = []

    def emit(tag, ra, rb, ckpt, cond, eseed, arm):
        rec = {'contrast': tag, 'checkpoint': ckpt, 'condition': cond,
               'eval_seed': eseed, 'arm': arm,
               'seed_a': int(ra['seed']), 'seed_b': int(rb['seed']),
               'dist_a': ra['dist'], 'dist_b': rb['dist']}
        for m in metrics:
            va, vb = ra.get(m), rb.get(m)
            if isinstance(va, float) and isinstance(vb, float):
                rec[f'd_{m}'] = va - vb
        cells.append(rec)

    # method contrasts: same dist, same train seed
    seeds = sorted({k[2] for k in by_cell})
    for dist in ('E', 'N'):
        for seed in seeds:
            for a, b in (('sga', 'colight'), ('concat', 'colight'),
                         ('sga', 'concat')):
                ca = by_cell.get((a, dist, seed))
                cb = by_cell.get((b, dist, seed))
                if not ca or not cb:
                    continue
                for key in sorted(set(ca) & set(cb)):
                    ckpt, cond, eseed = key
                    emit(f'{a}-{b}|{dist}', ca[key], cb[key], ckpt, cond,
                         eseed, ca[key].get('arm', '?'))
    # dist contrasts: same method, same train seed
    for meth in ('colight', 'concat', 'sga'):
        for seed in seeds:
            ce = by_cell.get((meth, 'E', seed))
            cn = by_cell.get((meth, 'N', seed))
            if not ce or not cn:
                continue
            for key in sorted(set(ce) & set(cn)):
                ckpt, cond, eseed = key
                emit(f'E-N|{meth}', ce[key], cn[key], ckpt, cond, eseed,
                     ce[key].get('arm', '?'))

    # colight seed-noise yardstick: std across train seeds per (dist,ckpt,cond,eseed)
    colight_vals = defaultdict(list)
    for (meth, dist, seed), cell in by_cell.items():
        if meth != 'colight':
            continue
        for (ckpt, cond, eseed), r in cell.items():
            for m in metrics:
                v = r.get(m)
                if isinstance(v, float):
                    colight_vals[(dist, ckpt, cond, m)].append(v)
    colight_std = {k: float(np.std(v)) if len(v) > 1 else float('nan')
                   for k, v in colight_vals.items()}

    agg = defaultdict(list)
    for r in cells:
        dist = r['dist_a'] if r['contrast'].endswith(('|E', '|N')) else 'E-N'
        agg[(r['contrast'], r['checkpoint'], r['condition'])].append(r)

    stat_rows = []
    for (contrast, ckpt, cond), rs in sorted(agg.items()):
        rec = {'contrast': contrast, 'checkpoint': ckpt, 'condition': cond,
               'n_cells': len(rs),
               'seeds': ','.join(str(s) for s in sorted({r['seed_a'] for r in rs}))}
        for m in METRICS:
            ds = [r[f'd_{m}'] for r in rs if f'd_{m}' in r]
            if not ds:
                continue
            dist = contrast.rsplit('|', 1)[-1]
            if dist in ('E', 'N'):
                sstd = colight_std.get((dist, ckpt, cond, m), float('nan'))
            else:
                sstd = float('nan')
            mean_d = float(np.mean(ds))
            rec[f'{m}_mean_delta'] = mean_d
            rec[f'{m}_delta_std'] = float(np.std(ds))
            rec[f'{m}_sign_consistency'] = float(
                max(np.mean([d > 0 for d in ds]),
                    np.mean([d < 0 for d in ds])))
            rec[f'{m}_colight_seed_std'] = sstd
            rec[f'{m}_abs_over_seedstd'] = (
                abs(mean_d) / sstd if sstd == sstd and sstd > 0 else float('nan'))
        stat_rows.append(rec)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (('paired_cells.csv', cells), ('paired_stats.csv', stat_rows)):
        fields = sorted({k for r in data for k in r})
        with (out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(data)

    print(f'{len(cells)} paired cells -> {out}/paired_cells.csv')
    print(f'{len(stat_rows)} stat rows -> {out}/paired_stats.csv')
    print('\n=== ep200 travel_time mean deltas ===')
    for r in stat_rows:
        if r['checkpoint'] == 200 and 'travel_time_mean_delta' in r:
            sstd = r.get('travel_time_colight_seed_std')
            ratio = r.get('travel_time_abs_over_seedstd')
            print(f"{r['contrast']:16s} {r['condition']:20s} "
                  f"n={r['n_cells']} seeds[{r['seeds']:12s}] "
                  f"d={r['travel_time_mean_delta']:+7.2f} "
                  f"sign={r['travel_time_sign_consistency']:.2f} "
                  f"seedstd={sstd if sstd==sstd else float('nan'):6.2f} "
                  f"|d|/std={ratio if ratio==ratio else float('nan'):6.2f}")


if __name__ == '__main__':
    main()
