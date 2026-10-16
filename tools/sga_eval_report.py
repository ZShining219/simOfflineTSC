#!/usr/bin/env python3
"""Aggregate ATT-ENTITY-002 evaluation packages into tables and figures.

Reads eval packages produced by `run.py --evaluation-manifest` (summary.csv +
records.jsonl DECISION_METRICS per attempt), joins run metadata (agent,
training distribution, epsilon arm, checkpoint), and writes:

  summary_long.csv      one row per attempt with all metrics
  paired_deltas.csv     per-cell values and delta vs colight (same dist/ckpt/
                        condition/eval_seed)
  figures/*.png         condition bars, event-window curves, ckpt stability,
                        training curves

Usage:
    python3 tools/sga_eval_report.py \
        --eval-root artifacts/sga_concat_colight_200_v1/eval_v1 \
        --queue-state artifacts/sga_concat_colight_200_v1/run_state/run_manifest.json \
        --out data/output_data/analysis/sga_concat_colight_200_v1
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

EVENT_WINDOW = (900.0, 1200.0)  # eval conditions fire inside [900,1200)
METRICS = ['travel_time', 'queue', 'throughput', 'unfinished_vehicles', 'delay']
DISPLAY = {'colight': 'colight', 'sga_colight': 'sga', 'concat_colight': 'concat'}
METHOD_ORDER = ['colight', 'concat', 'sga']
COND_ORDER = ['none', 'blockage', 'closure', 'rain',
              'blockage_closure', 'blockage_rain', 'closure_rain', 'all']


def parse_run(run_id):
    m = re.match(r'(colight|sga_colight|concat_colight)_(?:text_)?(event|normal)_\d+_s(\d+)$', run_id)
    if not m:
        return None
    return {'agent': m.group(1), 'dist': 'E' if m.group(2) == 'event' else 'N',
            'seed': int(m.group(3))}


def epsilon_arm(run_dir):
    cfg = run_dir / 'config' / 'resolved_config.yaml'
    if not cfg.is_file():
        return '?'
    decay = yaml.safe_load(cfg.read_text())['model'].get('epsilon_decay')
    return {0.9995: 'fast', 0.99995: 'slow'}.get(decay, str(decay))


_records_cache = {}


def window_means(records_path, controller_id, eval_seed):
    """Split per-decision records into before/during/after event windows."""
    key_path = str(records_path)
    parsed = _records_cache.get(key_path)
    if parsed is None:
        parsed = []
        for line in records_path.open():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('record_type') == 'DECISION_METRICS':
                parsed.append(r)
        _records_cache[key_path] = parsed
    buckets = {'before': [], 'during': [], 'after': []}
    for r in parsed:
        if (r.get('controller_id') != controller_id
                or r.get('evaluation_seed') != eval_seed):
            continue
        t = r['simulation_time_seconds']
        key = 'before' if t < EVENT_WINDOW[0] else (
            'during' if t < EVENT_WINDOW[1] else 'after')
        buckets[key].append(r)
    out = {}
    for key, rows in buckets.items():
        if not rows:
            continue
        for field in ('queue_network_mean', 'delay_network_weighted_mean',
                      'throughput_interval', 'reward_network_mean'):
            vals = [r[field] for r in rows if field in r]
            if vals:
                out[f'{key}_{field}'] = float(np.mean(vals))
    return out


def collect(eval_root, run_meta):
    rows = []
    for pkg in sorted((eval_root / 'packages').iterdir()):
        run_id = pkg.name
        meta = run_meta.get(run_id)
        if meta is None:
            continue
        # records.jsonl can be hundreds of MB per package; cache parsed
        # DECISION_METRICS only within the package loop to avoid OOM.
        _records_cache.clear()
        for row in csv.DictReader((pkg / 'summary.csv').open()):
            m = re.match(r'(.+)_ep(\d+)_([a-z_]+)$', row['controller_id'])
            cond = m.group(3)
            rec = {'run_id': run_id, 'agent': meta['agent'],
                   'method': DISPLAY[meta['agent']], 'dist': meta['dist'],
                   'seed': meta['seed'], 'arm': meta['arm'],
                   'checkpoint': int(row['checkpoint_episode']),
                   'condition': cond, 'eval_seed': int(row['evaluation_seed'])}
            for f in METRICS:
                rec[f] = float(row[f])
            rec.update(window_means(pkg / 'records.jsonl',
                                    row['controller_id'],
                                    int(row['evaluation_seed'])))
            rows.append(rec)
    return rows


def paired(rows):
    base = {}
    for r in rows:
        if r['method'] == 'colight':
            base[(r['dist'], r['arm'], r['checkpoint'], r['condition'],
                  r['eval_seed'])] = r
    out = []
    for r in rows:
        b = base.get((r['dist'], r['arm'], r['checkpoint'], r['condition'],
                      r['eval_seed']))
        if b is None:
            continue
        d = dict(r)
        for f in METRICS:
            d[f'd_{f}'] = r[f] - b[f]
        out.append(d)
    return out


def fig_condition_bars(rows, outdir):
    for dist in ('E', 'N'):
        sub = [r for r in rows if r['dist'] == dist and r['checkpoint'] == 200]
        if not sub:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        for ax, metric in zip(axes, ('travel_time', 'queue')):
            for i, meth in enumerate(METHOD_ORDER):
                xs, ys, es = [], [], []
                for cond in COND_ORDER:
                    vals = [r[metric] for r in sub
                            if r['method'] == meth and r['condition'] == cond]
                    if vals:
                        xs.append(cond); ys.append(np.mean(vals))
                        es.append(np.std(vals))
                ax.bar(np.arange(len(xs)) + i * 0.27, ys, 0.25,
                       yerr=es, capsize=2, label=meth)
            ax.set_xticks(np.arange(len(COND_ORDER)) + 0.27)
            ax.set_xticklabels(COND_ORDER, rotation=30, ha='right', fontsize=8)
            ax.set_title(f'{metric} (dist={dist}, ep200)')
            ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f'cond_bars_{dist}.png', dpi=150)
        plt.close(fig)


def fig_event_windows(rows, outdir):
    for cond in COND_ORDER[1:]:
        sub = [r for r in rows if r['condition'] == cond and r['checkpoint'] == 200
               and 'during_queue_network_mean' in r]
        if not sub:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.25
        labels = ['before', 'during', 'after']
        for i, meth in enumerate(METHOD_ORDER):
            ys = [np.mean([r[f'{w_}_queue_network_mean'] for r in sub
                           if r['method'] == meth]) for w_ in labels
                  if any(f'{w_}_queue_network_mean' in r for r in sub
                         if r['method'] == meth)]
            ax.bar(np.arange(len(ys)) + i * w, ys, w * 0.9, label=meth)
        ax.set_xticks(np.arange(len(labels)) + w)
        ax.set_xticklabels(labels)
        ax.axvline(0.5 + w, color='gray', ls='--', lw=0.8)
        ax.set_title(f'mean queue by event window ({cond})')
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f'window_queue_{cond}.png', dpi=150)
        plt.close(fig)


def fig_ckpt_stability(rows, outdir):
    pairs = {}
    for r in rows:
        if r['condition'] != 'all':
            continue
        key = (r['run_id'], r['eval_seed'])
        pairs.setdefault(key, {})[r['checkpoint']] = r['travel_time']
    xs, ys, names = [], [], []
    for key, v in pairs.items():
        if 150 in v and 200 in v:
            xs.append(v[150]); ys.append(v[200])
            names.append(key[0])
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xs, ys)
    lim = [min(xs + ys) * 0.95, max(xs + ys) * 1.05]
    ax.plot(lim, lim, '--', color='gray')
    ax.set_xlabel('ep150 travel_time'); ax.set_ylabel('ep200 travel_time')
    ax.set_title('checkpoint stability (all condition)')
    fig.tight_layout()
    fig.savefig(outdir / 'ckpt_stability.png', dpi=150)
    plt.close(fig)


def fig_training_curves(queue_state, outdir):
    state = json.loads(Path(queue_state).read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, meth in zip(axes, METHOD_ORDER):
        for task in state['tasks']:
            rid = task['run_id'].split('/')[-1]
            meta = parse_run(rid)
            if not meta or DISPLAY[meta['agent']] != meth:
                continue
            p = Path(task['output_path']) / 'metrics' / 'records.jsonl'
            if not p.is_file():
                continue
            eps, tts = [], []
            for line in p.open():
                r = json.loads(line)
                if r['record_type'] == 'TRAIN':
                    eps.append(r['episode']); tts.append(r['travel_time'])
            if eps:
                ax.plot(eps, tts, alpha=0.5, lw=0.8,
                        label=f"{meta['dist']}_s{meta['seed']}")
        ax.set_title(meth); ax.set_xlabel('episode'); ax.legend(fontsize=6)
    axes[0].set_ylabel('train travel_time')
    fig.tight_layout()
    fig.savefig(outdir / 'training_curves.png', dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-root', required=True, nargs='+',
                   help='one or more eval roots (e.g. eval_v1 eval_v2)')
    p.add_argument('--queue-state', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out); figdir = out / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)

    state = json.loads(Path(args.queue_state).read_text())
    run_meta = {}
    for task in state['tasks']:
        rid = task['run_id'].split('/')[-1]
        meta = parse_run(rid)
        if meta:
            meta['arm'] = epsilon_arm(Path(task['output_path']))
            run_meta[rid] = meta

    rows = []
    for root in args.eval_root:
        rows.extend(collect(Path(root), run_meta))
    if not rows:
        print('no evaluation packages found'); return
    fields = sorted({k for r in rows for k in r})
    with (out / 'summary_long.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
    deltas = paired(rows)
    with (out / 'paired_deltas.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, sorted({k for r in deltas for k in r}))
        w.writeheader(); w.writerows(deltas)
    fig_condition_bars(rows, figdir)
    fig_event_windows(rows, figdir)
    fig_ckpt_stability(rows, figdir)
    fig_training_curves(args.queue_state, figdir)

    print(f'{len(rows)} attempt rows -> {out}/summary_long.csv')
    for dist in ('E', 'N'):
        for meth in METHOD_ORDER:
            sub = [r for r in deltas if r['dist'] == dist
                   and r['method'] == meth and r['checkpoint'] == 200]
            if not sub:
                continue
            d = np.mean([r['d_travel_time'] for r in sub])
            print(f"dist={dist} {meth:8s} mean Δtravel_time vs colight: {d:+.2f} "
                  f"(n={len(sub)})")


if __name__ == '__main__':
    main()
