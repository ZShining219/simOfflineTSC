#!/usr/bin/env python3
"""Aggregate ATT-ENTITY-003 stage-2 evaluation packages into tables/figures.

Differences vs tools/sga_eval_report.py (ENTITY-002):
  - method set: colight / sga (v1) / flx (SGA-FLX)
  - training dist: randE (random plan hz4x4_random_v1) / fixE (fixed hz4x4.yml)
  - eval conditions: randeval_v1 12 held-out random schedules (strata
    normal/single/multi) + shared fixed 'none'/'all' calibration conditions
  - fixed-E reference baselines come from ENTITY-002 eval packages
    (--baseline-root, methods colight_002 / sga_002), seeds matched by name.
  - event windows are parsed per condition YAML (randeval windows differ per
    condition); fixed eval conditions are read from configs/events/eval/.

Outputs in --out:
  summary_long.csv    one row per (run, ckpt, cond, eval_seed)
  paired_cells.csv    per matched cell deltas for each contrast
  paired_stats.csv    per contrast x stratum/condition rollups
  figures/*.png

Usage:
    python3 tools/flx_eval_report.py \
        --eval-root artifacts/att_entity_003/eval_v1 \
        --baseline-root artifacts/sga_concat_colight_200_v1/eval_v1 \
        --queue-state artifacts/att_entity_003/eval_v1/run_state/run_manifest.json \
        --train-state artifacts/att_entity_003/run_state_stage2/run_manifest.json \
        --out data/output_data/analysis/att_entity_003/stage2
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats as sstats

METRICS = ['travel_time', 'queue', 'throughput', 'unfinished_vehicles', 'delay']
WINDOW_FIELDS = ['before_queue_network_mean', 'during_queue_network_mean',
                 'after_queue_network_mean', 'during_throughput_interval',
                 'during_delay_network_weighted_mean']
DISPLAY = {'colight': 'colight', 'sga_colight': 'sga',
           'sga_flx_colight': 'flx'}
# stage-3 alignment arms map to method names like flx_al0.1 / flx_al1.0
for _lam in ('0.1', '1.0'):
    DISPLAY[f'flx_al{_lam}'] = f'flx_al{_lam}'
METHOD_ORDER = ['colight', 'sga', 'flx', 'flx_al0.1', 'flx_al1.0']
BASELINE_METHODS = {'colight': 'colight_002', 'sga_colight': 'sga_002'}
FIXED_CONDS = ['none', 'blockage', 'closure', 'rain',
               'blockage_closure', 'blockage_rain', 'closure_rain', 'all']
EVAL_DIR = Path('configs/events/eval')
RANDEVAL_DIR = Path('configs/events/eval_random/randeval_v1')

RE_RUN = re.compile(
    r'(colight|sga_colight|sga_flx_colight)_(?:text_)?'
    r'(event_random|event|normal)_200_s(\d+)$')
RE_ALIGN = re.compile(r'align_l(\d+\.?\d*)_s(\d+)$')
RE_MAIN_ALIGN = re.compile(r'flx_align_l(\d+\.?\d*)_event_random_200_s(\d+)$')


def parse_run(run_id):
    m = RE_RUN.match(run_id)
    if m:
        agent, dist, seed = m.group(1), m.group(2), int(m.group(3))
        d = {'event_random': 'randE', 'event': 'fixE', 'normal': 'N'}[dist]
        return {'agent': agent, 'dist': d, 'seed': seed}
    for pattern in (RE_MAIN_ALIGN, RE_ALIGN):
        m = pattern.match(run_id)
        if m:
            # stage-3 alignment arm trains on the random-E plan.
            return {'agent': f"flx_al{m.group(1)}", 'dist': 'randE',
                    'seed': int(m.group(2))}
    return None


def cond_stratum(cond):
    m = re.match(r'randeval_v1_\d+_(normal|single|multi)$', cond)
    if m:
        return m.group(1)
    return 'fixed'


def cond_window(cond):
    """Union [begin,end] of the condition's events, or None for 'none'."""
    if cond == 'none':
        return None
    if cond.startswith('randeval_v1_'):
        path = RANDEVAL_DIR / f'{cond}.yml'
    else:
        path = EVAL_DIR / f'{cond}.yml'
    if not path.is_file():
        return None
    events = yaml.safe_load(path.read_text()).get('events') or []
    if not events:
        return None
    return (float(min(e['begin'] for e in events)),
            float(max(e['end'] for e in events)))


_records_cache = {}


def window_means(records_path, controller_id, eval_seed, window):
    if window is None:
        return {}
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
        key = 'before' if t < window[0] else (
            'during' if t < window[1] else 'after')
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


def collect(eval_root, method_map, dist_map, source):
    rows = []
    pkg_root = Path(eval_root) / 'packages'
    if not pkg_root.is_dir():
        return rows
    for pkg in sorted(pkg_root.iterdir()):
        run_id = pkg.name
        meta = parse_run(run_id)
        if meta is None:
            continue
        if meta['agent'] not in method_map:
            continue
        if dist_map is not None and meta['dist'] not in dist_map:
            continue
        method = method_map[meta['agent']]
        dist = meta['dist'] if dist_map is None else dist_map[meta['dist']]
        _records_cache.clear()
        for row in csv.DictReader((pkg / 'summary.csv').open()):
            m = re.match(r'(.+)_ep(\d+)_([a-z_0-9]+)$', row['controller_id'])
            cond = m.group(3)
            rec = {'run_id': run_id, 'source': source,
                   'method': method, 'dist': dist, 'seed': meta['seed'],
                   'checkpoint': int(row['checkpoint_episode']),
                   'condition': cond, 'stratum': cond_stratum(cond),
                   'eval_seed': int(row['evaluation_seed']),
                   'isolation': row.get('isolation_check', '')}
            for f in METRICS:
                rec[f] = float(row[f])
            rec.update(window_means(pkg / 'records.jsonl',
                                    row['controller_id'],
                                    int(row['evaluation_seed']),
                                    cond_window(cond)))
            rows.append(rec)
    return rows


def paired_cells(rows):
    by_cell = defaultdict(dict)
    for r in rows:
        by_cell[(r['method'], r['dist'], int(r['seed']))][
            (r['checkpoint'], r['condition'], r['eval_seed'])] = r
    contrasts = [('flx', 'colight', 'randE'), ('sga', 'colight', 'randE'),
                 ('flx', 'sga', 'randE'),
                 ('flx', 'colight_002', 'fixE'),
                 ('sga_002', 'colight_002', 'fixE'),
                 ('flx', 'sga_002', 'fixE')]
    align_methods = sorted({k[0] for k in by_cell
                            if k[0].startswith('flx_al')})
    for am in align_methods:
        contrasts += [(am, 'flx', 'randE'), (am, 'colight', 'randE'),
                      (am, 'colight_002', 'fixE')]
    cells = []
    metrics = METRICS + WINDOW_FIELDS
    for a, b, dist in contrasts:
        seeds = sorted({k[2] for k in by_cell})
        for seed in seeds:
            ca = by_cell.get((a, dist, seed))
            cb = by_cell.get((b, dist, seed))
            if not ca or not cb:
                continue
            for key in sorted(set(ca) & set(cb)):
                ckpt, cond, eseed = key
                ra, rb = ca[key], cb[key]
                rec = {'contrast': f'{a}-{b}|{dist}', 'checkpoint': ckpt,
                       'condition': cond, 'stratum': ra['stratum'],
                       'eval_seed': eseed, 'seed': seed}
                for m in metrics:
                    va, vb = ra.get(m), rb.get(m)
                    if isinstance(va, float) and isinstance(vb, float):
                        rec[f'd_{m}'] = va - vb
                cells.append(rec)
    return cells


def paired_stats(cells, rows):
    colight_vals = defaultdict(list)
    for r in rows:
        if r['method'] not in ('colight', 'colight_002'):
            continue
        for m in METRICS:
            colight_vals[(r['dist'], r['checkpoint'], r['condition'], m)].append(
                r[m])
    colight_std = {k: float(np.std(v)) if len(v) > 1 else float('nan')
                   for k, v in colight_vals.items()}

    # rollup granularity: stratum for randeval conds, single 'fixed' group too
    agg = defaultdict(list)
    for r in cells:
        agg[(r['contrast'], r['checkpoint'], r['stratum'])].append(r)
        agg[(r['contrast'], r['checkpoint'],
             f"cond:{r['condition']}")].append(r)

    stat_rows = []
    for (contrast, ckpt, group), rs in sorted(agg.items()):
        dist = contrast.rsplit('|', 1)[-1]
        rec = {'contrast': contrast, 'checkpoint': ckpt, 'group': group,
               'n_cells': len(rs),
               'seeds': ','.join(str(s) for s in sorted({r['seed'] for r in rs}))}
        for m in METRICS:
            ds = [r[f'd_{m}'] for r in rs if f'd_{m}' in r]
            if not ds:
                continue
            # seed-level rollup: average each seed's cells first so n = #seeds
            seed_d = defaultdict(list)
            for r in rs:
                if f'd_{m}' in r:
                    seed_d[r['seed']].append(r[f'd_{m}'])
            seed_means = np.array([np.mean(v) for v in seed_d.values()])
            mean_d = float(np.mean(ds))
            rec[f'{m}_mean_delta'] = mean_d
            rec[f'{m}_delta_std'] = float(np.std(ds))
            rec[f'{m}_seed_mean'] = float(seed_means.mean())
            rec[f'{m}_seed_std'] = float(seed_means.std(ddof=1)) \
                if len(seed_means) > 1 else float('nan')
            rec[f'{m}_sign_consistency'] = float(max(
                np.mean([d > 0 for d in ds]), np.mean([d < 0 for d in ds])))
            if len(seed_means) >= 2:
                ci = sstats.t.interval(
                    0.95, len(seed_means) - 1, loc=seed_means.mean(),
                    scale=sstats.sem(seed_means))
                rec[f'{m}_seed_ci_lo'] = float(ci[0])
                rec[f'{m}_seed_ci_hi'] = float(ci[1])
            if len(seed_means) >= 5 and np.any(seed_means != 0):
                try:
                    rec[f'{m}_wilcoxon_p'] = float(
                        sstats.wilcoxon(seed_means).pvalue)
                except ValueError:
                    pass
            if group.startswith('cond:'):
                sstd = colight_std.get((dist, ckpt, group[5:], m),
                                       float('nan'))
            else:
                sstd = float('nan')
            rec[f'{m}_abs_over_seedstd'] = (
                abs(mean_d) / sstd if sstd == sstd and sstd > 0
                else float('nan'))
        stat_rows.append(rec)
    return stat_rows


def fig_strata_bars(rows, outdir):
    sub = [r for r in rows if r['dist'] == 'randE' and r['checkpoint'] == 200
           and r['stratum'] in ('normal', 'single', 'multi')]
    if not sub:
        return
    strata = ['normal', 'single', 'multi']
    for metric in ('travel_time', 'queue'):
        fig, ax = plt.subplots(figsize=(7.5, 4))
        w = 0.25
        for i, meth in enumerate(METHOD_ORDER):
            ys, es = [], []
            for s in strata:
                vals = [r[metric] for r in sub
                        if r['method'] == meth and r['stratum'] == s]
                ys.append(np.mean(vals) if vals else np.nan)
                es.append(np.std(vals) if vals else np.nan)
            ax.bar(np.arange(len(strata)) + i * w, ys, w * 0.9,
                   yerr=es, capsize=2, label=meth)
        ax.set_xticks(np.arange(len(strata)) + w)
        ax.set_xticklabels(strata)
        ax.set_title(f'randeval strata {metric} (randE, ep200)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f'strata_{metric}_randE.png', dpi=150)
        plt.close(fig)


def fig_fixed_bars(rows, outdir):
    sub = [r for r in rows if r['checkpoint'] == 200
           and r['condition'] in FIXED_CONDS]
    if not sub:
        return
    groups = [('fixE-trained', ('colight_002', 'sga_002', 'flx'),
               lambda r: r['dist'] == 'fixE'),
              ('randE-trained', ('colight', 'sga', 'flx'),
               lambda r: r['dist'] == 'randE')]
    for tag, methods, keep in groups:
        sub2 = [r for r in sub if keep(r)]
        if not sub2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        for ax, metric in zip(axes, ('travel_time', 'queue')):
            for i, meth in enumerate(methods):
                xs, ys, es = [], [], []
                for cond in FIXED_CONDS:
                    vals = [r[metric] for r in sub2
                            if r['method'] == meth and r['condition'] == cond]
                    if vals:
                        xs.append(cond); ys.append(np.mean(vals))
                        es.append(np.std(vals))
                ax.bar(np.arange(len(xs)) + i * 0.27, ys, 0.25,
                       yerr=es, capsize=2, label=meth)
            ax.set_xticks(np.arange(len(FIXED_CONDS)) + 0.27)
            ax.set_xticklabels(FIXED_CONDS, rotation=30, ha='right',
                             fontsize=8)
            ax.set_title(f'{metric} ({tag}, ep200)')
            ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f'fixed_bars_{tag.split("-")[0]}.png', dpi=150)
        plt.close(fig)


def fig_paired_deltas(cells, outdir):
    sub = [c for c in cells if c['checkpoint'] == 200
           and 'd_travel_time' in c]
    if not sub:
        return
    align_cts = sorted({c['contrast'] for c in sub
                        if c['contrast'].startswith('flx_al')})
    for dist, contrasts in (('randE', ('flx-colight|randE', 'sga-colight|randE',
                                       *align_cts)),
                            ('fixE', ('flx-colight_002|fixE',
                                      'sga_002-colight_002|fixE'))):
        fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(contrasts) * 4), 4.2))
        groups = [c for c in sub if c['contrast'] in contrasts]
        strata = sorted({c['stratum'] for c in groups})
        pos, labels, tickpos, ticklabels = 0, [], [], []
        for s in strata:
            for ct in contrasts:
                vals = [c['d_travel_time'] for c in groups
                        if c['stratum'] == s and c['contrast'] == ct]
                if not vals:
                    continue
                bp = ax.boxplot([vals], positions=[pos], widths=0.6,
                                showfliers=False)
                ax.scatter(np.random.normal(pos, 0.05, len(vals)), vals,
                           s=10, alpha=0.7)
                tickpos.append(pos)
                ticklabels.append(f"{ct.split('-')[0]}\n{s}")
                pos += 1
            pos += 0.5
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xticks(tickpos)
        ax.set_xticklabels(ticklabels, fontsize=7)
        ax.set_ylabel('Δtravel_time vs baseline (per seed×cond)')
        ax.set_title(f'paired deltas ({dist}, ep200)')
        fig.tight_layout()
        fig.savefig(outdir / f'paired_deltas_{dist}.png', dpi=150)
        plt.close(fig)


def fig_event_windows(rows, outdir):
    for dist in ('randE', 'fixE'):
        sub = [r for r in rows if r['dist'] == dist and r['checkpoint'] == 200
               and 'during_queue_network_mean' in r]
        if not sub:
            continue
        methods = [m for m in (*METHOD_ORDER, 'colight_002', 'sga_002')
                   if any(r['method'] == m for r in sub)]
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = ['before', 'during', 'after']
        w = 0.8 / len(methods)
        for i, meth in enumerate(methods):
            ys = []
            for w_ in labels:
                vals = [r[f'{w_}_queue_network_mean'] for r in sub
                        if r['method'] == meth
                        and f'{w_}_queue_network_mean' in r]
                ys.append(np.mean(vals) if vals else np.nan)
            ax.bar(np.arange(len(labels)) + i * w, ys, w * 0.9, label=meth)
        ax.set_xticks(np.arange(len(labels)) + w * len(methods) / 2)
        ax.set_xticklabels(labels)
        ax.set_title(f'mean queue by event window ({dist}, ep200)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f'window_queue_{dist}.png', dpi=150)
        plt.close(fig)


def fig_ckpt_trend(rows, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for dist in ('randE', 'fixE'):
        for meth in (*METHOD_ORDER, 'colight_002', 'sga_002'):
            xs, ys = [], []
            for ckpt in (100, 200):
                vals = [r['travel_time'] for r in rows
                        if r['dist'] == dist and r['method'] == meth
                        and r['checkpoint'] == ckpt
                        and r['stratum'] in ('single', 'multi', 'fixed')
                        and r['condition'] != 'none']
                if vals:
                    xs.append(ckpt); ys.append(np.mean(vals))
            if xs:
                ax.plot(xs, ys, marker='o', label=f'{meth}({dist})')
    ax.set_xlabel('checkpoint episode')
    ax.set_ylabel('mean travel_time (event conds)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / 'ckpt_trend.png', dpi=150)
    plt.close(fig)


def fig_training_curves(train_state, outdir):
    state = json.loads(Path(train_state).read_text())
    fig, axes = plt.subplots(1, len(METHOD_ORDER), figsize=(5 * len(METHOD_ORDER), 4), sharey=True)
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


def fig_counterfactual(probe_root, outdir, csv_path=None):
    """Probe deltas vs checkpoint, computed from inline variant q-values.

    Each probe dir is `<run>_ep<ckpt>/counterfactual_records.jsonl`; every
    record carries `variants` = {correct, wrong_location, zero_g} each with
    `q` (nodes x actions) and `actions` (per-node argmax)."""
    root = Path(probe_root)
    if not root.is_dir():
        return
    # (dist, variant, ckpt, scope, seed) -> list of per-step deltas
    agg = defaultdict(list)
    for f in root.glob('*_ep*/counterfactual_records.jsonl'):
        dirname = f.parent.name
        run_id, ep = dirname.rsplit('_ep', 1)
        meta = parse_run(run_id)
        if meta is None or not str(DISPLAY.get(meta['agent'], '')).startswith('flx'):
            continue
        ckpt = int(ep)
        for line in f.open():
            if not line.strip():
                continue
            r = json.loads(line)
            variants = r.get('variants') or {}
            base = variants.get('correct')
            if not base or 'q' not in base:
                continue
            qc = np.asarray(base['q'])
            ac = np.asarray(base.get('actions', qc.argmax(-1)))
            scope = 'event_steps' if r.get('event_count', 0) > 0 else 'all'
            for var in ('wrong_location', 'zero_g'):
                v = variants.get(var)
                if not v:
                    continue
                qv = np.asarray(v['q'])
                av = np.asarray(v.get('actions', qv.argmax(-1)))
                ql1 = float(np.abs(qv - qc).mean())
                ach = float(np.mean(av != ac))
                for sc in (scope, 'all'):
                    agg[(meta['dist'], var, ckpt, sc,
                         meta['seed'])].append((ql1, ach))
    if not agg:
        return
    # seed-mean then dist-level mean/sd over seeds; also export a table
    table = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, mi, metric in zip(axes, (0, 1), ('q_l1', 'action_change_rate')):
        for dist in ('randE', 'fixE'):
            for var in ('wrong_location', 'zero_g'):
                for scope in ('event_steps',):
                    xs, ys, es = [], [], []
                    for ckpt in (50, 100, 150, 200):
                        seed_means = []
                        seeds = {k[4] for k in agg
                                 if k[0] == dist and k[1] == var
                                 and k[2] == ckpt and k[3] == scope}
                        for s in seeds:
                            vals = agg[(dist, var, ckpt, scope, s)]
                            if vals:
                                seed_means.append(np.mean(
                                    [v[mi] for v in vals]))
                        if seed_means:
                            xs.append(ckpt)
                            ys.append(np.mean(seed_means))
                            es.append(np.std(seed_means)
                                      if len(seed_means) > 1 else 0)
                            table.append({
                                'dist': dist, 'variant': var,
                                'scope': scope, 'checkpoint': ckpt,
                                'n_seeds': len(seed_means),
                                f'{metric}_mean': np.mean(seed_means),
                                f'{metric}_sd': np.std(seed_means)
                                if len(seed_means) > 1 else 0.0})
                    if xs:
                        ax.errorbar(xs, ys, yerr=es, marker='o', ms=3,
                                    capsize=2,
                                    label=f'{dist}:{var} (event steps)')
        ax.set_xlabel('checkpoint episode'); ax.set_title(metric)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / 'counterfactual_trajectories.png', dpi=150)
    plt.close(fig)
    if csv_path is not None:
        merged = {}
        for r in table:
            k = (r['dist'], r['variant'], r['scope'], r['checkpoint'])
            merged.setdefault(k, dict(r)).update(
                {kk: vv for kk, vv in r.items() if kk not in (
                    'dist', 'variant', 'scope', 'checkpoint')})
        rows = list(merged.values())
        with Path(csv_path).open('w', newline='') as f:
            w = csv.DictWriter(f, sorted({k for r in rows for k in r}))
            w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-root', required=True)
    p.add_argument('--baseline-root', default=None,
                   help='ENTITY-002 eval root for fixed-E reference methods')
    p.add_argument('--queue-state', required=True)
    p.add_argument('--train-state', default=None)
    p.add_argument('--probe-root',
                   default='data/output_data/analysis/att_entity_003/stage2/counterfactual')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out); figdir = out / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)

    rows = collect(args.eval_root, DISPLAY, None, 'e003')
    if args.baseline_root:
        rows += collect(args.baseline_root, BASELINE_METHODS,
                        {'fixE': 'fixE'}, 'e002')
    if not rows:
        print('no evaluation packages found'); return

    fields = sorted({k for r in rows for k in r})
    with (out / 'summary_long.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)

    cells = paired_cells(rows)
    with (out / 'paired_cells.csv').open('w', newline='') as f:
        fields = sorted({k for r in cells for k in r})
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(cells)

    stat_rows = paired_stats(cells, rows)
    with (out / 'paired_stats.csv').open('w', newline='') as f:
        fields = sorted({k for r in stat_rows for k in r})
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(stat_rows)

    fig_strata_bars(rows, figdir)
    fig_fixed_bars(rows, figdir)
    fig_paired_deltas(cells, figdir)
    fig_event_windows(rows, figdir)
    fig_ckpt_trend(rows, figdir)
    if args.train_state:
        fig_training_curves(args.train_state, figdir)
    fig_counterfactual(args.probe_root, figdir,
                       csv_path=out / 'counterfactual_summary.csv')

    print(f'{len(rows)} attempt rows -> {out}/summary_long.csv')
    print(f'{len(cells)} paired cells -> {out}/paired_cells.csv')
    print('\n=== ep200 strata rollups (seed-mean travel_time delta) ===')
    for r in stat_rows:
        if r['checkpoint'] == 200 and r['group'] in (
                'normal', 'single', 'multi', 'fixed') \
                and 'travel_time_seed_mean' in r:
            print(f"{r['contrast']:24s} {r['group']:8s} n={r['n_cells']:3d} "
                  f"seeds[{r['seeds']:15s}] "
                  f"d={r['travel_time_seed_mean']:+7.2f} "
                  f"sd={r.get('travel_time_seed_std', float('nan')):6.2f} "
                  f"ci=[{r.get('travel_time_seed_ci_lo', float('nan')):+7.2f},"
                  f"{r.get('travel_time_seed_ci_hi', float('nan')):+7.2f}] "
                  f"sign={r['travel_time_sign_consistency']:.2f}")


if __name__ == '__main__':
    main()
