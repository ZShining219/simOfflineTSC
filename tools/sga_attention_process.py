#!/usr/bin/env python3
"""Aggregate SGA/Concat per-decision scene diagnostics (sga_attention.jsonl).

Scans evaluation package attempts, splits decisions into before/during/after
the frozen event window, and summarizes whether the text/entity pathway is
actually used during control:

  attention_process.csv   one row per attempt x window with aggregates
  attention_summary.json  per (run_id, condition) rollup

Key quantities (SGA): event attention mass per node, share of that mass on
grounded (node,event) pairs, node/feature gate means.  Concat packages log
fusion=masked_mean_concat with scene_norm statistics instead of attention.

Usage:
    python3 tools/sga_attention_process.py \
        --eval-root artifacts/sga_concat_colight_200_v1/eval_v1 \
        --out data/output_data/analysis/sga_concat_colight_200_v1/attention
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

EVENT_WINDOW = (900.0, 1200.0)
ATTEMPT_RE = re.compile(
    r'(?P<run>.+)_ep(?P<ckpt>\d+)_(?P<cond>[a-z_]+)__eval_seed_(?P<seed>\d+)$')


def window_of(t):
    if t < EVENT_WINDOW[0]:
        return 'before'
    if t < EVENT_WINDOW[1]:
        return 'during'
    return 'after'


def aggregate_attempt(path):
    """Return per-window aggregates for one attempt's sga_attention.jsonl."""
    buckets = {'before': [], 'during': [], 'after': []}
    fusion = None
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        fusion = r.get('fusion', fusion)
        w = window_of(float(r['simulation_time']))
        entry = {'event_count': r['event_count'],
                 'scene_state': r.get('scene_state', ''),
                 'event_ids': ','.join(r.get('event_ids', []))}
        att = np.asarray(r.get('attention', [[]]), dtype=float)
        mask = np.asarray(r.get('direct_grounding_mask', [[]]), dtype=float)
        if att.ndim == 3 and att.shape[-1] > 0:
            # attention: [B][N][M]; direct_grounding_mask / node_event_relation
            # are stored per-record without batch dim: [N][M] and [N][M][3].
            att = att[0]
            entry['att_mass_mean'] = float(att.sum(-1).mean())
            grounded_mass = float((att * mask).sum()) if mask.shape == att.shape else 0.0
            entry['att_mass_on_grounded'] = grounded_mass
            entry['att_grounded_share'] = (
                grounded_mass / float(att.sum())) if att.sum() > 0 else None
            # attention on nodes that host at least one grounded event
            grounded_nodes = mask.sum(-1) > 0 if mask.ndim == 2 else None
            if grounded_nodes is not None and grounded_nodes.any():
                entry['att_mass_grounded_nodes'] = float(
                    att[grounded_nodes].sum(-1).mean())
            # per-event attention mass keyed by event_id (column order follows
            # the record's event_ids, i.e. the frozen schedule order)
            for col, ev_id in enumerate(r.get('event_ids', [])):
                if col < att.shape[1]:
                    entry[f'ev_att__{ev_id}'] = float(att[:, col].sum())
        if 'node_gate' in r:
            entry['node_gate_mean'] = float(np.asarray(r['node_gate']).mean())
        if 'feature_gate_mean' in r:
            entry['feature_gate_mean'] = float(
                np.asarray(r['feature_gate_mean']).mean())
        for stat in ('semantic_score', 'direct_bias', 'relation_bias', 'relevance'):
            s = r.get(stat + '_stats')
            if s:
                entry[stat + '_mean'] = s['mean']
        if 'scene_norm_mean' in r:
            entry['scene_norm_mean'] = r['scene_norm_mean']
            entry['scene_norm_max'] = r['scene_norm_max']
        buckets[w].append(entry)

    out = {'fusion': fusion or 'sga_attention'}
    for w, rows in buckets.items():
        if not rows:
            continue
        ev = [e for e in rows if e['event_count'] > 0]
        out[f'{w}_decisions'] = len(rows)
        out[f'{w}_event_decisions'] = len(ev)
        for src, suffix in ((rows, ''), (ev, '_ev')):
            fields = {f for e in src for f in e
                      if f not in ('scene_state', 'event_ids')
                      and isinstance(e.get(f), (int, float))}
            for field in sorted(fields):
                vals = [e[field] for e in src if e.get(field) is not None]
                if vals:
                    out[f'{w}{suffix}_{field}'] = float(np.mean(vals))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-root', required=True, nargs='+')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for root in args.eval_root:
        root = Path(root)
        for pkg in sorted((root / 'packages').iterdir()):
            for att_dir in sorted((pkg / 'attempts').iterdir()):
                m = ATTEMPT_RE.match(att_dir.name)
                f = att_dir / 'sga_attention.jsonl'
                if not m or not f.is_file():
                    continue
                agg = aggregate_attempt(f)
                agg.update({'run_id': m.group('run'),
                            'checkpoint': int(m.group('ckpt')),
                            'condition': m.group('cond'),
                            'eval_seed': int(m.group('seed')),
                            'eval_root': root.name})
                rows.append(agg)

    if not rows:
        print('no sga_attention.jsonl found')
        return
    fields = sorted({k for r in rows for k in r})
    with (out / 'attention_process.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # rollup: per run x condition x window(ev decisions only)
    summary = {}
    for r in rows:
        key = f"{r['run_id']}|{r['condition']}|ep{r['checkpoint']}"
        s = summary.setdefault(key, {'n_attempts': 0})
        s['n_attempts'] += 1
        for w in ('before', 'during', 'after'):
            for field in ('att_mass_mean', 'att_grounded_share',
                          'node_gate_mean', 'feature_gate_mean',
                          'scene_norm_mean'):
                v = r.get(f'{w}_ev_{field}')
                if v is not None:
                    s.setdefault(f'{w}_ev_{field}', []).append(v)
    flat = {}
    for key, s in summary.items():
        flat[key] = {k: (float(np.mean(v)) if isinstance(v, list) else v)
                     for k, v in s.items()}
    (out / 'attention_summary.json').write_text(
        json.dumps(flat, indent=1, ensure_ascii=False))
    print(f'{len(rows)} attempts -> {out}/attention_process.csv')
    print(f'{len(flat)} run-condition rollups -> attention_summary.json')


if __name__ == '__main__':
    main()
