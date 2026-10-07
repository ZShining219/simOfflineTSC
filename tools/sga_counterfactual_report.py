#!/usr/bin/env python3
"""Summarize ATT-ENTITY-002 counterfactual probe outputs.

Scans probe directories (counterfactual_records.jsonl produced by
tools/run_sga_counterfactual.py) and writes:

  counterfactual_variants.csv   per probe x variant: n, q_l1 mean/max,
                                action-change rate, gate means
  counterfactual_summary.json   same data keyed by probe dir name
  closed_loop_metrics.csv       per probe: episode traffic metrics of the
                                control variant that drove the sim, joined
                                from the run's tsc output dir via --run-root

Usage:
    python3 tools/sga_counterfactual_report.py \
        --probe-root data/output_data/analysis/sga_concat_colight_200_v1/counterfactual \
        --run-root data/output_data/tsc/sumo_sga_colight/hz4x4 \
        --out data/output_data/analysis/sga_concat_colight_200_v1/counterfactual
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

VARIANTS = ('correct', 'zero_z', 'zero_g', 'shuffle_z', 'shuffle_g',
            'wrong_location', 'remove_event_0', 'remove_event_1')


def summarize_records(path):
    recs = [json.loads(l) for l in path.open() if l.strip()]
    out = {'records': len(recs),
           'event_decisions': sum(1 for r in recs if r['event_count'] > 0)}
    for name in VARIANTS:
        qs, ac, ng, fg = [], [], [], []
        for r in recs:
            if r['event_count'] == 0:
                continue
            v = r.get('variants', {}).get(name)
            if v is None:
                continue
            qs.append(v['q_l1_vs_correct'])
            ac.append(v['action_changed_count'])
            ng.append(float(np.asarray(v['node_gate']).mean()))
            fg.append(float(np.asarray(v['feature_gate_mean']).mean()))
        if qs:
            out[name] = {
                'n': len(qs),
                'q_l1_mean': float(np.mean(qs)),
                'q_l1_max': float(np.max(qs)),
                'action_change_rate': float(np.mean([a > 0 for a in ac])),
                'action_changed_mean': float(np.mean(ac)),
                'node_gate_mean': float(np.mean(ng)),
                'feature_gate_mean': float(np.mean(fg)),
            }
    # The requested control variant comes from the probe dir name suffix;
    # per-record records fall back to 'correct' when a variant is inapplicable
    # (e.g. wrong_location needs a movable event), so recs[0] is unreliable.
    ctrl = recs[0].get('control_variant') if recs else None
    out['control_variant_first_record'] = ctrl
    from collections import Counter
    out['control_variant_used'] = dict(Counter(
        r.get('control_variant') for r in recs))
    return out


def episode_metrics(run_root, probe_name):
    """Find the run dir written by the probe (prefix sgacf_<probe>) and read
    its final episode metrics."""
    run_dir = Path(run_root) / f'sgacf_{probe_name}'
    f = run_dir / 'metrics' / 'records.jsonl'
    if not f.is_file():
        return {}
    for line in f.open():
        r = json.loads(line)
        if r.get('record_type') in ('FINAL_EVALUATION', 'TEST', 'EVALUATION'):
            return {k: r.get(k) for k in
                    ('travel_time', 'queue', 'throughput', 'delay',
                     'unfinished_vehicles', 'waiting_time')}
    return {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe-root', required=True)
    p.add_argument('--run-root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    var_rows, loop_rows = [], []
    for d in sorted(Path(args.probe_root).iterdir()):
        f = d / 'counterfactual_records.jsonl'
        if not f.is_file() or f.stat().st_size == 0:
            continue
        s = summarize_records(f)
        summary[d.name] = s
        for name in VARIANTS:
            if name in s:
                var_rows.append({'probe': d.name, 'variant': name, **s[name]})
        em = episode_metrics(args.run_root, d.name)
        if em:
            # requested control variant = dir-name suffix after the last '_t<seed>_'
            req = d.name.rsplit('_t', 1)[-1]
            req = req.split('_', 1)[-1] if '_' in req else 'correct'
            loop_rows.append({'probe': d.name,
                              'control_variant_requested': req,
                              'control_variant_used': json.dumps(
                                  s['control_variant_used']), **em})

    (out / 'counterfactual_summary.json').write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    if var_rows:
        with (out / 'counterfactual_variants.csv').open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(var_rows[0].keys()))
            w.writeheader()
            w.writerows(var_rows)
    if loop_rows:
        with (out / 'closed_loop_metrics.csv').open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(loop_rows[0].keys()))
            w.writeheader()
            w.writerows(loop_rows)
    print(f'{len(summary)} probes summarized -> {out}')
    for name in VARIANTS[1:]:
        vals = [s[name]['q_l1_mean'] for s in summary.values() if name in s]
        if vals:
            print(f'  {name:15s} q_l1_mean across {len(vals)} probes: '
                  f'mean={np.mean(vals):.4f} min={np.min(vals):.4f} max={np.max(vals):.4f}')


if __name__ == '__main__':
    main()
