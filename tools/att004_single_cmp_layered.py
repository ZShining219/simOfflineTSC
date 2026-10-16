#!/usr/bin/env python3
"""Layered metrics for the single-event comparison (ATT-ENTITY-004).

For every single_cmp cell (eval schedule = one lane_blockage 900-1200 s):
- queue: event-window mean / post-window mean (>1200 s) / full-episode mean
  (per-decision probe records, first 360 rows = eval episode 0)
- TT, delay, throughput, unfinished, phase_switches: run evaluation summary
- action flips (a3s only): count of steps where action != previous action,
  split window / post-window; baselines only have episode-level
  phase_switches (no per-step actions in the probe records).
Output: JSON + Markdown table under analysis/att_entity_004/single_cmp_layered/.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WIN = (900.0, 1200.0)
CELL_RE = re.compile(r'single_([a-z0-9]+)_((?:s\d+_)?(?:canonical|empty))?_?s*(\d+)$')


def parse_cell(name):
    # single_a3s_s7_canonical_ss7 / single_colight_s27_ss207 / single_fixedtime_ss7
    m = re.match(r'single_([a-z0-9]+?)_ss(\d+)$', name)
    if m:
        return m.group(1), None, int(m.group(2))
    m = re.match(r'single_([a-z0-9]+?)_s(\d+)_(canonical|empty)_ss(\d+)$', name)
    if m:
        return m.group(1), m.group(3), int(m.group(4))
    m = re.match(r'single_([a-z0-9]+?)_s(\d+)_ss(\d+)$', name)
    if m:
        return m.group(1), None, int(m.group(3))
    return None


def queue_layers(rows):
    q = np.asarray([r['queue_now'] for r in rows], dtype=np.float64)
    t = np.asarray([r['simulation_time'] for r in rows], dtype=np.float64)
    win = (t >= WIN[0]) & (t < WIN[1])
    post = t >= WIN[1]
    return {'queue_full': float(q.mean()),
            'queue_window': float(q[win].mean()),
            'queue_post': float(q[post].mean()),
            'queue_window_max': float(q[win].max()),
            'queue_post_max': float(q[post].max())}


def a3s_flips(rows):
    """Canonical action flips: per-step node-count flips, window vs post."""
    nf_win, nf_post = [], []
    prev = None
    for r in rows:
        a = np.asarray(r['variants']['canonical']['action']).reshape(-1)
        t = r['simulation_time']
        if prev is not None:
            n = int((a != prev).sum())
            if WIN[0] <= t < WIN[1]:
                nf_win.append(n)
            elif t >= WIN[1]:
                nf_post.append(n)
        prev = a
    return {'flips_window': float(np.mean(nf_win)) if nf_win else None,
            'flips_post': float(np.mean(nf_post)) if nf_post else None,
            'flips_window_total': int(sum(nf_win)),
            'flips_post_total': int(sum(nf_post))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',
                    default='data/output_data/analysis/att_entity_004/single_cmp')
    ap.add_argument('--tsc', default='data/output_data/tsc')
    ap.add_argument('--output',
                    default='data/output_data/analysis/att_entity_004/single_cmp_layered')
    args = ap.parse_args()
    base = (ROOT / args.root).resolve()
    out_dir = (ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        parsed = parse_cell(d.name)
        if not parsed:
            continue
        algo, cond, ss = parsed
        rec_file = (d / 'tarl_struct_cf_records.jsonl'
                    if algo == 'a3s' else d / 'baseline_probe_records.jsonl')
        if not rec_file.exists():
            continue
        rows = [json.loads(l) for l in rec_file.open()][:360]
        cell = {'cell': d.name, 'algo': algo, 'condition': cond or '-',
                'sumo_seed': ss, **queue_layers(rows)}
        if algo == 'a3s':
            cell.update(a3s_flips(rows))
        # run-side summary for TT / switches
        agent_dir = ('sumo_tarl_attention' if algo == 'a3s'
                     else f'sumo_{algo}')
        summ = (ROOT / args.tsc / agent_dir / 'hz4x4' / d.name
                / 'evaluation' / 'summary.json')
        e = None
        if summ.exists():
            e = json.load(summ.open())['evaluations'][0]
        else:
            mrec = (ROOT / args.tsc / agent_dir / 'hz4x4' / d.name
                    / 'metrics' / 'records.jsonl')
            if mrec.exists():
                e = json.loads(mrec.read_text().strip().splitlines()[-1])
        if e:
            cell.update({'TT': e.get('travel_time'),
                         'throughput': e.get('throughput'),
                         'unfinished': e.get('unfinished_vehicles'),
                         'phase_switches': e.get('phase_switches'),
                         'phase_switch_frequency': e.get('phase_switch_frequency')})
        cells.append(cell)

    # aggregate algo x condition
    agg = {}
    for c in cells:
        key = (c['algo'], c['condition'])
        agg.setdefault(key, []).append(c)
    table = []
    for (algo, cond), cs in sorted(agg.items()):
        row = {'algo': algo, 'condition': cond, 'n': len(cs)}
        for k in ('TT', 'queue_full', 'queue_window', 'queue_post',
                  'queue_window_max', 'queue_post_max', 'flips_window',
                  'flips_post', 'flips_window_total', 'flips_post_total',
                  'phase_switches', 'phase_switch_frequency'):
            vals = [c[k] for c in cs if c.get(k) is not None]
            if vals:
                row[k + '_mean'] = float(np.mean(vals))
                row[k + '_sd'] = float(np.std(vals))
        table.append(row)

    (out_dir / 'single_cmp_layered.json').write_text(json.dumps(
        {'window': WIN, 'cells': cells, 'table': table}, indent=1))

    lines = ['| algo | cond | n | TT | winQ | postQ | fullQ | winQmax | postQmax | flipsW | flipsP | phase_sw |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in table:
        def f(k, d=1):
            v = r.get(k + '_mean')
            return f'{v:.{d}f}±{r[k + "_sd"]:.{d}f}' if v is not None else '-'
        lines.append(
            f"| {r['algo']} | {r['condition']} | {r['n']} | {f('TT')} | {f('queue_window')} | "
            f"{f('queue_post')} | {f('queue_full')} | {f('queue_window_max')} | "
            f"{f('queue_post_max')} | {f('flips_window',0)} | {f('flips_post',0)} | {f('phase_switches',0)} |")
    (out_dir / 'single_cmp_layered.md').write_text(
        '# single_cmp layered metrics (event=lane_blockage 900-1200s)\n\n' + '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
