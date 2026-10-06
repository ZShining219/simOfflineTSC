#!/usr/bin/env python3
"""Paired per-episode utility readout for the attention line (att004).

Why this exists
---------------
Stage-2 utility on the attention line is normally read off a *single* eval
episode (`evaluation/summary.json`, `evaluation_episodes: [0, N]`).  The SUMO
seed is a single run-wide value (world/world_sumo.py `--seed`), so the eval
episode is fully determined by that seed: re-running the same config at two
different eval episode indices yields bit-identical traffic.  The practical
resolution of the standard protocol is therefore n = (number of sumo seeds),
which is 3.

Meanwhile every run already writes `metrics/records.jsonl` with one TRAIN row
per plan episode carrying the full three headline indicators
(travel_time / throughput / unfinished_vehicles).  For a frozen policy
(`learning_start` >> episodes) those rows are plain rollouts of the plan's
per-episode event schedules.  Comparing two runs that differ only in
`TARL_TEXT_CONDITION` therefore yields a *paired* per-episode contrast whose n
is the number of plan episodes, with the same metric definitions.

This tool computes that contrast.  It changes no training, evaluation or metric
definition: it only aggregates existing per-episode records.

Usage
-----
    python3 tools/att004_paired_util.py \
        --pair seen=<canon_run_dir>:<empty_run_dir> \
        --pair t3=<canon_run_dir>:<empty_run_dir> \
        --dumps seen=<canon_dump_dir>:<empty_dump_dir> \
        --out data/output_data/analysis/att_entity_004/<family>/paired

Run dirs are `data/output_data/tsc/sumo_tarl_attention/hz4x4/<tag>`; dump dirs
are the `--dump_dir` outputs of tools/run_tarl_struct_cf.py.  Dumps are optional
and only enable the event-window stratified queue split.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


HEADLINE = ('travel_time', 'throughput', 'unfinished_vehicles')
AUX = ('queue', 'delay', 'waiting_time', 'reward_mean')
ALL_METRICS = HEADLINE + AUX


def _read_records(run_dir):
    path = Path(run_dir) / 'metrics' / 'records.jsonl'
    if not path.exists():
        raise SystemExit(f'missing records.jsonl: {path}')
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rows[(rec.get('record_type'), rec.get('episode'))] = rec
    return rows


def _episode_events(dump_dir):
    """Map episode index -> merged [begin, end] window of that episode's events."""
    path = Path(dump_dir) / 'event_plan_log.jsonl'
    if not path.exists():
        return None
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ep = row.get('episode')
        if ep is None or row.get('role') == 'eval':
            continue
        events = row.get('events') or []
        if not events:
            out[ep] = None
            continue
        out[ep] = (min(e['begin'] for e in events),
                   max(e['end'] for e in events))
    return out


def _dump_windows(dump_dir):
    """Map dump-file episode index -> event window seen by the policy (z_task).

    `None` means the dump exists but the policy received no event row for the
    whole episode (channel-invisible), which is distinct from "episode absent".
    """
    out = {}
    for path in sorted(Path(dump_dir).glob('ep*.npz')):
        index = int(path.stem[2:])
        data = np.load(path)
        z = np.asarray(data['z_task'])
        present = np.abs(z).sum(axis=(1, 2)) > 0
        if not present.any():
            out[index] = None
            continue
        t = np.asarray(data['t'], dtype=float)
        out[index] = (float(t[present].min()), float(t[present].max()))
    return out


def _align_offset(plan_windows, dump_windows):
    """Dump index = plan episode + offset.  Detect it instead of assuming.

    Runs open with an eval reset (`evaluation_episodes` starts at 0), so the
    first dumped episode is the eval scenario and plan row 0 lands at dump 1;
    the offset is detected from the event windows so a config change cannot
    silently mis-pair the two series.
    """
    best, best_hits = 0, -1
    for offset in (0, 1, 2, -1):
        hits = 0
        for ep, window in plan_windows.items():
            if ep + offset not in dump_windows:
                continue
            got = dump_windows[ep + offset]
            if window is None and got is None:
                hits += 1
            elif window is None or got is None:
                continue
            elif abs(got[0] - window[0]) <= 20:
                hits += 1
        if hits > best_hits:
            best, best_hits = offset, hits
    return best, best_hits


def _episode_queue(dump_dir, dump_index):
    """Per-step (t, queue_now) of one dumped episode, or None."""
    path = Path(dump_dir) / f'ep{dump_index:03d}.npz'
    if not path.exists():
        return None
    data = np.load(path)
    return np.asarray(data['t'], dtype=float), np.asarray(data['queue_now'],
                                                          dtype=float)


def _bootstrap_ci(diffs, n_boot=10000, seed=0):
    if len(diffs) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _sign_test_p(diffs):
    """Two-sided exact binomial sign test, ties dropped."""
    nz = diffs[diffs != 0]
    n = len(nz)
    if n == 0:
        return float('nan')
    k = int((nz > 0).sum())
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1))
    return float(min(1.0, 2.0 * tail / 2 ** n))


def _mde(diffs, power_z=0.8416, alpha_z=1.959964):
    """Minimum detectable paired difference at 80% power / two-sided 5%."""
    n = len(diffs)
    if n < 2:
        return float('nan')
    sd = float(diffs.std(ddof=1))
    return (alpha_z + power_z) * sd / math.sqrt(n)


def _summarise(diffs, label):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0:
        return {'label': label, 'n': 0}
    lo, hi = _bootstrap_ci(diffs)
    return {
        'label': label,
        'n': n,
        'mean': float(diffs.mean()),
        'sd': float(diffs.std(ddof=1)) if n > 1 else 0.0,
        'ci95_low': lo,
        'ci95_high': hi,
        'positive': int((diffs > 0).sum()),
        'negative': int((diffs < 0).sum()),
        'zero': int((diffs == 0).sum()),
        'sign_test_p': _sign_test_p(diffs),
        'mde_80': _mde(diffs),
    }


def _select(vals, keep):
    return {key: value for key, value in vals.items() if key[1] in keep}


def _window_split(canon_dump, empty_dump, episode_events, metrics):
    """Per-episode in-window / out-window mean queue differences (paired).

    `episode_events` is keyed by record/dump episode index (see main()).
    """
    win_d, out_d, ok_eps = {}, {}, []
    for ep in sorted(set(episode_events) & {k[1] for k in metrics}):
        window = episode_events[ep]
        if window is None:
            continue
        a = _episode_queue(canon_dump, ep)
        b = _episode_queue(empty_dump, ep)
        if a is None or b is None:
            continue
        (ta, qa), (tb, qb) = a, b
        if ta.shape != tb.shape or not np.allclose(ta, tb):
            continue
        lo, hi = window
        inside = (ta >= lo) & (ta <= hi)
        outside = ~inside
        if not inside.any() or not outside.any():
            continue
        win_d[ep] = float(qa[inside].mean() - qb[inside].mean())
        out_d[ep] = float(qa[outside].mean() - qb[outside].mean())
        ok_eps.append(ep)
    return win_d, out_d, ok_eps


def _fmt(summary):
    if summary.get('n', 0) == 0:
        return '| (no episodes) | 0 | | | | | | | |'
    return ('| {label} | {n} | {mean:+.4f} | {sd:.4f} | '
            '[{ci95_low:+.4f}, {ci95_high:+.4f}] | {positive}/{negative}/{zero} | '
            '{sign_test_p:.3f} | {mde_80:.4f} |').format(**summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pair', action='append', required=True,
                    metavar='NAME=CANON_RUN_DIR:EMPTY_RUN_DIR',
                    help='one canonical/empty run-dir pair; repeatable')
    ap.add_argument('--dumps', action='append', default=[],
                    metavar='NAME=CANON_DUMP_DIR:EMPTY_DUMP_DIR',
                    help='optional dump dirs enabling event-window queue split')
    ap.add_argument('--out', required=True, help='output directory')
    args = ap.parse_args()

    dumps = {}
    for spec in args.dumps:
        name, rest = spec.split('=', 1)
        canon, empty = rest.split(':')
        dumps[name] = (canon, empty)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = {'pairs': {}, 'dumps': {k: list(v) for k, v in dumps.items()}}
    csv_lines = ['pair,episode,has_event,' + ','.join(ALL_METRICS) +
                 ',win_queue_in,win_queue_out']
    md = ['# att004 paired canonical vs empty — per-episode utility readout', '',
          'Unit: one plan episode per run pair (`metrics/records.jsonl` TRAIN rows).',
          'Sign convention: `diff = canonical - empty`; positive = canonical worse '
          '(higher travel_time / queue) or better (higher throughput).', '']

    for spec in args.pair:
        name, rest = spec.split('=', 1)
        canon_dir, empty_dir = rest.split(':')
        a = _read_records(canon_dir)
        b = _read_records(empty_dir)

        episodes = sorted({ep for (kind, ep) in a if kind == 'TRAIN'} &
                          {ep for (kind, ep) in b if kind == 'TRAIN'})
        per_metric = {m: {} for m in ALL_METRICS}
        for ep in episodes:
            ra, rb = a[('TRAIN', ep)], b[('TRAIN', ep)]
            for m in ALL_METRICS:
                if m in ra and m in rb:
                    per_metric[m][('TRAIN', ep)] = float(ra[m]) - float(rb[m])

        episode_events, offset, align_hits = None, 0, None
        if name in dumps:
            plan_windows = _episode_events(dumps[name][0])
            dump_windows = _dump_windows(dumps[name][0])
            offset, align_hits = _align_offset(plan_windows, dump_windows)
            # The trainer counts the opening eval as episode 0, so plan row `p`
            # is record/dump episode `p + offset`; re-key onto record episodes.
            episode_events = {ep + offset: w for ep, w in plan_windows.items()}
            plan_ev = {ep for ep, w in episode_events.items() if w is not None}
            visible = {ep for ep, w in episode_events.items()
                       if w is not None and dump_windows.get(ep) is not None}
            invisible = plan_ev - visible
            no_event = {ep for ep, w in episode_events.items() if w is None}
        else:
            plan_ev = visible = invisible = no_event = set()

        subsets = {
            'all': set(episodes),
            'event_planned': plan_ev,
            'event_visible': visible,
            'event_invisible': invisible,
            'no_event': no_event,
        }
        entries = {key: {m: _summarise(list(_select(vals, keep).values()), m)
                         for m, vals in per_metric.items()}
                   for key, keep in subsets.items()}

        pair_out = {'canonical_dir': canon_dir, 'empty_dir': empty_dir,
                    'n_episodes': len(episodes),
                    'dump_offset': offset, 'alignment_hits': align_hits,
                    'alignment_total': len(episode_events) if episode_events else None,
                    'subset_sizes': {k: len(v) for k, v in subsets.items()},
                    'subsets': entries}

        win_in, win_out = {}, {}
        if name in dumps and episode_events is not None:
            canon_dump, empty_dump = dumps[name]
            win_in, win_out, ok = _window_split(canon_dump, empty_dump,
                                                episode_events,
                                                per_metric['queue'])
            pair_out['window'] = {
                'episodes': ok,
                'queue_in_window': _summarise(list(win_in.values()),
                                              'queue_in_window'),
                'queue_out_window': _summarise(list(win_out.values()),
                                               'queue_out_window'),
            }

        report['pairs'][name] = pair_out

        md += [f'## pair `{name}`', '',
               f'- canonical: `{canon_dir}`', f'- empty: `{empty_dir}`',
               f'- paired plan episodes: {len(episodes)}', '']
        if episode_events is not None:
            md += [f'- dump/plan episode offset: {offset:+d} '
                   f'({align_hits}/{len(episode_events)} event windows consistent)',
                   f'- subset sizes: {pair_out["subset_sizes"]}', '']
        for subset in ('all', 'event_planned', 'event_visible',
                       'event_invisible', 'no_event'):
            if not entries[subset] or subset not in subsets:
                continue
            if not subsets[subset]:
                continue
            md += [f'### {subset}', '',
                   '| metric | n | mean diff | sd | 95% CI (bootstrap) | +/-/0 | '
                   'sign p | MDE(80%) |', '|---|---|---|---|---|---|---|---|']
            for m in ALL_METRICS:
                if m in entries[subset]:
                    md.append(_fmt(entries[subset][m]))
            md.append('')
        if win_in:
            md += ['### event-window stratified queue (per-episode event window)', '',
                   '| metric | n | mean diff | sd | 95% CI (bootstrap) | +/-/0 | '
                   'sign p | MDE(80%) |', '|---|---|---|---|---|---|---|---|',
                   _fmt(pair_out['window']['queue_in_window']),
                   _fmt(pair_out['window']['queue_out_window']), '']

        for ep in episodes:
            has = ''
            if episode_events is not None and ep in episode_events:
                has = 'yes' if episode_events[ep] is not None else 'no'
            row = [name, str(ep), has]
            row += [f'{per_metric[m][("TRAIN", ep)]:+.6f}' for m in ALL_METRICS]
            row += [f'{win_in[ep]:+.6f}' if ep in win_in else '',
                    f'{win_out[ep]:+.6f}' if ep in win_out else '']
            csv_lines.append(','.join(row))

    (out / 'paired_util.md').write_text('\n'.join(md) + '\n')
    (out / 'paired_util.json').write_text(json.dumps(report, indent=1) + '\n')
    (out / 'paired_episodes.csv').write_text('\n'.join(csv_lines) + '\n')
    print(f'wrote {out}/paired_util.md, paired_util.json, paired_episodes.csv')


if __name__ == '__main__':
    main()
