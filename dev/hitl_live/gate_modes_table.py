"""Gate trigger-mode comparison table (HITL gate ablation, phase 1).

探索件（dev/hitl_live/，登记 dev/INDEX.md）。横向比较同一预算下
不同触发策略的 live 会话：五种模式各 10 seed 全新采集
（gate_modes/<mode>/，hitl 臂，同 inject/冷却；on_demand 预算 36
实际花 ~7，window 预算 7——各臂实际查询量对齐）。

每种模式汇报：
  queries/ep、takeover、agree、budget_spent
  takeover_yield  = takeovers / queries（问到点子上的比例）
  in_window       = surge 事件窗（300-600s）内触发的查询占比
  tt / thr / unfin = 终端三指标（配对口径下在线接管收益本来就小，
                    主要看过程量）

用法（repo 根，conda env: colight）：

    python dev/hitl_live/gate_modes_table.py \
        --out output_data/hitl_probe/gate_modes/gate_modes_table.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
WINDOW = (300.0, 600.0)

SOURCES = {
    'on_demand': _REPO / 'output_data/hitl_probe/gate_modes/on_demand',
    'random': _REPO / 'output_data/hitl_probe/gate_modes/random',
    'margin': _REPO / 'output_data/hitl_probe/gate_modes/margin',
    'periodic': _REPO / 'output_data/hitl_probe/gate_modes/periodic',
    'window': _REPO / 'output_data/hitl_probe/gate_modes/window',
}


def _query_times(session_dir):
    path = Path(session_dir) / 'interventions.jsonl'
    ts = []
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('kind') == 'gate_query':
                ts.append(float(row.get('t') or 0.0))
    return ts


def _session_dir(root, run):
    return Path(root) / 'sessions' / run['session_id']


def collect(root):
    runs_path = Path(root) / 'runs.json'
    if not runs_path.is_file():
        return []
    runs = json.loads(runs_path.read_text(encoding='utf-8'))
    rows = []
    for run in runs:
        ts = _query_times(_session_dir(root, run))
        run = dict(run)
        run['in_window_frac'] = (
            float(np.mean([WINDOW[0] <= t <= WINDOW[1] for t in ts]))
            if ts else None)
        rows.append(run)
    return rows


def summarize(mode, rows):
    def _m(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (
            None, None)

    queries, _ = _m('queries')
    takeovers, _ = _m('takeovers')
    agrees, _ = _m('agrees')
    spent, _ = _m('budget_spent')
    win, _ = _m('in_window_frac')
    tt, tt_sd = _m('travel_time')
    thr, _ = _m('throughput')
    unfin, _ = _m('unfinished')
    return {
        'mode': mode,
        'n_sessions': len(rows),
        'queries_per_ep': queries,
        'takeovers_per_ep': takeovers,
        'agrees_per_ep': agrees,
        'budget_spent': spent,
        'takeover_yield': (
            takeovers / queries if queries else None),
        'in_window_frac': win,
        'travel_time': tt,
        'travel_time_sd': tt_sd,
        'throughput': thr,
        'unfinished': unfin,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    table = []
    for mode, root in SOURCES.items():
        rows = collect(root)
        table.append(summarize(mode, rows))

    header = (f"{'mode':<10} {'n':>3} {'q/ep':>5} {'to/ep':>6} "
              f"{'agree':>6} {'spent':>6} {'yield':>6} {'in-win':>7} "
              f"{'tt':>7} {'thr':>6} {'unfin':>6}")
    print(header)
    print('-' * len(header))
    for r in table:
        def f(v, nd=1):
            return f'{v:.{nd}f}' if v is not None else '-'
        print(f"{r['mode']:<10} {r['n_sessions']:>3} "
              f"{f(r['queries_per_ep']):>5} {f(r['takeovers_per_ep']):>6} "
              f"{f(r['agrees_per_ep']):>6} {f(r['budget_spent']):>6} "
              f"{f(r['takeover_yield'], 2):>6} "
              f"{f(r['in_window_frac'], 2):>7} "
              f"{f(r['travel_time']):>7} {f(r['throughput'], 0):>6} "
              f"{f(r['unfinished']):>6}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {'window': WINDOW, 'sources': {k: str(v) for k, v in
                                           SOURCES.items()},
             'table': table}, ensure_ascii=False, indent=2),
            encoding='utf-8')
        print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
