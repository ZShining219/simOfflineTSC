"""r3 multi-scene summary: per-scene paired bootstrap + live probe table.

探索件（hitl/figs/，登记 dev/INDEX.md）。两个产出：

1. 逐场景配对统计：对每个场景根调 paired_bootstrap.collect，
   各臂 vs 该场景自己的 HITL_C0（同 seed 配对，10000 重采样）；
2. 论文式终局表：场景×臂的 AULC Δ(travel_time/throughput/unfinished)
   + live 会话层三臂（student/hitl/expert）终端指标表。

用法（repo 根，conda env: colight）：

    python hitl/figs/multiscene_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from hitl.probe import paired_bootstrap as pb  # noqa: E402

ROOT = _REPO_ROOT / 'output_data/hitl_probe'
PLAN = ROOT / 'pilot5/plan_r3.json'
SCENES = {'qc': 'qc-yn (1417veh)', 'kn': 'kn-hz (743veh)',
          'sb': 'sb-sx (1671veh)'}
ARMS = ['HITL_APIQB', 'HITL_B2', 'HITL_EXPB']
METRICS = ['travel_time', 'throughput', 'unfinished_vehicles']
NAMES = {'HITL_APIQB': 'A:demo-pretrain', 'HITL_B2': 'B:RM-shape(0-shot)',
         'HITL_EXPB': 'C:expert-pool'}


def live_table():
    print('== live paired sessions (terminal metrics, mean of 10 seeds) ==')
    print(f'{"scene":18s} {"arm":8s} {"tt":>7s} {"thr":>6s} {"unfin":>6s} '
          f'{"queries":>7s} {"tookover":>8s}')
    for tag, name in SCENES.items():
        runs = json.loads((ROOT / f'probe_{tag}/runs.json').read_text())
        by = {}
        for r in runs:
            by.setdefault(r['arm'], []).append(r)
        for arm in ('student', 'hitl', 'expert'):
            rows = by.get(arm, [])
            if not rows:
                continue
            print(f'{name:18s} {arm:8s} '
                  f"{np.mean([x['travel_time'] for x in rows]):7.1f} "
                  f"{np.mean([x['throughput'] for x in rows]):6.0f} "
                  f"{np.mean([x['unfinished'] for x in rows]):6.1f} "
                  f"{np.mean([x['queries'] for x in rows]):7.1f} "
                  f"{np.mean([x['takeovers'] for x in rows]):8.1f}")


def scene_stats():
    out = {}
    rows_all = pb.collect(str(ROOT / 'multiscene'), str(PLAN))
    for tag in SCENES:
        rows = [r for r in rows_all if f'-{tag.upper()}-' in r['run_id']]
        conds = {}
        for metric in METRICS:
            deltas = pb.paired_deltas(
                rows, metric, dpe=120)
            for cond, dd in deltas.items():
                if cond == 'HITL_C0':
                    continue
                b = pb.bootstrap(
                    {s: d['aulc_delta'] for s, d in dd.items()},
                    10000, 11)
                conds.setdefault(cond, {})[metric] = b
        out[tag] = conds
    return out


def main():
    live_table()
    stats = scene_stats()
    print()
    print('== trained-arm AULC delta vs scene-local C0 '
          '(paired bootstrap, 95% CI) ==')
    print(f'{"scene":18s} {"arm":20s} {"tt Δ":>22s} {"thr Δ":>22s} '
          f'{"unfin Δ":>22s}')
    for tag, name in SCENES.items():
        for arm in ARMS:
            cells = []
            for m in METRICS:
                b = (stats.get(tag, {}).get(arm, {}) or {}).get(m)
                if not b:
                    cells.append(f'{"n/a":>22s}')
                    continue
                star = '*' if (b['ci95'][0] > 0 or b['ci95'][1] < 0) else ' '
                cells.append(f"{b['mean_delta']:+7.1f} "
                             f"[{b['ci95'][0]:+6.1f},{b['ci95'][1]:+6.1f}]"
                             f'{star}')
            print(f'{name:18s} {NAMES[arm]:20s} ' + ' '.join(cells))
    out = {t: s for t, s in stats.items()}
    path = ROOT / 'multiscene/summary_r3.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
