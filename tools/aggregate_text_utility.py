"""Aggregate closed-loop text utility runs (Test 3).

Per utility run the per-decision records sit in the --output dir
(text_utility_records.jsonl) while the episode metrics stream sits in the
normal run output dir named by --prefix (data/output_data/tsc/<agent_dir>/
<network>/<prefix>/metrics/records.jsonl).

Usage:
  python tools/aggregate_text_utility.py \
      --run-glob 'data/output_data/tsc/sumo_tarl_*/hz4x4/tarlutil_*' \
      --records-root artifacts/att_entity_003/text_utility
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


def episode_metrics(run_dir):
    rec = Path(run_dir) / 'metrics' / 'records.jsonl'
    if not rec.exists():
        return {}
    eps = {}
    for line in rec.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        ep = r.get('episode')
        if ep is None:
            continue
        cur = eps.setdefault(ep, {})
        for k in ('travel_time', 'real_delay', 'throughput', 'queue', 'delay',
                  'reward_sum', 'reward_mean', 'unfinished_vehicles'):
            if k in r and r[k] is not None:
                cur[k] = r[k]
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-glob', required=True)
    ap.add_argument('--records-root', required=True,
                    help='root of <arm>_s<seed>/<variant>/text_utility_records.jsonl')
    args = ap.parse_args()
    rows = []
    for run_dir in sorted(glob.glob(args.run_glob)):
        rd = Path(run_dir)
        prefix = rd.name
        # prefix convention: tarlutil_<arm><seed>_<variant>, arm in {att,gatg}
        m = re.match(r'tarlutil_([a-z]+)(\d+)_([a-z]+)$', prefix)
        recs_dir = (Path(args.records_root) / f'{m.group(1)}_s{m.group(2)}'
                    / m.group(3)) if m else None
        sens_f = ([recs_dir / 'text_utility_records.jsonl']
                  if recs_dir and (recs_dir / 'text_utility_records.jsonl').exists()
                  else list(rd.glob('text_utility_records.jsonl')))
        eps = episode_metrics(rd)
        sens = {}
        if sens_f:
            recs = [json.loads(l) for l in sens_f[0].open()]
            act = [r for r in recs if r['canonical_text']]
            sens = {
                'chg': float(np.mean([r['action_changed_vs_canonical']
                                      for r in recs])),
                'chg_event': float(np.mean([r['action_changed_vs_canonical']
                                            for r in act])) if act else 0.0,
                'q_l1': float(np.mean([r['q_l1_executed_vs_canonical']
                                       for r in recs])),
            }
        for ep, m in sorted(eps.items()):
            rows.append({'prefix': prefix, 'episode': ep, **m, **sens})
    for r in rows:
        print(f"{r['prefix']:28s} ep{r['episode']:<3} "
              f"TT={r.get('travel_time', float('nan')):7.2f} "
              f"tp={r.get('throughput', float('nan')):6.0f} "
              f"rew={r.get('reward_sum', float('nan')):9.0f} "
              f"unf={r.get('unfinished_vehicles', float('nan')):4.0f} "
              f"act_chg={r.get('chg', 0):.0f} q_l1={r.get('q_l1', 0):.2f}")
    return rows


if __name__ == '__main__':
    main()
