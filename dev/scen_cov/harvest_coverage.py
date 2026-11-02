"""Harvest the scen_cov_v1 pilot: per-cell metrics, Coverage / Margin / Delta.

Spec: docs/experiments/2026-11-02_scen_cov_v1.md (P4 card, dsh/design@e9eae61).
Formula frozen in docs/scenario_taxonomy.md:

    cov(pi,c)   = 1[ATT(pi,c) <= 1.05*ATT_MP(c) and
                    unfinished(pi,c) <= 1.10*unfinished_MP(c)]
    Coverage(pi)= (1/K) sum_k cov(pi,c_k)
    Margin(pi)  = (1/K) sum_k clip((1.05*ATT_MP - ATT)/(1.05*ATT_MP), -1, 1)

K=4 for L1 (training-seen demand regimes) and separately for L2 (unseen,
exploratory; never mixed with L1).  Every arm's eval attempt is an eval-only
run (episodes: 0) that writes one FINAL_EVALUATION record; the maxpressure
baseline is re-collected with the same eval sumo seeds 101/102/103, 1 ep each.

Reads (all under --run-root/data/output_data/tsc):
  sumo_dqn/<net>/scv1_<arm>_o2_sd<k>_cov_<tag>_es<e>/  final coverage eval
  sumo_dqn/<net>/scv1_<arm>_o2_sd<k>_ev<j>_<tag>_es<e>/
  sumo_maxpressure/<net>/scv1_mp_<tag>_es<e>/           baseline

Writes: coverage_report.json + coverage_cells.csv (under --out-dir).

Usage:
    python dev/scen_cov/harvest_coverage.py --run-root ~/worktrees/dsh-ops
"""
import argparse
import csv
import json
from pathlib import Path

ORDER = ['sumohz1x1', 'sumohz1x1_config4',
         'sumohz1x1_config2', 'sumohz1x1_config3']
NET_TAG = {'sumohz1x1': 's1', 'sumohz1x1_config4': 'c4',
           'sumohz1x1_config2': 'c2', 'sumohz1x1_config3': 'c3'}
L2_NETS = {'sumohz1x1_sparse_ultra': 'sparse_ultra',
           'sumohz1x1_dense_extreme': 'dense_extreme',
           'sumohz1x1_lt_extreme': 'lt_extreme',
           'sumohz1x1_asym_wonly': 'asym_wonly'}
ALL_NETS = {**NET_TAG, **L2_NETS}
AGR = ['ddqn_clear', 'soddqn']
METRICS = ['travel_time', 'throughput', 'unfinished_vehicles', 'queue', 'delay']


def load_final_eval(run_dir):
    """Return the last FINAL_EVALUATION metric dict, or None if unavailable."""
    run_dir = Path(run_dir)
    path = run_dir / 'evaluation' / 'summary.json'
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            data = {}
        evals = [e for e in data.get('evaluations', [])
                 if e.get('record_type') == 'FINAL_EVALUATION']
        if not evals:
            evals = data.get('evaluations') or []
        if evals:
            record = evals[-1]
            return {m: record.get(m) for m in METRICS}
    path = run_dir / 'metrics' / 'records.jsonl'
    if path.is_file():
        found = None
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get('record_type') == 'FINAL_EVALUATION':
                    found = record
        if found is not None:
            return {m: found.get(m) for m in METRICS}
    return None


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', required=True)
    ap.add_argument('--out-dir', default=None,
                    help='default: <run-root>/artifacts/scen_cov_v1')
    ap.add_argument('--train-seeds', default='0,1')
    ap.add_argument('--eval-seeds', default='101,102,103')
    ap.add_argument('--delta', type=float, default=0.05,
                    help='ATT tolerance vs maxpressure (frozen: 0.05)')
    ap.add_argument('--eta', type=float, default=0.10,
                    help='unfinished tolerance vs maxpressure (frozen: 0.10)')
    ap.add_argument('--allow-missing', action='store_true')
    args = ap.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    out_dir = (Path(args.out_dir).expanduser().resolve() if args.out_dir
               else run_root / 'artifacts' / 'scen_cov_v1')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_root = run_root / 'data' / 'output_data' / 'tsc'
    train_seeds = [int(x) for x in args.train_seeds.split(',') if x.strip()]
    eval_seeds = [int(x) for x in args.eval_seeds.split(',') if x.strip()]
    missing = []

    def eval_dir(agent, net, run_id):
        return out_root / f'sumo_{agent}' / net / run_id

    # ---- maxpressure baseline: mean over eval seeds, per cell ----------
    mp = {}
    for net, tag in ALL_NETS.items():
        per_seed = {}
        for es in eval_seeds:
            rid = f'scv1_mp_{tag}_es{es}'
            got = load_final_eval(eval_dir('maxpressure', net, rid))
            if got is None:
                missing.append(f'mp/{tag}/es{es}')
                continue
            per_seed[es] = got
        if per_seed:
            mp[tag] = {
                'net': net,
                'layer': 'L1' if net in NET_TAG else 'L2',
                'per_seed': {str(k): v for k, v in per_seed.items()},
                'metrics': {m: mean([v.get(m) for v in per_seed.values()])
                            for m in METRICS},
                'n_eval_seeds': len(per_seed),
            }
        else:
            mp[tag] = {'net': net,
                       'layer': 'L1' if net in NET_TAG else 'L2',
                       'per_seed': {}, 'metrics': {}, 'n_eval_seeds': 0}

    # ---- arms: per cell x eval seed, final coverage ckpt ---------------
    cell_metrics = {}
    for arm in AGR:
        cell_metrics[arm] = {}
        for net, tag in ALL_NETS.items():
            per_seed = {}
            for k in train_seeds:
                rid_seed = {}
                for es in eval_seeds:
                    rid = f'scv1_{arm}_o2_sd{k}_cov_{tag}_es{es}'
                    got = load_final_eval(eval_dir('dqn', net, rid))
                    if got is None:
                        missing.append(f'{arm}/{tag}/sd{k}/es{es}')
                        continue
                    rid_seed[es] = got
                if rid_seed:
                    per_seed[k] = {
                        'per_eval_seed': {str(e): v
                                          for e, v in rid_seed.items()},
                        'metrics': {m: mean([v.get(m)
                                             for v in rid_seed.values()])
                                    for m in METRICS},
                    }
            cell_metrics[arm][tag] = per_seed

    # ---- coverage / margin per arm, per layer --------------------------
    def layer_tags(layer):
        return [tag for net, tag in ALL_NETS.items()
                if ('L1' if net in NET_TAG else 'L2') == layer]

    def cov(value, base):
        if value is None or base is None:
            return None
        if value.get('travel_time') is None or base.get('travel_time') is None:
            return None
        if (value.get('unfinished_vehicles') is None or
                base.get('unfinished_vehicles') is None):
            return None
        return int(value['travel_time'] <= (1 + args.delta) * base['travel_time']
                   and value['unfinished_vehicles'] <=
                   (1 + args.eta) * base['unfinished_vehicles'])

    def margin(value, base):
        if value is None or base is None:
            return None
        if value.get('travel_time') is None or base.get('travel_time') is None:
            return None
        ref = (1 + args.delta) * base['travel_time']
        if not ref:
            return None
        return max(-1.0, min(1.0, (ref - value['travel_time']) / ref))

    summary = {}
    cells_by_layer = {}
    for layer in ('L1', 'L2'):
        tags = layer_tags(layer)
        cells_by_layer[layer] = tags
        for arm in AGR:
            by_seed = {}
            for k in train_seeds:
                covs, margins = [], []
                for tag in tags:
                    mp_metrics = mp[tag]['metrics']
                    cell = cell_metrics[arm][tag].get(k)
                    got = cell['metrics'] if cell else {}
                    c = cov(got, mp_metrics)
                    m = margin(got, mp_metrics)
                    if c is not None:
                        covs.append(c)
                    if m is not None:
                        margins.append(m)
                by_seed[str(k)] = {
                    'coverage': mean(covs),
                    'margin': mean(margins),
                    'n_cells': len(covs),
                }
            # coverage per eval seed (mean over train seeds first)
            by_eval_seed = {}
            for es in eval_seeds:
                covs, margins = [], []
                for tag in tags:
                    base = mp[tag]['per_seed'].get(str(es))
                    vals = []
                    for k in train_seeds:
                        cell = cell_metrics[arm][tag].get(k, {})
                        got = cell.get('per_eval_seed', {}).get(str(es))
                        if got is not None and base is not None:
                            vals.append(got)
                    got = {m: mean([v.get(m) for v in vals])
                           for m in METRICS} if vals else None
                    c = cov(got, base)
                    m = margin(got, base)
                    if c is not None:
                        covs.append(c)
                    if m is not None:
                        margins.append(m)
                if covs:
                    by_eval_seed[str(es)] = {'coverage': mean(covs),
                                             'margin': mean(margins),
                                             'n_cells': len(covs)}
            per_seed_cov = [v['coverage'] for v in by_seed.values()
                            if v['coverage'] is not None]
            es_cov = [v['coverage'] for v in by_eval_seed.values()
                      if v['coverage'] is not None]
            summary.setdefault(layer, {})[arm] = {
                'coverage': mean(per_seed_cov),
                'margin': mean([v['margin'] for v in by_seed.values()
                                if v['margin'] is not None]),
                'by_train_seed': by_seed,
                'by_eval_seed': by_eval_seed,
                'eval_seed_range': (
                    max(es_cov) - min(es_cov) if es_cov else None),
                'K': len(tags),
            }

    # ---- per-cell table + direction consistency ------------------------
    per_cell = {}
    for net, tag in ALL_NETS.items():
        entry = {'net': net, 'layer': mp[tag]['layer'],
                 'mp': mp[tag]['metrics'], 'arms': {}}
        for arm in AGR:
            vals = [cell_metrics[arm][tag][k]['metrics']
                    for k in train_seeds
                    if k in cell_metrics[arm][tag]]
            entry['arms'][arm] = {m: mean([v.get(m) for v in vals])
                                  for m in METRICS}
            entry['arms'][arm]['by_train_seed'] = {
                str(k): cell_metrics[arm][tag][k]['metrics']
                for k in train_seeds if k in cell_metrics[arm][tag]}
        per_cell[tag] = entry

    def delta_for(layer):
        semi = summary[layer]['soddqn']
        online = summary[layer]['ddqn_clear']
        if semi['coverage'] is None or online['coverage'] is None:
            return None
        return semi['coverage'] - online['coverage']

    directions = []
    for tag in cells_by_layer['L1']:
        base = mp[tag]['metrics']
        semi = per_cell[tag]['arms']['soddqn']
        online = per_cell[tag]['arms']['ddqn_clear']
        c_semi = cov(semi, base)
        c_online = cov(online, base)
        if c_semi is None or c_online is None:
            continue
        directions.append({'cell': tag, 'semi': c_semi, 'online': c_online,
                           'delta': c_semi - c_online})
    consistent = sum(1 for d in directions if d['delta'] >= 0)

    report = {
        'schema_version': 1,
        'campaign': 'scen_cov_v1',
        'spec': 'docs/experiments/2026-11-02_scen_cov_v1.md',
        'formula': {'delta': args.delta, 'eta': args.eta,
                    'coverage': '1/K sum cov(pi,c)',
                    'cov': '1[ATT<=1.05*ATT_MP and '
                           'unfinished<=1.10*unfinished_MP]',
                    'margin': '1/K sum clip((1.05*ATT_MP-ATT)/'
                              '(1.05*ATT_MP),-1,1)'},
        'run_root': str(run_root),
        'train_seeds': train_seeds,
        'eval_seeds': eval_seeds,
        'cells': {'L1': cells_by_layer['L1'], 'L2': cells_by_layer['L2']},
        'mp_baseline': mp,
        'summary': summary,
        'delta': {
            'coverage_L1': delta_for('L1'),
            'margin_L1': (
                summary['L1']['soddqn']['margin'] -
                summary['L1']['ddqn_clear']['margin']
                if summary['L1']['soddqn']['margin'] is not None and
                summary['L1']['ddqn_clear']['margin'] is not None else None),
            'coverage_L2_exploratory': delta_for('L2'),
            'per_cell_L1_direction': directions,
            'direction_consistent_cells': consistent,
            'direction_consistent_total': len(directions),
            'direction_consistent_ratio': (
                consistent / len(directions) if directions else None),
        },
        'per_cell': per_cell,
        'missing': missing,
    }
    report_path = out_dir / 'coverage_report.json'
    report_path.write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')

    csv_path = out_dir / 'coverage_cells.csv'
    cols = ['tag', 'layer', 'net', 'metric', 'mp']
    for arm in AGR:
        cols += [f'{arm}_mean', f'{arm}_min', f'{arm}_max']
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        for net, tag in ALL_NETS.items():
            entry = per_cell[tag]
            for metric in METRICS:
                row = {'tag': tag, 'layer': entry['layer'],
                       'net': entry['net'], 'metric': metric,
                       'mp': entry['mp'].get(metric)}
                for arm in AGR:
                    vals = [v.get(metric) for k, v in
                            entry['arms'][arm]['by_train_seed'].items()]
                    vals = [v for v in vals if v is not None]
                    row[f'{arm}_mean'] = mean(vals)
                    row[f'{arm}_min'] = min(vals) if vals else None
                    row[f'{arm}_max'] = max(vals) if vals else None
                writer.writerow(row)

    print('== Coverage (frozen delta=%.2f eta=%.2f) ==' % (args.delta,
                                                           args.eta))
    for layer in ('L1', 'L2'):
        for arm in AGR:
            s = summary[layer][arm]
            print(f'  {layer} {arm:11s} coverage={s["coverage"]} '
                  f'margin={s["margin"]} eval_seed_range={s["eval_seed_range"]}')
    d = report['delta']
    print(f'  Delta coverage L1 = {d["coverage_L1"]}  '
          f'(target >= +0.25)')
    print(f'  Delta margin   L1 = {d["margin_L1"]}')
    print(f'  cell direction consistent L1 = '
          f'{d["direction_consistent_cells"]}/{d["direction_consistent_total"]}'
          '  (target >= 3/4)')
    print(f'  Delta coverage L2 (exploratory) = {d["coverage_L2_exploratory"]}')
    print(f'report -> {report_path}')
    print(f'cells  -> {csv_path}')
    if missing:
        print(f'MISSING attempts: {len(missing)} (first 8: {missing[:8]})')
        if not args.allow_missing:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
