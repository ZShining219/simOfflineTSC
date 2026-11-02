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

L2 additionally reports a pre-registered guardrail (approved 2026-11-03):
uf_ok = uf <= max(1.10*uf_MP, uf_MP + 2), reported alongside the original
formula (both coverage/margin series); cells where the guard flips the
verdict are listed by name.  L1 keeps the frozen original formula.  The
report also carries cell-level eval-seed spread, delta_by_train_seed (for
the S5 resolution check), n_seeds/n_cells and a comparable flag.

Reads (all under --run-root/data/output_data/tsc):
  sumo_dqn/<net>/scv1_<arm>_o2_sd<k>_cov_<tag>_es<e>/  final coverage eval
  sumo_dqn/<net>/scv1_<arm>_o2_sd<k>_ev<j>_<tag>_es<e>/
  sumo_maxpressure/<net>/scv1_mp_<tag>_es<e>/           baseline

Writes: coverage_report.json + coverage_cells.csv (under --out-dir).

Usage:
    python dev/scen_cov/harvest_coverage.py --run-root <run-worktree>
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


def _pick(record):
    return {m: record.get(m) for m in METRICS} if record else None


def load_summary_final(run_dir):
    """Last FINAL_EVALUATION from evaluation/summary.json (own eval runs)."""
    path = Path(run_dir) / 'evaluation' / 'summary.json'
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return None
    evals = [e for e in data.get('evaluations', [])
             if e.get('record_type') == 'FINAL_EVALUATION']
    if not evals:
        evals = data.get('evaluations') or []
    return _pick(evals[-1]) if evals else None


def load_records_final(run_dir):
    """Last FINAL_EVALUATION from metrics/records.jsonl (maxpressure runs)."""
    path = Path(run_dir) / 'metrics' / 'records.jsonl'
    if not path.is_file():
        return None
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
    return _pick(found)


def load_final_eval(run_dir, prefer='summary'):
    """Read a run's final evaluation.

    P4 card r2 (dsh/design@23dfdfc) splits the two sources: our own eval
    attempts are read from evaluation/summary.json, maxpressure baselines
    from metrics/records.jsonl.  The other source stays as a fallback so a
    partially-written run degrades to "missing" instead of a crash.
    """
    first, second = ((load_summary_final, load_records_final)
                     if prefer == 'summary'
                     else (load_records_final, load_summary_final))
    return first(run_dir) or second(run_dir)


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
            got = load_final_eval(eval_dir('maxpressure', net, rid),
                                  prefer='records')
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
                    got = load_final_eval(eval_dir('dqn', net, rid),
                                          prefer='summary')
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

    # L2-only guardrail (approved 2026-11-03): a zero maxpressure
    # denominator must not turn a 1-vehicle difference into "not covered".
    # L1 keeps the frozen original formula bit for bit.
    def uf_limit(base_uf, guard):
        limit = (1 + args.eta) * base_uf
        if guard:
            limit = max(limit, base_uf + 2)
        return limit

    def cov(value, base, guard=False):
        if value is None or base is None:
            return None
        tv = value.get('travel_time')
        btv = base.get('travel_time')
        uf = value.get('unfinished_vehicles')
        buf = base.get('unfinished_vehicles')
        if None in (tv, btv, uf, buf):
            return None
        return int(tv <= (1 + args.delta) * btv and uf <= uf_limit(buf, guard))

    def margin(value, base):
        if value is None or base is None:
            return None
        if value.get('travel_time') is None or base.get('travel_time') is None:
            return None
        ref = (1 + args.delta) * base['travel_time']
        if not ref:
            return None
        return max(-1.0, min(1.0, (ref - value['travel_time']) / ref))

    def aggregate(layer, guard):
        tags = cells_by_layer[layer]
        out = {}
        for arm in AGR:
            by_seed = {}
            for k in train_seeds:
                covs, margins = [], []
                for tag in tags:
                    mp_metrics = mp[tag]['metrics']
                    cell = cell_metrics[arm][tag].get(k)
                    got = cell['metrics'] if cell else {}
                    c = cov(got, mp_metrics, guard)
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
                    got = ({m: mean([v.get(m) for v in vals])
                            for m in METRICS} if vals else None)
                    c = cov(got, base, guard)
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
            out[arm] = {
                'coverage': mean(per_seed_cov),
                'margin': mean([v['margin'] for v in by_seed.values()
                                if v['margin'] is not None]),
                'by_train_seed': by_seed,
                'by_eval_seed': by_eval_seed,
                'eval_seed_range': (
                    max(es_cov) - min(es_cov) if es_cov else None),
                'K': len(tags),
            }
        return out

    summary = {}
    cells_by_layer = {}
    for layer in ('L1', 'L2'):
        cells_by_layer[layer] = layer_tags(layer)
        summary[layer] = aggregate(layer, guard=False)
        if layer == 'L2':
            guarded = aggregate('L2', guard=True)
            for arm in AGR:
                summary['L2'][arm].update({
                    'coverage_guard': guarded[arm]['coverage'],
                    'margin_guard': guarded[arm]['margin'],
                    'by_train_seed_guard': guarded[arm]['by_train_seed'],
                    'by_eval_seed_guard': guarded[arm]['by_eval_seed'],
                })

    for layer in ('L1', 'L2'):
        for arm in AGR:
            summary[layer][arm]['n_seeds'] = sum(
                1 for v in summary[layer][arm]['by_train_seed'].values()
                if v.get('coverage') is not None)
            summary[layer][arm]['n_cells_per_seed'] = [
                v.get('n_cells') for v in
                summary[layer][arm]['by_train_seed'].values()]

    # ---- per-cell table (train-seed mean + eval-seed spread) -----------
    per_cell = {}
    for net, tag in ALL_NETS.items():
        entry = {'net': net, 'layer': mp[tag]['layer'],
                 'mp': mp[tag]['metrics'], 'arms': {}}
        for arm in AGR:
            vals = [cell_metrics[arm][tag][k]['metrics']
                    for k in train_seeds
                    if k in cell_metrics[arm][tag]]
            arm_entry = {m: mean([v.get(m) for v in vals]) for m in METRICS}
            arm_entry['by_train_seed'] = {
                str(k): cell_metrics[arm][tag][k]['metrics']
                for k in train_seeds if k in cell_metrics[arm][tag]}
            by_es = {}
            for es in eval_seeds:
                per_train = []
                for k in train_seeds:
                    cell = cell_metrics[arm][tag].get(k)
                    if not cell:
                        continue
                    got = cell.get('per_eval_seed', {}).get(str(es))
                    if got is not None:
                        per_train.append(got)
                if per_train:
                    by_es[str(es)] = {m: mean([v.get(m) for v in per_train])
                                      for m in METRICS}
            arm_entry['by_eval_seed'] = by_es
            es_range = {}
            for m in METRICS:
                series = [v.get(m) for v in by_es.values()
                          if v.get(m) is not None]
                es_range[m] = ({'min': min(series), 'max': max(series),
                                'range': max(series) - min(series)}
                               if len(series) > 1 else None)
            arm_entry['eval_seed_range'] = es_range
            base = mp[tag]['metrics']
            arm_entry['cov_raw'] = cov(arm_entry, base, False)
            if entry['layer'] == 'L2':
                arm_entry['cov_guard'] = cov(arm_entry, base, True)
            entry['arms'][arm] = arm_entry
        per_cell[tag] = entry

    # cells where the L2 guardrail actually flips the verdict (must be named)
    guard_effective = []
    for tag in cells_by_layer['L2']:
        for arm in AGR:
            a = per_cell[tag]['arms'][arm]
            if (a['cov_raw'] is not None and a.get('cov_guard') is not None
                    and a['cov_raw'] != a['cov_guard']):
                guard_effective.append({
                    'cell': tag, 'arm': arm,
                    'cov_raw': a['cov_raw'], 'cov_guard': a['cov_guard'],
                    'mp_unfinished': mp[tag]['metrics'].get(
                        'unfinished_vehicles'),
                })

    def delta_for(layer, key='coverage'):
        semi = summary[layer]['soddqn'].get(key)
        online = summary[layer]['ddqn_clear'].get(key)
        if semi is None or online is None:
            return None
        return semi - online

    def delta_by_train_seed(layer, key='coverage', guard=False):
        block = 'by_train_seed_guard' if guard else 'by_train_seed'
        out = {}
        for k in train_seeds:
            semi = summary[layer]['soddqn'][block].get(str(k), {}).get(key)
            online = summary[layer]['ddqn_clear'][block].get(str(k), {}).get(key)
            out[str(k)] = (semi - online
                           if semi is not None and online is not None else None)
        return out

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

    # S5 advisory: is |Delta| inside the same-arm train-seed spread?
    s5 = {'rule': 'S5: |Delta_coverage_L1| <= max same-arm train-seed '
                  'range -> insufficient resolution (advisory)'}
    if 'L1' in summary:
        arm_range = {}
        for arm in AGR:
            vals = [v['coverage'] for v in
                    summary['L1'][arm]['by_train_seed'].values()
                    if v['coverage'] is not None]
            arm_range[arm] = (max(vals) - min(vals)
                              if len(vals) > 1 else None)
        d_now = delta_for('L1')
        ranges = [r for r in arm_range.values() if r is not None]
        s5.update({
            'delta_by_train_seed': delta_by_train_seed('L1'),
            'delta_margin_by_train_seed': delta_by_train_seed('L1', 'margin'),
            'arm_seed_range': arm_range,
            'flag_s5_resolution_insufficient': bool(
                ranges and d_now is not None and abs(d_now) <= max(ranges)),
        })

    incomplete = False
    for layer in ('L1', 'L2'):
        for arm in AGR:
            k_expected = summary[layer][arm]['K']
            for blk in summary[layer][arm]['by_train_seed'].values():
                if blk.get('n_cells') != k_expected:
                    incomplete = True
    comparable = (not missing) and (not incomplete)

    report = {
        'schema_version': 2,
        'campaign': 'scen_cov_v1',
        'spec': 'docs/experiments/2026-11-02_scen_cov_v1.md',
        'formula': {
            'delta': args.delta, 'eta': args.eta,
            'coverage': '1/K sum cov(pi,c)',
            'cov': '1[ATT<=1.05*ATT_MP and '
                   'unfinished<=1.10*unfinished_MP]',
            'l2_guard': 'L2 only: uf_ok = uf <= max(1.10*uf_MP, uf_MP+2)',
            'margin': '1/K sum clip((1.05*ATT_MP-ATT)/'
                      '(1.05*ATT_MP),-1,1)',
        },
        'run_root': str(run_root),
        'train_seeds': train_seeds,
        'eval_seeds': eval_seeds,
        'comparable': comparable,
        'n_seeds': len(train_seeds),
        'n_eval_seeds': len(eval_seeds),
        'cells': {'L1': cells_by_layer['L1'], 'L2': cells_by_layer['L2']},
        'mp_baseline': mp,
        'summary': summary,
        'guard_effective_cells': guard_effective,
        'delta': {
            'coverage_L1': delta_for('L1'),
            'coverage_L1_by_train_seed': delta_by_train_seed('L1'),
            'margin_L1': delta_for('L1', 'margin'),
            'coverage_L2_exploratory': delta_for('L2'),
            'coverage_L2_guard': delta_for('L2', 'coverage_guard'),
            'margin_L2_guard': delta_for('L2', 'margin_guard'),
            'coverage_L2_guard_by_train_seed': delta_by_train_seed(
                'L2', guard=True),
            'per_cell_L1_direction': directions,
            'direction_consistent_cells': consistent,
            'direction_consistent_total': len(directions),
            'direction_consistent_ratio': (
                consistent / len(directions) if directions else None),
        },
        's5_advisory': s5,
        'per_cell': per_cell,
        'missing': missing,
    }
    report_path = out_dir / 'coverage_report.json'
    report_path.write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')

    csv_path = out_dir / 'coverage_cells.csv'
    cols = ['tag', 'layer', 'net', 'metric', 'mp']
    for arm in AGR:
        cols += [f'{arm}_mean', f'{arm}_min', f'{arm}_max',
                 f'{arm}_es_min', f'{arm}_es_max']
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
                    es_range = entry['arms'][arm]['eval_seed_range'].get(
                        metric) or {}
                    row[f'{arm}_es_min'] = es_range.get('min')
                    row[f'{arm}_es_max'] = es_range.get('max')
                writer.writerow(row)

    print('== Coverage (frozen delta=%.2f eta=%.2f) ==' % (args.delta,
                                                           args.eta))
    for layer in ('L1', 'L2'):
        for arm in AGR:
            s = summary[layer][arm]
            print(f'  {layer} {arm:11s} coverage={s["coverage"]} '
                  f'margin={s["margin"]} '
                  f'eval_seed_range={s["eval_seed_range"]}')
    print('  L2 guard (L2 only): '
          f'semi={summary["L2"]["soddqn"]["coverage_guard"]} '
          f'online={summary["L2"]["ddqn_clear"]["coverage_guard"]} '
          f'effective_cells={len(guard_effective)}')
    d = report['delta']
    print(f'  Delta coverage L1 = {d["coverage_L1"]}  '
          f'(target >= +0.25)')
    print(f'  Delta margin   L1 = {d["margin_L1"]}')
    print(f'  cell direction consistent L1 = '
          f'{d["direction_consistent_cells"]}/'
          f'{d["direction_consistent_total"]}  (target >= 3/4)')
    print(f'  Delta coverage L2 (exploratory, raw) = '
          f'{d["coverage_L2_exploratory"]}')
    print(f'  Delta coverage L2 (guard) = {d["coverage_L2_guard"]}')
    print(f'  S5 advisory flag = '
          f'{s5.get("flag_s5_resolution_insufficient")}')
    print(f'  comparable={comparable} missing={len(missing)}')
    print(f'report -> {report_path}')
    print(f'cells  -> {csv_path}')
    if missing:
        print(f'MISSING attempts: {len(missing)} (first 8: {missing[:8]})')
        if not args.allow_missing:
            raise SystemExit(2)

if __name__ == '__main__':
    main()
