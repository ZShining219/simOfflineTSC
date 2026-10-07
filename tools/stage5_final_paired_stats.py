#!/usr/bin/env python3
"""ATT-ENTITY-003 阶段五收尾：TARL 矩阵 5 种子配对统计（三指标口径）。

读取 eval_stage5_tarl + eval_stage4(mplight) + eval_v1(colight) 的 frozen-eval
summary.csv，按 (family, condition, training_seed) 配对，对 tarl_sensor 做
配对差；Wilcoxon 符号秩在条件×种子池上做（randeval12 / 锚点2 / 全部14 三种范围）。

用法：
    python3 tools/stage5_final_paired_stats.py \
        --outdir data/output_data/analysis/att_entity_003/stage5_tarl
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_eval_aggregate import infer_family, condition_key

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('travel_time', 'throughput', 'unfinished_vehicles')
BASELINE = 'tarl_sensor'
RANDEVAL_N = 12  # randeval_v1_00..11；none/all 为锚点


def load_rows():
    """返回 [{family, seed, condition, metric...}]，仅 eval_seed=400007、ep200。"""
    roots = {
        ROOT / 'artifacts/att_entity_003/eval_stage5_tarl/packages': None,
        ROOT / 'artifacts/att_entity_003/eval_stage4/packages': {'mplight'},
        ROOT / 'artifacts/att_entity_003/eval_v1/packages': {'colight'},
    }
    rows = []
    for root, allow in roots.items():
        for f in sorted(root.glob('*/summary.csv')):
            run_id = f.parent.name
            family = infer_family(run_id)
            if allow and family not in allow:
                continue
            for r in csv.DictReader(open(f, newline='')):
                if str(r.get('evaluation_seed')) != '400007':
                    continue
                if str(r.get('checkpoint_episode')) != '200':
                    continue
                rec = {'family': family, 'seed': r['training_seed'],
                       'condition': condition_key(r), 'run_id': run_id}
                for m in METRICS:
                    rec[m] = float(r[m]) if r.get(m) else None
                rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    # cell[(family, condition)][seed] = {metric: v}
    cell = defaultdict(dict)
    for r in rows:
        cell[(r['family'], r['condition'])][r['seed']] = {m: r[m] for m in METRICS}

    families = sorted({k[0] for k in cell})
    conditions = sorted({k[1] for k in cell})
    randeval = [c for c in conditions if c.startswith('randeval')]
    anchors = [c for c in conditions if c in ('none', 'all')]
    print('families:', families)
    print('conditions:', len(conditions), '| randeval:', len(randeval), '| anchors:', anchors)

    # 1) 长表：family × condition × seed × 3 指标
    with open(outdir / 'stage5_matrix_long.csv', 'w', newline='') as h:
        w = csv.writer(h)
        w.writerow(['family', 'condition', 'seed'] + list(METRICS))
        for (fam, cond), seeds in sorted(cell.items()):
            for seed, vals in sorted(seeds.items()):
                w.writerow([fam, cond, seed] + [vals[m] for m in METRICS])

    # 2) 逐条件配对表 + 3) 池化 Wilcoxon
    paired_rows, wil_rows = [], []
    for fam in families:
        if fam == BASELINE:
            continue
        for cond in conditions:
            vals, base = cell.get((fam, cond), {}), cell.get((BASELINE, cond), {})
            paired = [(s, vals[s], base[s]) for s in vals if s in base]
            if not paired:
                continue
            rec = {'family': fam, 'condition': cond, 'n_paired': len(paired)}
            for m in METRICS:
                deltas = {s: v[m] - b[m] for s, v, b in paired}
                rec[f'mean_{m}'] = sum(v[m] for _, v, _ in paired) / len(paired)
                rec[f'sen_mean_{m}'] = sum(b[m] for _, _, b in paired) / len(paired)
                rec[f'delta_{m}'] = sum(deltas.values()) / len(deltas)
                rec[f'delta_{m}_per_seed'] = json.dumps(deltas)
            paired_rows.append(rec)

        for scope_name, conds in (('randeval12', randeval),
                                  ('anchors2', anchors),
                                  ('all14', conditions)):
            for m in METRICS:
                deltas = []
                for cond in conds:
                    vals, base = cell.get((fam, cond), {}), cell.get((BASELINE, cond), {})
                    deltas += [vals[s][m] - base[s][m]
                               for s in vals if s in base]
                deltas = [d for d in deltas if d is not None]
                if not deltas:
                    continue
                try:
                    stat, p = wilcoxon(deltas)
                except ValueError:
                    stat, p = float('nan'), float('nan')
                wil_rows.append({'family': fam, 'metric': m, 'scope': scope_name,
                                 'n_pairs': len(deltas),
                                 'median_delta': sorted(deltas)[len(deltas) // 2],
                                 'mean_delta': sum(deltas) / len(deltas),
                                 'wilcoxon_W': stat, 'p_value': p})

    if paired_rows:
        with open(outdir / 'stage5_paired_vs_sensor.csv', 'w', newline='') as h:
            w = csv.DictWriter(h, fieldnames=list(paired_rows[0]))
            w.writeheader()
            w.writerows(paired_rows)
    if wil_rows:
        with open(outdir / 'stage5_wilcoxon_vs_sensor.csv', 'w', newline='') as h:
            w = csv.DictWriter(h, fieldnames=list(wil_rows[0]))
            w.writeheader()
            w.writerows(wil_rows)

    # 终端摘要：TT 配对 delta（all14 池化）
    print('\n== TT delta vs tarl_sensor（all14 池化）==')
    for r in wil_rows:
        if r['metric'] == 'travel_time' and r['scope'] == 'all14':
            print(f"{r['family']:12s} n={r['n_pairs']:3d} mean={r['mean_delta']:+7.2f} "
                  f"median={r['median_delta']:+7.2f} p={r['p_value']:.2e}")
    print(f"\nwritten: {outdir}/stage5_matrix_long.csv, stage5_paired_vs_sensor.csv, stage5_wilcoxon_vs_sensor.csv")


if __name__ == '__main__':
    main()
